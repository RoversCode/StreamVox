"""验证 GGUF 生成调度、ONNX 声码器状态和请求生命周期。"""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from streamvox.engine import TTSEngine
from streamvox.model_bundle import ModelBundleError, write_local_manifest
from streamvox.models.base import ExecutionPlan, ExecutionTarget, RuntimeInfo
from streamvox.models.breeze.adapter import BreezeTTS2GGUFAdapter
from streamvox.models.breeze.ort import OrtGraph
from streamvox.models.breeze.prompt import BreezePromptAsset
from streamvox.models.breeze.prefill import BackbonePrefill
from streamvox.models.breeze.runtime import BreezeRuntime
from streamvox.models.breeze.sampling import sample_token
from streamvox.models.breeze.templates import build_prompt_embeddings
from streamvox.models.registry import ModelRegistry


class FakeTokenizer:
    """记录特殊符号边界，并让带指令的提示长于无条件提示。"""

    def __init__(self):
        self.calls, self.rendered, self.tokens = [], {}, {}

    def encode(self, text, *, add_special_tokens):
        """返回能跟踪分段和指令长度的稳定编号。

        Args:
            text: 当前模板文本。
            add_special_tokens: 是否执行首轮特殊符号包装。
        Returns:
            含有编号列表的编码结果。
        """
        self.calls.append((text, add_special_tokens))
        token = self.tokens.setdefault(text, len(self.tokens) + 11)
        self.rendered[token] = text
        return SimpleNamespace(ids=[2, token, *([token, token] if "<ins_bos>" in text else []), 1])

    def decode(self, ids, *, skip_special_tokens):
        """保留起止标记，观察官方解码再编码边界。

        Args:
            ids: 首轮编码编号。
            skip_special_tokens: 是否省略特殊符号。
        Returns:
            含起止标记的原始模板。
        """
        assert skip_special_tokens is False
        return f"<bos>{self.rendered[ids[1]]}<eos>"


class FakeGraph:
    """提供文本、音频查表、条件投影及连续声码器的可观察替身。

    Args:
        kind: 当前 ONNX 组件的职责。
    """

    def __init__(self, kind):
        self.kind, self.calls = kind, []

    def warmup(self):
        """模拟加载阶段的声码器捕获，不生成请求音频。"""

    def close(self):
        """模拟释放声码器会话。"""

    def run(self, feeds):
        """记录独立输入快照，并生成可辨识的组件结果。

        Args:
            feeds: 当前 ONNX 图输入。
        Returns:
            与对应生产图接口一致的结果。
        """
        self.calls.append({name: value.copy() for name, value in feeds.items()})
        if self.kind == "text":
            ids = feeds["input_ids"]
            return {"embeddings": np.broadcast_to(ids[:, 1, None, None], (*ids.shape, 4)).astype(np.float32).copy()}
        if self.kind == "audio":
            return {"embeddings": np.full((*feeds["codes"].shape[:2], 4), 7, dtype=np.float32)}
        if self.kind == "conditioner":
            return {"embeddings": feeds["backbone_hidden"][:, :2] * 2}
        if self.kind == "encoder":
            return {"codes": np.full((1, 2, 16), 9, dtype=np.int64)}
        previous = feeds["codec_state"]
        level = float(previous.item()) + 1
        return {"audio": np.tile([[level, -level]], (1, 960)).astype(np.float32),
                "next_codec_state": previous + 1}


class FakeLlamaDecoder:
    """模拟独立序列和原生缓存生命周期，不加载 GGUF 或访问显卡。

    Args:
        model_path: 用于识别主干或深度解码器的模型路径。
        parameters: 原生构造器接受的后端、容量和输出设置。
    """

    def __init__(self, model_path, **parameters):
        self.kind = Path(model_path).stem
        self.parameters = parameters
        self.backend = parameters.get("backend", "cpu")
        self.n_ctx = parameters["n_ctx"]
        self.first_tokens = (3, 4, 2051)
        self.calls, self.reserves, self.sequences = [], [], {}
        self.reset_count = self.close_count = 0

    def reserve(self, n_ctx):
        """记录按请求扩容，保持既有模型实例。

        Args:
            n_ctx: 需要的每分支容量。
        """
        self.reserves.append(n_ctx)
        self.n_ctx = max(self.n_ctx, n_ctx)

    def reset(self):
        """清空所有分支的当前位置，保留历史调用记录。"""
        self.reset_count += 1
        self.sequences.clear()

    def close(self):
        """记录显式资源释放。"""
        self.close_count += 1
        self.sequences.clear()

    def remove_sequence(self, sequence, start, end=-1):
        """模拟请求收尾时移除目标后缀或整个分支。"""
        if start == 0:
            self.sequences.pop(sequence, None)
            return
        state = self.sequences[sequence]
        state["position"] = min(state["position"], start)
        state["step"] = 0

    def decode(self, *, embeddings=None, tokens=None, positions, sequence_ids, output=True,
               logit_start=0, logit_count=None):
        """推进独立分支，生成指定词表区间的可预测分数。

        Args:
            embeddings: 主干或深度条件嵌入。
            tokens: 带码本偏移的整数编号。
            positions: 各分支绝对起始位置。
            sequence_ids: 各分支独立序列编号。
            output: 是否需要输出分数及隐藏状态。
            logit_start: 联合词表的输出起点。
            logit_count: 所需输出区间长度。
        Returns:
            已复制的分数和主干最终隐藏状态。
        """
        values = embeddings if embeddings is not None else tokens
        call = {"positions": list(positions), "sequence_ids": list(sequence_ids), "output": output,
                "logit_start": logit_start, "logit_count": logit_count, "epoch": self.reset_count}
        call["embeddings" if embeddings is not None else "tokens"] = values.copy()
        self.calls.append(call)
        scores, hidden = [], []
        for row, (position, sequence) in enumerate(zip(positions, sequence_ids)):
            state = self.sequences.setdefault(sequence, {"position": 0, "step": 0})
            assert position == state["position"], "分支缓存位置发生串扰或未重置"
            state["position"] += values.shape[1]
            if self.kind == "backbone":
                if not output:
                    continue
                token = self.first_tokens[min(state["step"], len(self.first_tokens) - 1)]
                state["step"] += 1
                logits = np.full(2052, -1000, dtype=np.float32)
                logits[token] = 100
                logits[0] = values[row, -1, 0]
                hidden.append(values[row, -1])
            else:
                if not output:
                    state["condition"] = float(values[row, -1, 0])
                    continue
                logits = np.full(logit_count, -1000, dtype=np.float32)
                logits[5 + position] = 100
                logits[0] = state["condition"] / 100
                # 保留码拥有更高分数，强制验证调度仍屏蔽三枚无效音频码。
                logits[2048:2051] = 1000
            scores.append(logits)
        if not output:
            return {"hidden": None, "logits": None}
        return {"hidden": np.stack(hidden) if hidden else None, "logits": np.stack(scores)}


@pytest.fixture
def fake_runtime():
    """绕过模型加载，保留生产调度及所有请求锁和缓存方法。

    Returns:
        可观察原生及 ONNX 调用的运行时。
    """
    runtime = BreezeRuntime.__new__(BreezeRuntime)
    runtime._lock, runtime._closed, runtime._reference_text_cache = Lock(), False, {}
    runtime.ckpt, runtime.sample_rate = Path("unused"), 24000
    runtime.hidden_size, runtime.num_codebooks, runtime.codebook_size = 4, 16, 2048
    runtime.onnx_backend = "cpu"
    runtime.gguf_backend = "cpu"
    runtime._warmed_up = False
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze-test-codec")
    runtime._codec_executor = executor
    runtime.vocab_size = runtime.eos_token = 2051
    runtime.config = {"codebook_eos_token_id": 0, "eos_token_id": 1, "vocab_size": 2051}
    runtime.n_ctx = 10000
    runtime.codec_metadata = {"state_specs": [{"name": "codec_state", "shape": [1, 1]}]}
    runtime.tokenizer, runtime.text_encoder = FakeTokenizer(), FakeGraph("text")
    runtime.audio_embedding, runtime.depth_conditioner = FakeGraph("audio"), FakeGraph("conditioner")
    runtime.backbone = FakeLlamaDecoder(Path("backbone.gguf"), n_ctx=256)
    runtime.depth = FakeLlamaDecoder(Path("depth.gguf"), n_ctx=16)
    runtime._prefill = BackbonePrefill(runtime.backbone)
    runtime.decoder, runtime.audio_encoder = FakeGraph("decoder"), None
    yield runtime
    executor.shutdown(wait=True)


@pytest.fixture
def empty_prompt():
    """提供无参考的合法 GGUF 模型提示。"""
    return BreezePromptAsset(role_name="测试", model="breeze-tts-2-gguf", sample_rate=24000)


@pytest.fixture
def reference_prompt():
    """提供完整转录和两帧十六码本，验证跨请求转录复用。"""
    return BreezePromptAsset(role_name="缓存参考", model="breeze-tts-2-gguf", sample_rate=24000,
                             prompt_text="共享参考转录。", audio_codes=np.full((2, 16), 9, dtype=np.int64))


@pytest.fixture
def runtime_loader(tmp_path, monkeypatch):
    """保留真实清单验证，以替身截获两套原生模型和 ONNX 组件加载。

    Args:
        tmp_path: 隔离的临时模型目录。
        monkeypatch: 模型与后端入口替换工具。
    Returns:
        可指定资产及验收报告的加载器和构造记录。
    """
    import streamvox.models.breeze.runtime as module

    graphs, native, selections, directories = [], [], [], []
    names = {"text_encoder": "text", "audio_embedding": "audio", "depth_conditioner": "conditioner",
             "decoder_step": "decoder", "audio_encoder": "encoder"}

    def create_graph(path, backend, device_id=0):
        """记录独立组件的实际执行后端。"""
        graphs.append((path.name, backend, device_id))
        return FakeGraph(names[path.stem])

    monkeypatch.setattr(module, "OrtGraph", create_graph)
    monkeypatch.setattr(module, "CudaStreamingDecoder", lambda path, device_id: create_graph(path, "cuda", device_id))
    monkeypatch.setattr(module, "Tokenizer", SimpleNamespace(from_file=lambda _path: FakeTokenizer()))

    def load(*, backend="cpu", onnx_backend="cpu", report=None, omitted_assets=(), n_ctx=32,
             selected_backend=None, execution_plan=None, fail_component=None):
        """构造具有完整清单和已验收 GGUF 元数据的隔离模型。"""
        model_dir = tmp_path / f"bundle_{len(directories)}"
        directories.append(model_dir)
        ckpt = model_dir / "ckpt"
        ckpt.mkdir(parents=True)
        documents = {
            "config.json": {"hidden_size": 4, "num_codebooks": 16, "codec_config": {"codebook_size": 2048},
                            "vocab_size": 2051, "eos_token_id": 1, "codebook_eos_token_id": 0,
                            "max_position_embeddings": 4, "backbone_config": {"max_position_embeddings": 1024}},
            "codec_metadata.json": {"state_specs": []},
            "export-report.json": {"tiny": False, "generation": {"validated": True, "precision": "float32"}, **(report or {})},
        }
        assets = set(documents) | {"tokenizer.json", "text_encoder.onnx", "audio_embedding.onnx",
                                  "backbone.gguf", "depth.gguf", "depth_conditioner.onnx", "audio_encoder.onnx", "decoder_step.onnx"}
        assets.difference_update(omitted_assets)
        for name in assets:
            if name in documents:
                (ckpt / name).write_text(json.dumps(documents[name]), encoding="utf-8")
            else:
                (ckpt / name).write_bytes(b"test-model-resource")
        write_local_manifest(model="breeze-tts-2-gguf", root_dir=model_dir,
                             asset_paths=[ckpt / name for name in assets], adapter="BreezeTTS2GGUFAdapter")

        def create_native(path, **parameters):
            """允许模拟第二组件加载失败，检查第一组件释放。"""
            if path.stem == fail_component:
                raise RuntimeError("原生组件加载失败")
            decoder = FakeLlamaDecoder(path, **parameters)
            native.append(decoder)
            return decoder

        def select(**parameters):
            """记录原生后端请求，并模拟指定解析结果。"""
            selections.append(parameters)
            return SimpleNamespace(backend=selected_backend or backend, gpu_index=None if backend == "cpu" else 2)

        monkeypatch.setattr(module, "LlamaDecoder", create_native)
        monkeypatch.setattr(module, "select_llama_backend", select)
        monkeypatch.setattr(module, "resolve_backend", lambda _target: (onnx_backend, 2))
        plan = execution_plan or ExecutionPlan(raw_device="cpu", targets={
            "llama": ExecutionTarget("llama", "cpu" if backend == "cpu" else "gpu", 2, backend),
            "onnx": ExecutionTarget("onnx", "cpu" if onnx_backend == "cpu" else "gpu", 2, onnx_backend),
        })
        return BreezeRuntime(model_dir=model_dir, execution_plan=plan, n_ctx=n_ctx)

    return SimpleNamespace(load=load, graphs=graphs, native=native, selections=selections)


def test_loader_uses_complete_generation_models_and_independent_backends(runtime_loader):
    """生成使用显式 GGUF 后端，非逐帧编码和条件投影固定在处理器。"""
    runtime = runtime_loader.load(backend="cuda", onnx_backend="dml", n_ctx=512)
    try:
        assert runtime.eos_token == 2051 and runtime.n_ctx == 512
        assert runtime.audio_encoder is None
        assert runtime_loader.graphs == [("text_encoder.onnx", "cpu", 0), ("audio_embedding.onnx", "cpu", 0),
                                         ("depth_conditioner.onnx", "cpu", 0), ("decoder_step.onnx", "dml", 2)]
        main, depth = runtime_loader.native
        assert main.parameters == {"backend": "cuda", "device_id": 2, "n_ctx": 256, "n_seq_max": 2, "embeddings": True}
        assert depth.parameters == {"backend": "cuda", "device_id": 2, "n_ctx": 16, "n_seq_max": 2, "embeddings": False}
        assert runtime.runtime_info.model == "breeze-tts-2-gguf"
        assert runtime.runtime_info.metadata["generation_backend"] == "llama.cpp"
        assert runtime.runtime_info.metadata["kv_precision"] == "float32"
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("missing", ["backbone.gguf", "depth.gguf", "depth_conditioner.onnx", "audio_encoder.onnx"])
def test_loader_requires_complete_manifest(runtime_loader, missing):
    """任一必需 GGUF 或 ONNX 资产缺失时都不能加载部分模型。"""
    with pytest.raises(ModelBundleError, match="缺少运行资产"):
        runtime_loader.load(omitted_assets=(missing,))
    assert runtime_loader.native == [] and runtime_loader.graphs == []


@pytest.mark.parametrize("report,error", [({"generation": {"validated": False}}, "数值验收"),
                                         ({"tiny": True}, "随机小模型"),
                                         ({"files": {"undeclared.data": {}}}, "未纳入清单"),
                                         ({"files": {"../outside": {}}}, "路径越界")])
def test_loader_rejects_unvalidated_or_undeclared_assets(runtime_loader, report, error):
    """数值验收和全部外部资产校验必须发生在组件加载之前。"""
    with pytest.raises((ModelBundleError, ValueError), match=error):
        runtime_loader.load(report=report)
    assert runtime_loader.native == []


@pytest.mark.parametrize("n_ctx", [0, -1, 1025])
def test_loader_rejects_invalid_context_capacity(runtime_loader, n_ctx):
    """上下文约束必须来自实际内嵌 Qwen3 配置。"""
    with pytest.raises(ValueError, match="n_ctx"):
        runtime_loader.load(n_ctx=n_ctx)


def test_loader_never_silently_replaces_explicit_gpu(runtime_loader):
    """显式 CUDA 解析成处理器时拒绝继续加载。"""
    with pytest.raises(RuntimeError, match="未满足请求"):
        runtime_loader.load(backend="cuda", selected_backend="cpu")
    assert runtime_loader.native == []


def test_explicit_cpu_generation_keeps_gpu_codec_independent(runtime_loader):
    """用户可明确使用处理器生成，同时保留显卡声码器。"""
    plan = ExecutionPlan(raw_device="gpu:2", targets={
        "llama": ExecutionTarget("llama", "gpu", 2, "cpu"),
        "onnx": ExecutionTarget("onnx", "gpu", 2, "cuda"),
    })
    runtime = runtime_loader.load(backend="cpu", onnx_backend="cuda", execution_plan=plan)
    try:
        assert runtime.gguf_backend == "cpu" and runtime.onnx_backend == "cuda"
        assert all(decoder.parameters["backend"] == "cpu" for decoder in runtime_loader.native)
        assert ("decoder_step.onnx", "cuda", 2) in runtime_loader.graphs
    finally:
        runtime.shutdown()


def test_loader_failure_closes_already_loaded_native_model(runtime_loader):
    """第二组件加载失败时释放已加载的主干资源。"""
    with pytest.raises(RuntimeError, match="加载失败"):
        runtime_loader.load(fail_component="depth")
    assert len(runtime_loader.native) == 1 and runtime_loader.native[0].close_count == 1


def test_reference_encoder_is_lazy_cpu_and_reused(runtime_loader):
    """参考编码不占用生成设备，首次使用后复用同一会话。"""
    runtime = runtime_loader.load()
    try:
        for _ in range(2):
            codes = runtime.encode_reference(np.ones(24, np.float32))
            assert codes.shape == (2, 16) and codes.dtype == np.int64
        assert runtime_loader.graphs.count(("audio_encoder.onnx", "cpu", 0)) == 1
        assert len(runtime.audio_encoder.calls) == 2
    finally:
        runtime.shutdown()


def test_template_independently_encodes_reference_and_target_with_zero_audio_eos(fake_runtime):
    """参考转录和目标文本独立编码，音频结束帧所有码本均为零。"""
    prompt = BreezePromptAsset(
        role_name="参考", model="breeze-tts-2-gguf", sample_rate=24000,
        prompt_text="[笑]完整转录。", audio_codes=np.full((2, 16), 9, dtype=np.int64),
    )
    branches, reference_length = build_prompt_embeddings(
        fake_runtime.tokenizer, fake_runtime.text_encoder, fake_runtime.audio_embedding,
        text="目标<|sigh|>。", prompt=prompt, instructions=["自然女声。"], hidden_size=4,
        eos_code=0, num_codebooks=16,
    )
    embeddings = branches[0]
    assert fake_runtime.tokenizer.calls == [
        ("[S0][笑]完整转录。", True), ("<bos>[S0][笑]完整转录。<eos>", False),
        ("[S0]<ins_bos>自然女声。<ins_eos>目标<|sigh|>。", True),
        ("<bos>[S0]<ins_bos>自然女声。<ins_eos>目标<|sigh|>。<eos>", False),
    ]
    assert len(fake_runtime.text_encoder.calls) == 2
    for call in fake_runtime.text_encoder.calls:
        np.testing.assert_array_equal(call["attention_mask"], np.ones_like(call["input_ids"]))
    codes = fake_runtime.audio_embedding.calls[0]["codes"]
    np.testing.assert_array_equal(codes[0, :2], prompt.audio_codes)
    np.testing.assert_array_equal(codes[0, -1], np.zeros(16, dtype=np.int64))
    assert embeddings.shape == (1, 11, 4)
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(embeddings[0, 3:6], np.full((3, 4), 7))


def test_reference_cache_reuses_encoding_across_cfg_branches_and_requests(fake_runtime, reference_prompt):
    """相同参考只编码一次，且所有预填充逐点等于不使用缓存的结果。"""
    requests = []
    for _ in range(2):
        requests.append(list(fake_runtime.synthesize(
            "跨分支目标。", reference_prompt, instruction="温柔女声。", cfg_scale=4,
            max_new_tokens=1, do_sample=False,
        )))
    assert len(fake_runtime.text_encoder.calls) == 3
    reference_ids = fake_runtime.text_encoder.calls[0]["input_ids"]
    assert sum(np.array_equal(call["input_ids"], reference_ids) for call in fake_runtime.text_encoder.calls) == 1
    assert len(fake_runtime._reference_text_cache) == 1
    assert len(fake_runtime.backbone.calls) == 4

    uncached_encoder = FakeGraph("text")
    uncached_audio = FakeGraph("audio")
    expected = [
        build_prompt_embeddings(
            fake_runtime.tokenizer, uncached_encoder, uncached_audio,
            text="跨分支目标。", prompt=reference_prompt, instructions=[instruction],
            hidden_size=4, eos_code=0, num_codebooks=16, reference_cache=None,
        )[0][0]
        for instruction in ("温柔女声。", "")
    ]
    for index, call in enumerate(fake_runtime.backbone.calls):
        start = call["positions"][0]
        np.testing.assert_array_equal(call["embeddings"], expected[index % 2][:, start:])
    np.testing.assert_array_equal(np.concatenate(requests[0]), np.concatenate(requests[1]))


def test_target_text_is_reencoded_even_when_its_ids_match_cached_reference(fake_runtime, reference_prompt):
    """目标与参考完全相同时，也只能跳过参考段而不能复用目标段编码。"""
    for _ in range(2):
        list(fake_runtime.synthesize(reference_prompt.prompt_text, reference_prompt, max_new_tokens=1, do_sample=False))
    assert len(fake_runtime.text_encoder.calls) == 3
    first_ids = fake_runtime.text_encoder.calls[0]["input_ids"]
    assert all(np.array_equal(call["input_ids"], first_ids) for call in fake_runtime.text_encoder.calls)
    assert len(fake_runtime._reference_text_cache) == 1


def test_switching_reference_transcript_replaces_the_single_cached_entry(fake_runtime, reference_prompt):
    """角色切换替换最近转录，切回旧转录时重新编码且缓存仍只保留一条。"""
    list(fake_runtime.synthesize("相同目标。", reference_prompt, max_new_tokens=1, do_sample=False))
    first_key = next(iter(fake_runtime._reference_text_cache))
    other_prompt = replace(reference_prompt, prompt_text="另一段完整转录。")
    list(fake_runtime.synthesize("相同目标。", other_prompt, max_new_tokens=1, do_sample=False))
    assert len(fake_runtime._reference_text_cache) == 1
    assert first_key not in fake_runtime._reference_text_cache
    second_key = next(iter(fake_runtime._reference_text_cache))
    assert second_key == tuple(fake_runtime.text_encoder.calls[2]["input_ids"][0])
    list(fake_runtime.synthesize("相同目标。", reference_prompt, max_new_tokens=1, do_sample=False))
    assert len(fake_runtime.text_encoder.calls) == 6
    assert list(fake_runtime._reference_text_cache) == [first_key]


def test_oversized_reference_encoding_is_not_cached_and_evicts_previous_entry(fake_runtime, reference_prompt, monkeypatch):
    """超容量的新转录不能驻留，也不能让旧角色的缓存继续占用空间。"""
    import streamvox.models.breeze.templates as templates_module

    list(fake_runtime.synthesize("目标。", reference_prompt, max_new_tokens=1, do_sample=False))
    cached_size = next(iter(fake_runtime._reference_text_cache.values())).nbytes
    monkeypatch.setattr(templates_module, "REFERENCE_CACHE_LIMIT_BYTES", cached_size - 1)
    other_prompt = replace(reference_prompt, prompt_text="超出缓存容量的参考。")
    for _ in range(2):
        list(fake_runtime.synthesize("目标。", other_prompt, max_new_tokens=1, do_sample=False))
        assert fake_runtime._reference_text_cache == {}
    assert len(fake_runtime.text_encoder.calls) == 6


def test_shutdown_clears_reference_text_cache(fake_runtime, reference_prompt):
    """运行时关闭时释放跨请求保留的参考文本编码。"""
    list(fake_runtime.synthesize("目标。", reference_prompt, max_new_tokens=1, do_sample=False))
    assert len(fake_runtime._reference_text_cache) == 1
    fake_runtime.shutdown()
    assert fake_runtime._reference_text_cache == {}


def test_full_frame_preserves_fifteen_heads_and_projected_condition(fake_runtime, empty_prompt):
    """位置零注入投影条件，其后完整保留十五个码本输入与输出头。"""
    chunks = list(fake_runtime.synthesize("测试。", empty_prompt, max_new_tokens=3, chunk_frames=1, do_sample=False))
    assert len(chunks) == 2 and all(chunk.dtype == np.float32 for chunk in chunks)
    assert [chunk[0] for chunk in chunks] == [1, 2]
    assert [call["position"].item() for call in fake_runtime.decoder.calls] == [0, 1]
    for frame_index, codec in enumerate(fake_runtime.decoder.calls):
        frame = codec["codes"][0, 0]
        assert frame.shape == (16,) and frame.min() >= 0 and frame.max() < 2048
        assert frame[0] == (3, 4)[frame_index]
        calls = fake_runtime.depth.calls[frame_index * 16:(frame_index + 1) * 16]
        assert calls[0]["positions"] == [0] and calls[0]["output"] is False
        hidden = fake_runtime.depth_conditioner.calls[frame_index]["backbone_hidden"]
        np.testing.assert_array_equal(calls[0]["embeddings"], (hidden[:, :2] * 2)[:, None])
        for position, call in enumerate(calls[1:], 1):
            offset = (position - 1) * 2051
            assert call["positions"] == [position] and call["sequence_ids"] == [0]
            assert call["logit_start"] == offset and call["logit_count"] == 2051
            assert call["tokens"].item() == offset + frame[position - 1]
        assert len({call["epoch"] for call in calls}) == 1
    assert fake_runtime.depth.calls[0]["epoch"] != fake_runtime.depth.calls[16]["epoch"]


@pytest.mark.parametrize("first_token,frames", [(2051, 0), (1, 1)])
@pytest.mark.parametrize("backend", ["cpu", "cuda", "dml"])
def test_only_extra_backbone_class_terminates_audio(fake_runtime, empty_prompt, first_token, frames, backend):
    """额外类别才是语音结束符，文本结束编号一仍是有效音频码。"""
    fake_runtime.onnx_backend = backend
    fake_runtime.backbone.first_tokens = (first_token,)
    chunks = list(fake_runtime.synthesize("结束边界。", empty_prompt, max_new_tokens=1, do_sample=False))
    assert len(chunks) == frames and len(fake_runtime.decoder.calls) == frames
    assert len(fake_runtime.depth.calls) == frames * 16
    if frames:
        assert fake_runtime.decoder.calls[0]["codes"][0, 0, 0] == first_token


@pytest.mark.parametrize("backend", ["cpu", "cuda", "dml"])
def test_default_stream_groups_two_eighty_millisecond_frames(fake_runtime, empty_prompt, backend):
    """默认首包由两帧组成，在原始采样率下为一百六十毫秒。"""
    fake_runtime.onnx_backend = backend
    fake_runtime.backbone.first_tokens = (3, 4, 5, 6, 2051)
    chunks = list(fake_runtime.synthesize("四帧。", empty_prompt, max_new_tokens=5, do_sample=False))
    assert [chunk.size for chunk in chunks] == [3840, 3840]
    assert chunks[0].size / fake_runtime.sample_rate == 0.16
    assert np.concatenate(chunks).reshape(4, 1920)[:, 0].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("chunk_frames", [1, 2, 25])
@pytest.mark.parametrize("backend", ["cpu", "cuda", "dml"])
def test_first_chunk_waits_only_for_its_own_complete_frames(fake_runtime, empty_prompt, backend, chunk_frames):
    """首包凑齐完整帧后立即返回，后续仍按原分组和顺序输出。

    Args:
        fake_runtime: 保留真实生成调度的处理器替身。
        empty_prompt: 不含参考音频的提示资产。
        backend: 需要验证的声码器调度分支。
        chunk_frames: 首包需要包含的完整帧数。
    Returns:
        首包没有提前生成额外帧且剩余音频完整时正常返回。
    """
    fake_runtime.onnx_backend = backend
    fake_runtime.backbone.first_tokens = (3,) * (chunk_frames + 1) + (2051,)
    stream = fake_runtime.synthesize(
        "首包边界。", empty_prompt, chunk_frames=chunk_frames,
        max_new_tokens=chunk_frames + 2, do_sample=False,
    )
    try:
        first = next(stream)
        assert first.size == chunk_frames * 1920
        assert len(fake_runtime.depth.calls) == chunk_frames * 16
        assert len(fake_runtime.backbone.calls) == chunk_frames
        assert len(fake_runtime.audio_embedding.calls) == chunk_frames - 1
        assert [call["position"].item() for call in fake_runtime.decoder.calls] == list(range(chunk_frames))
        chunks = [first, *stream]
    finally:
        stream.close()
    assert [chunk.size for chunk in chunks] == [chunk_frames * 1920, 1920]
    assert np.concatenate(chunks).reshape(chunk_frames + 1, 1920)[:, 0].tolist() == list(range(1, chunk_frames + 2))


@pytest.mark.parametrize("stream,chunk_frames,chunk_sizes", [(True, 1, [1920, 1920]), (True, 25, [3840]), (False, 1, [3840])])
@pytest.mark.parametrize("backend", ["cpu", "cuda", "dml"])
def test_frame_order_and_last_frame_avoid_redundant_generation(fake_runtime, empty_prompt, stream, chunk_frames, chunk_sizes, backend):
    """流式与整段输出保持顺序，达到上限后不再预计算下一帧。"""
    fake_runtime.onnx_backend = backend
    chunks = list(fake_runtime.synthesize("两帧。", empty_prompt, stream=stream,
                                         max_new_tokens=2, chunk_frames=chunk_frames, do_sample=False))
    assert [chunk.size for chunk in chunks] == chunk_sizes
    assert np.concatenate(chunks).reshape(2, 1920)[:, 0].tolist() == [1, 2]
    assert len(fake_runtime.backbone.calls) == 2 and len(fake_runtime.audio_embedding.calls) == 1
    assert fake_runtime.backbone.reserves == [256]


@pytest.mark.parametrize("backend", ["cpu", "cuda", "dml"])
def test_reused_requests_reset_native_sequences_and_codec_state(fake_runtime, empty_prompt, backend):
    """主干、逐帧深度和声码器状态均不能带入下一请求。"""
    fake_runtime.onnx_backend = backend
    results = [list(fake_runtime.synthesize("复用。", empty_prompt, max_new_tokens=2, do_sample=False)) for _ in range(2)]
    np.testing.assert_array_equal(np.concatenate(results[0]), np.concatenate(results[1]))
    assert fake_runtime.backbone.reset_count == 4 and fake_runtime.depth.reset_count == 8
    assert fake_runtime.backbone.sequences == {} and fake_runtime.depth.sequences == {}
    assert [call["codec_state"].item() for call in fake_runtime.decoder.calls] == [0, 1, 0, 1]


@pytest.mark.parametrize("failure", ["cancel", "codec", "backbone"])
@pytest.mark.parametrize("backend", ["cpu", "cuda", "dml"])
def test_request_exit_releases_lock_and_discards_sequences(fake_runtime, empty_prompt, monkeypatch, failure, backend):
    """取消、声码器异常和主干异常都必须允许下一次完整请求。"""
    fake_runtime.onnx_backend = backend
    if failure == "cancel":
        stream = fake_runtime.synthesize("取消。", empty_prompt, max_new_tokens=2, chunk_frames=1, do_sample=False)
        assert next(stream)[0] == 1 and fake_runtime._lock.locked()
        stream.close()
    else:
        target, method = (fake_runtime.decoder, "run") if failure == "codec" else (fake_runtime.backbone, "decode")
        original = getattr(target, method)

        def fail(*_args, **_kwargs):
            """模拟生成阶段组件失败。"""
            raise RuntimeError("模型推理失败")

        monkeypatch.setattr(target, method, fail)
        with pytest.raises(RuntimeError, match="推理失败"):
            list(fake_runtime.synthesize("失败。", empty_prompt, max_new_tokens=1, do_sample=False))
        monkeypatch.setattr(target, method, original)
    assert not fake_runtime._lock.locked()
    assert fake_runtime.backbone.sequences == {} and fake_runtime.depth.sequences == {}
    chunks = list(fake_runtime.synthesize("恢复。", empty_prompt, max_new_tokens=1, do_sample=False))
    assert chunks[0][0] == 1 and fake_runtime.decoder.calls[-1]["codec_state"].item() == 0


@pytest.mark.parametrize("failure", ["cancel", "backbone"])
def test_cuda_exit_waits_for_inflight_codec_before_clearing_sequences(fake_runtime, empty_prompt, monkeypatch, failure):
    """取消或反馈失败时等待在途声码器，再清原生状态并允许新请求。"""
    fake_runtime.onnx_backend = "cuda"
    fake_runtime.backbone.first_tokens = (3, 4, 5, 2051)
    entered, release, completed, finished = Event(), Event(), Event(), Event()
    premature_cleanup, errors = [], []
    original_codec = fake_runtime.decoder.run
    original_backbone = fake_runtime.backbone.decode
    original_reset = fake_runtime.backbone.reset

    def blocked_codec(feeds):
        """将目标帧留在后台，观察退出是否等待状态写入完成。"""
        blocked_position = 2 if failure == "cancel" else 0
        if feeds["position"].item() != blocked_position:
            return original_codec(feeds)
        entered.set()
        try:
            if not release.wait(3):
                raise RuntimeError("测试未释放在途声码器")
            return original_codec(feeds)
        finally:
            completed.set()

    def fail_feedback(**feeds):
        """保留预填充，在首个反馈调用中制造原生执行异常。"""
        if feeds["positions"] != [0]:
            assert entered.wait(3)
            raise RuntimeError("反馈推理失败")
        return original_backbone(**feeds)

    def observe_reset():
        """记录任何发生在后台写入结束之前的原生状态清理。"""
        if entered.is_set() and not completed.is_set():
            premature_cleanup.append(True)
        original_reset()

    def consume_or_close():
        """在独立调用线程中进入请求清理路径。"""
        try:
            if failure == "cancel":
                stream.close()
            else:
                list(stream)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(fake_runtime.decoder, "run", blocked_codec)
    monkeypatch.setattr(fake_runtime.backbone, "reset", observe_reset)
    if failure == "backbone":
        monkeypatch.setattr(fake_runtime.backbone, "decode", fail_feedback)
    stream = fake_runtime.synthesize("在途退出。", empty_prompt, max_new_tokens=3, chunk_frames=1, do_sample=False)
    if failure == "cancel":
        assert next(stream)[0] == 1
        # 第二包恢复原流水线，取消时仍有下一帧正在后台推进状态。
        assert next(stream)[0] == 2
    worker = Thread(target=consume_or_close)
    worker.start()
    try:
        assert entered.wait(3)
        assert not finished.wait(0.05)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and completed.is_set() and finished.is_set()
    assert premature_cleanup == []
    assert [str(error) for error in errors] == ([] if failure == "cancel" else ["反馈推理失败"])
    assert not fake_runtime._lock.locked()
    assert fake_runtime.backbone.sequences == fake_runtime.depth.sequences == {}
    monkeypatch.setattr(fake_runtime.decoder, "run", original_codec)
    monkeypatch.setattr(fake_runtime.backbone, "decode", original_backbone)
    chunks = list(fake_runtime.synthesize("恢复。", empty_prompt, max_new_tokens=1, do_sample=False))
    assert chunks[0][0] == 1 and fake_runtime.decoder.calls[-1]["codec_state"].item() == 0


def test_context_rejection_precedes_native_mutation(fake_runtime, empty_prompt):
    """提示和生成预算不足时，不得扩容或运行任何生成组件。"""
    fake_runtime.n_ctx = 5
    with pytest.raises(ValueError, match="n_ctx"):
        list(fake_runtime.synthesize("超长。", empty_prompt, max_new_tokens=3, do_sample=False))
    assert fake_runtime.backbone.calls == fake_runtime.depth.calls == fake_runtime.decoder.calls == []
    assert fake_runtime.backbone.reserves == [] and not fake_runtime._lock.locked()


@pytest.mark.parametrize("scale,branch_count", [(0.0, 1), (1.0, 1), (4.0, 2)])
def test_cfg_preserves_independent_prompt_positions_and_batched_feedback(fake_runtime, empty_prompt, monkeypatch, scale, branch_count):
    """不等长条件提示独立预填充，音频反馈按各分支绝对位置合批。"""
    import streamvox.models.breeze.runtime as module

    samples = []

    def capture(logits, rng, **parameters):
        """记录真实采样入口的分支模式和主干历史快照。"""
        samples.append({**parameters, "history": list(parameters["history"]) if "history" in parameters else None,
                        "logits": np.asarray(logits).reshape(-1).copy()})
        return sample_token(logits, rng, **parameters)

    monkeypatch.setattr(module, "sample_token", capture)
    list(fake_runtime.synthesize("目标。", empty_prompt, instruction="自然女声。", cfg_scale=scale,
                                 max_new_tokens=2, do_sample=False, repetition_penalty=1.3))
    prefills, feedback = fake_runtime.backbone.calls[:-1], fake_runtime.backbone.calls[-1]
    assert len(prefills) == branch_count
    assert [call["sequence_ids"] for call in prefills] == [[index] for index in range(branch_count)]
    assert all(call["positions"] == [0] for call in prefills)
    assert feedback["positions"] == [call["embeddings"].shape[1] for call in prefills]
    assert feedback["sequence_ids"] == list(range(branch_count))
    np.testing.assert_array_equal(feedback["embeddings"], np.full((branch_count, 1, 4), 7))
    if branch_count == 2:
        assert feedback["positions"] == [5, 3]
    conditions = np.concatenate([call["embeddings"][:, -1] for call in prefills])
    np.testing.assert_array_equal(fake_runtime.depth_conditioner.calls[0]["backbone_hidden"], conditions)
    np.testing.assert_array_equal(fake_runtime.depth.calls[0]["embeddings"], (conditions[:, :2] * 2)[:, None])
    assert all(call["sequence_ids"] == list(range(branch_count)) for call in fake_runtime.depth.calls)
    expected = conditions[0, 0] if branch_count == 1 else conditions[1, 0] + scale * (conditions[0, 0] - conditions[1, 0])
    assert samples[0]["logits"][0] == expected
    assert [call["history"] for call in samples if call["history"] is not None] == [[], [3]]
    depth_samples = [call for call in samples if call["history"] is None]
    assert len(depth_samples) == 30
    assert depth_samples[0]["logits"][0] == pytest.approx(expected * 2 / 100)
    assert all(call["cfg_depth"] == (branch_count == 2) and "repetition_penalty" not in call for call in depth_samples)
    templates = [text for text, special in fake_runtime.tokenizer.calls if special]
    assert ["<ins_bos>" in text for text in templates] == ([False] if scale == 0 else [True] if scale == 1 else [True, False])


def test_shutdown_closes_native_models_once_and_rejects_new_work(fake_runtime, empty_prompt):
    """重复关闭安全，原生模型只释放一次，后续合成与参考编码立即拒绝。"""
    backbone, depth = fake_runtime.backbone, fake_runtime.depth
    fake_runtime.shutdown()
    fake_runtime.shutdown()
    assert backbone.close_count == depth.close_count == 1
    assert fake_runtime.backbone is fake_runtime.depth is fake_runtime.decoder is None
    with pytest.raises(RuntimeError, match="关闭"):
        list(fake_runtime.synthesize("关闭后。", empty_prompt, max_new_tokens=1))
    with pytest.raises(RuntimeError, match="关闭"):
        fake_runtime.encode_reference(np.ones(24, np.float32))
    assert not fake_runtime._lock.locked()


def test_shutdown_rejects_paused_generator_without_waiting(fake_runtime, empty_prompt):
    """生成器占锁时立即拒绝关闭，避免同线程等待自身释放资源。"""
    stream = fake_runtime.synthesize("暂停。", empty_prompt, max_new_tokens=2, chunk_frames=1, do_sample=False)
    next(stream)
    finished, errors = Event(), []

    def attempt():
        """在受控线程确认关闭操作不等待请求锁。"""
        try:
            fake_runtime.shutdown()
        except RuntimeError as error:
            errors.append(error)
        finally:
            finished.set()

    worker = Thread(target=attempt, daemon=True)
    worker.start()
    try:
        assert finished.wait(1) and len(errors) == 1 and "活动请求" in str(errors[0])
        assert fake_runtime._closed is False
    finally:
        stream.close()
        worker.join(1)
    assert not worker.is_alive()
    fake_runtime.shutdown()


def test_cuda_codec_keeps_streaming_state_on_device():
    """声码器状态直连下一帧，主机仅传入当前码本和位置。"""
    graph = OrtGraph.__new__(OrtGraph)
    graph.backend, graph.device_id = "cuda", 2
    graph.outputs = [SimpleNamespace(name="audio"), SimpleNamespace(name="next_codec_state")]
    actions, device_outputs = [], [object(), object()]
    binding = SimpleNamespace(
        bind_cpu_input=lambda name, value: actions.append(("cpu", name)),
        bind_ortvalue_input=lambda name, value: actions.append(("device", name, value)),
        bind_output=lambda name, device, index: actions.append(("output", name, device, index)),
        synchronize_outputs=lambda: actions.append(("synchronize",)),
        get_outputs=lambda: actions.append(("get_outputs",)) or device_outputs,
    )
    graph.session = SimpleNamespace(io_binding=lambda: binding, run_with_iobinding=lambda _binding: actions.append(("run",)))
    first = graph.run({"codes": np.ones((1, 1, 16), np.int64), "codec_state": np.zeros((1, 1), np.float32)})
    graph.run({"codes": np.ones((1, 1, 16), np.int64), "codec_state": first["next_codec_state"]})
    assert first["audio"] is device_outputs[0] and first["next_codec_state"] is device_outputs[1]
    assert ("device", "codec_state", device_outputs[1]) in actions
    assert ("output", "next_codec_state", "cuda", 2) in actions
    stages = [action[0] for action in actions]
    assert stages.index("run") < stages.index("synchronize") < stages.index("get_outputs")


def test_prepare_reference_precomputes_before_first_target_and_matches_fresh_audio(fake_runtime, reference_prompt):
    """角色准备先完成参考编码，首次目标仅追加后缀且波形匹配完整重算。"""
    fake_runtime.prepare_reference(reference_prompt)
    assert len(fake_runtime.text_encoder.calls) == 1
    assert len(fake_runtime.audio_embedding.calls) == 1
    assert fake_runtime.decoder.calls == []
    prepared = fake_runtime.backbone.calls[0]
    assert prepared["output"] is False and prepared["positions"] == [0]
    reference_length = prepared["embeddings"].shape[1]
    prepared_calls = len(fake_runtime.backbone.calls)
    warmed = np.concatenate(list(fake_runtime.synthesize(
        "首次目标。", reference_prompt, stream=False, do_sample=False, max_new_tokens=2,
    )))
    assert fake_runtime.backbone.calls[prepared_calls]["positions"] == [reference_length]

    fake_runtime.backbone.reset()
    fake_runtime._prefill = BackbonePrefill(fake_runtime.backbone)
    fresh = np.concatenate(list(fake_runtime.synthesize(
        "首次目标。", reference_prompt, stream=False, do_sample=False, max_new_tokens=2,
    )))
    np.testing.assert_array_equal(warmed, fresh)


def test_prepare_reference_reuses_same_content_and_rebuilds_changed_reference(fake_runtime, reference_prompt):
    """重复准备相同参考不重算原生前缀，转录变化后重新编码并建立缓存。"""
    fake_runtime.prepare_reference(reference_prompt)
    calls, resets = len(fake_runtime.backbone.calls), fake_runtime.backbone.reset_count
    fake_runtime.prepare_reference(reference_prompt)
    assert len(fake_runtime.backbone.calls) == calls
    assert fake_runtime.backbone.reset_count == resets
    reference_prompt.prompt_text = "另一段完整参考转录。"
    fake_runtime.prepare_reference(reference_prompt)
    assert len(fake_runtime.backbone.calls) == calls + 1
    assert fake_runtime.backbone.reset_count == resets + 1
    assert fake_runtime.backbone.calls[-1]["output"] is False
    assert len(fake_runtime.text_encoder.calls) == 2


def test_prepare_reference_rejects_full_context_before_native_mutation(fake_runtime, reference_prompt):
    """过长参考在扩容、清缓存和原生预填充之前被拒绝。"""
    fake_runtime.n_ctx = 1
    with pytest.raises(ValueError, match="参考前缀已达到 n_ctx"):
        fake_runtime.prepare_reference(reference_prompt)
    assert fake_runtime.backbone.calls == []
    assert fake_runtime.backbone.reserves == []
    assert fake_runtime.backbone.reset_count == 0


@pytest.mark.parametrize(("llama_backend", "onnx_backend", "fails"), [
    ("cpu", "cpu", False), ("cpu", "cuda", False), ("cuda", "cpu", False),
    ("cuda", "cuda", False), ("cuda", "cuda", True),
])
def test_runtime_warmup_runs_cuda_shapes_once_and_shuts_down_on_failure(
    fake_runtime, monkeypatch, llama_backend, onnx_backend, fails,
):
    """双 CUDA 才预热两种分支形状，重复调用复用结果，失败关闭资源。"""
    fake_runtime.gguf_backend, fake_runtime.onnx_backend = llama_backend, onnx_backend
    fake_runtime.model_name = "breeze-tts-2-gguf"
    calls, shutdowns = [], []

    def synthesize(text, prompt, **parameters):
        """记录预热合成参数，并按用例模拟执行失败。"""
        calls.append((text, prompt, parameters))
        if fails:
            raise RuntimeError("模拟预热失败")
        return iter(())

    monkeypatch.setattr(fake_runtime, "synthesize", synthesize)
    monkeypatch.setattr(fake_runtime, "shutdown", lambda: shutdowns.append(True))
    if fails:
        with pytest.raises(RuntimeError, match="模拟预热失败"):
            fake_runtime.warmup_for_init()
        assert shutdowns == [True] and fake_runtime._warmed_up is False
        return
    fake_runtime.warmup_for_init()
    fake_runtime.warmup_for_init()
    if llama_backend == onnx_backend == "cuda":
        assert [parameters["cfg_scale"] for _, _, parameters in calls] == [1.0, 1.0, 4.0, 4.0]
        assert all(parameters["stream"] is False and parameters["max_new_tokens"] == 2
                   and parameters["do_sample"] is False and parameters["seed"] == 0
                   for _, _, parameters in calls)
        assert fake_runtime._warmed_up is True
    else:
        assert calls == [] and fake_runtime._warmed_up is False
    assert shutdowns == []


@pytest.fixture
def engine_arguments(fake_runtime, tmp_path, monkeypatch):
    """保留真实引擎和预热入口，仅替换模型加载与显卡提供程序检查。

    Args:
        fake_runtime: 使用生产预热和合成调度的处理器替身。
        tmp_path: 隔离模型清单和角色缓存的临时目录。
        monkeypatch: 限定当前用例的加载与后端检查替换。
    Returns:
        可直接传给真实引擎构造器的参数。
    """
    adapter = BreezeTTS2GGUFAdapter()
    fake_runtime.model_name = adapter.model_name
    fake_runtime.gguf_backend = fake_runtime.onnx_backend = "cuda"
    fake_runtime.backbone.backend = fake_runtime.depth.backend = "cuda"
    fake_runtime.runtime_info = RuntimeInfo(
        model=adapter.model_name, adapter=type(adapter).__name__, bundle_loaded=True, asset_count=0,
    )
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(
        json.dumps({"model": adapter.model_name, "assets": []}), encoding="utf-8",
    )
    monkeypatch.setenv("STREAMVOX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(adapter, "load_runtime", lambda _request: fake_runtime)
    monkeypatch.setattr(adapter, "get_hardware_profile", lambda: None)
    monkeypatch.setattr(TTSEngine, "_verify_requested_backends", lambda _engine: None)
    return {
        "model": str(model_dir), "device": "gpu", "llama_backend": "cuda", "onnx_provider": "cuda",
        "verify_model_sha256": False, "registry": ModelRegistry({adapter.model_name: adapter}),
    }


def test_engine_initialization_runs_actual_breeze_warmup_hook(fake_runtime, engine_arguments):
    """真实引擎构造必须触发运行时预热，并执行两种分支的完整生成。

    Args:
        fake_runtime: 保留真实预热、采样和声码器调度的运行时。
        engine_arguments: 经适配器返回该运行时的真实引擎参数。
    Returns:
        引擎返回前已完成两种分支预热且状态可复用时正常返回。
    """
    engine = TTSEngine(**engine_arguments)
    try:
        assert engine.runtime is fake_runtime
        assert fake_runtime._warmed_up is True and fake_runtime._closed is False
        assert fake_runtime.backbone.calls
        assert {tuple(call["sequence_ids"]) for call in fake_runtime.depth.calls} == {(0,), (0, 1)}
        assert len(fake_runtime.decoder.calls) == 8
        assert fake_runtime.backbone.sequences == fake_runtime.depth.sequences == {}
    finally:
        engine.shutdown()


def test_engine_small_context_skips_warmup_and_still_synthesizes(fake_runtime, engine_arguments):
    """预热形状超过小上下文时跳过生成，用户可容纳的单字请求仍成功。

    Args:
        fake_runtime: 使用三词无指令提示的真实运行时调度。
        engine_arguments: 保留引擎初始化和流式入口的构造参数。
    Returns:
        四词上下文成功初始化并生成一帧用户音频时正常返回。
    """
    fake_runtime.n_ctx = 4
    engine = TTSEngine(n_ctx=4, **engine_arguments)
    try:
        assert fake_runtime._warmed_up is True and fake_runtime._closed is False
        assert fake_runtime.backbone.calls == fake_runtime.depth.calls == fake_runtime.decoder.calls == []
        chunks = list(engine.stream("啊", max_new_tokens=1, do_sample=False))
        assert [chunk.size for chunk in chunks] == [1920]
        assert len(fake_runtime.backbone.calls) == 1
        assert fake_runtime._closed is False
    finally:
        engine.shutdown()


def test_engine_warmup_failure_closes_actual_runtime_resources(fake_runtime, engine_arguments, monkeypatch):
    """引擎预热中的原生执行失败须上抛，并释放已加载的全部运行资源。

    Args:
        fake_runtime: 保留真实预热失败处理及关闭逻辑的运行时。
        engine_arguments: 通过真实引擎构造触发预热的参数。
        monkeypatch: 在原生解码入口注入执行失败。
    Returns:
        原异常保留且原生模型、辅助图和执行器均已关闭时正常返回。
    """
    backbone, depth = fake_runtime.backbone, fake_runtime.depth

    def fail_decode(**_feeds):
        """模拟预热首次进入原生解码时失败。"""
        raise RuntimeError("模拟预热原生解码失败")

    monkeypatch.setattr(backbone, "decode", fail_decode)
    with pytest.raises(RuntimeError, match="模拟预热原生解码失败"):
        TTSEngine(**engine_arguments)
    assert fake_runtime._closed is True and fake_runtime._warmed_up is False
    assert backbone.close_count == depth.close_count == 1
    assert fake_runtime.backbone is fake_runtime.depth is fake_runtime.decoder is fake_runtime.text_encoder is None
    assert fake_runtime._codec_executor is None and not fake_runtime._lock.locked()
