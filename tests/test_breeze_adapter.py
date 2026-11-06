"""验证 Breeze2 适配器的输入边界、缓存与引擎协作。"""

from __future__ import annotations

from dataclasses import replace
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from streamvox import TTSEngine
from streamvox.models.base import BasePromptAsset, RuntimeInfo, SerializedPromptAsset
from streamvox.models.breeze import BreezeTTS2GGUFAdapter
from streamvox.models.breeze.prompt import BreezePromptAsset
from streamvox.models.registry import ModelRegistry
from streamvox.prompt_cache import PromptCache


@pytest.fixture
def adapter():
    """创建不加载模型资源的适配器。

    Returns:
        当前用例独享的 Breeze2 适配器。
    """
    return BreezeTTS2GGUFAdapter()


@pytest.mark.parametrize("backend", ["auto", "cpu", "cuda"])
def test_text_backend_configuration_is_explicit(backend):
    """默认优先可用的显卡文本资产，并保留明确的处理器选择。"""
    assert BreezeTTS2GGUFAdapter().text_encoder_backend == "auto"
    assert BreezeTTS2GGUFAdapter(text_encoder_backend=backend).text_encoder_backend == backend


@pytest.mark.parametrize("backend", ["directml", "gpu", "unknown"])
def test_text_backend_rejects_unsupported_configuration(backend):
    """不支持的文本后端必须在模型加载之前拒绝。"""
    with pytest.raises(ValueError, match="text_encoder_backend"):
        BreezeTTS2GGUFAdapter(text_encoder_backend=backend)


@pytest.fixture
def prompt(adapter):
    """创建含有参考转录和合法编码的提示。

    Args:
        adapter: 当前模型适配器。

    Returns:
        用于缓存和合成边界测试的提示资产。
    """
    return BreezePromptAsset(
        role_name="参考角色", model=adapter.model_name, sample_rate=24000,
        metadata={"speaker": "测试"}, prompt_text="[笑]完整参考转录。",
        audio_codes=np.arange(32, dtype=np.int64).reshape(2, 16),
    )


def test_import_keeps_runtime_and_heavy_dependencies_lazy():
    """查询适配器不应加载模型运行时或张量框架。"""
    script = (
        "import sys; import streamvox; before = set(sys.modules); import streamvox.models.breeze; "
        "assert 'streamvox.models.breeze.runtime' not in sys.modules; "
        "assert 'onnxruntime' not in set(sys.modules) - before; assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_builtin_registry_uses_gguf_model_contract(adapter):
    """内置注册与公开导出应统一采用混合后端模型名称。"""
    import streamvox.models.breeze as breeze
    from streamvox.models.builtin_models import register_default_models
    from streamvox.models.registry import default_registry

    register_default_models()
    assert adapter.model_name == "breeze-tts-2-gguf"
    assert isinstance(default_registry.get(adapter.model_name), BreezeTTS2GGUFAdapter)
    assert "breeze-tts-2-onnx" not in default_registry.list_models()
    assert breeze.__all__ == ["BreezeTTS2GGUFAdapter"]
    assert not hasattr(breeze, "BreezeTTS2OnnxAdapter")


@pytest.mark.parametrize("device,mode,index", [
    ("auto", "auto", None), ("cpu", "cpu", None), ("gpu", "gpu", 0),
    ("gpu:3", "gpu", 3), (" GPU:2 ", "gpu", 2),
])
def test_execution_plan_preserves_device_intent(adapter, device, mode, index):
    """统一设备参数应分别传给主干与其余推理组件。"""
    plan = adapter.resolve_execution_plan(device)
    assert plan.raw_device == device
    assert set(plan.targets) == {"llama", "onnx"}
    for family in ("llama", "onnx"):
        target = plan.get(family)
        assert (target.mode, target.device_index) == (mode, index)
        assert target.backend_preference is None


@pytest.mark.parametrize("device", [None, 1, "", "cuda", "gpu:-1", "gpu:1.2", "gpu:", "gpu:²"])
def test_execution_plan_rejects_invalid_device(adapter, device):
    """错误的设备表达应在加载模型前被拒绝。"""
    with pytest.raises((TypeError, ValueError)):
        adapter.resolve_execution_plan(device)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "vulkan", "metal"])
def test_execution_plan_keeps_llama_backend_separate_from_onnx(adapter, backend):
    """主干后端不能覆盖其余组件的默认设备意图。"""
    plan = adapter.resolve_execution_plan("gpu:2", llama_backend=backend)
    assert plan.get("llama").backend_preference == backend
    assert (plan.get("llama").mode, plan.get("llama").device_index) == ("gpu", 2)
    assert plan.get("onnx").backend_preference is None
    assert (plan.get("onnx").mode, plan.get("onnx").device_index) == ("gpu", 2)


@pytest.mark.parametrize("backend", ["", "auto", "dml", "rocm", 1])
def test_execution_plan_rejects_unsupported_llama_backend(adapter, backend):
    """非法的主干后端应在加载模型资源之前失败。"""
    with pytest.raises((TypeError, ValueError), match="llama backend"):
        adapter.resolve_execution_plan("cpu", llama_backend=backend)


def test_hardware_advice_keeps_minimum_capacity_unverified(adapter):
    """混合后端的资源建议不能被描述为已验证的硬件门槛。"""
    profile = adapter.get_hardware_profile()
    assert "不代表已验证的最低门槛" in profile.notes
    assert "短句" in profile.notes
    assert profile.profile_id == "breeze-tts-2-gguf"
    assert profile.quantization == "gguf-q8_0/bf16/f32+onnx-f32"
    assert profile.min_vram_gb == profile.min_ram_gb == 0.0
    assert profile.recommended_ram_gb >= profile.min_ram_gb


def test_prompt_cache_round_trip_preserves_private_and_public_data(adapter, tmp_path, monkeypatch):
    """缓存往返保留编码、转录及同源的公共参考波形。"""
    monkeypatch.setenv("STREAMVOX_HOME", str(tmp_path))
    encoded_audio = []
    prepared = []

    def encode_reference(audio):
        """记录编码器实际接收到的规整波形。"""
        encoded_audio.append(audio)
        return np.arange(48, dtype=np.int64).reshape(3, 16)

    runtime = SimpleNamespace(sample_rate=24000, encode_reference=encode_reference, prepare_reference=prepared.append)
    stereo = np.asarray([[0.0, 0.2, 0.4], [1.0, 1.2, 1.4]], dtype=np.float64)
    asset = adapter.make_prompt(
        runtime, role_name="参考角色", sample_rate=24000, audio_data=stereo,
        prompt_text=" [笑]完整参考转录。 ",
    )
    cache = PromptCache(adapter.model_name)
    cache.store(adapter, asset, asset.prompt_cache_record)
    restored = cache.load(adapter, adapter.model_name, "参考角色")
    public = cache.load_prompt_data_record(adapter.model_name, "参考角色")

    np.testing.assert_allclose(encoded_audio[0], [0.5, 0.7, 0.9])
    assert encoded_audio[0].dtype == np.float32
    assert encoded_audio[0].flags.c_contiguous
    assert isinstance(restored, BreezePromptAsset)
    assert prepared[0] is asset
    assert restored.model == "breeze-tts-2-gguf"
    assert restored.prompt_text == " [笑]完整参考转录。 "
    np.testing.assert_array_equal(restored.audio_codes, asset.audio_codes)
    assert restored.audio_codes.dtype == np.int64
    assert public.prompt_text == [asset.prompt_text]
    assert len(public.references) == 1
    np.testing.assert_array_equal(public.references[0].audio, encoded_audio[0])


def test_default_prompt_round_trip_has_no_reference(adapter):
    """无参考资产供纯文本和指令设计共用。"""
    default = adapter.make_default_prompt(SimpleNamespace(sample_rate=24000))
    restored = adapter.deserialize_prompt_asset(adapter.serialize_prompt_asset(default))
    assert restored.audio_codes.shape == (0, 16)
    assert restored.audio_codes.dtype == np.int64
    assert restored.prompt_text is None


@pytest.mark.parametrize("changes,error", [
    ({"model": "another-model"}, ValueError),
    ({"sample_rate": 48000}, ValueError),
    ({"audio_codes": np.zeros((2, 15), dtype=np.int64)}, ValueError),
    ({"audio_codes": np.zeros((16,), dtype=np.int64)}, ValueError),
    ({"audio_codes": np.full((2, 16), -1, dtype=np.int64)}, ValueError),
    ({"audio_codes": np.full((2, 16), 2048, dtype=np.int64)}, ValueError),
    ({"audio_codes": np.full((2, 16), 2049, dtype=np.int64)}, ValueError),
    ({"audio_codes": np.full((2, 16), 2050, dtype=np.int64)}, ValueError),
    ({"audio_codes": np.zeros((2, 16), dtype=np.float32)}, TypeError),
    ({"prompt_text": None}, ValueError),
    ({"prompt_text": "  "}, ValueError),
    ({"prompt_text": 42}, TypeError),
])
def test_cache_rejects_incompatible_or_corrupt_prompt(adapter, prompt, changes, error):
    """序列化与反序列化必须应用相同的模型约束。"""
    with pytest.raises(error):
        adapter.serialize_prompt_asset(replace(prompt, **changes))
    payload = adapter.serialize_prompt_asset(prompt)
    for name, value in changes.items():
        if name == "audio_codes":
            payload.arrays[name] = value
        elif name == "prompt_text":
            payload.json_data[name] = value
        else:
            setattr(payload, name, value)
    with pytest.raises(error):
        adapter.deserialize_prompt_asset(payload)


def test_cache_rejects_wrong_asset_type_and_missing_codes(adapter):
    """缓存边界应报告类型与必需字段错误。"""
    with pytest.raises(TypeError, match="BreezePromptAsset"):
        adapter.serialize_prompt_asset(BasePromptAsset(role_name="其他", model=adapter.model_name, sample_rate=24000))
    with pytest.raises(ValueError, match="audio_codes"):
        adapter.deserialize_prompt_asset(SerializedPromptAsset(role_name="缺失", model=adapter.model_name, sample_rate=24000))


def test_cache_accepts_highest_codec_code(adapter, prompt):
    """合法的最高音频编码不能与保留的协议编码混淆。"""
    prompt.audio_codes[0, 0] = 2047
    restored = adapter.deserialize_prompt_asset(adapter.serialize_prompt_asset(prompt))
    assert restored.audio_codes[0, 0] == 2047


@pytest.mark.parametrize("changes", [
    {"audio_data": [np.ones(8)]}, {"audio_path": ["ref.wav"], "audio_data": None},
    {"sample_rate": [24000]}, {"prompt_text": ["完整转录"]},
    {"audio_path": "ref.wav"}, {"audio_data": None}, {"sample_rate": None},
    {"sample_rate": True}, {"sample_rate": 0}, {"sample_rate": -24000},
    {"sample_rate": 24000.0}, {"prompt_text": None}, {"prompt_text": " \n "},
    {"prompt_text": 1}, {"role_name": " "}, {"audio_data": np.asarray([])},
    {"audio_data": np.asarray([np.nan])}, {"audio_data": np.asarray([np.inf])},
    {"audio_data": np.zeros((1, 2, 3))}, {"audio_data": np.asarray([1j])},
    {"audio_data": np.asarray(["语音"])},
])
def test_make_prompt_rejects_bad_reference_before_encoding(adapter, changes):
    """非法输入不能触发昂贵的音频编码。"""
    parameters = dict(role_name="角色", sample_rate=24000, audio_data=np.ones(8, dtype=np.float32), prompt_text="完整转录")
    parameters.update(changes)
    runtime = SimpleNamespace(encode_reference=lambda _audio: pytest.fail("非法输入不应触发编码"))
    with pytest.raises((TypeError, ValueError)):
        adapter.make_prompt(runtime, **parameters)


def test_make_prompt_resamples_file_to_codec_rate(adapter, tmp_path):
    """文件头采样率优先于调用参数，并规整成编码器需要的单声道。"""
    import soundfile as sf

    path = tmp_path / "reference.wav"
    sf.write(path, np.full((240, 2), 0.25, dtype=np.float32), 12000, subtype="FLOAT")
    received = []
    runtime = SimpleNamespace(
        encode_reference=lambda audio: received.append(audio) or np.zeros((1, 16), dtype=np.int64),
        prepare_reference=lambda _prompt: None,
    )
    asset = adapter.make_prompt(runtime, "文件角色", 48000, audio_path=str(path), prompt_text="完整转录")
    assert asset.sample_rate == 24000
    assert received[0].shape == (480,)
    assert received[0].dtype == np.float32
    np.testing.assert_array_equal(asset.prompt_cache_record.references[0].audio, received[0])


def test_defaults_keep_two_sampling_distributions_independent(adapter):
    """时间层与深度层应保留各自的采样默认值。"""
    values = adapter.normalize_synthesize_kwargs()
    assert values == {
        "instruction": "", "cfg_scale": 1.0, "temperature": 0.9,
        "top_k": 50, "top_p": 1.0, "depth_temperature": 0.9,
        "depth_top_k": 50, "depth_top_p": 1.0, "repetition_penalty": 1.1,
        "max_new_tokens": 750, "seed": None, "do_sample": True, "chunk_frames": 2,
    }


@pytest.mark.parametrize("name,value", [
    ("instruction", None), ("instruction", 7), ("cfg_scale", -1), ("cfg_scale", float("inf")),
    ("temperature", 0), ("temperature", float("nan")), ("temperature", True),
    ("top_k", -1), ("top_k", 1.5), ("top_k", True), ("top_p", 0), ("top_p", 1.1),
    ("depth_temperature", 0), ("depth_temperature", float("inf")),
    ("depth_top_k", -1), ("depth_top_k", 1.5), ("depth_top_p", 0), ("depth_top_p", 1.01),
    ("repetition_penalty", 0), ("repetition_penalty", float("nan")),
    ("max_new_tokens", 0), ("max_new_tokens", 751), ("max_new_tokens", True),
    ("seed", -1), ("seed", 1.2), ("seed", True), ("do_sample", 1),
    ("chunk_frames", 0), ("chunk_frames", 26), ("chunk_frames", 2.0),
    ("mode", "voice_clone"), ("unknown", 1),
])
def test_sampling_parameters_reject_invalid_values(adapter, name, value):
    """采样数值越界和未知参数都应清晰失败。"""
    with pytest.raises((TypeError, ValueError)):
        adapter.normalize_synthesize_kwargs(**{name: value})


def test_synthesize_preserves_parameters_text_and_audio(adapter, prompt):
    """适配器不应改写两级分布、指令、事件符号或音频内容。"""
    expected = {
        "instruction": " [笑]柔和、自然。 ", "cfg_scale": 0,
        "temperature": 0.7, "top_k": 13, "top_p": 0.8,
        "depth_temperature": 1.2, "depth_top_k": 7, "depth_top_p": 0.6,
        "repetition_penalty": 1.05, "max_new_tokens": 123, "seed": 0,
        "do_sample": False, "chunk_frames": 25,
    }
    received = {}
    audio = np.asarray([0.0, -0.0, 1.25, -1.25], dtype=np.float32)

    def synthesize(**kwargs):
        """记录运行时请求，并原样返回可识别的音频。"""
        received.update(kwargs)
        yield audio

    runtime = SimpleNamespace(synthesize=synthesize)
    chunks = list(adapter.synthesize(runtime, "[sigh!]你好<|laugh|>。", prompt, stream=False, **expected))
    assert received == {"text": "[sigh!]你好<|laugh|>。", "prompt": prompt, "stream": False, **expected}
    assert chunks[0] is audio
    assert chunks[0].tobytes() == audio.tobytes()


def test_text_preprocess_preserves_event_markers(adapter):
    """额外清理与分段不能改变上游文本和标记。"""
    assert adapter.get_text_preprocess_config() is None


@pytest.mark.parametrize("device,llama_backend,onnx_provider,llama_mode,onnx_mode,index", [
    ("cpu", "cpu", "cpu", "cpu", "cpu", None),
    ("cpu", "cpu", "cuda", "cpu", "gpu", None),
    ("gpu:2", "vulkan", "cpu", "gpu", "cpu", 2),
])
@pytest.mark.parametrize("text_encoder_backend", ["cpu", "cuda"])
def test_local_bundle_engine_uses_fake_runtime_and_prompt_cache(
    tmp_path, monkeypatch, device, llama_backend, onnx_provider, llama_mode, onnx_mode, index, text_encoder_backend,
):
    """通过真实引擎入口验证目录、执行计划、参数和缓存协议一致。"""
    adapter = BreezeTTS2GGUFAdapter(text_encoder_backend=text_encoder_backend)
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(json.dumps({"model": adapter.model_name, "assets": []}), encoding="utf-8")
    monkeypatch.setenv("STREAMVOX_HOME", str(tmp_path / "home"))
    state = SimpleNamespace(load=None, requests=[], closed=False)

    class FakeRuntime:
        """替代模型资源，仅记录引擎与适配器之间的交互。"""

        sample_rate = 24000

        def __init__(self, **kwargs):
            """记录模型初始化时的完整公开请求。"""
            state.load = kwargs
            self.gguf_backend = kwargs["execution_plan"].get("llama").backend_preference
            self.onnx_backend = kwargs["execution_plan"].get("onnx").backend_preference
            self.runtime_info = RuntimeInfo(model=adapter.model_name, adapter=type(adapter).__name__, bundle_loaded=True, asset_count=0)

        def encode_reference(self, audio):
            """返回可持久化的参考编码。"""
            assert audio.dtype == np.float32 and audio.ndim == 1
            return np.zeros((2, 16), dtype=np.int64)

        def prepare_reference(self, prompt):
            """记录角色准备阶段收到的完整参考资产。"""
            state.prepared = prompt

        def synthesize(self, **kwargs):
            """输出单块音频并记录模型参数。"""
            state.requests.append(kwargs)
            yield np.asarray([0.2, -0.2], dtype=np.float32)

        def shutdown(self):
            """记录引擎释放运行时。"""
            state.closed = True

    runtime_module = ModuleType("streamvox.models.breeze.runtime")
    runtime_module.BreezeRuntime = FakeRuntime
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    monkeypatch.setattr(adapter, "get_hardware_profile", lambda: None)
    engine = TTSEngine(
        str(model_dir), device=device, llama_backend=llama_backend, onnx_provider=onnx_provider, n_ctx=2048,
        verify_model_sha256=False, registry=ModelRegistry({adapter.model_name: adapter}),
    )
    try:
        assert state.load["model_dir"] == model_dir.resolve()
        assert state.load["n_ctx"] == 2048
        assert state.load["text_encoder_backend"] == text_encoder_backend
        assert state.load["verify_model_sha256"] is False
        plan = state.load["execution_plan"]
        assert plan.get("llama").backend_preference == llama_backend
        assert plan.get("llama").mode == llama_mode
        assert plan.get("llama").device_index == index
        assert plan.get("onnx").backend_preference == onnx_provider
        assert plan.get("onnx").mode == onnx_mode
        assert plan.get("onnx").device_index == index
        assert engine.actual_backends["llama"] == llama_backend
        assert engine.actual_backends["onnx"] == onnx_provider
        engine.make_prompt("角色", audio_data=np.ones(16, dtype=np.float32), sample_rate=24000, prompt_text="完整转录。")
        public = engine.get_prompt_data("角色")
        assert public.model == public.prompt.model == "breeze-tts-2-gguf"
        assert public.prompt_text == ["完整转录。"]
        assert isinstance(public.prompt, BreezePromptAsset)
        assert len(public.references) == 1
        assert len(list(engine.stream("[笑]缓存角色。", role_name="角色", temperature=0.6, depth_temperature=1.2))) == 1
        assert len(list(engine.stream("无参考设计。", instruction="平静女声。"))) == 1
        assert len(list(engine.stream("纯文本。"))) == 1
        first, design, plain = state.requests
        assert first["prompt"].prompt_text == "完整转录。"
        assert first["temperature"] == 0.6 and first["depth_temperature"] == 1.2
        assert design["prompt"].audio_codes.shape == (0, 16)
        assert design["instruction"] == "平静女声。"
        assert plain["instruction"] == ""
        assert not {"max_length", "min_length", "remove_meaningless_chars", "track_performance"} & plain.keys()
    finally:
        engine.shutdown()
    assert state.closed
