"""用 llama.cpp 生成完整码本帧，再由 ONNX 组件连续解码音频。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer

from ...model_bundle import ModelBundleError, validate_local_bundle
from ..base import ExecutionPlan, RuntimeInfo
from ..llama_backend import select_llama_backend
from .llama import LlamaDecoder
from .codec import CudaStreamingDecoder
from .ort import OrtGraph, as_numpy, resolve_backend
from .prompt import BreezePromptAsset
from .prefill import BackbonePrefill
from .sampling import guided_logits, sample_token
from .spec import BREEZE_TTS_2_GGUF_SPEC
from .streaming import StreamingCodec
from .templates import build_prompt_embeddings, build_reference_prefix
from .text import HybridTextEncoder


class BreezeRuntime:
    """持有两套原生生成模型和独立的文本、音频 ONNX 组件。

    Args:
        model_dir: 带完整清单的 GGUF 与 ONNX 模型目录。
        device: 用户设备意图。
        execution_plan: 分别描述 llama.cpp 与 ONNX 的执行目标。
        verify_model_sha256: 是否校验资产摘要。
        n_ctx: 单个条件分支允许的最大上下文长度。
        text_encoder_backend: 文本编码使用处理器，或处理器查表与显卡主干。
    Returns:
        按请求隔离原生缓存的可复用运行时。
    """

    def __init__(
        self, *, model_dir: Path, device: str = "auto", execution_plan: ExecutionPlan | None = None,
        verify_model_sha256: bool = True, n_ctx: int | None = None,
        text_encoder_backend: str = "auto",
    ) -> None:
        if text_encoder_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("text_encoder_backend must be auto, cpu or cuda")
        self.model_name = BREEZE_TTS_2_GGUF_SPEC.model_name
        self.sample_rate = BREEZE_TTS_2_GGUF_SPEC.sample_rate
        bundle = validate_local_bundle(model_dir, self.model_name, verify_model_sha256)
        self.ckpt = Path(model_dir) / "ckpt"
        declared = {path.resolve() for path in bundle.asset_paths}
        required = {
            "config.json", "tokenizer.json", "codec_metadata.json", "export-report.json",
            "backbone.gguf", "depth.gguf", "depth_conditioner.onnx", "text_encoder.onnx",
            "audio_embedding.onnx", "audio_encoder.onnx", "decoder_step.onnx",
        }
        missing = [name for name in sorted(required) if (self.ckpt / name).resolve() not in declared]
        if missing:
            raise ModelBundleError(f"Breeze GGUF 清单缺少运行资产：{', '.join(missing)}")
        self.config = json.loads((self.ckpt / "config.json").read_text(encoding="utf-8"))
        self.codec_metadata = json.loads((self.ckpt / "codec_metadata.json").read_text(encoding="utf-8"))
        report = json.loads((self.ckpt / "export-report.json").read_text(encoding="utf-8"))
        for name in report.get("files", {}):
            path = (self.ckpt / name).resolve()
            if not path.is_relative_to(self.ckpt.resolve()) or path not in declared:
                raise ModelBundleError(f"Breeze 导出资产未纳入清单或路径越界：{name}")
        if report.get("tiny"):
            raise ValueError("随机小模型只能用于数值测试，不能作为 Breeze TTS 权重加载。")
        generation = report.get("generation", {})
        if generation.get("validated") is not True:
            raise ModelBundleError("Breeze GGUF 生成组件尚未完成数值验收。")
        self.hidden_size = int(self.config["hidden_size"])
        self.num_codebooks = int(self.config["num_codebooks"])
        self.codebook_size = int(self.config["codec_config"]["codebook_size"])
        self.vocab_size = int(self.config["vocab_size"])
        self.eos_token = self.vocab_size
        limit = int(self.config["backbone_config"]["max_position_embeddings"])
        if n_ctx is not None and (n_ctx <= 0 or n_ctx > limit):
            raise ValueError(f"Breeze n_ctx 必须位于 1..{limit}。")
        self.n_ctx = n_ctx or limit
        if execution_plan is None:
            from .adapter import BreezeTTS2GGUFAdapter

            execution_plan = BreezeTTS2GGUFAdapter().resolve_execution_plan(device)
        llama_target, onnx_target = execution_plan.get("llama"), execution_plan.get("onnx")
        if llama_target is None or onnx_target is None:
            raise ValueError("Breeze 执行计划必须包含 llama 与 ONNX 目标。")
        selected = select_llama_backend(
            mode=llama_target.mode, device_index=llama_target.device_index,
            model_label=self.model_name, explicit_backend=llama_target.backend_preference,
        )
        # 通用选择器允许自动回退；明确要求的模型后端不能因此被悄悄替换。
        explicit = llama_target.backend_preference
        unexpected_cpu = llama_target.mode == "gpu" and selected.backend == "cpu" and explicit != "cpu"
        if (explicit is not None and selected.backend != explicit) or unexpected_cpu:
            raise RuntimeError(f"Breeze llama.cpp 后端未满足请求：{explicit or 'gpu'} → {selected.backend}")
        self.gguf_backend = selected.backend
        self.llama_device_id = selected.gpu_index or 0
        self.onnx_backend, self.device_id = resolve_backend(onnx_target)
        fast_assets = ("text_encoder_trunk.onnx", "text_encoder_lookup.onnx", "text_encoder_lookup.json")
        fast_text_available = all((self.ckpt / name).resolve() in declared for name in fast_assets)
        if text_encoder_backend == "auto":
            text_encoder_backend = "cuda" if self.onnx_backend == "cuda" and fast_text_available else "cpu"
        if text_encoder_backend == "cuda" and not fast_text_available:
            raise ModelBundleError("Breeze CUDA 文本主干缺少快速资产，请先运行离线文本拆分工具。")
        self.text_encoder_backend = text_encoder_backend
        self._lock = Lock()
        self._closed = False
        self._warmed_up = False
        self._reference_text_cache: dict[tuple[int, ...], np.ndarray] = {}
        self.backbone = self.depth = None
        self.audio_encoder = self.text_encoder = self.audio_embedding = None
        self.depth_conditioner = self.decoder = self.tokenizer = None
        self._codec_executor = None
        try:
            if text_encoder_backend == "cuda":
                lookup = json.loads((self.ckpt / "text_encoder_lookup.json").read_text(encoding="utf-8"))
                if (self.ckpt / lookup["embedding"]["location"]).resolve() not in declared:
                    raise ModelBundleError("Breeze 文本查表权重未纳入模型清单。")
                self.text_encoder = HybridTextEncoder(self.ckpt, onnx_target.device_index or 0)
            else:
                self.text_encoder = OrtGraph(self.ckpt / "text_encoder.onnx", "cpu")
            self.audio_embedding = OrtGraph(self.ckpt / "audio_embedding.onnx", "cpu")
            # 主干状态经公开原生接口返回主机，投影直接在处理器完成，避免每帧往返显卡。
            self.depth_conditioner = OrtGraph(self.ckpt / "depth_conditioner.onnx", "cpu")
            if self.onnx_backend == "cuda":
                # 固定执行线程可跨请求复用声码器的线程本地运行资源。
                self.decoder = CudaStreamingDecoder(self.ckpt / "decoder_step.onnx", self.device_id)
                self._codec_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze-codec")
                self._codec_executor.submit(self.decoder.warmup).result()
            else:
                self.decoder = OrtGraph(self.ckpt / "decoder_step.onnx", self.onnx_backend, self.device_id)
            self.backbone = LlamaDecoder(
                self.ckpt / "backbone.gguf", backend=self.gguf_backend, device_id=self.llama_device_id,
                n_ctx=min(self.n_ctx, 256), n_seq_max=2, embeddings=True,
            )
            self.depth = LlamaDecoder(
                self.ckpt / "depth.gguf", backend=self.gguf_backend, device_id=self.llama_device_id,
                n_ctx=self.num_codebooks, n_seq_max=2, embeddings=False,
            )
            self._prefill = BackbonePrefill(self.backbone)
            self.tokenizer = Tokenizer.from_file(str(self.ckpt / "tokenizer.json"))
        except BaseException:
            self.shutdown()
            raise
        self.runtime_info = RuntimeInfo(
            model=self.model_name, adapter=BREEZE_TTS_2_GGUF_SPEC.adapter_name,
            bundle_loaded=True, asset_count=len(bundle.asset_paths), metadata={
                "gguf_backend": self.gguf_backend, "onnx_backend": self.onnx_backend,
                "generation_backend": "llama.cpp", "text_encoder_backend": self.text_encoder_backend,
                "text_lookup_backend": "cpu",
                "audio_encoder_backend": "cpu", "audio_embedding_backend": "cpu",
                "depth_conditioner_backend": "cpu", "kv_precision": "float32",
                "weight_precision": str(generation.get("precision", "")),
                "source_commit": str(report.get("source_commit", "")),
            },
        )

    def encode_reference(self, audio: np.ndarray) -> np.ndarray:
        """按需用 ONNX 编码参考音频，返回可持久缓存的码本帧。

        Args:
            audio: 24kHz 单声道音频。
        Returns:
            整数音频编码，形状为帧数、十六码本。
        """
        with self._lock:
            self._ensure_open()
            if self.audio_encoder is None:
                self.audio_encoder = OrtGraph(self.ckpt / "audio_encoder.onnx", "cpu")
            output = self.audio_encoder.run({"audio": np.asarray(audio, dtype=np.float32).reshape(1, -1)})
            return np.asarray(as_numpy(output["codes"])[0], dtype=np.int64)

    def prepare_reference(self, prompt: BreezePromptAsset) -> None:
        """在角色准备阶段编码已知参考并建立可复用的原生前缀。

        Args:
            prompt: 已包含完整转录和音频编码的合法参考资产。
        Returns:
            参考准备完成后返回，目标文本在实际请求中独立计算。
        """
        with self._lock:
            self._ensure_open()
            prefix = build_reference_prefix(
                self.tokenizer, self.text_encoder, self.audio_embedding, prompt=prompt,
                hidden_size=self.hidden_size, eos_code=int(self.config["codebook_eos_token_id"]),
                num_codebooks=self.num_codebooks, reference_cache=self._reference_text_cache,
            )
            if prefix is None:
                return
            if prefix.shape[1] >= self.n_ctx:
                raise ValueError("Breeze 参考前缀已达到 n_ctx，无法容纳目标文本。")
            required = prefix.shape[1] + BREEZE_TTS_2_GGUF_SPEC.max_new_tokens
            self._prefill.reserve(min(self.n_ctx, ((required + 255) // 256) * 256))
            self._prefill.warm(prefix)

    def warmup_for_init(self) -> None:
        """加载阶段预热 CUDA 文本、主干和两种深度分支形状。

        Returns:
            预热完成后返回，所有合成请求仍使用各自独立的采样状态。
        """
        if self._warmed_up or self.gguf_backend != "cuda" or self.onnx_backend != "cuda":
            return
        try:
            # 提前为常见短提示与默认生成上限分配容量，避免首次请求重建计算上下文。
            capacity = ((BREEZE_TTS_2_GGUF_SPEC.max_new_tokens + 255) // 256 + 1) * 256
            self._prefill.reserve(min(self.n_ctx, capacity))
            prompt = BreezePromptAsset(role_name="__warmup__", model=self.model_name, sample_rate=self.sample_rate)
            for scale in (1.0, 4.0):
                instructions = ["声音清晰自然。"] if scale == 1.0 else ["声音清晰自然。", ""]
                embeddings, _ = build_prompt_embeddings(
                    self.tokenizer, self.text_encoder, self.audio_embedding,
                    text="你好，欢迎使用语音合成。", prompt=prompt, instructions=instructions,
                    hidden_size=self.hidden_size, eos_code=int(self.config["codebook_eos_token_id"]),
                    num_codebooks=self.num_codebooks,
                )
                if max(value.shape[1] for value in embeddings) + 2 > self.n_ctx:
                    continue
                for _ in range(2):
                    list(self.synthesize(
                        "你好，欢迎使用语音合成。", prompt, stream=False,
                        instruction="声音清晰自然。", cfg_scale=scale, max_new_tokens=2,
                        do_sample=False, seed=0,
                    ))
            self._warmed_up = True
        except BaseException as error:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                error.add_note(f"Breeze 预热失败后的资源清理也发生异常：{cleanup_error}")
            raise

    def synthesize(self, text: str, prompt: BreezePromptAsset, stream: bool = True, **kwargs: object) -> Iterable[np.ndarray]:
        """生成完整码本帧，并连续输出由 ONNX 声码器还原的音频。

        Args:
            text: 保留事件标记的目标文本。
            prompt: 参考音频或空提示。
            stream: 是否按指定帧数逐块返回。
            kwargs: 经过适配器校验的采样和声音指令参数。
        Returns:
            逐块生成单声道浮点音频。
        """
        from .adapter import BreezeTTS2GGUFAdapter

        parameters = BreezeTTS2GGUFAdapter().normalize_synthesize_kwargs(**kwargs)
        with self._lock:
            self._ensure_open()
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Breeze 目标文本不能为空。")
            instruction, scale = parameters["instruction"], parameters["cfg_scale"]
            if scale != 1.0 and not instruction:
                raise ValueError("Breeze cfg_scale 不为一时需要 instruction。")
            branch_instructions = [instruction]
            if instruction and scale == 0.0:
                branch_instructions = [""]
            elif instruction and scale != 1.0:
                branch_instructions.append("")
            prompts, reference_length = build_prompt_embeddings(
                self.tokenizer, self.text_encoder, self.audio_embedding,
                text=text, prompt=prompt, instructions=branch_instructions, hidden_size=self.hidden_size,
                eos_code=int(self.config["codebook_eos_token_id"]), num_codebooks=self.num_codebooks,
                reference_cache=self._reference_text_cache,
            )
            lengths = [value.shape[1] for value in prompts]
            required = max(lengths) + parameters["max_new_tokens"]
            if required > self.n_ctx:
                raise ValueError("Breeze 提示与生成长度超过 n_ctx；请显式缩短请求或增大上下文。")
            # 按本次请求扩容而非按模型四万词上限预分配；扩容只重建缓存，不重复加载权重。
            self._prefill.reserve(min(self.n_ctx, ((required + 255) // 256) * 256))
            self.depth.reset()
            codec = None
            succeeded = False
            try:
                branches = self._prefill.decode(prompts, reference_length)
                rng = np.random.default_rng(parameters["seed"])
                history: list[int] = []
                if self.onnx_backend == "cuda":
                    codec = StreamingCodec(self.decoder, self.codec_metadata["state_specs"], self._codec_executor)
                else:
                    codec_states = {item["name"]: np.zeros(item["shape"], dtype=np.float32)
                                    for item in self.codec_metadata["state_specs"]}
                buffer: list[np.ndarray] = []
                for frame_index in range(parameters["max_new_tokens"]):
                    negative = None if len(branches) == 1 else branches[1]["logits"]
                    first = sample_token(
                        guided_logits(branches[0]["logits"], negative, scale), rng,
                        do_sample=parameters["do_sample"], temperature=parameters["temperature"],
                        top_k=parameters["top_k"], top_p=parameters["top_p"], history=history,
                        repetition_penalty=parameters["repetition_penalty"],
                        suppressed=tuple(range(self.codebook_size, self.vocab_size)),
                    )
                    if first == self.eos_token:
                        break
                    frame = self._depth_frame(first, branches, scale, rng, parameters)
                    history.append(first)
                    if codec is not None:
                        audio = codec.push(frame, frame_index)
                    else:
                        decoded = self.decoder.run({
                            "codes": np.asarray(frame, dtype=np.int64).reshape(1, 1, self.num_codebooks),
                            "position": np.asarray([frame_index], dtype=np.int64), **codec_states,
                        })
                        codec_states = {item["name"]: decoded[f"next_{item['name']}"]
                                        for item in self.codec_metadata["state_specs"]}
                        audio = as_numpy(decoded["audio"]).reshape(-1).astype(np.float32, copy=False)
                    if audio is not None:
                        buffer.append(audio)
                    # 首包凑齐自身帧数即收割当前音频，后续保持声码器与生成流水重叠。
                    if codec is not None and stream and frame_index + 1 == parameters["chunk_frames"]:
                        audio = codec.finish()
                        if audio is not None:
                            buffer.append(audio)
                    if stream and len(buffer) >= parameters["chunk_frames"]:
                        yield np.concatenate(buffer)
                        buffer.clear()
                    if frame_index + 1 == parameters["max_new_tokens"]:
                        break
                    embedded = self.audio_embedding.run({"codes": np.asarray(frame, dtype=np.int64).reshape(1, 1, -1)})["embeddings"]
                    output = self.backbone.decode(
                        embeddings=np.repeat(as_numpy(embedded), len(branches), axis=0),
                        positions=lengths, sequence_ids=list(range(len(branches))),
                    )
                    lengths = [length + 1 for length in lengths]
                    branches = [{name: value[index:index + 1] for name, value in output.items() if value is not None}
                                for index in range(len(branches))]
                if codec is not None:
                    audio = codec.finish()
                    if audio is not None:
                        buffer.append(audio)
                if buffer:
                    yield np.concatenate(buffer)
                succeeded = True
            finally:
                if codec is not None:
                    codec.close()
                # 正常结束、推理异常和生成器取消均清除本次请求的序列状态。
                self._prefill.finish(succeeded)
                self.depth.reset()

    def _depth_frame(self, first: int, branches: list[dict], scale: float, rng: np.random.Generator, parameters: dict) -> list[int]:
        """使用原生码本嵌入与全部位置输出头，生成当前帧剩余码本。

        Args:
            first: 主干已采样的首码本。
            branches: 各条件分支独立的主干输出。
            scale: 条件引导强度。
            rng: 本次请求的随机数生成器。
            parameters: 深度采样参数。
        Returns:
            顺序完整的十六码本编号。
        """
        batch_size = len(branches)
        sequence_ids = list(range(batch_size))
        hidden = np.concatenate([branch["hidden"] for branch in branches], axis=0)
        condition = self.depth_conditioner.run({"backbone_hidden": hidden})["embeddings"]
        self.depth.reset()
        self.depth.decode(
            embeddings=as_numpy(condition)[:, None], positions=[0] * batch_size,
            sequence_ids=sequence_ids, output=False,
        )
        frame = [first]
        for position in range(1, self.num_codebooks):
            offset = (position - 1) * self.vocab_size
            output = self.depth.decode(
                tokens=np.full((batch_size, 1), offset + frame[-1], dtype=np.int32),
                positions=[position] * batch_size, sequence_ids=sequence_ids,
                logit_start=offset, logit_count=self.vocab_size,
            )
            logits = output["logits"]
            frame.append(sample_token(
                guided_logits(logits[0], None if batch_size == 1 else logits[1], scale), rng,
                do_sample=parameters["do_sample"], temperature=parameters["depth_temperature"],
                top_k=parameters["depth_top_k"], top_p=parameters["depth_top_p"],
                suppressed=tuple(range(self.codebook_size, self.vocab_size)), cfg_depth=batch_size > 1,
            ))
        return frame

    def _ensure_open(self) -> None:
        """拒绝已关闭运行时的新请求。

        Returns:
            仍可使用时正常返回。
        """
        if self._closed:
            raise RuntimeError("Breeze 运行时已经关闭。")

    def shutdown(self) -> None:
        """释放原生模型与 ONNX 组件，允许重复关闭。

        Returns:
            所有资源关闭后返回。
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Breeze 仍有活动请求，请先关闭音频生成器再关闭运行时。")
        try:
            self._closed = True
            self._reference_text_cache.clear()
            if self._codec_executor is not None:
                self._codec_executor.shutdown(wait=True)
                self._codec_executor = None
            for name in ("decoder", "text_encoder", "audio_encoder", "audio_embedding", "depth_conditioner"):
                graph = getattr(self, name, None)
                if graph is not None:
                    graph.close()
                    setattr(self, name, None)
            for name in ("depth", "backbone"):
                decoder = getattr(self, name, None)
                if decoder is not None:
                    decoder.close()
                    setattr(self, name, None)
            self.audio_encoder = self.text_encoder = self.audio_embedding = None
            self.depth_conditioner = self.decoder = self.tokenizer = None
        finally:
            self._lock.release()
