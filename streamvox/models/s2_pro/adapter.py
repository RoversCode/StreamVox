from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import librosa
import numpy as np

from ...hardware import ModelFootprintProfile
from ...prompt_audio import build_prompt_reference
from ...utils.log import logger
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, PromptCacheRecord, RuntimeRequest, SerializedPromptAsset
from ..text_processing import TextPreprocessConfig
from .text import S2_PRO_TEXT_PREPROCESS_CONFIG
from .spec import S2_PRO_4B_GGUF_SPEC, S2ProModelSpec

if TYPE_CHECKING:
    from .runtime import S2ProRuntime


class S2Pro4BGGUFAdapter:
    spec: S2ProModelSpec = S2_PRO_4B_GGUF_SPEC

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def load_runtime(self, request: RuntimeRequest) -> "S2ProRuntime":
        from .runtime import S2ProRuntime

        logger.info("[StreamVox] loading s2-pro runtime")
        return S2ProRuntime(
            model_dir=Path(request.model_dir),
            model_spec=self.spec,
            device=request.device,
            n_ctx=request.n_ctx,
            verify_model_sha256=request.verify_model_sha256,
            execution_plan=request.execution_plan,
            text_preprocess_config=request.text_preprocess_config,
        )

    def get_hardware_profile(self) -> ModelFootprintProfile:
        return ModelFootprintProfile(
            profile_id=self.spec.hardware_profile_id,
            parameter_billion=self.spec.parameter_billion,
            quantization="gguf(q8_0)+onnx(fp16)",
            min_vram_gb=self.spec.min_vram_gb,
            recommended_vram_gb=self.spec.recommended_vram_gb,
            min_ram_gb=self.spec.min_ram_gb,
            recommended_ram_gb=self.spec.recommended_ram_gb,
            notes=self.spec.notes,
        )

    def get_text_preprocess_config(self) -> TextPreprocessConfig:
        # 返回 S2-Pro 专属文本规则；language 由用户提供，文本层只处理特殊 token 和分段规则。
        return S2_PRO_TEXT_PREPROCESS_CONFIG

    def resolve_execution_plan(self, device: str, llama_backend: str | None = None) -> ExecutionPlan:
        mode, index = self._parse_device_spec(device)
        # S2-Pro 的两个 runtime family 是 GGUF 和 ONNX codec，公开 device 语义仍保持统一入口。
        return ExecutionPlan(
            raw_device=device,
            targets={
                "llama": ExecutionTarget(
                    family="llama",
                    mode=mode,
                    device_index=index,
                    backend_preference=llama_backend,
                ),
                "onnx": ExecutionTarget(family="onnx", mode=mode, device_index=index),
            },
        )

    def _parse_device_spec(self, device: str) -> tuple[str, int | None]:
        if not isinstance(device, str):
            raise TypeError("device must be a string: auto, cpu, gpu or gpu:<index>")
        raw = device.strip().lower()
        if raw in {"auto", "cpu"}:
            return raw, None
        if raw == "gpu":
            return "gpu", 0
        if raw.startswith("gpu:"):
            index_text = raw.split(":", 1)[1]
            if not index_text.isdigit():
                raise ValueError(f"device index must be a non-negative integer: {device}")
            return "gpu", int(index_text)
        raise ValueError(f"unsupported device value: {device}")

    def normalize_synthesize_kwargs(self, **model_kwargs: object) -> dict[str, object]:
        valid_params = {
            "max_length",
            "min_length",
            "remove_meaningless_chars",
            "temperature",
            "top_p",
            "top_k",
        }
        normalized = {key: value for key, value in model_kwargs.items() if key in valid_params}
        for name in ("max_length", "min_length", "top_k"):
            if normalized.get(name) is not None and (not isinstance(normalized[name], int) or isinstance(normalized[name], bool)):
                raise ValueError(f"{name} must be an integer")
        if normalized.get("remove_meaningless_chars") is not None and not isinstance(normalized["remove_meaningless_chars"], bool):
            raise ValueError("remove_meaningless_chars must be a boolean")
        for name in ("temperature", "top_p"):
            if normalized.get(name) is not None and not isinstance(normalized[name], (int, float)):
                raise ValueError(f"{name} must be a number")
        return normalized

    def make_prompt(
        self,
        runtime: "S2ProRuntime",
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BasePromptAsset:
        """
        构建 S2-Pro 单参考或多参考 prompt。

        参数:
            runtime: 已加载的 S2-Pro runtime。
            role_name: 当前 prompt 的角色名。
            sample_rate: 单个采样率，或多条内存音频对应的采样率列表。
            audio_path: 单个音频路径或多个音频路径。
            audio_data: 单个音频数组或多个音频数组。
            prompt_text: 单条参考文本或多条参考文本。

        返回:
            返回 S2-Pro prompt 资产；其中 slow prefix cache 已在 make_prompt 阶段完成预填充。

        异常:
            当缺少参考文本、缺少音频、列表长度不匹配、音频编码失败或参考 prompt 超出上下文时抛出异常。
        """
        from .runtime import build_s2_pro_role_prefix_state, prompt_asset_from_payload

        # S2-Pro 的参考音频 token 必须和文本对齐，缺少 prompt_text 会导致 system reference prompt 不完整。
        if prompt_text is None:
            raise ValueError("s2-pro make_prompt requires prompt_text to align reference speech tokens")
        # 单条和多条参考音频统一归一化成列表，后续 codec 编码和拼接逻辑只需要处理一种形态。
        reference_items = self._normalize_prompt_references(
            runtime=runtime,
            sample_rate=sample_rate,
            audio_path=audio_path,
            audio_data=audio_data,
        )
        # 多条 prompt_text 按位置对应多条参考音频；单条 prompt_text 保持模型原有语义，由调用方自行组织完整参考文本。
        if isinstance(prompt_text, list):
            if len(prompt_text) != len(reference_items):
                raise ValueError("prompt_text list length must match prompt audio list length")
            if any(not item.strip() for item in prompt_text):
                raise ValueError("s2-pro prompt_text entries must be non-empty")
        elif not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("s2-pro make_prompt requires non-empty prompt_text")
        normalized_prompt_text = (
            prompt_text
            if isinstance(prompt_text, list)
            else [prompt_text] + [None for _ in range(len(reference_items) - 1)]
        )

        encoded_refs: list[np.ndarray] = []
        prompt_references = []
        # 每条参考音频分别编码，避免不同来源音频在重采样和形状规整时互相影响。
        for index, item in enumerate(reference_items):
            audio = item["audio"]
            input_rate = item["sample_rate"]
            # runtime codec encoder 只接受固定采样率；不一致时先按参考音频自己的采样率重采样。
            if input_rate != runtime.sample_rate:
                logger.info("S2-Pro prompt audio %s resampling from %sHz to %sHz", index, input_rate, runtime.sample_rate)
                audio = librosa.resample(audio, orig_sr=input_rate, target_sr=runtime.sample_rate)
            # codec encoder 要求三维 batch/channel/time 输入，这里每条参考音频独立编码成 (codebooks, T)。
            reference_audio = audio.reshape(1, 1, -1).astype(np.float32, copy=False)
            encoded_refs.append(runtime.codec_encoder.encode_audio(reference_audio)[0]) # shape [10, T]
            prompt_references.append(build_prompt_reference(audio=audio, sample_rate=runtime.sample_rate, index=index))
        # 多参考音频在 prompt 序列里按时间维拼接成一段 system_vq；文本侧由 prompt_builder 负责 speaker tag 组织。
        encoded = np.concatenate(encoded_refs, axis=1)

        # 只有单参考音频才用参考 token 预热 true-streaming decoder；多参考默认视作多说话人参考，避免把 decoder 初态偏向某一个说话人。
        if len(encoded_refs) == 1:
            decoder_state, decoder_tail_codes = runtime.decoder.prewarm_decoder_state(encoded, task_id=f"prompt:{role_name}")
        else:
            decoder_state = {}
            decoder_tail_codes = np.zeros((0, 0), dtype=np.int64)

        # slow model 的 system/reference prefill 与后续具体 role 无关，提前在 make_prompt 中完成并持久化，使用时直接恢复。
        slow_prefix_state, slow_prefix_position = build_s2_pro_role_prefix_state(
            runtime,
            prompt_text=prompt_text,
            prompt_codes=encoded,
        )

        prompt = prompt_asset_from_payload(
            role_name=role_name,
            model_name=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_text=prompt_text,
            prompt_codes=encoded,
            decoder_state=decoder_state,
            decoder_tail_codes=decoder_tail_codes,
            slow_prefix_state=slow_prefix_state,
            slow_prefix_position=slow_prefix_position,
            metadata={},
        )
        prompt.prompt_cache_record = PromptCacheRecord(
            prompt_text=list(normalized_prompt_text),
            references=tuple(prompt_references),
        )
        return prompt

    def _normalize_prompt_references(
        self,
        *,
        runtime: "S2ProRuntime",
        sample_rate: int | list[int],
        audio_path: str | list[str] | None,
        audio_data: np.ndarray | list[np.ndarray] | None,
    ) -> list[dict[str, np.ndarray | int]]:
        """
        将 S2-Pro prompt 音频输入统一成列表。

        参数:
            runtime: 已加载的 S2-Pro runtime，用于路径音频读取时确定目标采样率。
            sample_rate: 单个采样率或内存音频采样率列表。
            audio_path: 单个路径、路径列表或 None。
            audio_data: 单个数组、数组列表或 None。

        返回:
            返回若干字典，每个字典包含 audio 波形数组和 sample_rate 采样率。

        异常:
            当音频来源缺失、路径与内存音频混用、列表为空或采样率长度不匹配时抛出异常。
        """
        # 路径和内存音频是互斥来源；同时传入会让位置对齐和采样率解释变得不确定。
        if audio_path is not None and audio_data is not None:
            raise ValueError("provide either audio_path or audio_data, not both")
        # 路径模式下由 librosa 直接读取并转到 runtime 采样率，后续无需依赖调用方采样率。
        if audio_path is not None:
            paths = audio_path if isinstance(audio_path, list) else [audio_path]
            if not paths:
                raise ValueError("audio_path list must not be empty")
            references: list[dict[str, np.ndarray | int]] = []
            for path in paths:
                if not isinstance(path, str) or not path:
                    raise TypeError("each audio_path item must be a non-empty string")
                audio, input_rate = librosa.load(path, sr=runtime.sample_rate, mono=True)
                references.append({"audio": np.asarray(audio, dtype=np.float32).reshape(-1), "sample_rate": int(input_rate)})
            return references
        # 缺少路径时必须提供内存音频，否则无法构建参考 speech token。
        if audio_data is None:
            raise ValueError("audio_path or audio_data is required to build a s2-pro prompt")
        # 内存音频同样统一成列表，单条输入走单元素列表以复用后续校验和转换逻辑。
        audios = audio_data if isinstance(audio_data, list) else [audio_data]
        if not audios:
            raise ValueError("audio_data list must not be empty")
        # 多条内存音频可以使用等长采样率列表，也可以使用一个采样率广播到全部音频。
        if isinstance(sample_rate, list):
            if len(sample_rate) != len(audios):
                raise ValueError("sample_rate list length must match prompt audio list length")
            rates = sample_rate
        else:
            rates = [sample_rate for _ in audios]
        references = []
        for audio, rate in zip(audios, rates, strict=True):
            if not isinstance(audio, np.ndarray):
                raise TypeError("each audio_data item must be a numpy.ndarray")
            if not isinstance(rate, int):
                raise TypeError("each sample_rate item must be an int")
            references.append({"audio": np.asarray(audio, dtype=np.float32).reshape(-1), "sample_rate": int(rate)})
        return references

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        from .runtime import S2ProPromptAsset, serialize_prompt_asset

        if not isinstance(asset, S2ProPromptAsset):
            raise TypeError("s2-pro adapter can only serialize S2ProPromptAsset")
        return serialize_prompt_asset(asset)

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BasePromptAsset:
        from .runtime import deserialize_prompt_asset

        return deserialize_prompt_asset(payload)

    def synthesize(
        self,
        runtime: "S2ProRuntime",
        text: str,
        prompt: BasePromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        max_length: int | None = None,
        min_length: int | None = None,
        remove_meaningless_chars: bool = False,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        session = runtime.create_session()
        yield from session.synthesize(
            text=text,
            prompt=prompt,
            stream=stream,
            track_performance=track_performance,
            max_length=max_length,
            min_length=min_length,
            remove_meaningless_chars=remove_meaningless_chars,
            generation_options=model_kwargs,
        )
