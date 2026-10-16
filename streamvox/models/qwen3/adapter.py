from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import librosa
from ...utils.log import logger
from ...prompt_audio import build_prompt_reference
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, PromptCacheRecord, RuntimeRequest, SerializedPromptAsset
from ..text_processing import TextPreprocessConfig
from ...hardware import ModelFootprintProfile
from .schema.constant import LANGUAGE_MAP
from .spec import QWEN3_CLONE_06B_SPEC, QWEN3_CLONE_17B_SPEC, Qwen3CloneModelSpec

if TYPE_CHECKING:
    from .runtime import Qwen3Runtime


MODEL_NAME = QWEN3_CLONE_17B_SPEC.model_name


class Qwen3TTSCloneAdapter:
    spec: Qwen3CloneModelSpec = QWEN3_CLONE_17B_SPEC

    def load_runtime(self, request: RuntimeRequest) -> "Qwen3Runtime":
        from .runtime import Qwen3Runtime

        logger.info("[StreamVox] loading qwen3 clone runtime")
        return Qwen3Runtime(
            model_dir=Path(request.model_dir),
            model_spec=self.spec,
            device=request.device,
            n_ctx=request.n_ctx,
            verify_model_sha256=request.verify_model_sha256,
            execution_plan=request.execution_plan,
            text_preprocess_config=request.text_preprocess_config,
        )

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def get_hardware_profile(self) -> ModelFootprintProfile:
        """
        qwen3 adapter 自己声明该具体规格的最低/推荐硬件要求。
        engine 初始化时只对比“当前模型要求”和“当前机器资源”。
        """
        # 同一 qwen3 clone 系列共享推理路径，但不同尺寸的资源占用不同，硬件档位必须跟着规格走。
        return ModelFootprintProfile(
            profile_id=self.spec.hardware_profile_id,
            parameter_billion=self.spec.parameter_billion,
            quantization="mixed(q5/q8/fp16)",
            min_vram_gb=self.spec.min_vram_gb, # 最低显存要求
            recommended_vram_gb=self.spec.recommended_vram_gb, # 推荐显存要求
            min_ram_gb=self.spec.min_ram_gb, # 最低内存要求
            recommended_ram_gb=self.spec.recommended_ram_gb, # 推荐内存要求
            notes=self.spec.notes,
        )

    def get_text_preprocess_config(self) -> TextPreprocessConfig:
        """
        qwen3 只声明文本处理阶段需要原样保留的特殊 token。
        language 由用户显式传入，文本层不再做语言检测或语言专属清理。
        """
        return TextPreprocessConfig(
            special_tokens=(
                "<|im_start|>",
                "<|im_end|>",
                # "<laughter>",
                # "<laugh>",
                # "<cry>",
                # "<cough>",
                # "<sigh>",
                # "<breath>",
            ),
            # special_token_wrappers=(
            #     ("<", ">"),
            #     ("[", "]"),
            # ),
        )

    def resolve_execution_plan(
        self,
        device: str,
        llama_backend: str | None = None,
    ) -> ExecutionPlan:
        '''
            qwen3 当前包含两个 runtime family：
            - llama: GGUF / llama.cpp
            - onnx: encoder / decoder
            引擎层只吃通用的 auto / cpu / gpu，细节不泄漏到 TTSEngine。
        '''
        default_mode, default_index = self._parse_device_spec(device) # mode, index
        targets: dict[str, ExecutionTarget] = {}
        for family in ("llama", "onnx"):
            # 当前公开 device 语义只有一个全局意图，llama.cpp 和 ONNX decoder 各自在 runtime 内落到具体后端。
            targets[family] = ExecutionTarget(
                family=family,
                mode=default_mode,
                device_index=default_index,
                backend_preference=llama_backend if family == "llama" else None,
            )
        return ExecutionPlan(raw_device=device, targets=targets)

    def _parse_device_spec(
        self,
        device: str,
    ) -> tuple[str, int | None]:
        '''
            只支持简单入口：auto / cpu / gpu / gpu:1。
        '''
        if not isinstance(device, str):
            raise TypeError("device must be a string: auto, cpu, gpu or gpu:<index>")
        return self._parse_device_value(device)

    def _parse_device_value(self, value: str) -> tuple[str, int | None]:
        raw = value.strip().lower()
        if raw in {"auto", "cpu"}:
            return raw, None
        if raw == "gpu":
            return "gpu", 0
        if raw.startswith("gpu:"):
            index_text = raw.split(":", 1)[1]
            if not index_text.isdigit():
                raise ValueError(f"device index must be a non-negative integer: {value}")
            return "gpu", int(index_text)
        raise ValueError(f"unsupported device value: {value}")

    def normalize_synthesize_kwargs(
        self,
        **model_kwargs: object,
    ) -> dict[str, object]:
        '''规范化模型私有参数'''
        valid_params = ["icl", "stream", "language", "max_length", "min_length", "remove_meaningless_chars"]
        normalized_model_kwargs = {
            key: value for key, value in model_kwargs.items() if key in valid_params
        }
        # 验证icl和stream以及language是否合法
        if normalized_model_kwargs.get("language") is not None:
            if normalized_model_kwargs.get("language") not in LANGUAGE_MAP and normalized_model_kwargs.get("language") != "auto":
                raise ValueError(f"language {normalized_model_kwargs.get('language')} is not supported")
        if normalized_model_kwargs.get("icl") is not None and not isinstance(normalized_model_kwargs.get("icl"), bool):
            raise ValueError(f"icl must be a boolean")
        if normalized_model_kwargs.get("stream") is not None and not isinstance(normalized_model_kwargs.get("stream"), bool):
            raise ValueError(f"stream must be a boolean")
        if normalized_model_kwargs.get("remove_meaningless_chars") is not None and not isinstance(normalized_model_kwargs.get("remove_meaningless_chars"), bool):
            raise ValueError("remove_meaningless_chars must be a boolean")
        # 提前在 adapter 层校验分段阈值，避免 runtime 跑到一半才因为非法参数中断。
        if normalized_model_kwargs.get("max_length") is not None:
            max_length = normalized_model_kwargs.get("max_length")
            if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
                raise ValueError("max_length must be a positive integer")
        if normalized_model_kwargs.get("min_length") is not None:
            min_length = normalized_model_kwargs.get("min_length")
            if not isinstance(min_length, int) or isinstance(min_length, bool) or min_length < 0:
                raise ValueError("min_length must be a non-negative integer")
        if (
            normalized_model_kwargs.get("max_length") is not None
            and normalized_model_kwargs.get("min_length") is not None
            and int(normalized_model_kwargs["min_length"]) > int(normalized_model_kwargs["max_length"])
        ):
            raise ValueError("min_length must be less than or equal to max_length")

        return normalized_model_kwargs

    def make_prompt(
        self,
        runtime: "Qwen3Runtime",
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BasePromptAsset:
        """
        构建 Qwen3 单参考音频 prompt。

        参数:
            runtime: 已加载的 Qwen3 runtime。
            role_name: 当前 prompt 的角色名。
            sample_rate: 单条内存音频采样率。
            audio_path: 单条参考音频路径。
            audio_data: 单条参考音频数组。
            prompt_text: 单条参考文本。

        返回:
            返回 Qwen3 clone prompt 资产。

        异常:
            当 Qwen3 编码器未加载、缺少音频输入、传入列表输入或采样率不合法时抛出异常。
        """
        from .runtime import prompt_asset_from_clone_payload

        # Qwen3 clone 当前模型结构只支持一个参考音频；列表输入在模型语义上不可降级，必须显式拒绝。
        if any(isinstance(value, list) for value in (sample_rate, audio_path, audio_data, prompt_text)):
            raise ValueError("qwen3 make_prompt supports exactly one prompt audio; list inputs are not supported")
        # prompt_text 参与 prompt builder 的参考文本字段，Qwen3 这里只允许单条字符串或缺省值。
        if prompt_text is not None and not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string or None")
        # 内存音频采样率必须是单个整数；列表已经在上方拒绝，这里防止其它非整数类型漏入重采样流程。
        if not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an int")
        # Qwen3 clone prompt 依赖 codec encoder 和 speaker encoder 同时存在，缺任一组件都无法构建完整 prompt。
        if runtime.codec_encoder is None or runtime.speaker_encoder is None:
            raise RuntimeError("qwen3 clone prompt extraction requires codec and speaker encoders")
        # 未直接传入内存音频时，从文件路径读取；路径也缺失则无法构建 clone prompt。
        if audio_data is None:
            if audio_path is None:
                raise ValueError("audio_path or audio_data is required to build a clone prompt")
            audio_data, input_rate = librosa.load(audio_path, sr=runtime.sample_rate)
        # 直接传入内存音频时，采样率来自调用方显式提供的 sample_rate。
        else:
            input_rate = sample_rate

        # 模型 encoder 只接受 runtime 采样率，输入不一致时先重采样保证 codec token 时间尺度正确。
        if input_rate != runtime.sample_rate:
            logger.info(f"⚠️ 音频采样率不匹配，正在重新采样到 {runtime.sample_rate}Hz")
            audio_data = librosa.resample(audio_data, orig_sr=input_rate, target_sr=runtime.sample_rate)

        # codec token、speaker embedding 和 decoder warm state 共同构成后续流式合成可复用的 clone prompt。
        prompt_codes = runtime.codec_encoder.encode(audio_data)
        speaker_embedding = runtime.speaker_encoder.encode(audio_data)
        decoder_state = runtime.warm_decoder_state(prompt_codes)

        prompt = prompt_asset_from_clone_payload(
            role_name=role_name,
            model_name=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_text=prompt_text,
            speaker_embedding=speaker_embedding,
            prompt_codes=prompt_codes,
            decoder_state=decoder_state,
            metadata={},
        )
        prompt.prompt_cache_record = PromptCacheRecord(
            prompt_text=[prompt_text],
            references=(
                build_prompt_reference(audio=audio_data, sample_rate=runtime.sample_rate, index=0),
            ),
        )
        return prompt

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        from .runtime import Qwen3ClonePromptAsset, serialize_prompt_asset

        if not isinstance(asset, Qwen3ClonePromptAsset):
            raise TypeError("qwen3 adapter can only serialize Qwen3ClonePromptAsset")
        return serialize_prompt_asset(asset)

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BasePromptAsset:
        from .runtime import deserialize_prompt_asset

        return deserialize_prompt_asset(payload)

    def synthesize(
        self,
        runtime: "Qwen3Runtime",
        text: str,
        prompt: BasePromptAsset,
        icl: bool = False,
        stream: bool = True,
        language: str = "auto",
        track_performance: bool = False,
        max_length: int|None = None,
        min_length: int|None = None,
        remove_meaningless_chars: bool = False,
    ) -> Iterable[np.ndarray]:
        session = runtime.create_session()
        yield from session.synthesize(
            text=text,
            prompt=prompt,
            language=language,
            stream=stream,
            icl=icl,
            track_performance=track_performance,
            max_length=max_length,
            min_length=min_length,
            remove_meaningless_chars=remove_meaningless_chars,
        )


class Qwen3TTSClone06BAdapter(Qwen3TTSCloneAdapter):
    spec = QWEN3_CLONE_06B_SPEC


class Qwen3TTSClone17BAdapter(Qwen3TTSCloneAdapter):
    spec = QWEN3_CLONE_17B_SPEC
