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
from .spec import VOXCPM2_GGUF_SPEC, VoxCPM2GGUFModelSpec
from .text import VOXCPM2_TEXT_PREPROCESS_CONFIG

if TYPE_CHECKING:
    from .runtime import VoxCPM2GGUFRuntime


SUPPORTED_MODES = {"text", "ref", "continuation", "ref_continuation"}
CONTROL_TEXT_MODES = {"text", "ref"}
# 数字模式按统一的零基序号组织，避免 voxcpm2 在公开 API 上额外保留重复编号。
MODE_CODE_TO_NAME = {
    0: "text",
    1: "ref",
    2: "continuation",
    3: "ref_continuation",
}


class VoxCPM2GGUFAdapter:
    """
    VoxCPM2-GGUF 的 StreamVox 模型适配器。

    参数:
        spec: 模型规格，默认使用 voxcpm2-gguf。

    返回:
        adapter 负责被 registry 调用，不直接返回音频；音频由 synthesize 生成器产出。

    边界异常:
        mode、prompt、control_text 和采样参数非法时在 adapter 层提前抛出清晰异常。
    """

    spec: VoxCPM2GGUFModelSpec = VOXCPM2_GGUF_SPEC

    @property
    def model_name(self) -> str:
        """返回 StreamVox 对外模型名。"""

        return self.spec.model_name

    def load_runtime(self, request: RuntimeRequest) -> "VoxCPM2GGUFRuntime":
        """按 RuntimeRequest 加载 VoxCPM2-GGUF runtime。"""

        from .runtime import VoxCPM2GGUFRuntime

        logger.info("[StreamVox] loading voxcpm2-gguf runtime")
        return VoxCPM2GGUFRuntime(
            model_dir=Path(request.model_dir),
            model_spec=self.spec,
            device=request.device,
            n_ctx=request.n_ctx,
            verify_model_sha256=request.verify_model_sha256,
            execution_plan=request.execution_plan,
            text_preprocess_config=request.text_preprocess_config,
        )

    def get_text_preprocess_config(self) -> TextPreprocessConfig:
        """返回 VoxCPM2-GGUF 的文本预处理配置声明。"""

        return VOXCPM2_TEXT_PREPROCESS_CONFIG

    def get_hardware_profile(self) -> ModelFootprintProfile:
        """返回硬件资源建议，用于 TTSEngine 初始化阶段打印提示。"""

        return ModelFootprintProfile(
            profile_id=self.spec.hardware_profile_id,
            parameter_billion=self.spec.parameter_billion,
            # 当前 ONNX 资产已经是混合精度：LocEnc / LocDiT estimator 为 FP32，
            # AudioVAE encoder / decoder 仍为 FP16。这里必须如实暴露，避免设备评估误判。
            quantization="gguf(q8_0)+onnx(mixed: locenc/locdit fp32, audiovae fp16)",
            min_vram_gb=self.spec.min_vram_gb,
            recommended_vram_gb=self.spec.recommended_vram_gb,
            min_ram_gb=self.spec.min_ram_gb,
            recommended_ram_gb=self.spec.recommended_ram_gb,
            notes=self.spec.notes,
        )

    def resolve_execution_plan(self, device: str, llama_backend: str | None = None) -> ExecutionPlan:
        """把用户 device 字符串解析为 llama 与 onnx 两类 runtime family。"""

        mode, index = self._parse_device_spec(device)
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
        """解析 auto/cpu/gpu/gpu:N 设备字符串。"""

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
        """
        校验并归一化 VoxCPM2-GGUF 私有合成参数。

        参数:
            model_kwargs: TTSEngine.stream 透传的模型私有参数。
            mode: 支持文本模式名，也支持数字模式值；数字模式按
                `0=text, 1=ref, 2=continuation, 3=ref_continuation` 解释。

        返回:
            返回只包含受支持字段的参数字典；未显式传入的 mode/min_length/max_length
            继续由 runtime 配置和 prompt 实际形态决定默认值。

        边界异常:
            类型错误、mode 不支持或数值参数越界时抛出 ValueError/TypeError。
        """

        valid_params = {
            "mode",
            "control_text",
            "min_length",
            "max_length",
            "remove_meaningless_chars",
            "seed",
        }
        # 过滤非合法参数
        normalized = {key: value for key, value in model_kwargs.items() if key in valid_params and value is not None}
        if "mode" in normalized:
            normalized["mode"] = self._normalize_mode_value(normalized["mode"])
        if "control_text" in normalized and not isinstance(normalized["control_text"], str):
            raise TypeError("control_text must be a string")
        for name in ("min_length", "max_length", "seed"):
            if name in normalized and (not isinstance(normalized[name], int) or isinstance(normalized[name], bool)):
                raise TypeError(f"{name} must be an integer")
        if "min_length" in normalized and int(normalized["min_length"]) < 0:
            raise ValueError("min_length must be greater than or equal to 0")
        if "max_length" in normalized and int(normalized["max_length"]) <= 0:
            raise ValueError("max_length must be greater than 0")
        if (
            "min_length" in normalized
            and "max_length" in normalized
            and int(normalized["min_length"]) > int(normalized["max_length"])
        ):
            raise ValueError("min_length must be less than or equal to max_length")
        if "remove_meaningless_chars" in normalized and not isinstance(normalized["remove_meaningless_chars"], bool):
            raise TypeError("remove_meaningless_chars must be a boolean")
        return normalized

    def _normalize_mode_value(self, raw_mode: object) -> str:
        """
        将用户传入的文本或数字 mode 统一归一化为内部模式名。

        参数:
            raw_mode: 用户传入的模式值，允许字符串模式名、数字字符串或整数。

        返回:
            返回 `text/ref/continuation/ref_continuation` 之一。

        边界异常:
            不支持的数字、空字符串或布尔值会抛出清晰异常，避免静默落到错误模式。
        """

        if isinstance(raw_mode, bool):
            raise TypeError("mode must be a string or integer code")
        if isinstance(raw_mode, int):
            normalized_mode = MODE_CODE_TO_NAME.get(int(raw_mode))
            if normalized_mode is None:
                raise ValueError("mode integer must be one of: 0, 1, 2, 3")
            return normalized_mode
        if not isinstance(raw_mode, str):
            raise TypeError("mode must be a string or integer code")

        normalized_text = raw_mode.strip().lower()
        if not normalized_text:
            raise ValueError("mode must be a non-empty string or integer code")
        if normalized_text.isdigit():
            mode_code = int(normalized_text)
            normalized_mode = MODE_CODE_TO_NAME.get(mode_code)
            if normalized_mode is None:
                raise ValueError("mode integer must be one of: 0, 1, 2, 3")
            return normalized_mode
        if normalized_text not in SUPPORTED_MODES:
            raise ValueError("mode must be one of: text, ref, continuation, ref_continuation")
        return normalized_text

    def _resolve_effective_mode(
        self,
        *,
        runtime: "VoxCPM2GGUFRuntime",
        prompt: BasePromptAsset,
        mode: str | int | None,
        prompt_text: str | None,
    ) -> str:
        """
        结合显式 mode、prompt 资产和 runtime 配置计算最终模式。

        参数:
            runtime: 当前 voxcpm2 runtime，用于读取配置默认值。
            prompt: 已解析好的 prompt 资产。
            mode: 用户显式传入的模式；为空时按 prompt 形态推导默认值。
            prompt_text: 本次 synthesize 额外透传的 prompt_text；为空时回退到 prompt 资产里的文本。

        返回:
            返回最终内部模式名。

        边界异常:
            prompt 类型错误会在 `_validate_mode_prompt` 中统一处理；这里专注默认模式推导。
        """

        from .runtime import VoxCPM2PromptAsset

        if not isinstance(prompt, VoxCPM2PromptAsset):
            raise TypeError("voxcpm2-gguf synthesize requires VoxCPM2PromptAsset")
        if mode is not None:
            normalized_mode = self._normalize_mode_value(mode)
            effective_prompt_text = prompt_text if prompt_text not in (None, "") else prompt.prompt_text
            if normalized_mode in {"continuation", "ref_continuation"}:
                if effective_prompt_text not in (None, ""):
                    return normalized_mode
                # 只有存在参考音频时，才能把 continuation 类模式安全回退到 ref；
                # 否则继续保留原 mode，让后续统一校验给出明确错误。
                if self._has_reference_audio(prompt):
                    logger.warning(
                        "voxcpm2-gguf prompt has audio but no prompt_text; requested mode=%s will fall back to mode=ref",
                        normalized_mode,
                    )
                    return "ref"
            return normalized_mode

        # 无 prompt 音频时默认走 text；这也是 voxcpm2 与其它必须参考音频模型的主要差异。
        if not self._has_prompt_audio(prompt) and not self._has_reference_audio(prompt):
            return "text"
        # 当前调用如果明确给了 prompt_text，优先理解成 continuation 语义，避免 prompt_text 被静默丢弃。
        effective_prompt_text = prompt_text if prompt_text not in (None, "") else prompt.prompt_text
        if effective_prompt_text not in (None, ""):
            return "continuation"
        # 只有参考音频、没有 prompt_text 时，回退到配置文件声明的默认模式。
        normalized_mode = self._normalize_mode_value(runtime.config.default_mode)
        if normalized_mode in {"continuation", "ref_continuation"} and self._has_reference_audio(prompt):
            logger.warning(
                "voxcpm2-gguf prompt has audio but no prompt_text; requested mode=%s will fall back to mode=ref",
                normalized_mode,
            )
            return "ref"
        return normalized_mode

    def _has_prompt_audio(self, prompt: "VoxCPM2PromptAsset") -> bool:
        """
        判断 prompt 是否携带 continuation 路径所需的提示音频。

        参数:
            prompt: 当前请求使用的 prompt 资产。

        返回:
            只要存在路径音频、内存音频或预编码特征，就返回 True。

        边界异常:
            本函数只做轻量字段判断，不触发文件读取。
        """

        return (
            prompt.prompt_audio_feat is not None
            or prompt.prompt_locenc_hidden is not None
        )

    def _has_reference_audio(self, prompt: "VoxCPM2PromptAsset") -> bool:
        """
        判断 prompt 是否携带 reference 路径所需的参考音频。

        参数:
            prompt: 当前请求使用的 prompt 资产。

        返回:
            只要存在路径音频、内存音频或预编码特征，就返回 True。

        边界异常:
            本函数只做轻量字段判断，不触发文件读取。
        """

        return (
            prompt.reference_audio_feat is not None
            or prompt.reference_locenc_hidden is not None
        )

    def _normalize_control_text_for_mode(self, *, mode: str, control_text: str | None) -> str | None:
        """
        根据最终模式决定是否保留 control_text。

        参数:
            mode: 已归一化的最终模式名。
            control_text: 用户传入的控制文本。

        返回:
            支持模式下原样返回 control_text；不支持模式下返回 None。

        边界异常:
            不支持模式不会抛错，而是记录警告后忽略，符合 voxcpm2 当前公开 API 约定。
        """

        if control_text in (None, ""):
            return control_text
        if mode in CONTROL_TEXT_MODES:
            return control_text
        # 当前用户已经显式传了 control_text，直接抛错会让 continuation/ref_continuation
        # 与参考实现的“忽略但继续跑”语义不一致；这里统一改成警告并丢弃。
        logger.warning(
            "voxcpm2-gguf ignores control_text when mode=%s; control_text is only effective in mode=text/ref",
            mode,
        )
        return None

    def make_default_prompt(self, runtime: "VoxCPM2GGUFRuntime") -> BasePromptAsset:
        """构建 text 模式使用的空 prompt，绕过通用引擎对 prompt 音频的默认要求。"""

        from .runtime import prompt_asset_from_payload

        return prompt_asset_from_payload(
            role_name="__voxcpm2_text__",
            model_name=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_audio_feat=None,
            prompt_locenc_hidden=None,
            prompt_text=None,
            reference_audio_feat=None,
            reference_locenc_hidden=None,
            metadata={"mode": "text"},
        )

    def make_prompt(
        self,
        runtime: "VoxCPM2GGUFRuntime",
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BasePromptAsset:
        """
        构建 VoxCPM2-GGUF 单人 prompt 资产。

        参数:
            runtime: 已加载的 VoxCPM2-GGUF runtime。
            role_name: prompt 角色名。
            sample_rate: prompt 音频采样率；路径模式不直接使用，内存音频模式用于必要的重采样。
            audio_path: 单个参考/续写音频路径。
            audio_data: 单个参考/续写内存音频数组；允许不提供 prompt_text。
            prompt_text: continuation 类模式使用的提示文本，可为空。

        返回:
            返回保存单条音频来源、预编码特征和提示文本的 VoxCPM2PromptAsset。

        边界异常:
            多音频、路径/内存音频混用、空音频或非法采样率会被拒绝。
        """

        from .runtime import prompt_asset_from_payload

        if isinstance(audio_path, list) or isinstance(audio_data, list) or isinstance(prompt_text, list):
            raise ValueError("voxcpm2-gguf make_prompt only supports one speaker and one audio file")
        if audio_path is not None and audio_data is not None:
            raise ValueError("voxcpm2-gguf make_prompt accepts either audio_path or audio_data, not both")
        if prompt_text is not None and not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string or None")
        if not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an int")

        normalized_audio_path: str | None = None
        normalized_audio_data: np.ndarray | None = None
        if audio_data is not None:
            if not isinstance(audio_data, np.ndarray):
                raise TypeError("audio_data must be a numpy.ndarray or None")
            if audio_data.size == 0:
                raise ValueError("audio_data must not be empty")
            # 内存音频统一转到 runtime 采样率，避免后续 prompt cache 落盘后再区分原始采样率。
            flattened_audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            normalized_audio_data = flattened_audio
            if sample_rate != runtime.sample_rate:
                logger.info("⚠️ 音频采样率不匹配，正在重新采样到 %sHz", runtime.sample_rate)
                normalized_audio_data = np.asarray(
                    librosa.resample(flattened_audio, orig_sr=sample_rate, target_sr=runtime.sample_rate),
                    dtype=np.float32,
                )
        else:
            if not isinstance(audio_path, str) or not audio_path:
                raise ValueError("voxcpm2-gguf make_prompt requires a non-empty audio_path or audio_data")
            normalized_audio_path = audio_path
            normalized_audio_data, _ = librosa.load(audio_path, sr=runtime.sample_rate, mono=True)
            normalized_audio_data = np.asarray(normalized_audio_data, dtype=np.float32).reshape(-1)

        # make_prompt 阶段提前把同一条音频分别按 continuation/reference 的补零语义编码，
        # 后续 stream 直接读取 prompt asset 里的缓存，不再重复跑 AudioVAE encoder 与 locenc。
        prompt_audio_feat, prompt_locenc_hidden = runtime.input_builder.prepare_prompt_source_features(
            wav_path=normalized_audio_path,
            audio=normalized_audio_data,
            padding_mode="left",
        )
        reference_audio_feat, reference_locenc_hidden = runtime.input_builder.prepare_prompt_source_features(
            wav_path=normalized_audio_path,
            audio=normalized_audio_data,
            padding_mode="right",
        )
        # # 同时把后续真正参与生成首包时最容易抖动的 locdit 与 decoder 预热掉。
        # runtime.ensure_prompt_pipeline_warm()

        # 同一条音频既可在 ref 模式作为音色参考，也可在 continuation 模式作为续写 prompt；
        # 最终如何消费由 synthesize 的 mode 决定，避免 make_prompt 过早绑定用途。
        prompt = prompt_asset_from_payload(
            role_name=role_name,
            model_name=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_audio_feat=prompt_audio_feat,
            prompt_locenc_hidden=prompt_locenc_hidden,
            prompt_text=prompt_text,
            reference_audio_feat=reference_audio_feat,
            reference_locenc_hidden=reference_locenc_hidden,
            metadata={},
        )
        prompt.prompt_cache_record = PromptCacheRecord(
            prompt_text=[prompt_text],
            references=(
                build_prompt_reference(audio=normalized_audio_data, sample_rate=runtime.sample_rate, index=0),
            ),
        )
        return prompt

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        """序列化 VoxCPM2PromptAsset 到本地 PromptCache。"""

        from .runtime import VoxCPM2PromptAsset, serialize_prompt_asset

        if not isinstance(asset, VoxCPM2PromptAsset):
            raise TypeError("voxcpm2-gguf adapter can only serialize VoxCPM2PromptAsset")
        return serialize_prompt_asset(asset)

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BasePromptAsset:
        """从 PromptCache 载荷恢复 VoxCPM2PromptAsset。"""

        from .runtime import deserialize_prompt_asset

        return deserialize_prompt_asset(payload)

    def synthesize(
        self,
        runtime: "VoxCPM2GGUFRuntime",
        text: str,
        prompt: BasePromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        mode: str | int | None = None,
        control_text: str | None = None,
        prompt_text: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        remove_meaningless_chars: bool = True,
        seed: int | None = None,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        """创建 session 并按指定 mode 生成音频块。"""
        from .runtime import VoxCPM2GGUFSession

        if model_kwargs:
            # adapter.synthesize 通常只接收 normalize_synthesize_kwargs 的结果；这里兜底拒绝绕过归一化的参数。
            names = ", ".join(sorted(model_kwargs))
            raise ValueError(f"unsupported voxcpm2-gguf synthesize parameters: {names}")
        # 识别当前有效的mode
        effective_mode = self._resolve_effective_mode(
            runtime=runtime,
            prompt=prompt,
            mode=mode,
            prompt_text=prompt_text,
        )
        normalized_control_text = self._normalize_control_text_for_mode(
            mode=effective_mode,
            control_text=control_text,
        )
        # 验证每个mode，所需的data是否齐全
        self._validate_mode_prompt(mode=effective_mode, prompt=prompt, prompt_text=prompt_text)

        session = VoxCPM2GGUFSession(runtime)
        yield from session.synthesize(
            text=text,
            prompt=prompt,
            track_performance=track_performance,
            mode=effective_mode,
            control_text=normalized_control_text,
            prompt_text=prompt_text,
            min_length=min_length,
            max_length=max_length,
            remove_meaningless_chars=remove_meaningless_chars,
            seed=seed,
        )

    def _validate_mode_prompt(self, *, mode: str, prompt: BasePromptAsset, prompt_text: str | None) -> None:
        """按 mode 校验 prompt 是否具备所需音频和文本。"""

        from .runtime import VoxCPM2PromptAsset

        if not isinstance(prompt, VoxCPM2PromptAsset):
            raise TypeError("voxcpm2-gguf synthesize requires VoxCPM2PromptAsset")
        if mode == "text":
            return
        if mode == "ref":
            if not self._has_reference_audio(prompt):
                raise ValueError("mode=ref requires prompt/reference audio")
            return
        if mode == "continuation":
            if not self._has_prompt_audio(prompt):
                raise ValueError("mode=continuation requires prompt audio")
            if not (prompt_text or prompt.prompt_text):
                raise ValueError("mode=continuation requires prompt_text")
            return
        if mode == "ref_continuation":
            if not self._has_reference_audio(prompt) or not self._has_prompt_audio(prompt):
                raise ValueError("mode=ref_continuation requires reference and prompt audio")
            if not (prompt_text or prompt.prompt_text):
                raise ValueError("mode=ref_continuation requires prompt_text")
            return
        raise ValueError("mode must be one of: text, ref, continuation, ref_continuation")
