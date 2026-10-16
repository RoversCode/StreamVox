from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import librosa
import numpy as np

from ...hardware import ModelFootprintProfile
from ...prompt_audio import build_prompt_reference
from ...utils.log import logger
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, PromptCacheRecord, RuntimeRequest, SerializedPromptAsset
from .common import resample_and_match_channels
from ..text_processing import TextPreprocessConfig
from .spec import MOSS_TTS_NANO_ONNX_SPEC, MossTTSNanoOnnxModelSpec
from .text import MOSS_TTS_NANO_TEXT_PREPROCESS_CONFIG

if TYPE_CHECKING:
    from .runtime import MossTTSNanoRuntime


SUPPORTED_MODES = {"voice_clone", "continuation"}
MODE_CODE_TO_NAME = {
    0: "voice_clone",
    1: "continuation",
}


class MossTTSNanoOnnxAdapter:
    """
    MOSS-TTS-Nano ONNX 的 StreamVox 模型适配器。

    参数:
        spec: 模型规格，默认使用 moss-tts-nano-onnx。

    返回:
        adapter 被 ModelRegistry 调用，本身不直接返回音频；音频由 synthesize 生成器产出。

    边界异常:
        device、prompt、mode 和采样参数非法时会在 adapter 层提前抛出清晰异常。
    """

    spec: MossTTSNanoOnnxModelSpec = MOSS_TTS_NANO_ONNX_SPEC

    @property
    def model_name(self) -> str:
        """
        返回 StreamVox 对外模型名。

        参数:
            本属性无参数。

        返回:
            返回 moss-tts-nano-onnx。

        边界异常:
            本属性不主动抛出异常。
        """

        return self.spec.model_name

    def load_runtime(self, request: RuntimeRequest) -> "MossTTSNanoRuntime":
        """
        按 RuntimeRequest 加载 MOSS-TTS-Nano runtime。

        参数:
            request: TTSEngine 传入的模型目录、设备计划和校验开关。

        返回:
            返回已初始化的 MossTTSNanoRuntime。

        边界异常:
            模型 bundle 缺失、ONNX 权重缺失或 provider 不可用时抛出异常。
        """

        from .runtime import MossTTSNanoRuntime

        # 延迟导入 runtime，避免只查询 registry 时加载 onnxruntime 和 sentencepiece。
        logger.info("[Moss Nano] loading moss-tts-nano-onnx runtime")
        return MossTTSNanoRuntime(
            model_dir=Path(request.model_dir),
            model_spec=self.spec,
            device=request.device,
            verify_model_sha256=request.verify_model_sha256,
            execution_plan=request.execution_plan,
            text_preprocess_config=request.text_preprocess_config,
        )

    def get_text_preprocess_config(self) -> TextPreprocessConfig:
        """
        返回 MOSS-TTS-Nano 的文本预处理配置。

        参数:
            本函数无显式参数。

        返回:
            返回只包含轻量清理和分段规则的 TextPreprocessConfig。

        边界异常:
            本函数不主动抛出异常。
        """

        # 当前模型不移植参考项目的 WeText，只声明 StreamVox 通用分段规则。
        return MOSS_TTS_NANO_TEXT_PREPROCESS_CONFIG

    def get_hardware_profile(self) -> ModelFootprintProfile:
        """
        返回 MOSS-TTS-Nano 硬件资源建议。

        参数:
            本函数无显式参数。

        返回:
            返回 ModelFootprintProfile，用于 TTSEngine 初始化阶段硬件提示。

        边界异常:
            本函数不主动抛出异常。
        """

        # ONNX 资产由 TTS 100M 和 Audio Tokenizer Nano 两部分组成，硬件提示按整体运行链路描述。
        return ModelFootprintProfile(
            profile_id=self.spec.hardware_profile_id,
            parameter_billion=self.spec.parameter_billion,
            quantization="onnx(fp32/fp16 external-data)",
            min_vram_gb=self.spec.min_vram_gb,
            recommended_vram_gb=self.spec.recommended_vram_gb,
            min_ram_gb=self.spec.min_ram_gb,
            recommended_ram_gb=self.spec.recommended_ram_gb,
            notes=self.spec.notes,
        )

    def resolve_execution_plan(self, device: str, llama_backend: str | None = None) -> ExecutionPlan:
        """
        将用户 device 字符串解析为 MOSS ONNX runtime family。

        参数:
            device: auto、cpu、gpu 或 gpu:N。

        返回:
            返回只包含 onnx target 的 ExecutionPlan。

        边界异常:
            device 类型非法、GPU 序号非法或值不支持时抛出 TypeError/ValueError。
        """

        # MOSS 没有 llama.cpp / GGUF 链路，因此 llama_backend 在这里没有业务意义，直接忽略。
        _ = llama_backend
        # MOSS ONNX 首版只需要一个 onnx family；具体 CUDA/DML/CPU 回退在 runtime 中结合环境解析。
        mode, index = self._parse_device_spec(device)
        return ExecutionPlan(
            raw_device=device,
            targets={"onnx": ExecutionTarget(family="onnx", mode=mode, device_index=index)},
        )

    def _parse_device_spec(self, device: str) -> tuple[str, int | None]:
        """
        解析 StreamVox 统一 device 字符串。

        参数:
            device: auto、cpu、gpu 或 gpu:N。

        返回:
            返回规范化模式和可选设备序号。

        边界异常:
            device 不是字符串、GPU 序号不是非负整数或值不支持时抛出异常。
        """

        # device 是用户公开 API 的一部分，必须先做类型判断，避免 None 等值被 str() 静默吞掉。
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
        校验并归一化 MOSS-TTS-Nano 私有合成参数。

        参数:
            model_kwargs: TTSEngine.stream 透传的模型私有参数。

        返回:
            返回只包含 MOSS 支持字段的字典；未传入字段继续使用 runtime yaml 默认值。

        边界异常:
            mode 不支持、seed 类型非法或 remove_meaningless_chars 类型非法时抛出异常。
        """

        # 只接受 MOSS 公开支持的少量请求参数；分段和采样全部使用 runtime 默认值，避免调用面过宽。
        valid_params = {
            "mode",
            "remove_meaningless_chars",
            "seed",
        }
        normalized = {key: value for key, value in model_kwargs.items() if key in valid_params and value is not None}

        # mode 直接决定 prompt 结构，必须在 adapter 层提前规范化。
        if "mode" in normalized:
            normalized["mode"] = self._normalize_mode(normalized["mode"])

        # seed 只用于重置音频采样随机数，必须是明确整数，不能接受 bool 的隐式 0/1。
        if "seed" in normalized and (not isinstance(normalized["seed"], int) or isinstance(normalized["seed"], bool)):
            raise TypeError("seed must be an integer")

        # 布尔字段只接受 bool，避免 "false" 字符串在 Python 中被当成真值。
        if "remove_meaningless_chars" in normalized and not isinstance(normalized["remove_meaningless_chars"], bool):
            raise TypeError("remove_meaningless_chars must be a boolean")
        return normalized

    def _normalize_mode(self, raw_mode: object) -> str:
        """
        归一化 MOSS 合成模式。

        参数:
            raw_mode: 用户传入的模式值；支持字符串模式名，也支持数字模式 0/1。

        返回:
            返回 voice_clone 或 continuation。

        边界异常:
            raw_mode 不是字符串或整数、布尔值、数字越界、空字符串或不在支持集合中时抛出异常。
        """

        # 布尔值是 int 的子类，但把 True/False 当作模式编号没有业务可读性，因此必须先拒绝。
        if isinstance(raw_mode, bool):
            raise TypeError("mode must be a string or integer code")

        # 数字模式沿用 StreamVox 调用习惯：0 表示 voice_clone，1 表示 continuation。
        if isinstance(raw_mode, int):
            normalized_mode = MODE_CODE_TO_NAME.get(int(raw_mode))
            if normalized_mode is None:
                raise ValueError("mode integer must be one of: 0, 1")
            return normalized_mode

        # 字符串既允许直接传模式名，也允许传数字字符串，便于命令行和配置文件复用同一套语义。
        if not isinstance(raw_mode, str):
            raise TypeError("mode must be a string or integer code")
        normalized = raw_mode.strip().lower()
        if normalized.isdigit():
            normalized_mode = MODE_CODE_TO_NAME.get(int(normalized))
            if normalized_mode is None:
                raise ValueError("mode integer must be one of: 0, 1")
            return normalized_mode

        # 只支持参考实现明确存在的两种文本模式。
        if normalized not in SUPPORTED_MODES:
            raise ValueError("mode must be one of: voice_clone, continuation, 0, 1")
        return normalized

    def make_default_prompt(self, runtime: "MossTTSNanoRuntime") -> BasePromptAsset:
        """
        构造 MOSS 纯文本 continuation 默认 prompt。

        参数:
            runtime: 已加载的 MOSS runtime。

        返回:
            返回不含参考音频的临时 prompt，使 TTSEngine 可支持无 prompt 纯文本合成。

        边界异常:
            本函数不主动抛出异常。
        """

        from .runtime import prompt_asset_from_codes

        # 空 audio codes 明确表示 plain continuation；默认角色不落盘，只服务当前 stream 调用。
        return prompt_asset_from_codes(
            role_name="__default__",
            model_name=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_audio_codes=np.zeros((0, 16), dtype=np.int32),
            prompt_text=None,
            default_mode="continuation",
            metadata={"source": "default_continuation"},
        )

    def make_prompt(
        self,
        runtime: "MossTTSNanoRuntime",
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BasePromptAsset:
        """
        构建 MOSS-TTS-Nano 单参考 prompt。

        参数:
            runtime: 已加载的 MOSS runtime。
            role_name: 当前 prompt 角色名。
            sample_rate: 内存音频采样率；路径音频会从文件读取真实采样率。
            audio_path: 单个音频路径。
            audio_data: 单个内存音频数组。
            prompt_text: 可选参考文本；默认 voice_clone 不使用，continuation 模式可使用。

        返回:
            返回 MossTTSNanoPromptAsset，其中 prompt_audio_codes 已经由 codec encoder 生成。

        边界异常:
            多参考输入、路径与内存音频混用、采样率非法或音频编码失败时抛出异常。
        """

        from .runtime import prompt_asset_from_codes

        # MOSS ONNX 当前 voice_clone prompt 只支持单条参考音频，多参考需要上层先自行拼接或另行设计 prompt 结构。
        if isinstance(audio_path, list) or isinstance(audio_data, list) or isinstance(sample_rate, list):
            raise ValueError("moss-tts-nano make_prompt supports exactly one reference audio")
        if isinstance(prompt_text, list):
            raise ValueError("moss-tts-nano make_prompt supports at most one prompt_text")
        if audio_path is not None and audio_data is not None:
            raise ValueError("provide either audio_path or audio_data, not both")

        if audio_path is not None:
            audio_for_record, input_rate = librosa.load(str(Path(audio_path).expanduser().resolve()), sr=None, mono=False)
            prompt_audio_codes = runtime.core.encode_reference_audio_path(audio_path)
        else:
            # 内存音频必须有采样率，否则无法规整到 codec 48k 双声道输入。
            if audio_data is None:
                raise ValueError("audio_path or audio_data is required to build a moss-tts-nano prompt")
            if not isinstance(sample_rate, int):
                raise TypeError("sample_rate must be an int when audio_data is provided")
            if not isinstance(audio_data, np.ndarray):
                raise TypeError("audio_data must be a numpy.ndarray")
            audio_for_record = np.asarray(audio_data, dtype=np.float32)
            input_rate = int(sample_rate)
            prompt_audio_codes = runtime.core.encode_reference_audio_array(audio_data, sample_rate)

        # PromptData 保存 codec 目标采样率下的一维参考音频，和实际编码音频保持同源。
        target_channels = int(runtime.core.bundle.codec_meta["codec_config"]["channels"])
        reference_audio = resample_and_match_channels(
            np.asarray(audio_for_record, dtype=np.float32),
            int(input_rate),
            int(runtime.sample_rate),
            target_channels,
        ).reshape(target_channels, -1)

        # 有参考音频的 prompt 默认用于 voice_clone；用户显式 mode=continuation 时会复用同一份 audio codes。
        prompt = prompt_asset_from_codes(
            role_name=role_name,
            model_name=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_audio_codes=prompt_audio_codes,
            prompt_text=prompt_text if isinstance(prompt_text, str) and prompt_text.strip() else None,
            default_mode="voice_clone",
            metadata={},
        )
        prompt.prompt_cache_record = PromptCacheRecord(
            prompt_text=[prompt_text],
            references=(
                build_prompt_reference(audio=reference_audio, sample_rate=runtime.sample_rate, index=0),
            ),
        )
        return prompt

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        """
        序列化 MOSS prompt 资产。

        参数:
            asset: adapter.make_prompt 返回的 BasePromptAsset。

        返回:
            返回 PromptCache 可保存的 SerializedPromptAsset。

        边界异常:
            asset 不是 MossTTSNanoPromptAsset 时抛出 TypeError。
        """

        from .runtime import MossTTSNanoPromptAsset, serialize_prompt_asset

        # prompt 类型必须严格匹配，避免把其它模型缓存误写入 MOSS cache。
        if not isinstance(asset, MossTTSNanoPromptAsset):
            raise TypeError("moss-tts-nano adapter can only serialize MossTTSNanoPromptAsset")
        return serialize_prompt_asset(asset)

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BasePromptAsset:
        """
        反序列化 MOSS prompt 资产。

        参数:
            payload: PromptCache 读取出的 SerializedPromptAsset。

        返回:
            返回 MossTTSNanoPromptAsset。

        边界异常:
            payload 中数组维度非法时抛出 ValueError。
        """

        from .runtime import deserialize_prompt_asset

        # 具体字段恢复逻辑放在 runtime，adapter 只负责接上 PromptCache 契约。
        return deserialize_prompt_asset(payload)

    def synthesize(
        self,
        runtime: "MossTTSNanoRuntime",
        text: str,
        prompt: BasePromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        """
        调用 MOSS runtime 执行合成。

        参数:
            runtime: 已加载的 MOSS runtime。
            text: 待合成文本。
            prompt: MOSS prompt 资产。
            stream: 是否流式输出；当前模型始终按流式块输出。
            track_performance: 兼容统一接口；当前模型暂不输出额外性能日志。
            model_kwargs: normalize_synthesize_kwargs 返回的模型私有参数。

        返回:
            逐块 yield float32 PCM ndarray。

        边界异常:
            prompt 类型不匹配或底层 ONNX 推理失败时抛出异常。
        """

        from .runtime import MossTTSNanoPromptAsset

        # synthesize 入口再次校验 prompt 类型，防止调用方绕过 PromptCache 直接传错对象。
        if not isinstance(prompt, MossTTSNanoPromptAsset):
            raise TypeError("moss-tts-nano synthesize requires MossTTSNanoPromptAsset")
        session = runtime.create_session()
        yield from session.synthesize(
            text=text,
            prompt=prompt,
            stream=stream,
            track_performance=track_performance,
            **model_kwargs,
        )
