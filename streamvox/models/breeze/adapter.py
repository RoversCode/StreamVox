"""将 Breeze2 GGUF 混合后端接入引擎、提示缓存与统一执行计划。"""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np

from ...hardware import ModelFootprintProfile
from ...prompt_audio import build_prompt_reference, normalize_prompt_audio
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, PromptCacheRecord, RuntimeRequest, SerializedPromptAsset
from ..llama_backend import normalize_explicit_llama_backend
from ..text_processing import TextPreprocessConfig
from .prompt import BreezePromptAsset
from .spec import BREEZE_TTS_2_GGUF_SPEC, BreezeTTS2GGUFModelSpec

if TYPE_CHECKING:
    from .runtime import BreezeRuntime


class BreezeTTS2GGUFAdapter:
    """管理 Breeze2 的公开输入边界与运行时调用。

    Args:
        text_encoder_backend: 文本编码器使用处理器，或使用处理器查表与显卡主干。
    Returns:
        适配器实例；模型资源在调用加载方法时创建。
    """

    spec: BreezeTTS2GGUFModelSpec = BREEZE_TTS_2_GGUF_SPEC

    def __init__(self, *, text_encoder_backend: Literal["auto", "cpu", "cuda"] = "auto") -> None:
        if text_encoder_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("text_encoder_backend must be auto, cpu or cuda")
        self.text_encoder_backend = text_encoder_backend

    @property
    def model_name(self) -> str:
        """返回注册使用的模型名称。

        Returns:
            固定的 Breeze2 GGUF 模型名。
        """
        return self.spec.model_name

    def load_runtime(self, request: RuntimeRequest) -> BreezeRuntime:
        """按照引擎请求延迟加载模型运行时。

        Args:
            request: 本地模型目录、设备计划、上下文与校验选项。

        Returns:
            已完成初始化的运行时。
        """
        from .runtime import BreezeRuntime

        return BreezeRuntime(
            model_dir=Path(request.model_dir),
            device=request.device,
            execution_plan=request.execution_plan,
            verify_model_sha256=request.verify_model_sha256,
            n_ctx=request.n_ctx,
            text_encoder_backend=self.text_encoder_backend,
        )

    def get_text_preprocess_config(self) -> TextPreprocessConfig | None:
        """保留上游文本模板与标记，不引入额外预处理。

        Returns:
            空值，表示不声明通用文本预处理配置。
        """
        return None

    def get_hardware_profile(self) -> ModelFootprintProfile:
        """提供附带测量范围说明的保守资源建议。

        Returns:
            带有测量状态说明的模型资源档位。
        """
        return ModelFootprintProfile(
            profile_id=self.spec.hardware_profile_id,
            parameter_billion=self.spec.parameter_billion,
            quantization="gguf-q8_0/bf16/f32+onnx-f32",
            min_vram_gb=self.spec.min_vram_gb,
            recommended_vram_gb=self.spec.recommended_vram_gb,
            min_ram_gb=self.spec.min_ram_gb,
            recommended_ram_gb=self.spec.recommended_ram_gb,
            notes=self.spec.notes + (" 使用 CUDA 文本主干时另需约 2.61GiB 的 FP32 权重显存。" if self.text_encoder_backend != "cpu" else ""),
        )

    def resolve_execution_plan(self, device: str, llama_backend: str | None = None) -> ExecutionPlan:
        """解析统一设备意图，保留具体执行提供程序的运行时选择权。

        Args:
            device: 自动、处理器或显卡设备字符串。
            llama_backend: 主干使用的显式后端，空值保留运行时自动选择。

        Returns:
            分别包含主干与其余组件设备意图的执行计划。
        """
        backend = normalize_explicit_llama_backend(llama_backend)
        if not isinstance(device, str):
            raise TypeError("device must be a string: auto, cpu, gpu or gpu:<index>")
        normalized = device.strip().lower()
        index = None
        if normalized in {"auto", "cpu"}:
            mode = normalized
        elif normalized == "gpu":
            mode, index = "gpu", 0
        elif normalized.startswith("gpu:") and normalized[4:].isascii() and normalized[4:].isdigit():
            mode, index = "gpu", int(normalized[4:])
        else:
            raise ValueError(f"unsupported device value: {device}")
        return ExecutionPlan(
            raw_device=device,
            targets={
                "llama": ExecutionTarget(
                    family="llama", mode=mode, device_index=index, backend_preference=backend,
                ),
                "onnx": ExecutionTarget(family="onnx", mode=mode, device_index=index),
            },
        )

    def validate_make_prompt_args(
        self,
        *,
        role_name: str,
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        sample_rate: int | list[int] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> None:
        """在音频读取与编码前校验单参考和完整转录要求。

        Args:
            role_name: 非空的缓存角色名。
            audio_path: 单条参考音频路径。
            audio_data: 单条参考音频数组。
            sample_rate: 内存音频采样率，路径输入可省略。
            prompt_text: 对应整段音频的非空转录。

        Returns:
            校验通过时无返回值。
        """
        if not isinstance(role_name, str) or not role_name.strip():
            raise TypeError("role_name must be a non-empty string")
        if any(isinstance(value, list) for value in (audio_path, audio_data, sample_rate, prompt_text)):
            raise ValueError("breeze-tts-2-gguf requires exactly one reference audio and one complete transcript")
        if audio_path is not None and audio_data is not None:
            raise ValueError("provide either audio_path or audio_data, not both")
        if not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a non-empty complete transcript string")
        if not prompt_text.strip():
            raise ValueError("prompt_text must be a non-empty complete transcript")
        if sample_rate is not None and (not isinstance(sample_rate, int) or isinstance(sample_rate, bool)):
            raise TypeError("sample_rate must be an integer")
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if audio_path is not None:
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise TypeError("audio_path must be a non-empty string")
            return
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be a numpy.ndarray when audio_path is absent")
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio_data is provided")
        if audio_data.ndim not in {1, 2} or audio_data.size == 0:
            raise ValueError("audio_data must be a non-empty mono or stereo waveform")
        if not np.issubdtype(audio_data.dtype, np.number) or np.iscomplexobj(audio_data):
            raise TypeError("audio_data must contain real numeric samples")
        if not np.isfinite(audio_data).all():
            raise ValueError("audio_data must contain only finite samples")

    def make_prompt(
        self,
        runtime: BreezeRuntime,
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BreezePromptAsset:
        """将单条参考音频编码成可复用的提示资产。

        Args:
            runtime: 已加载的模型运行时。
            role_name: 提示角色名。
            sample_rate: 内存音频采样率，文件输入读取真实采样率。
            audio_path: 与内存输入互斥的文件路径。
            audio_data: 单声道或双声道音频数组。
            prompt_text: 与参考音频完整对应的转录。

        Returns:
            包含整数编码与公共参考资料的提示资产。
        """
        self.validate_make_prompt_args(
            role_name=role_name, audio_path=audio_path, audio_data=audio_data,
            sample_rate=sample_rate, prompt_text=prompt_text,
        )
        if audio_path is not None:
            import librosa

            audio_data, sample_rate = librosa.load(str(Path(audio_path).expanduser().resolve()), sr=None, mono=False)
        # 在混合声道前检查原始数据，避免非法样本被聚合或类型转换掩盖。
        if audio_data is None or audio_data.size == 0 or not np.isfinite(audio_data).all():
            raise ValueError("reference audio must contain non-empty finite samples")
        audio = normalize_prompt_audio(audio_data)
        if sample_rate != self.spec.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.spec.sample_rate)
        audio = normalize_prompt_audio(audio)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("normalized reference audio must contain non-empty finite samples")
        prompt = BreezePromptAsset(
            role_name=role_name,
            model=self.model_name,
            sample_rate=self.spec.sample_rate,
            prompt_text=prompt_text,
            audio_codes=runtime.encode_reference(audio),
        )
        self._validate_prompt(prompt)
        if prompt.audio_codes.shape[0] == 0:
            raise ValueError("reference audio encoder returned no audio frames")
        runtime.prepare_reference(prompt)
        prompt.prompt_cache_record = PromptCacheRecord(
            prompt_text=[prompt_text],
            references=(build_prompt_reference(audio=audio, sample_rate=self.spec.sample_rate, index=0),),
        )
        return prompt

    def make_default_prompt(self, runtime: BreezeRuntime) -> BreezePromptAsset:
        """创建纯文本与声音设计共用的无参考提示。

        Args:
            runtime: 已加载的运行时。

        Returns:
            不含音频编码和参考转录的临时提示。
        """
        return BreezePromptAsset(role_name="__default__", model=self.model_name, sample_rate=self.spec.sample_rate)

    def _validate_prompt(self, asset: BasePromptAsset) -> None:
        """统一检查缓存读写和合成入口的提示约束。

        Args:
            asset: 待检查的模型提示资产。

        Returns:
            类型、模型、采样率和码本数据均合法时无返回值。
        """
        if not isinstance(asset, BreezePromptAsset):
            raise TypeError("breeze-tts-2-gguf requires BreezePromptAsset")
        if asset.model != self.model_name:
            raise ValueError(f"prompt model must be {self.model_name}")
        if asset.sample_rate != self.spec.sample_rate:
            raise ValueError(f"prompt sample_rate must be {self.spec.sample_rate}")
        codes = asset.audio_codes
        if not isinstance(codes, np.ndarray) or codes.dtype != np.int64:
            raise TypeError("audio_codes must be an int64 numpy.ndarray")
        if codes.ndim != 2 or codes.shape[1] != self.spec.num_codebooks:
            raise ValueError(f"audio_codes must have shape (frames, {self.spec.num_codebooks})")
        if codes.size and (np.any(codes < 0) or np.any(codes >= self.spec.codebook_size)):
            raise ValueError(f"audio_codes must be in [0, {self.spec.codebook_size})")
        if asset.prompt_text is not None and not isinstance(asset.prompt_text, str):
            raise TypeError("prompt_text must be a string or None")
        if codes.shape[0] and (asset.prompt_text is None or not asset.prompt_text.strip()):
            raise ValueError("reference audio_codes require a non-empty complete prompt_text")

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        """将经过校验的提示转为通用缓存载荷。

        Args:
            asset: 当前模型的提示资产。

        Returns:
            仅保存编码、转录及通用元信息的缓存载荷。
        """
        self._validate_prompt(asset)
        return SerializedPromptAsset(
            role_name=asset.role_name, model=asset.model, sample_rate=asset.sample_rate,
            metadata=dict(asset.metadata), arrays={"audio_codes": asset.audio_codes.copy()},
            json_data={"prompt_text": asset.prompt_text},
        )

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BreezePromptAsset:
        """恢复缓存资产并拒绝错模型或损坏的编码。

        Args:
            payload: 通用缓存读取出的序列化载荷。

        Returns:
            可直接用于当前模型合成的提示资产。
        """
        if "audio_codes" not in payload.arrays:
            raise ValueError("breeze prompt cache is missing audio_codes")
        prompt = BreezePromptAsset(
            role_name=payload.role_name, model=payload.model, sample_rate=payload.sample_rate,
            metadata=dict(payload.metadata), prompt_text=payload.json_data.get("prompt_text"),
            audio_codes=payload.arrays["audio_codes"].copy(),
        )
        self._validate_prompt(prompt)
        return prompt

    def normalize_synthesize_kwargs(
        self,
        *,
        max_length: object = None,
        min_length: object = None,
        remove_meaningless_chars: object = True,
        **model_kwargs: object,
    ) -> dict[str, object]:
        """严格校验 Breeze2 的采样和流式参数。

        Args:
            max_length: 引擎通用文本字段，不参与本模型的音频采样。
            min_length: 引擎通用文本字段，不参与本模型的音频采样。
            remove_meaningless_chars: 引擎通用清理字段，保留本模型语气事件标记。
            model_kwargs: 声音指令、两级采样参数、随机种子和分块帧数。

        Returns:
            具有完整默认值且不含未知字段的运行时参数。
        """
        defaults: dict[str, object] = {
            "instruction": "", "cfg_scale": 1.0, "temperature": 0.9,
            "top_k": 50, "top_p": 1.0, "depth_temperature": 0.9,
            "depth_top_k": 50, "depth_top_p": 1.0, "repetition_penalty": 1.1,
            "max_new_tokens": self.spec.max_new_tokens, "seed": None, "do_sample": True, "chunk_frames": 2,
        }
        unknown = model_kwargs.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"unsupported breeze-tts-2-gguf synthesize parameters: {', '.join(sorted(unknown))}")
        normalized = {**defaults, **model_kwargs}
        if not isinstance(normalized["instruction"], str):
            raise TypeError("instruction must be a string")
        if not isinstance(normalized["do_sample"], bool):
            raise TypeError("do_sample must be a boolean")
        # 两级采样各自使用独立的温度与截断分布，禁止归一化时互相覆盖。
        for name in ("cfg_scale", "temperature", "top_p", "depth_temperature", "depth_top_p", "repetition_penalty"):
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "cfg_scale" and value < 0:
                raise ValueError("cfg_scale must be non-negative")
            if name != "cfg_scale" and value <= 0:
                raise ValueError(f"{name} must be positive")
            if name in {"top_p", "depth_top_p"} and value > 1:
                raise ValueError(f"{name} must not exceed 1")
            normalized[name] = value
        for name in ("top_k", "depth_top_k", "max_new_tokens", "chunk_frames", "seed"):
            value = normalized[name]
            if name == "seed" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            minimum = 1 if name in {"max_new_tokens", "chunk_frames"} else 0
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            maximum = {"max_new_tokens": self.spec.max_new_tokens, "chunk_frames": 25}.get(name)
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} must not exceed {maximum}")
            normalized[name] = value
        return normalized

    def synthesize(
        self,
        runtime: BreezeRuntime,
        text: str,
        prompt: BasePromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        """校验提示与私有参数后逐块透传运行时音频。

        Args:
            runtime: 已加载的模型运行时。
            text: 保留语气与事件标记的目标文本。
            prompt: 单参考或无参考提示资产。
            stream: 是否以流式形式生成音频。
            track_performance: 引擎通用字段，本适配器不记录额外性能日志。
            model_kwargs: 公开的声音指令和采样参数。

        Returns:
            保持运行时顺序与内容的音频块迭代器。
        """
        self._validate_prompt(prompt)
        parameters = self.normalize_synthesize_kwargs(**model_kwargs)
        yield from runtime.synthesize(text=text, prompt=prompt, stream=stream, **parameters)
