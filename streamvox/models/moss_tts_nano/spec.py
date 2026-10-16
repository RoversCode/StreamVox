from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MossTTSNanoOnnxModelSpec:
    """
    MOSS-TTS-Nano ONNX 模型规格。

    参数:
        model_name: StreamVox 对外暴露的模型名称，同时对应默认模型目录名。
        adapter_name: 当前适配器类名，用于 runtime_info 和 manifest 追踪。
        parameter_billion: TTS 主模型参数量估算值，用于硬件提示。
        hardware_profile_id: 硬件档位标识，包含模型尺寸和 ONNX 后端事实。
        min_vram_gb: GPU 推理时的最低显存建议。
        recommended_vram_gb: GPU 推理时的推荐显存建议。
        min_ram_gb: CPU 推理时的最低内存建议。
        recommended_ram_gb: CPU 推理时的推荐内存建议。
        notes: 面向日志和用户提示的资源说明。

    返回:
        该类只作为不可变配置容器，不执行模型加载，也不返回音频。

    边界异常:
        dataclass 本身不做额外校验；非法字段会在 adapter 或 runtime 使用时暴露。
    """

    model_name: str
    adapter_name: str
    parameter_billion: float
    hardware_profile_id: str
    min_vram_gb: float
    recommended_vram_gb: float
    min_ram_gb: float
    recommended_ram_gb: float
    notes: str


# 关键常量只描述模型事实，避免 adapter、runtime、manifest 各自硬编码模型名。
MOSS_TTS_NANO_ONNX_SPEC = MossTTSNanoOnnxModelSpec(
    model_name="moss-tts-nano-onnx",
    adapter_name="MossTTSNanoOnnxAdapter",
    parameter_billion=0.1,
    hardware_profile_id="moss-tts-nano-100m-onnx",
    min_vram_gb=0.0,
    recommended_vram_gb=2.0,
    min_ram_gb=3.0,
    recommended_ram_gb=5.0,
    notes="MOSS-TTS-Nano 100M ONNX + MOSS-Audio-Tokenizer-Nano ONNX 的经验档位。",
)
