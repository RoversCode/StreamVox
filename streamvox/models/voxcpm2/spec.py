from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VoxCPM2GGUFModelSpec:
    """
    VoxCPM2-GGUF 模型规格。

    参数:
        model_name: StreamVox 对外暴露的模型名称，也对应默认 ckpt 目录名。
        adapter_name: 当前模型适配器类名，用于 runtime_info 和 manifest 追踪。
        parameter_billion: 模型参数量估算值，用于硬件提示。
        hardware_profile_id: 硬件档位标识。
        min_vram_gb: 最低显存建议。
        recommended_vram_gb: 推荐显存建议。
        min_ram_gb: 最低内存建议。
        recommended_ram_gb: 推荐内存建议。
        notes: 给用户和日志展示的模型资源说明。

    返回:
        该类只作为不可变配置容器，不提供运行时返回值。

    边界异常:
        dataclass 本身不做额外校验；非法值会在 adapter/runtime 使用时暴露。
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


VOXCPM2_GGUF_SPEC = VoxCPM2GGUFModelSpec(
    model_name="voxcpm2-gguf",
    adapter_name="VoxCPM2GGUFAdapter",
    parameter_billion=4.5,
    # 关键常量：当前 bundle 仍是 GGUF q8_0，但 ONNX 子模块已经变成混合精度，
    # 其中 LocEnc / LocDiT estimator 为 FP32，AudioVAE encoder / decoder 仍为 FP16。
    # 这里显式改名，避免后续日志、硬件提示或缓存键继续误导为“全 ONNX FP16”。
    hardware_profile_id="voxcpm2-q8-mixed-onnx",
    min_vram_gb=7.0,
    recommended_vram_gb=8.0,
    min_ram_gb=8.0,
    recommended_ram_gb=12.0,
    notes="VoxCPM2 base/residual GGUF q8_0 + LocEnc/LocDiT FP32 + AudioVAE FP16 ONNX 的经验档位。",
)
