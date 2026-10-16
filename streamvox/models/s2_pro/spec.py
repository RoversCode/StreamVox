from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class S2ProModelSpec:
    """
    S2-Pro GGUF 模型规格。

    这里单独声明规格，是为了让 adapter/runtime 只消费稳定的模型事实，
    后续如果增加不同量化或尺寸，不需要复制整套推理代码。
    """

    model_name: str
    adapter_name: str
    size_label: str
    parameter_billion: float
    hardware_profile_id: str
    min_vram_gb: float
    recommended_vram_gb: float
    min_ram_gb: float
    recommended_ram_gb: float
    notes: str


S2_PRO_4B_GGUF_SPEC = S2ProModelSpec(
    model_name="s2-pro-4b-gguf",
    adapter_name="S2Pro4BGGUFAdapter",
    size_label="4B",
    parameter_billion=4.5,
    hardware_profile_id="4.5B-q8-fp16",
    min_vram_gb=7.0,
    recommended_vram_gb=8.0,
    min_ram_gb=6.0,
    recommended_ram_gb=8.0,
    notes="S2-Pro slow/fast GGUF q8_0 + true-streaming codec FP16 ONNX 的经验档位。",
)
