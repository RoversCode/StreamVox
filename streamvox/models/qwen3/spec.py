from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Qwen3CloneModelSpec:
    """
    qwen3 clone GGUF 系列的模型规格。

    为什么单独抽出规格层：
    - 0.6B 和 1.7B 的推理流程一致，不应该复制 runtime；
    - 模型名、权重子目录和硬件档位不同，必须从硬编码里拆出来。
    """

    model_name: str
    adapter_name: str
    size_label: str
    core_subdir: str
    parameter_billion: float
    hardware_profile_id: str
    min_vram_gb: float
    recommended_vram_gb: float
    min_ram_gb: float
    recommended_ram_gb: float
    notes: str


QWEN3_CLONE_06B_SPEC = Qwen3CloneModelSpec(
    model_name="qwen3-tts-clone-0.6b-gguf",
    adapter_name="Qwen3TTSClone06BAdapter",
    size_label="0.6B",
    core_subdir="clone_0.6B",
    parameter_billion=0.6,
    hardware_profile_id="0.6B-mixed-q5q8-fp16",
    min_vram_gb=2.0,
    recommended_vram_gb=3.0,
    min_ram_gb=3.0,
    recommended_ram_gb=5.0,
    notes="对应当前 qwen3 clone 0.6B 一类模型的经验档位。",
)


QWEN3_CLONE_17B_SPEC = Qwen3CloneModelSpec(
    model_name="qwen3-tts-clone-1.7b-gguf",
    adapter_name="Qwen3TTSClone17BAdapter",
    size_label="1.7B",
    core_subdir="clone_1.7B",
    parameter_billion=1.7,
    hardware_profile_id="1.7B-mixed-q5q8-fp16",
    min_vram_gb=3.0,
    recommended_vram_gb=4.0,
    min_ram_gb=4.0,
    recommended_ram_gb=6.0,
    notes="对应当前 qwen3 clone 1.7B 一类模型的经验档位。",
)


QWEN3_CLONE_SPECS: dict[str, Qwen3CloneModelSpec] = {
    QWEN3_CLONE_06B_SPEC.model_name: QWEN3_CLONE_06B_SPEC,
    QWEN3_CLONE_17B_SPEC.model_name: QWEN3_CLONE_17B_SPEC,
}
