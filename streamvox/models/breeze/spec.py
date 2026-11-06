"""集中保存 Breeze2 GGUF 混合后端的固定模型规格。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BreezeTTS2GGUFModelSpec:
    """描述适配器与运行时共用的模型约束。

    Args:
        model_name: 对外模型名及默认模型目录名。
        adapter_name: 运行信息使用的适配器名称。
        sample_rate: 编解码器的音频采样率。
        num_codebooks: 每个音频帧的码本数量。
        codebook_size: 单个码本中的有效编码数量。
        max_new_tokens: 默认及允许的最大音频帧数。
        parameter_billion: 语音生成主模型的参数量估计。
        hardware_profile_id: 硬件建议档位标识。
        min_vram_gb: 显存检查下限，零表示不作最低容量判断。
        recommended_vram_gb: 建议使用的显存容量。
        min_ram_gb: 建议的最低物理内存容量。
        recommended_ram_gb: 建议使用的物理内存容量。
        notes: 资源建议的适用范围与测量状态。

    Returns:
        只保存配置，不加载模型资源。
    """

    model_name: str = "breeze-tts-2-gguf"
    adapter_name: str = "BreezeTTS2GGUFAdapter"
    sample_rate: int = 24000
    num_codebooks: int = 16
    codebook_size: int = 2048
    max_new_tokens: int = 750
    parameter_billion: float = 3.0
    hardware_profile_id: str = "breeze-tts-2-gguf"
    min_vram_gb: float = 0.0
    recommended_vram_gb: float = 12.0
    min_ram_gb: float = 0.0
    recommended_ram_gb: float = 16.0
    notes: str = "RTX 4090 的 BF16 权重、FP32 缓存混合链路已实测短句；建议预留 16GB 内存与 12GB 显存，FP32 权重和长上下文的用量需分别测量，不代表已验证的最低门槛。"


BREEZE_TTS_2_GGUF_SPEC = BreezeTTS2GGUFModelSpec()
