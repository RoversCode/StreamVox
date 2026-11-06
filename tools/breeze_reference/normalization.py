"""保留原始 RMS 的 FP32 运算与 BF16 权重乘法之间的舍入边界。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.onnx import symbolic_helper


class _Float32Rms(torch.autograd.Function):
    """将 FP32 归一化前缀映射到 ONNX Runtime 已有的原生 RMS 内核。"""

    @staticmethod
    def forward(context: Any, hidden: torch.Tensor, width: int, epsilon: float) -> torch.Tensor:
        """按原始 Torch 公式执行 FP32 均方根归一化。

        Args:
            context: Torch 导出所需的自动求导上下文。
            hidden: 已提升为 FP32 的输入。
            width: 固定的归一化维度，仅供导出创建单位权重。
            epsilon: 原始模型的稳定常量。

        Returns:
            尚未舍入或乘以模型权重的 FP32 结果。
        """
        variance = hidden.pow(2).mean(-1, keepdim=True)
        return hidden * torch.rsqrt(variance + epsilon)

    @staticmethod
    def symbolic(graph: Any, hidden: Any, width: Any, epsilon: Any) -> Any:
        """显式导出单位权重的 FP32 RMS，避免通用融合吞掉 BF16 舍入。

        Args:
            graph: Torch ONNX 导出上下文。
            hidden: FP32 输入的图表示。
            width: 固定归一化维度。
            epsilon: 原始稳定常量。

        Returns:
            形状与输入相同的 FP32 原生内核输出。
        """
        width = symbolic_helper._parse_arg(width, "i")
        epsilon = symbolic_helper._parse_arg(epsilon, "f")
        unit = graph.op("Constant", value_t=torch.ones(width, dtype=torch.float32))
        result = graph.op(
            "SimplifiedLayerNormalization", hidden, unit,
            axis_i=-1, epsilon_f=epsilon, stash_type_i=1,
        )
        return result.setType(hidden.type())


def rms_norm(normalization: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """保留上游公式，并为 BF16 图明确划分归一化与权重乘法。

    Args:
        normalization: 持有原始权重和稳定常量的归一化层。
        hidden: 当前隐藏状态或查询、键张量。

    Returns:
        与上游精度边界一致的归一化结果。
    """
    if hidden.dtype != torch.bfloat16:
        return normalization(hidden)
    # 只融合 FP32 前缀；模型权重必须在恢复 BF16 后相乘，不能并入融合内核。
    normalized = _Float32Rms.apply(
        hidden.float(), normalization.weight.shape[0], normalization.variance_epsilon,
    )
    return normalization.weight * normalized.to(hidden.dtype)
