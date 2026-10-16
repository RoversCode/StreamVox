"""从开发环境的 Torch 模型抽取 NumPy 权重，发行包不包含本模块。"""

import torch
from torch import nn

from streamvox.models.voxcpm2.lightweight_ops import (
    LinearLayerRuntime, ScalarQuantizationLayerRuntime, StopPredictorRuntime,
)


def export_linear(module: nn.Linear) -> LinearLayerRuntime:
    """复制线性层权重，让导出资源与训练模型互不影响。

    Args:
        module: 开发环境中的线性层。
    Returns:
        使用独立 float32 数组的运行层。
    """
    weight = module.weight.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    bias = None if module.bias is None else module.bias.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    return LinearLayerRuntime(weight=weight, bias=bias)


def export_scalar_quantization(module: nn.Module) -> ScalarQuantizationLayerRuntime:
    """抽取标量量化层的两次投影与缩放值。

    Args:
        module: 官方模型中的标量量化层。
    Returns:
        保留原始缩放规则的 NumPy 运行层。
    """
    return ScalarQuantizationLayerRuntime(
        in_proj=export_linear(module.in_proj), out_proj=export_linear(module.out_proj),
        scale=float(module.scale),
    )


def export_stop_predictor(stop_proj: nn.Linear, stop_head: nn.Linear) -> StopPredictorRuntime:
    """抽取停止预测器的两层权重。

    Args:
        stop_proj: 输入投影层。
        stop_head: 输出投影层。
    Returns:
        保留原有激活计算的 NumPy 停止预测器。
    """
    return StopPredictorRuntime(stop_proj=export_linear(stop_proj), stop_head=export_linear(stop_head))
