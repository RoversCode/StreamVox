"""精确提升深度条件投影的原始权重，并导出固定 FP32 的 ONNX 接口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch
from torch import nn

from .bundle import export_isolated_graph
from .numerics import FP32_POLICY, compare_fp32


class DepthConditionerGraph(nn.Module):
    """将主干隐藏状态投影到深度解码器使用的嵌入空间。

    Args:
        projector: 保留原始参数且无偏置的 FP32 线性投影。
    """

    def __init__(self, projector: nn.Linear) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        """投影一批帧条件，不改变批次数或引入额外归一化。

        Args:
            backbone_hidden: 形状为批次乘主干隐藏维度的 FP32 张量。

        Returns:
            形状为批次乘深度隐藏维度的 FP32 嵌入。
        """
        return self.projector(backbone_hidden)


def export_depth_conditioner(checkpoint: Path, destination: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """只加载原始条件投影，导出后验证动态批次的数值与接口。

    Args:
        checkpoint: 含原始 BF16 或 FP32 Safetensors 的本地检查点。
        destination: 接收深度条件 ONNX 图的目录。
        raw: 提供主干与深度解码器真实维度的完整配置。

    Returns:
        FP32 提升来源、投影公式及真实 ONNX Runtime 数值报告。
    """
    import onnxruntime as ort

    checkpoint = checkpoint.resolve(strict=True)
    depth = raw["depth_decoder_config"]
    input_size, output_size = int(raw["hidden_size"]), int(depth["hidden_size"])
    if input_size != int(depth["audio_embed_size"]) or input_size != int(depth["backbone_hidden_size"]):
        raise ValueError("固定 Breeze 检查点不允许额外的主干隐藏状态投影")
    source = "depth_decoder.model.inputs_embeds_projector.weight"
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
            weight_map = {name: "model.safetensors" for name in handle.keys()}
    if any(name.startswith("depth_decoder.model.backbone_hidden_state_projector.") for name in weight_map):
        raise ValueError("固定 Breeze 检查点不允许额外的主干隐藏状态投影")
    if source not in weight_map:
        raise ValueError(f"检查点缺少深度条件投影：{source}")
    path = (checkpoint / weight_map[source]).resolve(strict=True)
    if not path.is_relative_to(checkpoint):
        raise ValueError("深度条件投影的参数路径越过检查点目录")
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(source)
        if tuple(weight.shape) != (output_size, input_size):
            raise ValueError(f"深度条件投影形状必须为 {(output_size, input_size)}")
        if weight.dtype not in (torch.bfloat16, torch.float32) or not torch.isfinite(weight).all():
            raise ValueError("深度条件投影必须使用有限的原始 BF16 或 FP32 权重")
        source_dtype = str(weight.dtype).removeprefix("torch.")
        projector = nn.Linear(input_size, output_size, bias=False, device="meta")
        # BF16 可被 FP32 精确表示，提升权重不会发生二次舍入。
        projector.weight = nn.Parameter(weight.float().contiguous(), requires_grad=False)
    graph = DepthConditionerGraph(projector).eval()
    path = destination / "depth_conditioner.onnx"
    with torch.inference_mode():
        export_isolated_graph(
            graph, (torch.zeros(1, input_size),), path,
            input_names=["backbone_hidden"], output_names=["embeddings"],
            dynamic_axes={"backbone_hidden": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=17, dynamo=False, external_data=True,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        maximum = 0.0
        for batch in (1, 2, 3):
            hidden = torch.randn(batch, input_size)
            actual = session.run(["embeddings"], {"backbone_hidden": hidden.numpy()})[0]
            expected = graph(hidden).numpy()
            metrics = compare_fp32(actual, expected)
            maximum = max(maximum, float(metrics["max_abs"]))
    return {
        "validated": True, "source": source, "source_dtype": source_dtype, "precision": "float32",
        "formula": "backbone_hidden @ inputs_embeds_projector.weight.T",
        "input_size": input_size, "output_size": output_size, "cases": 3,
        "maximum_absolute_error": maximum, "numeric_policy": FP32_POLICY,
        "inputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_outputs()],
    }
