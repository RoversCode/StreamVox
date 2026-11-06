"""从本地 Safetensors 读取 Breeze 图实际使用的参数并保留原始精度。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


def load_graph_weights(graph: nn.Module, name: str, checkpoint: Path) -> None:
    """从本地 Safetensors 逐张量加载实际参与推理的权重。

    Args:
        graph: 无存储参数图。
        name: 图名称，用于解析共享音频嵌入。
        checkpoint: 含配置与 Safetensors 的本地目录。
    """
    from safetensors import safe_open

    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        single = checkpoint / "model.safetensors"
        with safe_open(single, framework="pt", device="cpu") as handle:
            weight_map = {key: single.name for key in handle.keys()}
    required: dict[str, list[tuple[str, str]]] = {}
    for target in graph.state_dict():
        source = target
        if name == "audio_embedding":
            source = target.replace("embedding.", "backbone_model.embed_tokens.", 1)
            if source not in weight_map and target == "embedding.embed_audio_tokens.weight":
                source = "depth_decoder.model.embed_tokens.weight"
        if source not in weight_map:
            raise ValueError(f"检查点缺少推理权重：{source}")
        filename = weight_map[source]
        path = (checkpoint / filename).resolve()
        if not path.is_relative_to(checkpoint.resolve()):
            raise ValueError(f"检查点索引包含目录外文件：{filename}")
        required.setdefault(filename, []).append((target, source))
    parameters = dict(graph.named_parameters())
    buffers = dict(graph.named_buffers())
    for filename, entries in required.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for target, source in entries:
                tensor = handle.get_tensor(source)
                expected = parameters.get(target, buffers.get(target))
                if expected is None or tensor.shape != expected.shape:
                    raise ValueError(f"检查点参数形状不匹配：{source}")
                if tensor.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"固定检查点只允许 BF16 或 FP32 权重：{source}")
                if expected.dtype == torch.bfloat16 and tensor.dtype != torch.bfloat16:
                    raise ValueError(f"BF16 图要求检查点原始权重为 BF16，不能舍入 FP32 权重：{source}")
                # BF16 原始图不改变权重；FP32 基准图的提升也不会引入舍入或量化。
                graph.load_state_dict({target: tensor.to(expected.dtype)}, strict=False, assign=True)
    missing = [key for key, value in graph.state_dict().items() if value.is_meta]
    if missing or any(value.is_meta for value in graph.buffers()):
        raise ValueError(f"仍有未加载参数或缓冲区：{missing}")
