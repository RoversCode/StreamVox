"""对照固定上游与保留的 FP32 ONNX 编码组件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .factory import empty_cache, graph_inputs
from .numerics import FP32_POLICY, compare_fp32


def verify_reference(name: str, graph: nn.Module, raw: dict[str, Any]) -> float:
    """将手动缓存接口与原始主干、深度前向计算逐步对照。

    Args:
        name: 图名称。
        graph: 同一组权重的导出图。
        raw: 模型配置。

    Returns:
        隐藏状态、分数与每层缓存的最大绝对误差。
    """
    from transformers.cache_utils import DynamicCache

    maximum = 0.0

    def compare(actual: torch.Tensor, expected: torch.Tensor) -> None:
        """校验一组对应输出并累计最大绝对误差。

        Args:
            actual: 导出接口的结果。
            expected: 上游原始接口的结果。
        """
        nonlocal maximum
        torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-4, atol=3e-5)
        maximum = max(maximum, float((actual.float() - expected.float()).abs().max()))

    if name == "backbone":
        model = graph.backbone_model
        reference_cache = DynamicCache()
        cache = empty_cache(model, 2)
        for length in (5, 1, 3):
            embeddings = torch.randn(2, length, raw["hidden_size"])
            actual = graph(embeddings, *cache)
            expected = model(inputs_embeds=embeddings.to(model.norm.weight.dtype), past_key_values=reference_cache, use_cache=True)
            compare(actual[1], expected.last_hidden_state[:, -1])
            compare(actual[0], graph.lm_head(expected.last_hidden_state[:, -1]))
            cache = actual[2:]
            for index, layer in enumerate(reference_cache.layers):
                compare(cache[index * 2], layer.keys)
                compare(cache[index * 2 + 1], layer.values)
    elif name == "depth":
        model = graph.depth_decoder.model
        count = raw["num_codebooks"]
        tokens = torch.randint(0, raw["vocab_size"], (2, count))
        condition = torch.randn(2, raw["hidden_size"], dtype=model.norm.weight.dtype)
        expected = model(input_ids=tokens, backbone_last_hidden_state=condition, use_cache=False).last_hidden_state
        expected_logits = graph.depth_decoder.codebooks_head(expected[:, 1:])
        cache = empty_cache(model, 2)
        for position in range(count):
            output = graph(tokens[:, position:position + 1], condition, torch.tensor([position]), *cache)
            cache = output[1:]
            if position:
                compare(output[0], expected_logits[:, position - 1])
    return maximum


def verify_onnx(path: Path, name: str, graph: nn.Module, raw: dict[str, Any]) -> dict[str, Any]:
    """在真实 ONNX Runtime 上对照动态长度、空缓存及帧内各位置。

    Args:
        path: 已导出的 ONNX 文件。
        name: 图名称。
        graph: 使用同一权重的 Torch 参考图。
        raw: 模型配置。

    Returns:
        数值误差、验证样本数以及实际运行时输入输出签名。
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    maximum = 0.0
    cases = 0
    input_names = [value.name for value in session.get_inputs()]
    output_names = [value.name for value in session.get_outputs()]
    output_metrics: dict[str, dict[str, float | int]] = {}
    codebook_size = int(raw.get("codec_config", {}).get("codebook_size", raw["vocab_size"]))

    def compare(arguments: tuple[torch.Tensor, ...], runtime_arguments: tuple[Any, ...] | None = None) -> tuple[tuple[torch.Tensor, ...], list[np.ndarray]]:
        """比较一次运行，同时保留各后端独立生成的缓存。

        Args:
            arguments: Torch 参考输入。
            runtime_arguments: 可选的 ONNX 独立缓存输入。

        Returns:
            两个后端的输出，用于连续推进各自的缓存。
        """
        nonlocal maximum, cases
        expected = graph(*arguments)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        values = arguments if runtime_arguments is None else runtime_arguments
        feed = {key: value.numpy() if isinstance(value, torch.Tensor) else value for key, value in zip(input_names, values, strict=True)}
        output = session.run(None, feed)
        for index, (actual, reference) in enumerate(zip(output, expected, strict=True)):
            metrics = compare_fp32(
                actual, reference.numpy(), logits=index == 0 and name in ("backbone", "depth"),
                codebook_size=codebook_size, backbone=name == "backbone",
            )
            aggregate = output_metrics.setdefault(output_names[index], {})
            for key, value in metrics.items():
                target = "minimum_top1_agreement" if key == "top1_agreement" else key
                reducer = min if key in ("top1_agreement", "min_top1_margin") else max
                aggregate[target] = reducer(aggregate.get(target, value), value)
            maximum = max(maximum, float(metrics["max_abs"]))
        cases += 1
        return tuple(expected), output

    scenarios = [(1, 1, 0), (2, 7, 0), (2, 3, 4)]
    if name == "text_encoder":
        window = int(raw["text_encoder_config"].get("sliding_window", 0) or 0)
        # 全尺寸模型的窗口比默认短样本长，必须实际跨窗验证动态掩码，不能只依赖小配置。
        if window and all(length <= window for _, length, _ in scenarios):
            scenarios.append((1, window + 1, 0))
    if name == "depth":
        scenarios = [(2, 1, position) for position in range(raw["num_codebooks"])]
    for batch, length, previous in scenarios:
        arguments = graph_inputs(name, graph, raw, batch, length, previous)
        compare(arguments)
    # 分别携带两个后端自己的真实输出，验证连续解码不会靠共享参考缓存掩盖误差。
    if name in ("backbone", "depth"):
        model = graph.backbone_model if name == "backbone" else graph.depth_decoder.model
        reference_cache = empty_cache(model, 2)
        runtime_cache = tuple(value.numpy() for value in reference_cache)
        steps = (5, 1, 3) if name == "backbone" else range(raw["num_codebooks"])
        condition = torch.randn(2, raw["hidden_size"])
        for step in steps:
            if name == "backbone":
                inputs = (torch.randn(2, step, raw["hidden_size"]),)
                offset = 2
            else:
                inputs = (torch.randint(0, raw["vocab_size"], (2, 1)), condition, torch.tensor([step]))
                offset = 1
            expected, actual = compare((*inputs, *reference_cache), (*inputs, *runtime_cache))
            reference_cache = expected[offset:]
            runtime_cache = tuple(actual[offset:])
    return {
        "maximum_absolute_error": maximum, "cases": cases,
        "numeric_policy": FP32_POLICY, "output_metrics": output_metrics,
        "inputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_outputs()],
    }
