"""在无需 Torch 的解释器中验收 GGUF 的分支、缓存及码本位置。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from streamvox.models.breeze.llama import LlamaDecoder

from .numerics import BF16_POLICY, FP32_POLICY, compare_bf16, compare_fp32


def validate(model: Path, cases: Path, backend: str) -> dict[str, Any]:
    """分别推进原生缓存并与同源 Torch 数学参考比较。

    Args:
        model: 主干或深度 GGUF 模型路径。
        cases: 带计划、输入和参考输出的隔离目录。
        backend: 原生 CPU、CUDA 或 Vulkan 后端。
    Returns:
        输出误差、实际后端、缓存精度和独立分支的验证记录。
    """
    plan = json.loads((cases / "plan.json").read_text(encoding="utf-8"))
    precision = plan["weight_precision"]
    compare = compare_fp32 if precision == "float32" else compare_bf16
    policy = FP32_POLICY if precision == "float32" else BF16_POLICY
    metrics: dict[str, dict[str, float | int]] = {}
    native = LlamaDecoder(
        model, backend=backend, n_ctx=64, n_seq_max=2, embeddings=plan["component"] == "backbone",
    )
    try:
        for index, entry in enumerate(plan["cases"]):
            if entry["reset"]:
                native.reset()
            path = (cases / entry["file"]).resolve(strict=True)
            if not path.is_relative_to(cases.resolve()):
                raise ValueError("GGUF 验收样本不能越过隔离目录。")
            with np.load(path, allow_pickle=False) as arrays:
                arguments = {
                    entry["input_kind"]: arrays["input"], "positions": entry["positions"],
                    "sequence_ids": entry["sequence_ids"], "logit_start": entry["logit_start"],
                    "logit_count": entry["logit_count"],
                }
                actual = native.decode(**arguments)
                for name in ("logits", "hidden"):
                    key = "expected_" + name
                    if key not in arrays:
                        continue
                    try:
                        current = compare(
                            actual[name], arrays[key], logits=name == "logits",
                            codebook_size=plan["codebook_size"], backbone=plan["component"] == "backbone",
                        )
                    except (AssertionError, ValueError) as error:
                        raise type(error)(f"{model.name} 样本 {index} 输出 {name}：{error}") from error
                    aggregate = metrics.setdefault(name, {})
                    for key, value in current.items():
                        if key in ("pointwise_violations", "element_count"):
                            aggregate[key] = aggregate.get(key, 0) + value
                        else:
                            reducer = min if key in ("top1_agreement", "min_top1_margin") else max
                            aggregate[key] = reducer(aggregate.get(key, value), value)
    finally:
        native.close()
    return {
        "component": plan["component"], "cases": len(plan["cases"]),
        "backend": backend, "weight_precision": precision, "kv_precision": "float32",
        "flash_attention": False, "numeric_policy": policy, "output_metrics": metrics,
        "independent_native_and_reference_caches": True,
        "reference": "固定上游权重精确提升的 Torch FP32 前向；BF16 存储使用独立低精度误差界限。",
    }


def main() -> None:
    """解析原生验证任务并在成功后写入报告。

    Returns:
        无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = validate(arguments.model, arguments.cases, arguments.backend)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
