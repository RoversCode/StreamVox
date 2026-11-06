"""建立同源数学参考，再由独立原生解释器验收两个 GGUF 生成组件。"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

from . import factory, weights
from .comparison import verify_reference


def verify_gguf_generation(
    destination: Path, raw: dict[str, Any], modules: tuple, checkpoint: Path,
    *, python: Path, backend: str = "cpu", precision: str = "float32",
) -> dict[str, Any]:
    """以完整码本与独立缓存验证 GGUF 的计算及布局转换。

    Args:
        destination: 包含主干与深度 GGUF 的目录。
        raw: 同一检查点的完整模型配置。
        modules: 通过固定提交检查的原始模型模块。
        checkpoint: 完整检查点或随机小模型检查点。
        python: 安装 SDK 与原生库的独立解释器。
        backend: 数值验收使用的原生后端。
        precision: GGUF 的原始权重存储精度。
    Returns:
        两个生成组件分别对照的数值报告。
    """
    if precision not in ("float32", "bfloat16"):
        raise ValueError("GGUF 仅支持未经量化的 FP32 或原始 BF16 权重。")
    reports: dict[str, Any] = {}
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="breeze-gguf-verify-") as temporary, torch.inference_mode():
        directory = Path(temporary)
        for name in ("backbone", "depth"):
            cases = directory / name
            cases.mkdir()
            graph = factory.build_graph(name, raw, modules, meta=True)
            weights.load_graph_weights(graph, name, checkpoint)
            # 独立核对原始模型接口，避免导出与参考共用同一处缓存实现错误。
            with torch.random.fork_rng(devices=[]):
                upstream_error = verify_reference(name, graph, raw)
            plan: dict[str, Any] = {
                "component": name, "weight_precision": precision,
                "codebook_size": int(raw.get("codec_config", {}).get("codebook_size", raw["vocab_size"])),
                "cases": [],
            }

            def save(
                value: torch.Tensor, expected: dict[str, torch.Tensor], *, reset: bool,
                positions: list[int], sequences: list[int], kind: str = "embeddings", offset: int = 0,
            ) -> None:
                """记录独立前向的输入与结果，不共享原生缓存。

                Args:
                    value: 原生输入的嵌入或词表编号。
                    expected: 同源参考输出。
                    reset: 是否开始新的缓存场景。
                    positions: 各分支的绝对位置。
                    sequences: 各分支的独立序列编号。
                    kind: 嵌入输入或词表编号输入。
                    offset: 深度位置对应的输出头起点。
                Returns:
                    无。
                """
                filename = f"case-{len(plan['cases'])}.npz"
                arrays = {"input": value.numpy(), **{"expected_" + key: result.numpy() for key, result in expected.items()}}
                np.savez(cases / filename, **arrays)
                plan["cases"].append({
                    "file": filename, "reset": reset, "positions": positions, "sequence_ids": sequences,
                    "input_kind": kind, "logit_start": offset,
                    "logit_count": int(raw["vocab_size"]) + int(name == "backbone"),
                })

            if name == "backbone":
                model = graph.backbone_model
                caches: dict[int, tuple] = {}
                scenarios = [
                    (True, [0], [0], 1), (True, [0, 1], [0, 0], 7),
                    (False, [0, 1], [7, 7], 1), (False, [0, 1], [8, 8], 3),
                    (True, [0], [0], 5), (False, [1], [0], 3), (False, [0, 1], [5, 3], 1),
                ]
                for reset, sequences, positions, length in scenarios:
                    if reset:
                        caches.clear()
                    value = torch.randn(len(sequences), length, int(raw["hidden_size"]))
                    logits, hidden = [], []
                    for row, sequence in enumerate(sequences):
                        cache = caches.get(sequence, factory.empty_cache(model))
                        output = graph(value[row:row + 1], *cache)
                        caches[sequence] = output[2:]
                        logits.append(output[0])
                        hidden.append(output[1])
                    save(value, {"logits": torch.cat(logits), "hidden": torch.cat(hidden)},
                         reset=reset, positions=positions, sequences=sequences)
            else:
                model = graph.depth_decoder.model
                for batch_size in (1, 2):
                    condition = torch.randn(batch_size, int(raw["hidden_size"]))
                    cache = factory.empty_cache(model, batch_size)
                    for position in range(int(raw["num_codebooks"])):
                        tokens = torch.randint(0, int(raw["vocab_size"]), (batch_size, 1))
                        output = graph(tokens, condition, torch.tensor([position]), *cache)
                        cache = output[1:]
                        offset = max(0, position - 1) * int(raw["vocab_size"])
                        if position == 0:
                            value = model.inputs_embeds_projector(condition)[:, None]
                            kind = "embeddings"
                        else:
                            value = (tokens + offset).to(torch.int32)
                            kind = "tokens"
                        save(value, {"logits": output[0]}, reset=position == 0,
                             positions=[position] * batch_size, sequences=list(range(batch_size)), kind=kind, offset=offset)
            (cases / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            del graph
            gc.collect()
            environment = os.environ.copy()
            environment["NVIDIA_TF32_OVERRIDE"] = "0"
            environment["GGML_VK_DISABLE_F16"] = "1"
            path = cases / "report.json"
            try:
                subprocess.run([
                    str(python.resolve(strict=True)), "-m", "tools.breeze_reference.gguf_validate",
                    "--model", str((destination / f"{name}.gguf").resolve(strict=True)),
                    "--cases", str(cases), "--backend", backend, "--report", str(path),
                ], cwd=root, env=environment, check=True)
            except (subprocess.CalledProcessError, OSError) as error:
                failure = directory.with_name(directory.name + "-failed")
                directory.rename(failure)
                raise RuntimeError(f"GGUF 数值验收失败，原始样本已保留：{failure}") from error
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
            reports[name]["upstream_reference_maximum_absolute_error"] = upstream_error
    return reports
