"""以独立缓存验收 Q8_0 相对原始 BF16 GGUF 的增量数值误差。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from streamvox.models.breeze.llama import LlamaDecoder

from .numerics import _logits_statistics, _output_statistics


# 量化验收单独维护，避免把增量误差门限误用于原有数学参考验收。
Q8_POLICY = {
    "pointwise_atol": 0.01,
    "pointwise_rtol": 0.02,
    "pointwise_hard_gate": False,
    "relative_l2_max": 0.025,
    "reference_norm_floor": 1e-6,
    "logits_temperature": 0.9,
    "kl_reference_to_actual_max": 0.003,
    "max_probability_difference": 0.035,
}

SEED = 20261106
CODEBOOK_SIZE = 2048
NUM_CODEBOOKS = 16
EXPECTED_CASES = 57


def _file_record(path: Path) -> dict[str, Any]:
    """读取被验证文件的完整摘要。

    Args:
        path: 已存在的 GGUF 文件。
    Returns:
        绝对路径、字节数和 SHA256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _build_cases(component: str, input_size: int, codec_vocab_size: int) -> list[dict[str, Any]]:
    """建立固定输入的完整主干与十六码本重放计划。

    Args:
        component: 主干或深度组件名称。
        input_size: 原始模型的实际嵌入维度。
        codec_vocab_size: 每码本的物理词表宽度，包含保留符号。
    Returns:
        包含输入数组、独立序列编号和输出头区间的有序计划。
    """
    specifications = []
    if component == "backbone":
        scenarios = [("batch-1", 1, 7), ("batch-2", 2, 7)]
        scenarios.extend((f"length-{length}", 1, length) for length in (511, 512, 513))
        for scenario, batch_size, length in scenarios:
            sequences = [0] if batch_size == 1 else [1, 0]
            for step in range(4):
                specifications.append({
                    "name": f"backbone/{scenario}/" + ("prefill" if step == 0 else f"increment-{step}"),
                    "reset": step == 0, "sequence_ids": sequences,
                    "positions": [0 if step == 0 else length + step - 1] * batch_size,
                    "length": length if step == 0 else 1,
                    "input_kind": "embeddings", "logit_start": 0, "logit_count": codec_vocab_size + 1,
                })
        # 两个不同长度的前缀分别建立缓存，再合批推进，覆盖真实条件分支的错位位置。
        for sequence, length in ((0, 5), (1, 3)):
            specifications.append({
                "name": f"backbone/misaligned/prefill-{sequence}", "reset": sequence == 0,
                "sequence_ids": [sequence], "positions": [0], "length": length,
                "input_kind": "embeddings", "logit_start": 0, "logit_count": codec_vocab_size + 1,
            })
        for step in range(3):
            specifications.append({
                "name": f"backbone/misaligned/increment-{step + 1}", "reset": False,
                "sequence_ids": [0, 1], "positions": [5 + step, 3 + step], "length": 1,
                "input_kind": "embeddings", "logit_start": 0, "logit_count": codec_vocab_size + 1,
            })
    else:
        for batch_size in (1, 2):
            for position in range(NUM_CODEBOOKS):
                specifications.append({
                    "name": f"depth/batch-{batch_size}/position-{position}", "reset": position == 0,
                    "sequence_ids": list(range(batch_size)), "positions": [position] * batch_size,
                    "length": 1, "input_kind": "embeddings" if position == 0 else "tokens",
                    "logit_start": max(0, position - 1) * codec_vocab_size, "logit_count": codec_vocab_size,
                })

    rng = np.random.default_rng(SEED + int(component == "depth"))
    for case in specifications:
        shape = (len(case["sequence_ids"]), case["length"])
        if case["input_kind"] == "embeddings":
            values = rng.normal(0.0, 0.3, (*shape, input_size)).astype(np.float32)
        else:
            values = rng.integers(0, CODEBOOK_SIZE, size=shape, dtype=np.int32) + case["logit_start"]
        # 同一份只读数组供两个模型顺序重放，避免随机输入或意外写入引入额外差异。
        values.flags.writeable = False
        case["values"] = values
    return specifications


def _execute_cases(native: LlamaDecoder, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """独立推进一个模型，并保留所有场景的输出和执行错误。

    Args:
        native: 当前唯一加载的原生模型。
        cases: 同源只读输入与缓存操作计划。
    Returns:
        每项的输出副本或明确错误，数量和计划一致。
    """
    results = []
    cache_valid = True
    for case in cases:
        try:
            if case["reset"]:
                native.reset()
                cache_valid = True
            output = native.decode(
                **{case["input_kind"]: case["values"]},
                positions=case["positions"], sequence_ids=case["sequence_ids"],
                logit_start=case["logit_start"], logit_count=case["logit_count"],
            )
            if not cache_valid:
                results.append({"error": "同一场景先前解码失败，缓存状态无法用于验收。"})
                continue
            results.append({"outputs": {
                name: np.array(value, copy=True) for name, value in output.items() if value is not None
            }})
        except Exception as error:
            cache_valid = False
            results.append({"error": f"{type(error).__name__}: {error}"})
    return results


def _compare_output(
    actual: Any, reference: Any, *, expected_shape: tuple[int, int], logits: bool, backbone: bool,
) -> dict[str, Any]:
    """记录完整输出指标，并严格应用预先确定的量化增量误差门限。

    Args:
        actual: 候选模型输出。
        reference: 原始 BF16 模型输出。
        expected_shape: 当前请求规定的批次及输出宽度。
        logits: 是否校验合法码本上的概率。
        backbone: 是否纳入主干末列的结束类别。
    Returns:
        形状、有限值、误差、概率、逐元素审计和通过状态。
    """
    actual_array, reference_array = np.asarray(actual), np.asarray(reference)
    metrics: dict[str, Any] = {
        "actual_shape": list(actual_array.shape), "reference_shape": list(reference_array.shape),
        "expected_shape": list(expected_shape),
        "shape_match": actual_array.shape == reference_array.shape == expected_shape,
        "passed": False, "errors": [],
    }
    for name, value in (("actual", actual_array), ("reference", reference_array)):
        real = np.issubdtype(value.dtype, np.number) and not np.iscomplexobj(value)
        metrics[name + "_finite"] = bool(real and np.isfinite(value).all())
    if not metrics["shape_match"]:
        metrics["errors"].append("输出形状不符合请求。")
    if not metrics["actual_finite"] or not metrics["reference_finite"]:
        metrics["errors"].append("输出必须是有限实数数组。")
    if metrics["errors"]:
        return metrics

    actual_values, reference_values, statistics = _output_statistics(
        actual_array, reference_array, norm_floor=Q8_POLICY["reference_norm_floor"],
    )
    metrics.update(statistics)
    threshold = Q8_POLICY["pointwise_atol"] + Q8_POLICY["pointwise_rtol"] * np.abs(reference_values)
    metrics["pointwise_violations"] = int(np.count_nonzero(np.abs(actual_values - reference_values) > threshold))
    metrics["element_count"] = int(reference_values.size)
    gates = [("relative_l2", Q8_POLICY["relative_l2_max"])]
    if logits:
        metrics.update(_logits_statistics(
            actual_values, reference_values, codebook_size=CODEBOOK_SIZE, backbone=backbone,
            temperature=Q8_POLICY["logits_temperature"],
        ))
        gates.extend([
            ("kl_reference_to_actual", Q8_POLICY["kl_reference_to_actual_max"]),
            ("max_probability_difference", Q8_POLICY["max_probability_difference"]),
        ])
    for name, limit in gates:
        if not np.isfinite(metrics[name]) or metrics[name] > limit:
            metrics["errors"].append(f"{name} 超过预设门限 {limit}。")
    metrics["passed"] = not metrics["errors"]
    return metrics


def _validate_component(reference: Path, candidate: Path, component: str, backend: str) -> dict[str, Any]:
    """先释放原始模型，再用完全相同的输入验证候选模型。

    Args:
        reference: 原始组件文件。
        candidate: 量化组件文件。
        component: 主干或深度组件名称。
        backend: 明确选择的原生后端。
    Returns:
        每项数值报告、接口元数据和模型级错误。
    """
    cases: list[dict[str, Any]] = []
    executions: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {"cases": [], "models": {}, "errors": []}
    for side, path in (("reference", reference), ("candidate", candidate)):
        native = None
        try:
            native = LlamaDecoder(path, backend=backend, n_ctx=520, n_seq_max=2, embeddings=component == "backbone")
            interface = {key: int(getattr(native, key)) for key in ("input_size", "hidden_size", "vocab_size")}
            report["models"][side] = interface
            if component == "depth" and interface["vocab_size"] % NUM_CODEBOOKS:
                raise ValueError("深度联合词表必须能完整分为十六码本。")
            # 物理词表包含保留符号；合法音频码仅决定输入采样范围及概率统计范围。
            codec_vocab_size = interface["vocab_size"] - 1 if component == "backbone" else interface["vocab_size"] // NUM_CODEBOOKS
            interface["codec_vocab_size"] = codec_vocab_size
            if codec_vocab_size < CODEBOOK_SIZE:
                raise ValueError(f"每码本物理词表不能小于 {CODEBOOK_SIZE} 个合法音频码。")
            if side == "reference":
                cases = _build_cases(component, interface["input_size"], codec_vocab_size)
            elif interface != report["models"]["reference"]:
                raise ValueError("原始模型与候选模型的输入、隐藏状态或词表维度不同。")
            executions[side] = _execute_cases(native, cases)
        except Exception as error:
            message = f"{side}: {type(error).__name__}: {error}"
            report["errors"].append(message)
            executions[side] = [{"error": message} for _ in cases]
        finally:
            if native is not None:
                native.close()
        if not cases:
            return report

    for index, case in enumerate(cases):
        entry = {key: value for key, value in case.items() if key != "values"}
        entry.update({
            "component": component, "input_shape": list(case["values"].shape),
            "input_dtype": str(case["values"].dtype),
            "input_sha256": hashlib.sha256(case["values"].tobytes()).hexdigest(),
            "outputs": {}, "errors": [],
        })
        for side in ("reference", "candidate"):
            if "error" in executions[side][index]:
                entry["errors"].append(f"{side}: {executions[side][index]['error']}")
        if entry["errors"]:
            entry["passed"] = False
            report["cases"].append(entry)
            continue
        for name in (("logits", "hidden") if component == "backbone" else ("logits",)):
            width = case["logit_count"] if name == "logits" else report["models"]["reference"]["hidden_size"]
            entry["outputs"][name] = _compare_output(
                executions["candidate"][index]["outputs"].get(name),
                executions["reference"][index]["outputs"].get(name),
                expected_shape=(len(case["sequence_ids"]), width), logits=name == "logits",
                backbone=component == "backbone",
            )
        entry["passed"] = all(value["passed"] for value in entry["outputs"].values())
        report["cases"].append(entry)
    return report


def validate_q8(reference: Path, candidate: Path, backend: str) -> dict[str, Any]:
    """验收完整两个 Q8_0 组件相对原始 BF16 资产的增量误差。

    Args:
        reference: 原始 bundle 根目录，或直接包含两个 GGUF 的 ckpt 目录。
        candidate: 直接包含两个候选 GGUF 的目录。
        backend: 同时用于原始及候选模型的 cpu 或 cuda 后端。
    Returns:
        文件摘要、固定门限、完整场景和最终通过状态；数值失败仍返回完整报告。
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("Q8 验收后端只能为 cpu 或 cuda。")
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    if (reference / "ckpt").is_dir():
        reference = reference / "ckpt"
    report: dict[str, Any] = {
        "schema_version": 1, "quantization": "q8_0", "backend": backend,
        "reference": {"directory": str(reference), "files": {}},
        "candidate": {"directory": str(candidate), "files": {}},
        "policy": dict(Q8_POLICY), "seed": SEED, "cases": [], "components": {}, "errors": [],
        "comparison": "同一 llama.cpp b8683 后端下，Q8_0 对原始 BF16 GGUF 的增量误差验收；门限不表示数学等价。",
        "kv_precision": "float32", "flash_attention": False,
        "sequential_model_loading": True, "independent_native_caches": True,
        "coverage": {
            "full_suite": True, "backbone_prefill_lengths": [3, 5, 7, 511, 512, 513],
            "backbone_scenarios": 6, "incremental_steps": 3, "batch_sizes": [1, 2],
            "misaligned_positions": True, "depth_positions": list(range(NUM_CODEBOOKS)),
            "depth_heads": list(range(NUM_CODEBOOKS - 1)),
        },
    }
    for side, directory in (("reference", reference), ("candidate", candidate)):
        for component in ("backbone", "depth"):
            filename = component + ".gguf"
            try:
                report[side]["files"][filename] = _file_record(directory / filename)
            except OSError as error:
                report["errors"].append(f"{side}/{filename}: {type(error).__name__}: {error}")
    if not report["errors"]:
        for component in ("backbone", "depth"):
            result = _validate_component(reference / f"{component}.gguf", candidate / f"{component}.gguf", component, backend)
            report["cases"].extend(result.pop("cases"))
            report["components"][component] = result
            report["errors"].extend(f"{component}/{error}" for error in result["errors"])
        vocab_sizes = {
            component["models"]["reference"]["codec_vocab_size"]
            for component in report["components"].values()
            if "codec_vocab_size" in component["models"].get("reference", {})
        }
        if len(vocab_sizes) > 1:
            report["errors"].append("主干与深度模型的每码本物理词表宽度不同。")
    failed = sum(not case["passed"] for case in report["cases"])
    report["coverage"]["full_suite"] = len(report["cases"]) == EXPECTED_CASES
    passed = not report["errors"] and len(report["cases"]) == EXPECTED_CASES and failed == 0
    report["passed"] = passed
    report["summary"] = {"passed": passed, "cases": len(report["cases"]), "failed_cases": failed, "expected_cases": EXPECTED_CASES}
    return report


def main() -> None:
    """保存完整验收报告，并用退出码标记量化验收失败。

    Returns:
        成功时正常退出，失败时在报告落盘后以状态码一退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", type=Path, required=True, help="原始 bundle 根目录或 ckpt 目录")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="直接包含两个 Q8_0 GGUF 的目录")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = validate_q8(arguments.reference_bundle, arguments.candidate_dir, arguments.backend)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(arguments.report.resolve()), **report["summary"]}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
