"""完整重放冻结的真实任务，以既定门限验收 Q8_0 的全部消费输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from streamvox.models.breeze.llama import LlamaDecoder
from streamvox.models.breeze.sampling import guided_logits

from .q8_validate import CODEBOOK_SIZE, NUM_CODEBOOKS, Q8_POLICY, _compare_output, _file_record


def _resolve_arrays(value: Any, arrays: Any) -> Any:
    """解析嵌套调用参数中的显式数组引用。

    Args:
        value: 原生调用的普通参数或数组引用。
        arrays: 禁止对象反序列化的数组归档。
    Returns:
        保留普通参数结构的实际调用值。
    """
    if isinstance(value, dict):
        if "array" in value:
            if set(value) != {"array"} or not isinstance(value["array"], str):
                raise ValueError("数组引用必须仅包含一个字符串 array 字段。")
            result = np.asarray(arrays[value["array"]])
            if result.dtype.hasobject:
                raise ValueError("轨迹不能包含对象数组。")
            return result
        return {key: _resolve_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_arrays(item, arrays) for item in value]
    return value


def _guided_row(
    reference: Any, event: dict[str, Any], events: list[dict[str, Any]], retained: dict[int, Any],
) -> np.ndarray:
    """从同组件、同样本的先前解码输出提取一条条件分支。

    Args:
        reference: 解码事件编号和行号。
        event: 当前引导事件。
        events: 完整冻结轨迹。
        retained: 仍被后续引导引用的候选输出。
    Returns:
        当前条件分支的完整物理词表分数。
    """
    if not isinstance(reference, dict) or set(reference) != {"event", "row"}:
        raise ValueError("引导引用必须包含 event 和 row。")
    index, row = reference["event"], reference["row"]
    if type(index) is not int or not 0 <= index < event["id"] or type(row) is not int or row < 0:
        raise ValueError("引导必须引用先前事件中的有效非负行号。")
    source = events[index]
    if source["operation"] != "decode" or source["component"] != event["component"] or source["sample"] != event["sample"]:
        raise ValueError("引导引用必须来自同组件和同样本的解码事件。")
    output = retained.get(index)
    if not isinstance(output, dict) or "logits" not in output or output["logits"] is None:
        raise ValueError("引导引用的解码分数不可用。")
    logits = np.asarray(output["logits"])
    if logits.ndim != 2 or row >= logits.shape[0]:
        raise ValueError("引导引用行号超出解码分数范围。")
    return logits[row]


def _replay_component(
    component: str, path: Path, plan: dict[str, Any], arrays: Any, backend: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """顺序重放一个组件，并在失败后保留其余全部消费输出的报告。

    Args:
        component: 主干或深度组件。
        path: 候选组件模型。
        plan: 含全局事件顺序的冻结计划。
        arrays: 输入和 BF16 参考数组。
        backend: 与参考捕获一致的原生后端。
    Returns:
        每个消费事件的报告及组件执行错误。
    """
    cases, errors = [], []
    retained: dict[int, Any] = {}
    references: dict[int, int] = {}
    for event in plan["events"]:
        if event["component"] != component or event["operation"] != "guided":
            continue
        for source in (event.get("positive_ref"), event.get("negative_ref")):
            if isinstance(source, dict) and type(source.get("event")) is int:
                index = source["event"]
                references[index] = references.get(index, 0) + 1
    native = None
    load_error = None
    try:
        native = LlamaDecoder(
            path, backend=backend, n_ctx=plan["components"][component]["n_ctx"], n_seq_max=2,
            embeddings=component == "backbone",
        )
        if component == "depth" and native.vocab_size % NUM_CODEBOOKS:
            raise ValueError("深度联合词表必须能完整分为十六码本。")
        codec_vocab_size = native.vocab_size - 1 if component == "backbone" else native.vocab_size // NUM_CODEBOOKS
        if codec_vocab_size < CODEBOOK_SIZE:
            raise ValueError("候选物理词表不能小于合法音频码范围。")
    except Exception as error:
        load_error = f"{component}: {type(error).__name__}: {error}"
        errors.append(load_error)
    cache_valid = True
    try:
        for event in plan["events"]:
            if event["component"] != component:
                continue
            operation = event["operation"]
            entry = {key: event[key] for key in ("id", "component", "sample", "phase", "operation")}
            entry.update({"event": event["id"], "outputs": {}, "errors": []})
            expected = event.get("expected", {})
            consumed = operation == "guided" or operation == "decode" and bool(expected)
            try:
                if load_error is not None:
                    raise RuntimeError(load_error)
                if operation == "guided":
                    positive = _guided_row(event.get("positive_ref"), event, plan["events"], retained)
                    negative = None if event.get("negative_ref") is None else _guided_row(event["negative_ref"], event, plan["events"], retained)
                    reference = np.asarray(arrays[event["expected"]])
                    if reference.ndim != 1:
                        raise ValueError("引导参考必须保留原始一维词表。")
                    actual = np.asarray(guided_logits(positive, negative, event["scale"]))
                    reference, actual = reference.reshape(1, -1), actual.reshape(1, -1)
                    entry["outputs"]["logits"] = _compare_output(
                        actual, reference, expected_shape=reference.shape, logits=True, backbone=component == "backbone",
                    )
                else:
                    arguments = _resolve_arrays(event.get("arguments", {}), arrays)
                    args = _resolve_arrays(event.get("args", []), arrays)
                    if operation == "decode":
                        if isinstance(expected, dict) and "logits" in expected:
                            if component == "backbone":
                                if arguments.get("logit_start", 0) != 0 or arguments.get("logit_count") not in (None, native.vocab_size):
                                    raise ValueError("主干消费输出必须包含完整物理词表及末列结束类别。")
                            else:
                                positions = arguments.get("positions", [])
                                if not positions or len(set(positions)) != 1 or not 1 <= positions[0] < NUM_CODEBOOKS:
                                    raise ValueError("深度消费输出必须对应位置一至十五的同一码本头。")
                                if arguments.get("logit_start", 0) != (positions[0] - 1) * codec_vocab_size or arguments.get("logit_count") != codec_vocab_size:
                                    raise ValueError("深度消费输出未请求对应的完整物理码本头。")
                        output = native.decode(**arguments)
                        if not cache_valid:
                            raise RuntimeError("同一场景先前缓存操作失败，后续结果无法用于验收。")
                        if event["id"] in references:
                            retained[event["id"]] = output
                        if not isinstance(expected, dict) or set(expected) - {"logits", "hidden"}:
                            raise ValueError("解码参考只能包含 logits 和 hidden。")
                        actual_names = {name for name, value in output.items() if value is not None}
                        if actual_names != set(expected):
                            raise ValueError("候选返回的消费输出集合与冻结参考不同。")
                        for name, key in expected.items():
                            reference = np.asarray(arrays[key])
                            if reference.ndim != 2:
                                raise ValueError("解码参考必须保留二维批次输出。")
                            entry["outputs"][name] = _compare_output(
                                output.get(name), reference, expected_shape=reference.shape,
                                logits=name == "logits", backbone=component == "backbone",
                            )
                    else:
                        getattr(native, operation)(*args, **arguments)
                        if operation == "reset":
                            cache_valid = True
                            retained.clear()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                entry["errors"].append(message)
                if operation != "guided":
                    cache_valid = False
                errors.append(f"事件 {event['id']}: {message}")
            entry["passed"] = not entry["errors"] and all(value["passed"] for value in entry["outputs"].values())
            if consumed or entry["errors"]:
                cases.append(entry)
            if operation == "guided":
                for source in (event.get("positive_ref"), event.get("negative_ref")):
                    if not isinstance(source, dict) or type(source.get("event")) is not int:
                        continue
                    index = source["event"]
                    references[index] -= 1
                    if references[index] == 0:
                        retained.pop(index, None)
    finally:
        if native is not None:
            try:
                native.close()
            except Exception as error:
                errors.append(f"{component} 释放失败: {type(error).__name__}: {error}")
    return cases, errors


def _coverage(plan: dict[str, Any], arrays: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """核对六个真实任务、全部深度头和长参考都实际参与重放。

    Args:
        plan: 冻结样本及事件。
        arrays: 用于核对实际输入长度的归档。
        cases: 已执行的消费事件报告。
    Returns:
        完整覆盖范围及可直接用于最终判定的完整性状态。
    """
    names = [sample["name"] for sample in plan["samples"]]
    per_sample = {name: {"backbone": False, "depth_positions": set(), "depth_heads": set(), "guided_components": set()} for name in names}
    cfg, prefix, context = False, 0, 0
    consumed = set()
    for event in plan["events"]:
        component, operation = event["component"], event["operation"]
        if operation == "guided":
            consumed.add(event["id"])
            if event["phase"] == "synthesis":
                per_sample[event["sample"]]["guided_components"].add(component)
                cfg = cfg or event.get("negative_ref") is not None and event.get("scale") != 1.0
        if operation != "decode":
            continue
        arguments = _resolve_arrays(event.get("arguments", {}), arrays)
        values = np.asarray(arguments.get("embeddings", arguments.get("tokens")))
        if values.ndim not in (2, 3):
            raise ValueError("解码输入缺少有效的批次和长度。")
        positions = arguments["positions"]
        if len(positions) != values.shape[0]:
            raise ValueError("轨迹位置数量与输入批次不同。")
        if event.get("expected"):
            consumed.add(event["id"])
        if component == "backbone":
            end = max(positions) + values.shape[1]
            context = max(context, end)
            if event["phase"] == "reference":
                prefix = max(prefix, end)
            if event["phase"] == "synthesis" and {"hidden", "logits"} <= set(event.get("expected", {})):
                per_sample[event["sample"]]["backbone"] = True
        elif event["phase"] == "synthesis":
            current = per_sample[event["sample"]]
            current["depth_positions"].update(positions)
            if "logits" in event.get("expected", {}):
                current["depth_heads"].update(position for position in positions if 1 <= position < NUM_CODEBOOKS)
    reported = {case["event"] for case in cases}
    complete = len(names) == 6 and len(set(names)) == 6 and consumed <= reported
    for sample in plan["samples"]:
        current = per_sample[sample["name"]]
        complete = complete and sample.get("finite_audio") is True and type(sample.get("frames")) is int and sample["frames"] > 0
        complete = complete and bool(sample.get("termination")) and current["backbone"]
        complete = complete and current["depth_positions"] == set(range(NUM_CODEBOOKS))
        complete = complete and current["depth_heads"] == set(range(1, NUM_CODEBOOKS))
        complete = complete and current["guided_components"] == {"backbone", "depth"}
        for key in ("depth_positions", "depth_heads", "guided_components"):
            current[key] = sorted(current[key])
    return {
        "samples": names, "per_sample": per_sample, "sample_count": len(names),
        "depth_positions": sorted({position for value in per_sample.values() for position in value["depth_positions"]}),
        "depth_heads": sorted({head for value in per_sample.values() for head in value["depth_heads"]}),
        "all_consumed_outputs": consumed <= reported and all(not case["errors"] for case in cases),
        "cfg": bool(cfg), "long_reference_prefix_length": prefix, "max_backbone_context_length": context,
        "complete": bool(complete and cfg and prefix > 512),
    }


def validate_trajectory(reference: Path, candidate: Path, trace_dir: Path, backend: str) -> dict[str, Any]:
    """以冻结 BF16 轨迹验收候选模型的全部真实消费输出。

    Args:
        reference: 原始 bundle 根目录或直接包含 GGUF 的检查点目录。
        candidate: 两个 Q8_0 候选组件所在目录。
        trace_dir: 含 plan.json 和 arrays.npz 的冻结轨迹目录。
        backend: 重放使用的 cpu 或 cuda 后端。
    Returns:
        文件摘要、完整消费输出指标、覆盖范围及严格通过状态。
    """
    reference, candidate, trace_dir = Path(reference).resolve(), Path(candidate).resolve(), Path(trace_dir).resolve()
    if (reference / "ckpt").is_dir():
        reference = reference / "ckpt"
    report: dict[str, Any] = {
        "schema_version": 1, "basis": "fixed_real_task_replay", "quantization": "q8_0", "backend": backend,
        "reference": {"directory": str(reference), "files": {}}, "candidate": {"directory": str(candidate), "files": {}},
        "trace": {"directory": str(trace_dir), "files": {}}, "policy": dict(Q8_POLICY),
        "cases": [], "errors": [], "coverage": {"complete": False, "all_consumed_outputs": False},
        "sequential_model_loading": True, "independent_native_caches": True,
        "comparison": "固定真实任务的全部 BF16 消费输出与同输入 Q8_0 重放，指标描述教师强制条件轨迹上的量化增量误差。",
    }
    event_count = 0
    try:
        if backend not in ("cpu", "cuda"):
            raise ValueError("轨迹重放后端只能为 cpu 或 cuda。")
        for filename in ("plan.json", "arrays.npz"):
            report["trace"]["files"][filename] = _file_record(trace_dir / filename)
        plan = json.loads((trace_dir / "plan.json").read_text(encoding="utf-8"))
        # 将冻结计划和规范序列化摘要一起保留，发布审计可以重新核对全部消费事件。
        canonical_plan = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        report["trace"]["plan"] = plan
        report["trace"]["canonical_sha256"] = hashlib.sha256(canonical_plan).hexdigest()
        if (
            report["trace"]["files"]["plan.json"]["bytes"] != len(canonical_plan)
            or report["trace"]["files"]["plan.json"]["sha256"] != report["trace"]["canonical_sha256"]
        ):
            raise ValueError("冻结计划文件必须与保存的规范序列化内容一致。")
        report["samples"] = plan.get("samples", [])
        if plan.get("schema_version") != 1 or plan.get("kind") != "breeze_real_trajectory":
            raise ValueError("轨迹格式或版本不受支持。")
        if not isinstance(plan.get("events"), list) or not isinstance(plan.get("samples"), list):
            raise ValueError("轨迹必须提供样本和事件列表。")
        names = [sample["name"] for sample in plan["samples"]]
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("验收要求六个名称唯一的冻结样本。")
        for side, directory in (("reference", reference), ("candidate", candidate)):
            for component in ("backbone", "depth"):
                filename = component + ".gguf"
                record = _file_record(directory / filename)
                report[side]["files"][filename] = record
                if side == "reference":
                    expected = plan["reference"]["files"][filename]
                    if any(record[key] != expected[key] for key in ("bytes", "sha256")):
                        report["errors"].append(f"原始文件摘要与冻结轨迹不一致: {filename}")
        expected_arrays = plan.get("arrays")
        if not isinstance(expected_arrays, dict) or any(
            report["trace"]["files"]["arrays.npz"][key] != expected_arrays.get(key) for key in ("bytes", "sha256")
        ):
            report["errors"].append("数组归档摘要与冻结计划不一致或缺失。")
        allowed = {"decode", "reset", "reserve", "remove_sequence", "guided"}
        for index, event in enumerate(plan["events"]):
            if type(event.get("id")) is not int or event["id"] != index:
                raise ValueError("事件编号必须与完整列表顺序一致。")
            if event.get("operation") not in allowed or event.get("component") not in ("backbone", "depth"):
                raise ValueError(f"事件 {index} 包含不受支持的操作或组件。")
            if event.get("sample") not in names or event.get("phase") not in ("reference", "synthesis"):
                raise ValueError(f"事件 {index} 缺少有效样本或阶段。")
        event_count = len(plan["events"])
        if not report["errors"]:
            with np.load(trace_dir / "arrays.npz", allow_pickle=False) as arrays:
                for component in ("backbone", "depth"):
                    cases, errors = _replay_component(component, candidate / f"{component}.gguf", plan, arrays, backend)
                    report["cases"].extend(cases)
                    report["errors"].extend(errors)
                report["cases"].sort(key=lambda case: case["event"])
                report["coverage"] = _coverage(plan, arrays, report["cases"])
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    failed = sum(not case["passed"] for case in report["cases"])
    checked = sum(len(case["outputs"]) for case in report["cases"])
    passed = not report["errors"] and failed == 0 and checked > 0 and report["coverage"]["complete"]
    report["passed"] = bool(passed)
    report["summary"] = {
        "passed": bool(passed), "cases": len(report["cases"]), "failed_cases": failed,
        "checked_outputs": checked, "events": event_count,
    }
    return report


def main() -> None:
    """先保存完整报告，再通过退出码标记真实轨迹验收结果。

    Returns:
        验收失败时以状态码一退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = validate_trajectory(arguments.reference_bundle, arguments.candidate_dir, arguments.trace_dir, arguments.backend)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(arguments.report.resolve()), **report["summary"]}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
