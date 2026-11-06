"""核验 Q8 发布报告的数值范围、完整任务证据和来源绑定。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .q8_validate import Q8_POLICY


LOSSY_TASK_AUDIT_SCOPE = "q8_0_lossy_task_audit_v1"
_COMPONENTS = ("backbone.gguf", "depth.gguf")


def _validate_strict_report(report: dict[str, Any]) -> None:
    """按既有门限要求完整五十七项数值验收通过。

    Args:
        report: 独立量化数值验收报告。
    Returns:
        完整通过时正常返回，否则拒绝派生。
    """
    cases, summary, coverage = report.get("cases", []), report.get("summary", {}), report.get("coverage", {})
    if (
        report.get("passed") is not True or report.get("quantization") != "q8_0"
        or summary.get("passed") is not True or summary.get("failed_cases") != 0
        or summary.get("cases") != 57 or summary.get("expected_cases") != 57
        or len(cases) != 57 or any(case.get("passed") is not True for case in cases) or report.get("errors")
        or coverage.get("full_suite") is not True or coverage.get("depth_positions") != list(range(16))
        or coverage.get("depth_heads") != list(range(15)) or coverage.get("batch_sizes") != [1, 2]
        or coverage.get("backbone_prefill_lengths") != [3, 5, 7, 511, 512, 513]
        or coverage.get("misaligned_positions") is not True or coverage.get("incremental_steps") != 3
        or coverage.get("backbone_scenarios") != 6
    ):
        raise ValueError("必须提供完整通过的独立 Q8_0 量化对照报告。")
    _audit_numerical_report(report, report, expected_passed=True)


def _audit_numerical_report(
    audit: dict[str, Any], report: dict[str, Any], *, expected_passed: bool,
) -> tuple[int, int]:
    """重算数值审计结果，拒绝结构错误、摘要漂移及结果删改。

    Args:
        audit: 保留原始指标和门限的数值审计报告。
        report: 绑定同一组源文件和量化文件的顶层验收报告。
        expected_passed: 严格验收要求全部通过，有损来源审计要求保留数值失败。
    Returns:
        实际数值失败用例数和已检查输出数。
    """
    cases, summary = audit.get("cases", []), audit.get("summary", {})
    if (
        audit.get("passed") is not expected_passed or audit.get("quantization") != "q8_0"
        or audit.get("errors") != [] or audit.get("policy") != Q8_POLICY
        or summary.get("passed") is not expected_passed or summary.get("cases") != len(cases) or not cases
    ):
        raise ValueError("数值审计状态、完整输出及原始门限不一致。")
    for kind in ("reference", "candidate"):
        files = audit.get(kind, {}).get("files", {})
        if set(files) != set(_COMPONENTS):
            raise ValueError("数值审计缺少完整生成组件摘要。")
        for filename in _COMPONENTS:
            expected = report.get(kind, {}).get("files", {}).get(filename, {})
            if any(files[filename].get(key) != expected.get(key) for key in ("bytes", "sha256")):
                raise ValueError("数值审计与发布验收的模型摘要不一致。")
    failed, checked_outputs = 0, 0
    for case in cases:
        outputs = case.get("outputs", {})
        if (
            case.get("errors") != [] or not outputs or not set(outputs).issubset({"hidden", "logits"})
            or type(case.get("passed")) is not bool
        ):
            raise ValueError("数值审计包含执行错误或缺失输出。")
        for name, value in outputs.items():
            if (
                value.get("actual_finite") is not True or value.get("reference_finite") is not True
                or value.get("shape_match") is not True or not value.get("expected_shape")
                or value.get("actual_shape") != value.get("expected_shape")
                or value.get("reference_shape") != value.get("expected_shape")
                or type(value.get("passed")) is not bool
            ):
                raise ValueError("数值审计存在非有限输出或形状错误。")
            limits = {"relative_l2": Q8_POLICY["relative_l2_max"]}
            if name == "logits":
                limits.update({
                    "kl_reference_to_actual": Q8_POLICY["kl_reference_to_actual_max"],
                    "max_probability_difference": Q8_POLICY["max_probability_difference"],
                })
            if any(not math.isfinite(value.get(key, math.nan)) for key in limits):
                raise ValueError("数值审计缺少有限的误差指标。")
            numerical_passed = all(value[key] <= limit for key, limit in limits.items())
            if value["passed"] != numerical_passed:
                raise ValueError("数值审计通过标记与原始门限不一致。")
        case_passed = all(value["passed"] for value in outputs.values())
        if case["passed"] != case_passed:
            raise ValueError("数值审计用例标记与逐输出结果不一致。")
        failed += int(not case_passed)
        checked_outputs += len(outputs)
    if (failed == 0) != expected_passed or summary.get("failed_cases") != failed:
        raise ValueError("数值审计失败数量与声明的验收状态不一致。")
    return failed, checked_outputs


def _edit_distance(reference: str, transcript: str) -> int:
    """重算已经规范化的转录与原文之间的字符编辑距离。

    Args:
        reference: 报告中保存的规范化原文。
        transcript: 使用相同规则规范化的转录。
    Returns:
        插入、删除和替换所需的最少操作数。
    """
    previous = list(range(len(transcript) + 1))
    for row, character in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(transcript, start=1):
            current.append(min(previous[column] + 1, current[-1] + 1, previous[column - 1] + (character != actual)))
        previous = current
    return previous[-1]


def _validate_lossy_tasks(report: dict[str, Any], sample_names: set[str]) -> tuple[int, int, int]:
    """核对多种子自然结束、冻结转录比较与用户试听的音频绑定。

    Args:
        report: 绑定候选模型、合成、内容转录和用户试听的完整报告。
        sample_names: 完整真实轨迹中的六个固定任务名称。
    Returns:
        合成、转录及人工试听的实际样本数量。
    """
    reports = report["reports"]
    synthesis = reports.get("synthesis", {})
    samples = synthesis.get("samples", [])
    audio = {sample.get("name"): sample.get("audio_sha256") for sample in samples}
    synthesized = {sample.get("name"): sample for sample in samples}
    candidate_files = synthesis.get("candidate", {}).get("files", {})
    if set(candidate_files) != set(_COMPONENTS) or any(
        candidate_files[name].get(key) != report["candidate"]["files"][name].get(key)
        for name in _COMPONENTS for key in ("bytes", "sha256")
    ):
        raise ValueError("合成证据必须绑定本次发布的两个完整候选组件。")
    if (
        synthesis.get("synthesis_checks_passed") is not True or len(samples) != 8 or len(audio) != 8
        or not sample_names.issubset(audio) or len({sample.get("seed") for sample in samples}) < 2
    ):
        raise ValueError("有损验收必须包含完整八个多种子合成样本。")
    for sample in samples:
        if (
            not isinstance(sample.get("name"), str) or not sample["name"]
            or sample.get("finite_audio") is not True or sample.get("termination") != "eos"
            or type(sample.get("seed")) is not int or type(sample.get("frames")) is not int or sample["frames"] <= 0
            or not math.isfinite(sample.get("seconds", math.nan)) or sample.get("seconds", 0) <= 0
            or re.fullmatch(r"[0-9a-f]{64}", sample.get("audio_sha256", "")) is None
            or re.fullmatch(r"[0-9a-f]{64}", sample.get("wav_sha256", "")) is None
        ):
            raise ValueError("合成样本存在无效音频、异常结束或缺失摘要。")
    asr = reports.get("asr", {})
    comparisons = asr.get("samples", [])
    if (
        asr.get("model_revision") != "973afd24965f72e36ca33b3055d56a652f456b4d"
        or asr.get("normalization") != "NFKC+OpenCC-t2s+punctuation-whitespace"
        or len(comparisons) != 6 or {sample.get("name") for sample in comparisons} != sample_names
    ):
        raise ValueError("内容复核必须使用冻结的转录版本、规范化规则和六个固定样本。")
    originals = {}
    for side in ("baseline", "candidate"):
        original = asr.get(side, {})
        entries = original.get("samples", [])
        if (
            original.get("status") != "complete"
            or original.get("model") != {"id": "openai/whisper-small", "revision": asr["model_revision"]}
            or not original.get("settings") or original.get("settings") != asr.get("baseline", {}).get("settings")
            or len(entries) != 6 or {entry.get("name") for entry in entries} != sample_names
        ):
            raise ValueError("原始转录记录缺少完整样本，或模型版本与推理设置不同。")
        originals[side] = {entry["name"]: entry for entry in entries}
        for entry in entries:
            reference = entry.get("canonical_normalized_reference")
            transcript = entry.get("canonical_normalized_transcript")
            if (
                entry.get("status") != "complete" or not isinstance(reference, str) or not reference
                or not isinstance(transcript, str)
                or re.fullmatch(r"[0-9a-f]{64}", entry.get("wav_sha256", "")) is None
                or type(entry.get("canonical_reference_characters")) is not int
                or entry["canonical_reference_characters"] != len(reference)
                or type(entry.get("canonical_edit_distance")) is not int
                or entry["canonical_edit_distance"] != _edit_distance(reference, transcript)
            ):
                raise ValueError("原始转录的音频摘要、字符数或编辑距离与规范化文本不符。")
    for sample in comparisons:
        name = sample["name"]
        baseline, candidate = originals["baseline"][name], originals["candidate"][name]
        counts = [sample.get(key) for key in ("reference_edits", "candidate_edits", "reference_characters", "candidate_characters")]
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or sample["reference_characters"] <= 0 or sample["candidate_characters"] != sample["reference_characters"]
            or sample["candidate_edits"] > sample["reference_edits"]
            or baseline["canonical_normalized_reference"] != candidate["canonical_normalized_reference"]
            or sample["reference_edits"] != baseline["canonical_edit_distance"]
            or sample["candidate_edits"] != candidate["canonical_edit_distance"]
            or sample["reference_characters"] != baseline["canonical_reference_characters"]
            or sample["candidate_characters"] != candidate["canonical_reference_characters"]
            or sample.get("candidate_audio_sha256") != audio[name]
            or sample.get("candidate_wav_sha256") != synthesized[name]["wav_sha256"]
            or sample.get("candidate_wav_sha256") != candidate["wav_sha256"]
        ):
            raise ValueError("内容复核与原始转录、实际合成音频不一致，或量化样本错误数增加。")
    expected_totals = {
        "reference_edits": sum(sample["reference_edits"] for sample in comparisons),
        "candidate_edits": sum(sample["candidate_edits"] for sample in comparisons),
        "characters": sum(sample["reference_characters"] for sample in comparisons),
    }
    if asr.get("aggregate") != expected_totals:
        raise ValueError("内容复核汇总与逐样本结果不一致。")
    listening = reports.get("listening", {})
    listened = listening.get("samples", [])
    if (
        listening.get("confirmed") is not True or not listening.get("feedback") or len(listened) != 2
        or len({sample.get("name") for sample in listened}) != 2
        or any(
            sample.get("name") not in audio or audio[sample["name"]] != sample.get("audio_sha256")
            for sample in listened
        )
    ):
        raise ValueError("用户试听必须明确确认并绑定两个实际合成样本的摘要。")
    return len(samples), len(comparisons), len(listened)


def _validate_trajectory_evidence(trajectory: dict[str, Any], report: dict[str, Any]) -> set[str]:
    """从冻结计划重建消费事件及码本覆盖，拒绝删减或移植的轨迹。

    Args:
        trajectory: 保留完整计划与原始指标的真实轨迹报告。
        report: 绑定原始和候选组件摘要的发布报告。
    Returns:
        已验证的六个固定任务名称。
    """
    trace = trajectory.get("trace", {})
    plan = trace.get("plan", {})
    canonical = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    record = trace.get("files", {}).get("plan.json", {})
    if (
        plan.get("schema_version") != 1 or plan.get("kind") != "breeze_real_trajectory"
        or trace.get("canonical_sha256") != digest or record.get("sha256") != digest
        or record.get("bytes") != len(canonical)
        or any(
            plan.get("arrays", {}).get(key) != trace.get("files", {}).get("arrays.npz", {}).get(key)
            or plan.get("arrays", {}).get(key) is None for key in ("bytes", "sha256")
        )
    ):
        raise ValueError("真实轨迹缺少与原始文件和数组摘要一致的完整冻结计划。")
    for name in _COMPONENTS:
        if any(
            plan.get("reference", {}).get("files", {}).get(name, {}).get(key)
            != report["reference"]["files"][name].get(key) for key in ("bytes", "sha256")
        ):
            raise ValueError("冻结计划与发布验收的原始模型摘要不同。")
    samples, events = plan.get("samples", []), plan.get("events", [])
    names = [sample.get("name") for sample in samples]
    if len(names) != 6 or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 6:
        raise ValueError("冻结计划必须包含六个名称唯一的固定任务。")
    sample_names = set(names)
    per_sample = {
        name: {"backbone": False, "depth_positions": set(), "depth_heads": set(), "guided_components": set()}
        for name in names
    }
    cases = trajectory["cases"]
    by_event = {case.get("event"): case for case in cases}
    if (
        len(by_event) != len(cases) or any(type(case.get("event")) is not int for case in cases)
        or trajectory["summary"].get("events") != len(events)
        or trajectory["summary"].get("cases") != len(cases)
    ):
        raise ValueError("真实轨迹事件编号重复或汇总与冻结计划不符。")
    consumed, cfg, reference_pending, prefix = set(), False, set(), 0
    codec_widths, backbone_widths = set(), set()
    for index, event in enumerate(events):
        component, operation, sample = event.get("component"), event.get("operation"), event.get("sample")
        if (
            type(event.get("id")) is not int or event["id"] != index
            or component not in {"backbone", "depth"} or sample not in sample_names
            or event.get("phase") not in {"reference", "synthesis"}
            or operation not in {"decode", "guided", "reset", "reserve", "remove_sequence"}
        ):
            raise ValueError("冻结计划包含无效的事件编号、组件、样本或操作。")
        expected = event.get("expected", {})
        is_consumed = operation == "guided" or operation == "decode" and bool(expected)
        case = by_event.get(index)
        if is_consumed:
            consumed.add(index)
            output_names = {"logits"} if operation == "guided" else set(expected)
            if (
                not output_names or not output_names <= {"logits", "hidden"} or case is None
                or any(case.get(key) != event[key] for key in ("id", "component", "operation", "sample", "phase"))
                or set(case["outputs"]) != output_names
            ):
                raise ValueError("真实轨迹遗漏消费事件，或事件归属及输出集合与冻结计划不同。")
        if operation == "guided":
            if not isinstance(expected, str) or not expected or not math.isfinite(event.get("scale", math.nan)):
                raise ValueError("冻结引导事件缺少参考数组或有限的引导系数。")
            for source in (event.get("positive_ref"), event.get("negative_ref")):
                if source is None and event.get("positive_ref") is not None:
                    continue
                if (
                    not isinstance(source, dict) or set(source) != {"event", "row"}
                    or type(source["event"]) is not int or not 0 <= source["event"] < index
                    or type(source["row"]) is not int or source["row"] < 0
                ):
                    raise ValueError("冻结引导引用必须指向先前的有效解码行。")
                producer = events[source["event"]]
                output = by_event.get(source["event"], {}).get("outputs", {}).get("logits", {})
                shape = output.get("expected_shape", [])
                if (
                    producer["operation"] != "decode" or producer["component"] != component or producer["sample"] != sample
                    or len(shape) != 2 or source["row"] >= shape[0]
                    or case["outputs"]["logits"]["expected_shape"] != [1, shape[1]]
                ):
                    raise ValueError("冻结引导引用与同组件、同样本的完整输出头不同。")
            if event["phase"] == "synthesis":
                per_sample[sample]["guided_components"].add(component)
                cfg = cfg or event.get("negative_ref") is not None and event["scale"] != 1.0
            continue
        if operation != "decode":
            continue
        arguments = event.get("arguments", {})
        positions = arguments.get("positions", [])
        if not positions or any(type(position) is not int or position < 0 for position in positions):
            raise ValueError("冻结解码事件缺少有效的批次位置。")
        if not isinstance(expected, dict) or set(expected) - {"logits", "hidden"}:
            raise ValueError("冻结解码输出集合不受支持。")
        if case is not None and any(
            len(value["expected_shape"]) != 2 or value["expected_shape"][0] != len(positions)
            for value in case["outputs"].values()
        ):
            raise ValueError("真实消费输出的批次与冻结解码位置不同。")
        if component == "backbone":
            if event["phase"] == "reference":
                reference_pending.add(sample)
            elif sample in reference_pending:
                prefix = max(prefix, max(positions))
                reference_pending.remove(sample)
            if "logits" in expected:
                width = case["outputs"]["logits"]["expected_shape"][1]
                if arguments.get("logit_start", 0) != 0 or arguments.get("logit_count") not in (None, width):
                    raise ValueError("冻结主干消费输出没有覆盖完整词表及结束类别。")
                backbone_widths.add(width)
            if event["phase"] == "synthesis" and {"hidden", "logits"} <= set(expected):
                per_sample[sample]["backbone"] = True
        elif event["phase"] == "synthesis":
            if len(set(positions)) != 1 or not 0 <= positions[0] < 16:
                raise ValueError("冻结深度事件没有对应同一个有效码本位置。")
            per_sample[sample]["depth_positions"].update(positions)
            if "logits" in expected:
                position = positions[0]
                width = case["outputs"]["logits"]["expected_shape"][1]
                if (
                    not 1 <= position < 16 or width < 2048 or arguments.get("logit_count") != width
                    or arguments.get("logit_start", 0) != (position - 1) * width
                ):
                    raise ValueError("冻结深度消费输出遗漏完整码本头或请求了错误偏移。")
                codec_widths.add(width)
                per_sample[sample]["depth_heads"].add(position)
    coverage = trajectory.get("coverage", {})
    if (
        consumed != set(by_event) or not cfg or prefix <= 512
        or len(codec_widths) != 1 or backbone_widths != {next(iter(codec_widths)) + 1}
        or coverage.get("complete") is not True or coverage.get("all_consumed_outputs") is not True
        or coverage.get("cfg") is not True or coverage.get("sample_count") != 6
        or coverage.get("samples") != names or set(coverage.get("per_sample", {})) != sample_names
        or coverage.get("depth_positions") != list(range(16)) or coverage.get("depth_heads") != list(range(1, 16))
        or coverage.get("long_reference_prefix_length", 0) < prefix
    ):
        raise ValueError("真实轨迹没有完整覆盖冻结计划的全部消费输出、条件引导及长参考。")
    for sample in samples:
        current = per_sample[sample["name"]]
        if (
            sample.get("finite_audio") is not True or type(sample.get("frames")) is not int or sample["frames"] <= 0
            or sample.get("termination") != "eos" or not current["backbone"]
            or current["depth_positions"] != set(range(16)) or current["depth_heads"] != set(range(1, 16))
            or current["guided_components"] != {"backbone", "depth"}
        ):
            raise ValueError("冻结任务遗漏有效音频、主干、深度码本或条件引导分支。")
        actual = {key: sorted(value) if isinstance(value, set) else value for key, value in current.items()}
        if coverage["per_sample"][sample["name"]] != actual:
            raise ValueError("逐样本覆盖声明与冻结事件实际覆盖不同。")
    return sample_names


def _validate_lossy_report(report: dict[str, Any]) -> None:
    """验收明确限定范围的有损 Q8 发布证据，保留数值审计失败。

    Args:
        report: 使用有损任务验收范围的完整报告。
    Returns:
        结构、完整覆盖、内容与试听证据一致时正常返回。
    """
    summary, reports = report.get("summary", {}), report.get("reports", {})
    if (
        report.get("schema_version") != 1 or report.get("passed") is not True
        or report.get("quantization") != "q8_0" or report.get("basis") != LOSSY_TASK_AUDIT_SCOPE
        or report.get("numerical_equivalence_claimed") is not False
        or report.get("strict_numerical_gates_passed") is not False
        or summary.get("passed") is not True or summary.get("structural_failed_cases") != 0
    ):
        raise ValueError("有损 Q8 验收必须明确范围及未通过原数值门限的状态。")
    synthetic = reports.get("synthetic_stress", {})
    synthetic_failed, _ = _audit_numerical_report(synthetic, report, expected_passed=False)
    if (
        len(synthetic["cases"]) != 57 or synthetic.get("coverage", {}).get("full_suite") is not True
        or synthetic["summary"].get("expected_cases") != 57
    ):
        raise ValueError("有损验收必须保留完整五十七项合成输入审计。")
    trajectory = reports.get("real_trajectory", {})
    numerical_failed, checked_outputs = _audit_numerical_report(trajectory, report, expected_passed=False)
    sample_names = _validate_trajectory_evidence(trajectory, report)
    if trajectory["summary"].get("checked_outputs") != checked_outputs:
        raise ValueError("真实轨迹的消费输出数量与逐事件结果不同。")
    synthesis_count, asr_count, listened_count = _validate_lossy_tasks(report, sample_names)
    expected = {
        "numerical_failed_cases": numerical_failed, "synthetic_failed_cases": synthetic_failed,
        "checked_outputs": checked_outputs, "synthesis_samples": synthesis_count,
        "asr_samples": asr_count, "listened_samples": listened_count,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("有损验收汇总与原始审计和任务结果不一致。")


def validate_q8_audit(report: dict[str, Any]) -> None:
    """按报告声明的范围核验完整证据，并检查两个生成组件的摘要记录。

    Args:
        report: 严格数值验收或明确有损任务验收的完整报告。
    Returns:
        范围、指标及证据完整时正常返回，否则拒绝派生。
    """
    if report.get("basis") == LOSSY_TASK_AUDIT_SCOPE:
        _validate_lossy_report(report)
    else:
        _validate_strict_report(report)
    for kind in ("reference", "candidate"):
        if set(report.get(kind, {}).get("files", {})) != set(_COMPONENTS):
            raise ValueError("Q8 对照报告必须精确绑定两个完整生成组件。")
