"""验证真实轨迹重放的消费输出、缓存操作、证据摘要和完整验收边界。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from tools.breeze_reference import q8_trajectory


SAMPLE_NAMES = (
    "01-clear-female", "02-low-male", "03-bright-female", "04-gentle-story",
    "05-reference-clone", "06-long-reference",
)


def _file_record(path: Path) -> dict[str, Any]:
    """计算夹具证据文件的真实摘要。

    Args:
        path: 当前测试中的文件。
    Returns:
        文件字节数和 SHA256。
    """
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class _TrajectoryDecoder:
    """模拟可扩容、截断且具有独立分支状态的原生解码器。

    Args:
        path: 用于区分组件和候选模型的文件路径。
        configuration: 用于注入受控失败的设置。
        parameters: 原生模型初始化参数。
    """

    def __init__(self, path: Path, configuration: dict[str, Any], **parameters: Any) -> None:
        self.path, self.configuration, self.parameters = path, configuration, parameters
        self.input_size = self.hidden_size = 8 if path.stem == "backbone" else 4
        self.vocab_size = 2052 if path.stem == "backbone" else 32816
        self.n_ctx = parameters["n_ctx"]
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self.history: dict[int, list[float]] = {}

    def reset(self) -> None:
        """清除所有分支缓存。

        Returns:
            无。
        """
        self.calls.append({"operation": "reset"})
        self.history.clear()

    def reserve(self, n_ctx: int) -> None:
        """仅在扩容时重建上下文并清除缓存。

        Args:
            n_ctx: 每条分支的新容量。
        Returns:
            无。
        """
        self.calls.append({"operation": "reserve", "n_ctx": n_ctx})
        if n_ctx > self.n_ctx:
            self.n_ctx = n_ctx
            self.history.clear()

    def remove_sequence(self, sequence: int, start: int, end: int = -1) -> None:
        """删除指定分支的缓存区间。

        Args:
            sequence: 独立分支编号。
            start: 删除起始位置。
            end: 删除结束位置，负一表示直到末尾。
        Returns:
            无。
        """
        self.calls.append({"operation": "remove_sequence", "sequence": sequence, "start": start, "end": end})
        history = self.history.setdefault(sequence, [])
        del history[start:None if end == -1 else end]

    def decode(self, **arguments: Any) -> dict[str, Any]:
        """校验历史位置，并按同源输入产生确定性输出。

        Args:
            arguments: 输入数组、分支位置和输出区间。
        Returns:
            当前最后位置的隐藏状态与分数。
        """
        kind = "embeddings" if "embeddings" in arguments else "tokens"
        values = np.asarray(arguments[kind])
        self.calls.append({"operation": "decode", **arguments, kind: values.copy()})
        assert values.ndim == (3 if kind == "embeddings" else 2)
        if kind == "embeddings":
            assert values.shape[2] == self.input_size
        assert np.isfinite(values).all()
        hidden = []
        for row, (position, sequence) in enumerate(zip(arguments["positions"], arguments["sequence_ids"])):
            history = self.history.setdefault(sequence, [])
            assert position == len(history)
            assert position + values.shape[1] <= self.n_ctx
            features = values[row].mean(axis=-1) if kind == "embeddings" else values[row] / 32816.0
            history.extend(float(value) for value in features)
            scalar = 1.0 + np.mean(history) * 0.01 + len(history) * 1e-4 + sequence * 0.003
            hidden.append(scalar + np.arange(self.hidden_size, dtype=np.float32) * 0.001)
        if not arguments.get("output", True):
            return {"hidden": None, "logits": None}
        hidden_array = np.asarray(hidden, dtype=np.float32)
        count = arguments.get("logit_count", self.vocab_size - arguments.get("logit_start", 0))
        columns = np.sin(np.arange(count, dtype=np.float32) * 0.01)
        logits = hidden_array[:, :1] + columns[None, :] + arguments.get("logit_start", 0) * 1e-6
        if self.path.parent.name == "candidate":
            if self.configuration.get("drift"):
                logits += 0.2
            if self.configuration.get("nonfinite"):
                logits[:, 0] = np.nan
            if self.configuration.get("shape"):
                logits = logits[:, :-1]
        return {"hidden": hidden_array if self.parameters["embeddings"] else None, "logits": logits}

    def close(self) -> None:
        """标记当前模型已释放。

        Returns:
            无。
        """
        self.closed = True


def _write_trace(fixture: dict[str, Any]) -> None:
    """保存修改后的轨迹，并同步数组文件的真实摘要。

    Args:
        fixture: 当前测试中的目录、计划和数组。
    Returns:
        无。
    """
    trace = fixture["trace"]
    np.savez(trace / "arrays.npz", **fixture["arrays"])
    fixture["plan"]["arrays"] = _file_record(trace / "arrays.npz")
    (trace / "plan.json").write_bytes((json.dumps(fixture["plan"], ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


@pytest.fixture
def trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """生成六个样本、全部码本和长参考的完整真实文件夹具。

    Args:
        tmp_path: 当前测试的隔离目录。
        monkeypatch: 原生模型替换工具。
    Returns:
        模型目录、轨迹证据、假模型实例和失败配置。
    """
    reference, candidate, trace = tmp_path / "reference", tmp_path / "candidate", tmp_path / "trace"
    for directory in (reference / "ckpt", candidate, trace):
        directory.mkdir(parents=True)
    for directory in (reference / "ckpt", candidate):
        for name in ("backbone.gguf", "depth.gguf"):
            (directory / name).write_bytes(f"{directory.name}/{name}".encode())
    plan: dict[str, Any] = {
        "schema_version": 1, "kind": "breeze_real_trajectory",
        "reference": {"files": {name: _file_record(reference / "ckpt" / name) for name in ("backbone.gguf", "depth.gguf")}},
        "components": {"backbone": {"n_ctx": 16}, "depth": {"n_ctx": 16}},
        "samples": [
            {"name": name, "seed": 42, "cfg_scale": 4.0, "frames": 1, "finite_audio": True,
             "termination": "eos", "clone": index >= 4, "long_reference": index == 5}
            for index, name in enumerate(SAMPLE_NAMES)
        ],
        "events": [],
    }
    arrays: dict[str, np.ndarray] = {}
    producers = {
        component: _TrajectoryDecoder(Path("producer") / f"{component}.gguf", {}, n_ctx=16, embeddings=component == "backbone")
        for component in ("backbone", "depth")
    }
    rng = np.random.default_rng(42)

    def operation(component: str, name: str, sample: str, phase: str = "synthesis", *args: Any, **arguments: Any) -> None:
        """将缓存操作及其原始参数写入轨迹。

        Args:
            component: 当前原生组件。
            name: 缓存操作名称。
            sample: 当前样本名称。
            phase: 当前参考或合成阶段。
            args: 操作的位置参数。
            arguments: 操作的关键字参数。
        Returns:
            无。
        """
        getattr(producers[component], name)(*args, **arguments)
        plan["events"].append({
            "id": len(plan["events"]), "component": component, "operation": name,
            "sample": sample, "phase": phase, "args": list(args), "arguments": arguments,
        })

    def decode(component: str, sample: str, phase: str, **arguments: Any) -> tuple[int, dict[str, Any]]:
        """保存输入及全部原始消费输出。

        Args:
            component: 当前原生组件。
            sample: 当前样本名称。
            phase: 当前参考或合成阶段。
            arguments: 解码输入及缓存位置。
        Returns:
            解码事件编号和输出数组。
        """
        event_id = len(plan["events"])
        event = {"id": event_id, "component": component, "operation": "decode", "sample": sample, "phase": phase,
                 "arguments": {}, "expected": {}}
        for name, value in arguments.items():
            if isinstance(value, np.ndarray):
                key = f"e{event_id}_{name}"
                arrays[key] = value.copy()
                event["arguments"][name] = {"array": key}
            else:
                event["arguments"][name] = value
        output = producers[component].decode(**arguments)
        for name, value in output.items():
            if value is not None:
                key = f"e{event_id}_{name}"
                arrays[key] = value.copy()
                event["expected"][name] = key
        plan["events"].append(event)
        return event_id, output

    def guided(component: str, sample: str, source: int, output: dict[str, Any]) -> None:
        """记录使用两个真实输出分支的条件引导结果。

        Args:
            component: 分数所属组件。
            sample: 当前样本名称。
            source: 被消费的解码事件编号。
            output: 该事件的原始输出。
        Returns:
            无。
        """
        event_id = len(plan["events"])
        key = f"e{event_id}_guided"
        positive, negative = output["logits"]
        arrays[key] = negative + 4.0 * (positive - negative)
        plan["events"].append({
            "id": event_id, "component": component, "operation": "guided", "sample": sample, "phase": "synthesis",
            "positive_ref": {"event": source, "row": 0}, "negative_ref": {"event": source, "row": 1},
            "scale": 4.0, "expected": key,
        })

    for sample in SAMPLE_NAMES:
        operation("backbone", "reserve", sample, "synthesis", 600)
        operation("backbone", "reset", sample)
        operation("depth", "reset", sample)
        if sample == SAMPLE_NAMES[-1]:
            decode("backbone", sample, "reference", embeddings=rng.normal(0, 0.3, (1, 513, 8)).astype(np.float32),
                   positions=[0], sequence_ids=[0], output=False)
            operation("backbone", "remove_sequence", sample, "synthesis", 0, 512)
            decode("backbone", sample, "synthesis", embeddings=np.ones((1, 3, 8), np.float32),
                   positions=[0], sequence_ids=[1], output=False)
            positions = [512, 3]
        else:
            decode("backbone", sample, "synthesis", embeddings=np.ones((2, 2, 8), np.float32),
                   positions=[0, 0], sequence_ids=[0, 1], output=False)
            operation("backbone", "remove_sequence", sample, sequence=0, start=1, end=-1)
            positions = [1, 2]
        event_id, output = decode("backbone", sample, "synthesis", embeddings=rng.normal(0, 0.3, (2, 1, 8)).astype(np.float32),
                                  positions=positions, sequence_ids=[0, 1], logit_start=0, logit_count=2052)
        guided("backbone", sample, event_id, output)
        decode("depth", sample, "synthesis", embeddings=rng.normal(0, 0.3, (2, 1, 4)).astype(np.float32),
               positions=[0, 0], sequence_ids=[0, 1], output=False)
        for position in range(1, 16):
            offset = (position - 1) * 2051
            event_id, output = decode("depth", sample, "synthesis",
                                      tokens=rng.integers(0, 2048, (2, 1), dtype=np.int32) + offset,
                                      positions=[position, position], sequence_ids=[0, 1], logit_start=offset, logit_count=2051)
            guided("depth", sample, event_id, output)

    models: list[_TrajectoryDecoder] = []
    configuration: dict[str, Any] = {}

    def factory(path: Path, **parameters: Any) -> _TrajectoryDecoder:
        """禁止同时加载多个组件，并记录被验证的候选模型。

        Args:
            path: 当前候选 GGUF 文件。
            parameters: 原生初始化参数。
        Returns:
            唯一存活的当前假解码器。
        """
        assert all(model.closed for model in models)
        assert path.parent == candidate
        native = _TrajectoryDecoder(path, configuration, **parameters)
        models.append(native)
        return native

    monkeypatch.setattr(q8_trajectory, "LlamaDecoder", factory)
    fixture = {"reference": reference, "candidate": candidate, "trace": trace, "plan": plan, "arrays": arrays,
               "models": models, "configuration": configuration}
    _write_trace(fixture)
    return fixture


def test_complete_trajectory_checks_every_consumed_output_and_cache_operation(trajectory: dict[str, Any]) -> None:
    """六个样本的全部消费输出、条件引导和缓存操作都进入验收。

    Args:
        trajectory: 完整同源轨迹夹具。
    """
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is True
    assert report["summary"]["passed"] is True
    assert report["summary"]["cases"] == 192
    assert report["summary"]["failed_cases"] == 0
    assert report["summary"]["checked_outputs"] == 198
    assert report["summary"]["events"] == len(trajectory["plan"]["events"])
    assert set(report["coverage"]["samples"]) == set(SAMPLE_NAMES)
    assert report["coverage"]["depth_positions"] == list(range(16))
    assert report["coverage"]["depth_heads"] == list(range(1, 16))
    assert report["coverage"]["cfg"] is True
    assert report["coverage"]["all_consumed_outputs"] is True
    assert report["coverage"]["long_reference_prefix_length"] >= 513
    assert report["coverage"]["max_backbone_context_length"] >= 513
    assert len(trajectory["models"]) == 2
    assert [model.path.stem for model in trajectory["models"]] == ["backbone", "depth"]
    for native in trajectory["models"]:
        assert native.closed
        expected_operations = [event["operation"] for event in trajectory["plan"]["events"]
                               if event["component"] == native.path.stem and event["operation"] != "guided"]
        assert [call["operation"] for call in native.calls] == expected_operations
    assert all(case["passed"] and case["outputs"] for case in report["cases"])
    assert {case["operation"] for case in report["cases"]} == {"decode", "guided"}
    for filename in ("plan.json", "arrays.npz"):
        expected = _file_record(trajectory["trace"] / filename)
        assert report["trace"]["files"][filename]["sha256"] == expected["sha256"]
        assert report["trace"]["files"][filename]["bytes"] == expected["bytes"]
    assert report["trace"]["plan"] == trajectory["plan"]
    assert report["trace"]["canonical_sha256"] == _file_record(trajectory["trace"] / "plan.json")["sha256"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["source_hash", "arrays_hash", "arrays_bytes", "arrays_manifest", "event_id"])
def test_corrupt_provenance_or_event_order_fails_before_replay(trajectory: dict[str, Any], failure: str) -> None:
    """源文件、数组摘要或事件编号不一致时拒绝原生重放。

    Args:
        trajectory: 可受控破坏的完整轨迹。
        failure: 当前破坏的证据约束。
    """
    if failure == "source_hash":
        trajectory["plan"]["reference"]["files"]["backbone.gguf"]["sha256"] = "0" * 64
    elif failure == "event_id":
        trajectory["plan"]["events"][3]["id"] = 999
    elif failure == "arrays_bytes":
        trajectory["plan"]["arrays"]["bytes"] += 1
    elif failure == "arrays_manifest":
        del trajectory["plan"]["arrays"]
    else:
        trajectory["plan"]["arrays"]["sha256"] = "0" * 64
    (trajectory["trace"] / "plan.json").write_text(json.dumps(trajectory["plan"]), encoding="utf-8")
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert report["errors"]
    assert not trajectory["models"]
    json.dumps(report, allow_nan=False)


def test_plan_file_must_match_embedded_canonical_serialization(trajectory: dict[str, Any]) -> None:
    """冻结计划的原文件摘要必须能够由报告中的计划内容直接重算。

    Args:
        trajectory: 保存完整规范序列化计划的轨迹。
    Returns:
        非规范文件在原生重放前被拒绝时正常返回。
    """
    (trajectory["trace"] / "plan.json").write_bytes(json.dumps(trajectory["plan"]).encode("utf-8"))
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert any("规范序列化" in error for error in report["errors"])
    assert report["trace"]["plan"] == trajectory["plan"]
    assert not trajectory["models"]


@pytest.mark.parametrize("location", ["input", "output", "guided"])
def test_missing_array_keys_fail_with_complete_report(trajectory: dict[str, Any], location: str) -> None:
    """完整文件摘要不能掩盖输入、输出或引导数组缺失。

    Args:
        trajectory: 当前完整轨迹。
        location: 要删除的数组引用类别。
    """
    if location == "guided":
        event = next(event for event in trajectory["plan"]["events"] if event["operation"] == "guided")
        key = event["expected"]
    else:
        event = next(event for event in trajectory["plan"]["events"] if event["operation"] == "decode" and event["expected"])
        key = event["arguments"]["embeddings"]["array"] if location == "input" else event["expected"]["hidden"]
    del trajectory["arrays"][key]
    _write_trace(trajectory)
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert report["errors"] or report["summary"]["failed_cases"] > 0
    assert all(model.closed for model in trajectory["models"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["nonfinite", "shape", "drift", "reference_nonfinite", "input_dimension"])
def test_invalid_arrays_and_numeric_drift_remain_failures(trajectory: dict[str, Any], failure: str) -> None:
    """数值漂移、非有限值和维度错误不能通过轨迹验收。

    Args:
        trajectory: 当前完整轨迹。
        failure: 当前注入的数组问题。
    """
    if failure in ("nonfinite", "shape", "drift"):
        trajectory["configuration"][failure] = True
    else:
        event = next(event for event in trajectory["plan"]["events"] if event["operation"] == "decode" and event["expected"])
        if failure == "reference_nonfinite":
            trajectory["arrays"][event["expected"]["hidden"]][0, 0] = np.inf
        else:
            key = event["arguments"]["embeddings"]["array"]
            trajectory["arrays"][key] = trajectory["arrays"][key][..., :-1]
        _write_trace(trajectory)
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert report["errors"] or report["summary"]["failed_cases"] > 0
    assert all(model.closed for model in trajectory["models"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["future_event", "nondecode_event", "negative_row", "oversized_row", "expected_shape", "expected_drift"])
def test_guided_references_and_expected_output_are_strictly_checked(trajectory: dict[str, Any], failure: str) -> None:
    """条件引导必须消费合法历史事件及行，并满足同一数值标准。

    Args:
        trajectory: 当前完整轨迹。
        failure: 要破坏的引用或输出。
    """
    event = next(event for event in trajectory["plan"]["events"] if event["operation"] == "guided")
    if failure == "future_event":
        event["positive_ref"]["event"] = len(trajectory["plan"]["events"])
    elif failure == "nondecode_event":
        event["positive_ref"]["event"] = 0
    elif failure == "negative_row":
        event["negative_ref"]["row"] = -1
    elif failure == "oversized_row":
        event["negative_ref"]["row"] = 2
    elif failure == "expected_shape":
        trajectory["arrays"][event["expected"]] = trajectory["arrays"][event["expected"]][None, :]
    else:
        trajectory["arrays"][event["expected"]] += 1.0
    _write_trace(trajectory)
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert report["errors"] or report["summary"]["failed_cases"] > 0
    assert all(model.closed for model in trajectory["models"])


@pytest.mark.parametrize("failure", ["empty", "sample", "hidden", "depth_head", "guided", "long_reference"])
def test_incomplete_coverage_cannot_pass(trajectory: dict[str, Any], failure: str) -> None:
    """空集和局部通过不能替代六个样本、全部输出头及长参考的完整覆盖。

    Args:
        trajectory: 当前完整轨迹。
        failure: 要删除的覆盖要素。
    """
    events = trajectory["plan"]["events"]
    if failure == "empty":
        trajectory["plan"]["events"] = []
    elif failure == "sample":
        trajectory["plan"]["samples"] = trajectory["plan"]["samples"][:-1]
    elif failure == "hidden":
        event = next(event for event in events if event["component"] == "backbone" and event["operation"] == "decode" and event["expected"])
        del event["expected"]["hidden"]
    elif failure == "depth_head":
        event = next(event for event in events if event["component"] == "depth" and event["operation"] == "decode"
                     and event["arguments"]["positions"] == [15, 15])
        event["arguments"]["output"] = False
        event["expected"] = {}
    elif failure == "guided":
        for event in events:
            if event["operation"] == "guided":
                event["negative_ref"] = None
                source = events[event["positive_ref"]["event"]]
                trajectory["arrays"][event["expected"]] = trajectory["arrays"][source["expected"]["logits"]][0].copy()
    else:
        event = next(event for event in events if event["phase"] == "reference" and event["operation"] == "decode")
        event["phase"] = "synthesis"
    _write_trace(trajectory)
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert report["coverage"]["complete"] is False


@pytest.mark.parametrize("failure", ["depth_offset", "depth_width", "backbone_width"])
def test_physical_vocabulary_slices_cannot_silently_select_a_different_head(
    trajectory: dict[str, Any], failure: str,
) -> None:
    """输出数值近似相同，也必须严格保留完整物理词表与深度头位置。

    Args:
        trajectory: 当前完整轨迹。
        failure: 当前破坏的物理词表边界。
    """
    component = "backbone" if failure == "backbone_width" else "depth"
    event = next(event for event in trajectory["plan"]["events"]
                 if event["component"] == component and event["operation"] == "decode" and event["expected"])
    if failure == "depth_offset":
        event["arguments"]["logit_start"] += 1
    else:
        event["arguments"]["logit_count"] -= 1
        key = event["expected"]["logits"]
        trajectory["arrays"][key] = trajectory["arrays"][key][:, :-1]
        guided_event = next(value for value in trajectory["plan"]["events"]
                            if value["operation"] == "guided" and value["positive_ref"]["event"] == event["id"])
        key = guided_event["expected"]
        trajectory["arrays"][key] = trajectory["arrays"][key][:-1]
    _write_trace(trajectory)
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    assert any(case["event"] == event["id"] and not case["passed"] for case in report["cases"])


def test_guided_cannot_reuse_a_decode_from_before_the_same_sample_reset(trajectory: dict[str, Any]) -> None:
    """同样本清缓存之后，旧分数引用也必须失效。

    Args:
        trajectory: 当前完整轨迹。
    """
    events = trajectory["plan"]["events"]
    guided_event = next(event for event in events if event["operation"] == "guided")
    insertion = guided_event["id"]
    for event in events:
        if event["id"] >= insertion:
            event["id"] += 1
        if event["operation"] != "guided":
            continue
        for key in ("positive_ref", "negative_ref"):
            if event.get(key) is not None and event[key]["event"] >= insertion:
                event[key]["event"] += 1
    events.insert(insertion, {
        "id": insertion, "component": "backbone", "operation": "reset", "sample": guided_event["sample"],
        "phase": "synthesis", "args": [], "arguments": {},
    })
    _write_trace(trajectory)
    report = q8_trajectory.validate_trajectory(trajectory["reference"], trajectory["candidate"], trajectory["trace"], "cpu")
    assert report["passed"] is False
    case = next(case for case in report["cases"] if case["event"] == guided_event["id"])
    assert case["passed"] is False and case["errors"]


@pytest.mark.parametrize("failure", [False, True])
def test_cli_saves_report_before_returning_failure(
    trajectory: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool,
) -> None:
    """命令先保存完整轨迹报告，随后用退出码传达失败。

    Args:
        trajectory: 当前完整轨迹。
        tmp_path: 报告隔离目录。
        monkeypatch: 命令行参数替换工具。
        failure: 是否注入候选数值漂移。
    """
    trajectory["configuration"]["drift"] = failure
    destination = tmp_path / "reports" / "trajectory.json"
    monkeypatch.setattr(sys, "argv", [
        "q8_trajectory", "--reference-bundle", str(trajectory["reference"]),
        "--candidate-dir", str(trajectory["candidate"]), "--trace-dir", str(trajectory["trace"]),
        "--backend", "cpu", "--report", str(destination),
    ])
    if failure:
        with pytest.raises(SystemExit) as raised:
            q8_trajectory.main()
        assert raised.value.code == 1
    else:
        q8_trajectory.main()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["passed"] is not failure
    assert report["summary"]["events"] == len(trajectory["plan"]["events"])
