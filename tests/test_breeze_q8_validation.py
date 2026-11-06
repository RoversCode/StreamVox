"""用无 Torch 的轻量解码器验证独立量化门限、缓存计划和完整失败报告。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from tools.breeze_reference import q8_validate
from tools.breeze_reference.numerics import BF16_POLICY, FP32_POLICY


class _FakeDecoder:
    """模拟具有独立分支缓存的原生解码器。

    Args:
        path: 用文件名区分组件、用目录区分原始模型与候选模型。
        configuration: 当前测试注入的受控失败配置。
        parameters: 被测工具传递的原生初始化参数。
    """

    def __init__(self, path: Path, configuration: dict[str, Any], **parameters: Any) -> None:
        self.path, self.configuration, self.parameters = path, configuration, parameters
        self.candidate = path.parent.name == "candidate"
        self.input_size = 8 if path.stem == "backbone" else 4
        self.hidden_size = self.input_size
        self.vocab_size = 2052 if path.stem == "backbone" else 32816
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self.resets = 0
        self.positions: dict[int, int] = {}
        self.states: dict[int, float] = {}

    def reset(self) -> None:
        """清除独立分支状态。

        Returns:
            无。
        """
        self.resets += 1
        self.positions.clear()
        self.states.clear()

    def decode(self, **arguments: Any) -> dict[str, Any]:
        """以分支历史生成可复现输出，并按配置注入失败。

        Args:
            arguments: 原生输入、绝对位置、序列编号及输出头切片。
        Returns:
            最后位置的隐藏状态和分数。
        """
        kind = "embeddings" if "embeddings" in arguments else "tokens"
        values = arguments[kind]
        assert not values.flags.writeable
        self.calls.append({**arguments, kind: values.copy()})
        if self.candidate and self.configuration.get("decode_once") and len(self.calls) == 1:
            raise RuntimeError("受控解码失败")
        hidden = []
        for row, (position, sequence) in enumerate(zip(arguments["positions"], arguments["sequence_ids"])):
            assert position == self.positions.get(sequence, 0)
            self.positions[sequence] = position + values.shape[1]
            self.states[sequence] = self.states.get(sequence, 0.0) + float(values[row].mean()) * 1e-4
            hidden.append(np.full(self.hidden_size, 1.0 + self.states[sequence], dtype=np.float32))
        hidden_array = np.stack(hidden)
        columns = np.sin(np.arange(arguments["logit_count"], dtype=np.float32) * 0.1)
        logits = hidden_array[:, :1] + columns[None, :] + arguments["logit_start"] * 1e-6
        if self.candidate and self.configuration.get("drift"):
            logits += 0.1
        if self.candidate and self.configuration.get("nonfinite"):
            logits[:, 0] = np.nan
        return {"logits": logits, "hidden": hidden_array if self.parameters["embeddings"] else None}

    def close(self) -> None:
        """标记模型释放，供工厂检测顺序加载。

        Returns:
            无。
        """
        self.closed = True


@pytest.fixture
def fake_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """准备可哈希的模型文件和禁止同时加载模型的假工厂。

    Args:
        tmp_path: 当前测试临时目录。
        monkeypatch: 测试替换工具。
    Returns:
        原始 bundle、候选目录、已加载模型及受控失败配置。
    """
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for directory in (reference / "ckpt", candidate):
        directory.mkdir(parents=True)
        for component in ("backbone", "depth"):
            (directory / f"{component}.gguf").write_bytes(f"{directory.name}/{component}".encode())
    models: list[_FakeDecoder] = []
    configuration: dict[str, Any] = {}

    def factory(path: Path, **parameters: Any) -> _FakeDecoder:
        """校验生命周期并构建当前唯一模型。

        Args:
            path: 本次加载的文件。
            parameters: 原生初始化参数。
        Returns:
            可以确定性重放输入的假解码器。
        """
        assert all(model.closed for model in models)
        if configuration.get("load_failure") and path.parent == candidate and path.stem == "backbone":
            raise RuntimeError("受控加载失败")
        native = _FakeDecoder(path, configuration, **parameters)
        models.append(native)
        return native

    monkeypatch.setattr(q8_validate, "LlamaDecoder", factory)
    return reference, candidate, models, configuration


def test_full_suite_replays_identical_inputs_with_sequential_models(fake_validation: tuple) -> None:
    """完整计划覆盖长预填充、三个续写、错位缓存和全部深度输出头。

    Args:
        fake_validation: 含假解码器的隔离模型目录。
    """
    reference, candidate, models, _configuration = fake_validation
    report = q8_validate.validate_q8(reference, candidate, "cpu")
    assert report["passed"] is True
    assert report["summary"] == {"passed": True, "cases": 57, "failed_cases": 0, "expected_cases": 57}
    assert report["coverage"]["full_suite"] is True
    assert report["quantization"] == "q8_0"
    assert [model.path.stem for model in models] == ["backbone", "backbone", "depth", "depth"]
    assert [model.candidate for model in models] == [False, True, False, True]
    assert all(model.closed for model in models)
    assert [len(model.calls) for model in models] == [25, 25, 32, 32]
    for native in models:
        assert native.parameters == {"backend": "cpu", "n_ctx": 520, "n_seq_max": 2, "embeddings": native.path.stem == "backbone"}
    for original, quantized in ((models[0], models[1]), (models[2], models[3])):
        for expected, actual in zip(original.calls, quantized.calls):
            assert expected.keys() == actual.keys()
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key])
    main_cases = [case for case in report["cases"] if case["component"] == "backbone"]
    assert {case["length"] for case in main_cases} >= {511, 512, 513}
    assert {tuple(case["positions"]) for case in main_cases} >= {(5, 3), (6, 4), (7, 5)}
    for batch_size in (1, 2):
        depth = [case for case in report["cases"] if case["name"].startswith(f"depth/batch-{batch_size}/")]
        assert [case["positions"][0] for case in depth] == list(range(16))
        assert {case["logit_start"] // 2051 for case in depth} == set(range(15))
        assert all(case["logit_count"] == 2051 for case in depth)
        assert depth[0]["input_kind"] == "embeddings"
        assert all(case["input_kind"] == "tokens" for case in depth[1:])
    for case in report["cases"]:
        assert case["passed"] is True
        assert case["logit_count"] == (2052 if case["component"] == "backbone" else 2051)
        assert case["outputs"]["logits"]["valid_vocab_size"] == (2049 if case["component"] == "backbone" else 2048)
    for side, directory in (("reference", reference / "ckpt"), ("candidate", candidate)):
        for filename, asset in report[side]["files"].items():
            raw = (directory / filename).read_bytes()
            assert asset["sha256"] == hashlib.sha256(raw).hexdigest()
            assert asset["bytes"] == len(raw)
    json.dumps(report, allow_nan=False)


def test_direct_checkpoint_path_and_fixed_seed_are_reproducible(fake_validation: tuple) -> None:
    """直接指定检查点目录时，固定种子仍生成完全一致的报告。

    Args:
        fake_validation: 具有确定性假解码器的模型目录。
    """
    reference, candidate, _models, _configuration = fake_validation
    first = q8_validate.validate_q8(reference, candidate, "cpu")
    second = q8_validate.validate_q8(reference / "ckpt", candidate, "cpu")
    assert first == second


def test_q8_policy_is_independent_and_pointwise_error_is_only_audited() -> None:
    """独立的量化门限保留逐元素审计，并维持原有两个精度策略。

    Returns:
        无。
    """
    assert q8_validate.Q8_POLICY is not BF16_POLICY
    assert q8_validate.Q8_POLICY["relative_l2_max"] == 0.025
    assert q8_validate.Q8_POLICY["kl_reference_to_actual_max"] == 0.003
    assert q8_validate.Q8_POLICY["max_probability_difference"] == 0.035
    assert FP32_POLICY["relative_l2_max"] == 1e-4
    reference = np.zeros((1, 2049), dtype=np.float32)
    reference[0, 0] = 10
    actual = reference.copy()
    actual[0, 1] = 0.02
    metrics = q8_validate._compare_output(actual, reference, expected_shape=(1, 2049), logits=True, backbone=True)
    assert metrics["passed"] is True
    assert metrics["pointwise_violations"] == 1


@pytest.mark.parametrize("gate", ["relative_l2", "kl_reference_to_actual", "max_probability_difference"])
def test_each_prespecified_hard_gate_rejects_drift(gate: str) -> None:
    """整体漂移、分布漂移和单项概率差分别保留独立拒绝能力。

    Args:
        gate: 当前独立触发的验收约束。
    """
    reference = np.full((1, 2048), 100.0)
    actual = reference.copy()
    if gate == "relative_l2":
        actual += 3.0
    elif gate == "kl_reference_to_actual":
        actual[:, :1024] += 0.08
        actual[:, 1024:] -= 0.08
    else:
        reference[:, 2:] = actual[:, 2:] = 80.0
        actual[:, 0] += 0.065
        actual[:, 1] -= 0.065
    metrics = q8_validate._compare_output(actual, reference, expected_shape=(1, 2048), logits=True, backbone=False)
    assert metrics["passed"] is False
    assert any(gate in message for message in metrics["errors"])
    assert all(name in metrics for name in ("max_abs", "rms", "relative_l2", "top1_agreement", "min_top1_margin"))


def test_backbone_eos_is_included_in_probability_validation() -> None:
    """主干结束类别的概率变化必须参与量化验收。

    Returns:
        无。
    """
    reference = np.full((1, 2052), 80.0)
    reference[0, 0] = reference[0, -1] = 100.0
    actual = reference.copy()
    actual[0, 0] += 0.065
    actual[0, -1] -= 0.065
    metrics = q8_validate._compare_output(actual, reference, expected_shape=(1, 2052), logits=True, backbone=True)
    assert metrics["passed"] is False
    assert metrics["valid_vocab_size"] == 2049
    assert metrics["max_probability_difference"] > 0.035


def test_reserved_columns_are_audited_without_entering_legal_probabilities() -> None:
    """物理词表保留列参与原始误差统计，但不改变合法码本和末列结束概率。

    Returns:
        无。
    """
    reference = np.full((1, 2052), 100.0)
    actual = reference.copy()
    actual[:, 2048:2051] += 1.0
    metrics = q8_validate._compare_output(actual, reference, expected_shape=(1, 2052), logits=True, backbone=True)
    assert metrics["passed"] is True
    assert metrics["max_abs"] == 1.0
    assert metrics["valid_vocab_size"] == 2049
    assert metrics["kl_reference_to_actual"] == metrics["max_probability_difference"] == 0.0


@pytest.mark.parametrize("side", ["actual", "reference"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_outputs_are_rejected_and_report_stays_serializable(side: str, bad: float) -> None:
    """两侧非有限值都明确失败，报告本身保持严格 JSON 可保存。

    Args:
        side: 被破坏的原始或候选数组。
        bad: 注入的非有限值。
    """
    arrays = {"actual": np.ones((1, 4)), "reference": np.ones((1, 4))}
    arrays[side][0, 0] = bad
    metrics = q8_validate._compare_output(**arrays, expected_shape=(1, 4), logits=False, backbone=False)
    assert metrics["passed"] is False
    assert metrics[side + "_finite"] is False
    json.dumps(metrics, allow_nan=False)


def test_identically_wrong_output_shapes_cannot_pass() -> None:
    """两侧相同但违反接口契约的形状同样不能通过。

    Returns:
        无。
    """
    metrics = q8_validate._compare_output(np.ones(4), np.ones(4), expected_shape=(1, 4), logits=False, backbone=False)
    assert metrics["shape_match"] is False
    assert metrics["passed"] is False


@pytest.mark.parametrize("failure", ["drift", "nonfinite", "decode_once", "load_failure"])
def test_failures_preserve_full_case_report_and_release_models(fake_validation: tuple, failure: str) -> None:
    """数值、解码或候选加载失败后仍保存完整计划并释放已加载模型。

    Args:
        fake_validation: 含受控失败配置的假解码器环境。
        failure: 当前注入的失败方式。
    """
    reference, candidate, models, configuration = fake_validation
    configuration[failure] = True
    report = q8_validate.validate_q8(reference, candidate, "cpu")
    assert report["passed"] is False
    assert report["summary"]["cases"] == 57
    assert report["summary"]["failed_cases"] > 0
    assert any(not case["passed"] for case in report["cases"])
    assert all(model.closed for model in models)
    if failure == "drift":
        assert report["cases"][0]["outputs"]["logits"]["relative_l2"] > 0.025
    if failure == "load_failure":
        assert all(case["passed"] for case in report["cases"] if case["component"] == "depth")
    json.dumps(report, allow_nan=False)


def test_missing_asset_cannot_be_reported_as_a_full_success(fake_validation: tuple) -> None:
    """缺少模型文件时保存失败原因，并明确完整套件尚未执行。

    Args:
        fake_validation: 当前隔离的模型目录。
    """
    reference, candidate, models, _configuration = fake_validation
    (candidate / "depth.gguf").unlink()
    report = q8_validate.validate_q8(reference, candidate, "cpu")
    assert report["passed"] is False
    assert report["coverage"]["full_suite"] is False
    assert report["errors"] and not models


@pytest.mark.parametrize("failure", [False, True])
def test_cli_writes_complete_report_before_failure_exit(
    fake_validation: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool,
) -> None:
    """命令先落盘可复核报告，失败时再返回非零退出码。

    Args:
        fake_validation: 当前隔离的模型和假解码器。
        tmp_path: 报告输出目录。
        monkeypatch: 命令行参数替换工具。
        failure: 是否注入数值失败。
    """
    reference, candidate, _models, configuration = fake_validation
    configuration["drift"] = failure
    output = tmp_path / "reports" / "q8.json"
    monkeypatch.setattr(sys, "argv", [
        "q8_validate", "--reference-bundle", str(reference), "--candidate-dir", str(candidate),
        "--backend", "cpu", "--report", str(output),
    ])
    if failure:
        with pytest.raises(SystemExit) as raised:
            q8_validate.main()
        assert raised.value.code == 1
    else:
        q8_validate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is not failure
    assert len(report["cases"]) == 57
