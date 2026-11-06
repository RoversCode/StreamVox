"""验证 FP32 验收的整体误差、概率约束和合法语音词表。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.breeze_reference.numerics import BF16_POLICY, FP32_POLICY, compare_bf16, compare_fp32


def test_full_scale_near_zero_difference_passes_with_small_global_and_probability_error():
    """真实全尺寸近零分数的微小舍入差异可通过明确的多重约束。"""
    reference = np.asarray([[10.0, 0.02800273895263672, -3.0]], dtype=np.float32)
    actual = reference.copy()
    actual[0, 1] = 0.027951225638389587
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, reference, rtol=4e-4, atol=4e-5)
    metrics = compare_fp32(actual, reference, logits=True, codebook_size=3)
    assert metrics["max_abs"] == pytest.approx(5.1513314e-5)
    assert metrics["relative_l2"] < 1e-4
    assert metrics["kl_reference_to_actual"] < 1e-7
    assert metrics["top1_agreement"] == 1.0


def test_relative_l2_rejects_global_drift_despite_pointwise_pass():
    """逐元素均在容差内的整体偏移仍必须被整体误差约束拦截。"""
    reference = np.ones(128)
    actual = reference + 1.5e-4
    np.testing.assert_allclose(actual, reference, rtol=4e-4, atol=2e-4)
    with pytest.raises(AssertionError, match="relative L2"):
        compare_fp32(actual, reference)


def test_pointwise_guard_rejects_local_error_hidden_by_large_reference_norm():
    """大范数不能掩盖单个低幅度输出上的明显偏差。"""
    reference = np.asarray([1000.0, 0.0])
    actual = np.asarray([1000.0, 3e-4])
    with pytest.raises(AssertionError, match="pointwise"):
        compare_fp32(actual, reference)


def test_probability_guard_rejects_changed_sampling_with_small_raw_relative_error():
    """共同的大分数偏置不能掩盖两个高概率符号之间的概率变化。"""
    reference = np.asarray([[100.0, 100.0]])
    actual = np.asarray([[100.00015, 99.99985]])
    compare_fp32(actual, reference)
    with pytest.raises(AssertionError, match="probability difference"):
        compare_fp32(actual, reference, logits=True, codebook_size=2)


def test_kl_guard_rejects_distributed_probability_drift():
    """大量小概率项的集体偏移也必须受相对熵约束。"""
    reference = np.full((1, 1024), 100.0)
    actual = reference.copy()
    actual[:, :512] += 0.001
    actual[:, 512:] -= 0.001
    compare_fp32(actual, reference)
    with pytest.raises(AssertionError, match="KL divergence"):
        compare_fp32(actual, reference, logits=True, codebook_size=1024)


def test_backbone_vocabulary_includes_extra_eos_and_excludes_reserved_columns():
    """主干结束类别参与概率验收，保留的协议符号不参与采样。"""
    reference = np.full((1, 5), 100.0)
    actual = reference.copy()
    actual[0, -1] += 0.001
    depth = compare_fp32(actual, reference, logits=True, codebook_size=2)
    assert depth["valid_vocab_size"] == 2
    assert depth["max_probability_difference"] == depth["kl_reference_to_actual"] == 0.0
    with pytest.raises(AssertionError, match="KL divergence|probability difference"):
        compare_fp32(actual, reference, logits=True, codebook_size=2, backbone=True)


def test_reserved_column_changes_do_not_change_probability_metrics():
    """保留符号仍检查原始数值，但不会影响合法符号概率。"""
    reference = np.full((2, 5), 100.0)
    actual = reference.copy()
    actual[:, 2] += 0.001
    actual[:, 3] -= 0.001
    metrics = compare_fp32(actual, reference, logits=True, codebook_size=2, backbone=True)
    assert metrics["valid_vocab_size"] == 3
    assert metrics["max_probability_difference"] == 0.0
    assert metrics["kl_reference_to_actual"] == 0.0


def test_top1_change_at_near_tie_is_recorded_without_hard_failure():
    """近似平局可以交换首选项，但必须记录一致比例和参考分差。"""
    reference = np.asarray([[1.0, 1.0000001], [1.0, 1.2]])
    actual = reference.copy()
    actual[0] = reference[0, ::-1]
    metrics = compare_fp32(actual, reference, logits=True, codebook_size=2)
    assert metrics["top1_agreement"] == 0.5
    assert metrics["min_top1_margin"] == pytest.approx(1e-7)


def test_eos_contributes_to_reference_top1_margin():
    """结束类别与第二名音频码的分差用于报告，而非协议保留列。"""
    reference = np.asarray([[1.0, 2.0, 100.0, 200.0, 2.25]])
    metrics = compare_fp32(reference, reference, logits=True, codebook_size=2, backbone=True)
    assert metrics["valid_vocab_size"] == 3
    assert metrics["min_top1_margin"] == 0.25


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["actual", "reference"])
def test_non_finite_values_fail_before_numerical_comparison(bad, side):
    """两侧非有限值都不能被相等判断或概率归一化掩盖。"""
    actual, reference = np.asarray([1.0]), np.asarray([1.0])
    (actual if side == "actual" else reference)[0] = bad
    with pytest.raises(ValueError, match="finite"):
        compare_fp32(actual, reference)


def test_shape_mismatch_cannot_broadcast_into_a_comparison():
    """输出布局必须完全相同，禁止通过广播伪装成数值一致。"""
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_fp32(np.zeros((1, 2)), np.zeros((2,)))


@pytest.mark.parametrize("shape", [(0,), (0, 2), (2, 0, 3)])
def test_empty_outputs_have_zero_error_without_dividing_by_zero(shape):
    """动态空输出不会触发最大值归约错误或零范数除法。"""
    metrics = compare_fp32(np.empty(shape), np.empty(shape))
    assert metrics == {"max_abs": 0.0, "rms": 0.0, "relative_l2": 0.0}


def test_empty_logits_keep_json_safe_probability_metrics():
    """无批次分数仍保留词表契约，并返回可序列化的空样本指标。"""
    metrics = compare_fp32(np.empty((0, 3)), np.empty((0, 3)), logits=True, codebook_size=2, backbone=True)
    assert metrics["valid_vocab_size"] == 3
    assert metrics["top1_agreement"] == 1.0
    assert metrics["min_top1_margin"] == metrics["kl_reference_to_actual"] == 0.0
    json.dumps(metrics, allow_nan=False)


def test_zero_reference_norm_has_a_finite_floor():
    """参考值全部为零时仍能区分可忽略误差与相对偏移。"""
    assert compare_fp32(np.asarray([5e-11]), np.asarray([0.0]))["relative_l2"] == pytest.approx(5e-5)
    with pytest.raises(AssertionError, match="relative L2"):
        compare_fp32(np.asarray([2e-10]), np.asarray([0.0]))


@pytest.mark.parametrize("size,backbone", [(0, False), (-1, False), (3, False), (2, True)])
def test_invalid_codec_vocabulary_is_rejected(size, backbone):
    """合法码本范围和额外结束类别必须适配输出列数。"""
    with pytest.raises(ValueError, match="codebook_size"):
        compare_fp32(np.zeros((1, 2)), np.zeros((1, 2)), logits=True, codebook_size=size, backbone=backbone)


def test_policy_and_metrics_are_plain_serializable_data():
    """报告可以直接保存阈值与结果，无需张量框架或自定义序列化。"""
    metrics = compare_fp32(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
    assert metrics == {"max_abs": 0.0, "rms": 0.0, "relative_l2": 0.0}
    json.dumps({"policy": FP32_POLICY, "metrics": metrics}, allow_nan=False)


def test_bf16_audits_near_zero_pointwise_violation_under_global_probability_limits():
    """近零项可违反旧逐元素门限，但必须保留审计并通过整体和概率验收。"""
    reference = np.asarray([[10.0, 0.001]], dtype=np.float32)
    actual = np.asarray([[10.0, 0.02]], dtype=np.float32)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, reference, rtol=0.02, atol=0.01)
    metrics = compare_bf16(actual, reference, logits=True, codebook_size=2)
    assert metrics["pointwise_violations"] == 1
    assert metrics["element_count"] == 2
    assert metrics["max_abs"] == pytest.approx(0.019)
    assert metrics["relative_l2"] < 0.025
    assert metrics["kl_reference_to_actual"] < 0.003
    assert metrics["max_probability_difference"] < 0.035
    assert metrics["top1_agreement"] == 1.0
    json.dumps({"policy": BF16_POLICY, "metrics": metrics}, allow_nan=False)


def test_bf16_global_error_rejects_shift_even_when_old_pointwise_check_passes():
    """逐元素仍在旧容差内的整体漂移不能通过新验收。"""
    reference = np.ones(128)
    actual = reference + 0.026
    np.testing.assert_allclose(actual, reference, rtol=0.02, atol=0.01)
    with pytest.raises(AssertionError, match="BF16 relative L2"):
        compare_bf16(actual, reference)


def test_bf16_kl_limit_rejects_distributed_probability_shift():
    """整体分数误差很小也不能放过大量符号上的集体概率偏移。"""
    reference = np.full((1, 1024), 100.0)
    actual = reference.copy()
    actual[:, :512] += 0.08
    actual[:, 512:] -= 0.08
    compare_bf16(actual, reference)
    with pytest.raises(AssertionError, match="BF16 logits KL"):
        compare_bf16(actual, reference, logits=True, codebook_size=1024)


def test_bf16_probability_limit_rejects_large_change_with_acceptable_kl():
    """高概率符号的单项变化还必须单独满足最大概率差门限。"""
    reference = np.asarray([[100.0, 100.0]])
    actual = np.asarray([[100.065, 99.935]])
    compare_bf16(actual, reference)
    with pytest.raises(AssertionError, match="BF16 logits probability difference"):
        compare_bf16(actual, reference, logits=True, codebook_size=2)


def test_bf16_policy_does_not_widen_fp32_acceptance():
    """原生低精度的验收范围不能被误用于基准图。"""
    reference = np.asarray([10.0, 0.001])
    actual = np.asarray([10.0, 0.02])
    compare_bf16(actual, reference)
    with pytest.raises(AssertionError, match="FP32 pointwise"):
        compare_fp32(actual, reference)
    assert FP32_POLICY["pointwise_atol"] == 2e-4
    assert FP32_POLICY["relative_l2_max"] == 1e-4


def test_bf16_records_top1_change_without_an_extra_hard_gate():
    """允许近似平局交换首选项，同时完整记录参考分差与一致比例。"""
    reference = np.asarray([[1.0, 1.01]])
    actual = np.asarray([[1.01, 1.0]])
    metrics = compare_bf16(actual, reference, logits=True, codebook_size=2)
    assert metrics["top1_agreement"] == 0.0
    assert metrics["min_top1_margin"] == pytest.approx(0.01)


@pytest.mark.parametrize("actual,reference,kwargs,message", [
    (np.asarray([np.nan]), np.asarray([1.0]), {}, "finite"),
    (np.asarray([1.0]), np.asarray([np.inf]), {}, "finite"),
    (np.zeros((1, 2)), np.zeros(2), {}, "shape mismatch"),
    (np.zeros((1, 2)), np.zeros((1, 2)), {"logits": True, "codebook_size": 0}, "codebook_size"),
    (np.zeros((1, 2)), np.zeros((1, 2)), {"logits": True, "codebook_size": 3}, "codebook_size"),
    (np.zeros((1, 2)), np.zeros((1, 2)), {"logits": True, "codebook_size": 2, "backbone": True}, "codebook_size"),
])
def test_bf16_rejects_invalid_arrays_and_vocabularies(actual, reference, kwargs, message):
    """低精度验收同样严格检查形状、有限值与合法词表边界。"""
    with pytest.raises(ValueError, match=message):
        compare_bf16(actual, reference, **kwargs)


def test_bf16_empty_logits_return_zero_audit_and_json_safe_metrics():
    """空分数输出不会产生无效归约或无法保存的审计值。"""
    metrics = compare_bf16(np.empty((0, 3)), np.empty((0, 3)), logits=True, codebook_size=2, backbone=True)
    assert metrics["pointwise_violations"] == metrics["element_count"] == 0
    assert metrics["relative_l2"] == metrics["kl_reference_to_actual"] == 0.0
    assert metrics["top1_agreement"] == 1.0
    assert metrics["valid_vocab_size"] == 3
    json.dumps(metrics, allow_nan=False)
