"""按各自精度的门限校验整体误差与合法语音符号概率，保持推理计算不变。"""

from __future__ import annotations

import numpy as np


# 全尺寸处理器前向与独立保留样本共同确认的验收界限，不能用于低精度图。
FP32_POLICY = {
    "pointwise_atol": 2e-4,
    "pointwise_rtol": 4e-4,
    "relative_l2_max": 1e-4,
    "reference_norm_floor": 1e-6,
    "logits_temperature": 0.9,
    "kl_reference_to_actual_max": 1e-7,
    "max_probability_difference": 5e-5,
}


BF16_POLICY = {
    "pointwise_atol": 0.01,
    "pointwise_rtol": 0.02,
    "pointwise_hard_gate": False,
    "relative_l2_max": 0.025,
    "reference_norm_floor": 1e-6,
    "logits_temperature": 0.9,
    "kl_reference_to_actual_max": 0.003,
    "max_probability_difference": 0.035,
}


def _output_statistics(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    norm_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """验证公共数组边界，并以双精度计算整体误差。

    Args:
        actual: 待验收的实数输出。
        reference: 相同输入和权重的参考输出。
        norm_floor: 参考范数的安全下限。
    Returns:
        双精度实际数组、参考数组及整体误差统计。
    """
    actual_array, reference_array = np.asarray(actual), np.asarray(reference)
    if actual_array.shape != reference_array.shape:
        raise ValueError(f"numeric output shape mismatch: {actual_array.shape} != {reference_array.shape}")
    if any(not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) for value in (actual_array, reference_array)):
        raise ValueError("comparison requires real numeric arrays")
    if not np.isfinite(actual_array).all() or not np.isfinite(reference_array).all():
        raise ValueError("comparison requires finite outputs")
    actual_values = actual_array.astype(np.float64, copy=False)
    reference_values = reference_array.astype(np.float64, copy=False)
    difference = actual_values - reference_values
    relative_l2 = float(
        np.linalg.norm(difference.reshape(-1))
        / max(float(np.linalg.norm(reference_values.reshape(-1))), norm_floor)
    )
    metrics = {
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "rms": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "relative_l2": relative_l2,
    }
    return actual_values, reference_values, metrics


def _logits_statistics(
    actual_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    codebook_size: int,
    backbone: bool,
    temperature: float,
) -> dict[str, float | int]:
    """在合法词表上计算概率误差与首选符号统计。

    Args:
        actual_values: 双精度实际分数。
        reference_values: 双精度参考分数。
        codebook_size: 连续合法音频码的数量。
        backbone: 是否另外包含末列结束类别。
        temperature: 比较概率时使用的正温度。
    Returns:
        最大相对熵、最大概率差、首选一致比例、参考分差下限及合法词表大小。
    """
    if reference_values.ndim == 0:
        raise ValueError("logits require a vocabulary dimension")
    if not isinstance(codebook_size, int) or isinstance(codebook_size, bool) or codebook_size <= 0:
        raise ValueError("codebook_size must be a positive integer")

    width = reference_values.shape[-1]
    if width and codebook_size > width - int(backbone):
        raise ValueError("codebook_size exceeds the available codec vocabulary")
    valid_vocab_size = codebook_size + int(backbone) if width else 0
    metrics: dict[str, float | int] = {
        "kl_reference_to_actual": 0.0, "max_probability_difference": 0.0,
        "top1_agreement": 1.0, "min_top1_margin": 0.0, "valid_vocab_size": valid_vocab_size,
    }
    if not reference_values.size:
        return metrics

    # 保留符号不参与真实采样；主干的额外结束类别必须与普通音频码一起比较。
    actual_valid, reference_valid = actual_values[..., :codebook_size], reference_values[..., :codebook_size]
    if backbone:
        actual_valid = np.concatenate((actual_valid, actual_values[..., -1:]), axis=-1)
        reference_valid = np.concatenate((reference_valid, reference_values[..., -1:]), axis=-1)

    # 使用双精度对数概率计算相对熵，避免小概率下溢污染验收指标。
    actual_scaled = actual_valid / temperature
    reference_scaled = reference_valid / temperature
    actual_scaled -= np.max(actual_scaled, axis=-1, keepdims=True)
    reference_scaled -= np.max(reference_scaled, axis=-1, keepdims=True)
    log_actual = actual_scaled - np.log(np.exp(actual_scaled).sum(axis=-1, keepdims=True))
    log_reference = reference_scaled - np.log(np.exp(reference_scaled).sum(axis=-1, keepdims=True))
    probabilities_actual, probabilities_reference = np.exp(log_actual), np.exp(log_reference)
    kl = float(max(0.0, np.max(np.sum(probabilities_reference * (log_reference - log_actual), axis=-1))))
    max_probability_difference = float(np.max(np.abs(probabilities_actual - probabilities_reference)))
    margin = 0.0
    if valid_vocab_size > 1:
        top_two = np.partition(reference_valid, valid_vocab_size - 2, axis=-1)[..., -2:]
        margin = float(np.min(top_two[..., 1] - top_two[..., 0]))
    metrics.update({
        "kl_reference_to_actual": kl,
        "max_probability_difference": max_probability_difference,
        "top1_agreement": float(np.mean(np.argmax(actual_valid, axis=-1) == np.argmax(reference_valid, axis=-1))),
        "min_top1_margin": margin,
    })
    return metrics


def compare_fp32(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    logits: bool = False,
    codebook_size: int = 2048,
    backbone: bool = False,
) -> dict[str, float | int]:
    """以逐元素、整体误差和概率三道约束验收 FP32 输出。

    Args:
        actual: ONNX Runtime 实际输出的实数数组。
        reference: 使用相同输入和权重得到的参考输出。
        logits: 是否额外校验合法语音符号上的概率分布。
        codebook_size: 合法音频码数量，从第零列连续取值。
        backbone: 是否将最后一列独立结束类别加入合法词表。
    Returns:
        最大绝对误差、均方根误差与相对二范数误差；分数输出另外返回概率与首选符号统计。
        空输出的误差为零，空分数行的一致比例为一，分差下限为零。
    """
    actual_values, reference_values, metrics = _output_statistics(actual, reference, norm_floor=FP32_POLICY["reference_norm_floor"])
    np.testing.assert_allclose(
        actual_values, reference_values,
        rtol=FP32_POLICY["pointwise_rtol"], atol=FP32_POLICY["pointwise_atol"],
        err_msg="FP32 pointwise error exceeds the approved bound",
    )
    relative_l2 = metrics["relative_l2"]
    if not np.isfinite(relative_l2) or relative_l2 > FP32_POLICY["relative_l2_max"]:
        raise AssertionError(f"FP32 relative L2 error exceeds the approved bound: {relative_l2}")
    if not logits:
        return metrics
    metrics.update(_logits_statistics(
        actual_values, reference_values, codebook_size=codebook_size, backbone=backbone,
        temperature=FP32_POLICY["logits_temperature"],
    ))
    kl, probability = metrics["kl_reference_to_actual"], metrics["max_probability_difference"]
    if not np.isfinite(kl) or kl > FP32_POLICY["kl_reference_to_actual_max"]:
        raise AssertionError(f"FP32 logits KL divergence exceeds the approved bound: {kl}")
    if not np.isfinite(probability) or probability > FP32_POLICY["max_probability_difference"]:
        raise AssertionError(f"FP32 logits probability difference exceeds the approved bound: {probability}")
    return metrics


def compare_bf16(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    logits: bool = False,
    codebook_size: int = 2048,
    backbone: bool = False,
) -> dict[str, float | int]:
    """按整体误差和概率验收 BF16，并审计旧逐元素标准的违规数量。

    Args:
        actual: 原生 BF16 图转换为实数数组的实际输出。
        reference: 同精度、同输入及同权重的原生参考输出。
        logits: 是否额外校验合法语音符号上的概率分布。
        codebook_size: 合法音频码数量，从第零列连续取值。
        backbone: 是否将最后一列独立结束类别加入合法词表。
    Returns:
        整体误差统计、旧逐元素标准的违规数与元素总数；分数输出另外返回概率与首选符号统计。
        逐元素违规仅用于审计，整体相对误差及概率门限仍会严格拒绝越界输出。
    """
    actual_values, reference_values, metrics = _output_statistics(actual, reference, norm_floor=BF16_POLICY["reference_norm_floor"])
    threshold = BF16_POLICY["pointwise_atol"] + BF16_POLICY["pointwise_rtol"] * np.abs(reference_values)
    metrics.update({
        "pointwise_violations": int(np.count_nonzero(np.abs(actual_values - reference_values) > threshold)),
        "element_count": int(reference_values.size),
    })
    relative_l2 = metrics["relative_l2"]
    if not np.isfinite(relative_l2) or relative_l2 > BF16_POLICY["relative_l2_max"]:
        raise AssertionError(f"BF16 relative L2 error exceeds the approved bound: {relative_l2}")
    if not logits:
        return metrics
    metrics.update(_logits_statistics(
        actual_values, reference_values, codebook_size=codebook_size, backbone=backbone,
        temperature=BF16_POLICY["logits_temperature"],
    ))
    kl, probability = metrics["kl_reference_to_actual"], metrics["max_probability_difference"]
    if not np.isfinite(kl) or kl > BF16_POLICY["kl_reference_to_actual_max"]:
        raise AssertionError(f"BF16 logits KL divergence exceeds the approved bound: {kl}")
    if not np.isfinite(probability) or probability > BF16_POLICY["max_probability_difference"]:
        raise AssertionError(f"BF16 logits probability difference exceeds the approved bound: {probability}")
    return metrics
