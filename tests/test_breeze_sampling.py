"""验证 Breeze 官方采样分布、条件引导和贪心边界。"""

from __future__ import annotations

import numpy as np
import pytest

from streamvox.models.breeze.sampling import guided_logits, sample_token, sampling_probabilities


@pytest.mark.parametrize("scale,expected", [
    (0.0, [3.0, -1.0, 2.0]), (1.0, [1.0, 5.0, -2.0]), (4.0, [-5.0, 23.0, -14.0]),
])
def test_cfg_uses_official_unconditional_plus_scaled_difference(scale, expected):
    """零、一和增强引导分别对应无条件、条件和外推分数。"""
    positive = np.asarray([1, 5, -2], dtype=np.float32)
    negative = np.asarray([3, -1, 2], dtype=np.float32)
    np.testing.assert_array_equal(guided_logits(positive, negative, scale), expected)
    np.testing.assert_array_equal(guided_logits(positive, None, scale), positive)


def test_standard_top_p_removes_low_probability_complement():
    """普通采样按固定版本规则，从小概率端剔除累计不超过补集的项。"""
    probabilities = sampling_probabilities(
        np.log(np.asarray([0.55, 0.25, 0.15, 0.05], dtype=np.float32)),
        temperature=1, top_k=0, top_p=0.6,
    )
    np.testing.assert_allclose(probabilities, [0.6875, 0.3125, 0, 0], atol=1e-7)


def test_standard_top_p_removes_exact_complement_boundary():
    """普通采样在补集累计恰好等于阈值时仍淘汰该项。"""
    probabilities = sampling_probabilities(np.asarray([0, 0]), temperature=1, top_k=0, top_p=0.5)
    np.testing.assert_array_equal(probabilities, [0, 1])


def test_cfg_depth_top_p_preserves_exact_descending_boundary():
    """条件引导深度路径在降序累计恰好等于阈值时继续保留下一项。"""
    probabilities = sampling_probabilities(np.asarray([0, 0]), temperature=1, top_k=0, top_p=0.5, cfg_depth=True)
    np.testing.assert_array_equal(probabilities, [0.5, 0.5])


def test_top_k_preserves_all_ties_at_the_kth_score():
    """相同阈值分数的符号不能被任意裁掉。"""
    probabilities = sampling_probabilities(np.asarray([4, 3, 3, 0]), temperature=1, top_k=2, top_p=1)
    assert np.flatnonzero(probabilities).tolist() == [0, 1, 2]
    assert probabilities[1] == probabilities[2]


def test_cfg_depth_top_k_keeps_exact_count_instead_of_all_ties():
    """条件引导深度采样按概率选取精确数量，区别于普通阈值筛选。"""
    probabilities = sampling_probabilities(
        np.asarray([4, 3, 3, 0]), temperature=1, top_k=2, top_p=1, cfg_depth=True,
    )
    assert np.count_nonzero(probabilities) == 2
    assert probabilities[0] > 0
    expected = np.exp(np.asarray([4, 3], dtype=np.float32) - 4)
    expected /= expected.sum()
    np.testing.assert_allclose(np.sort(probabilities[probabilities > 0])[::-1], expected)


def test_cfg_depth_top_p_keeps_first_probability_crossing_after_top_k():
    """条件引导深度层先重新归一化前若干项，再按累计概率筛选。"""
    probabilities = sampling_probabilities(
        np.log(np.asarray([0.55, 0.25, 0.15, 0.05], dtype=np.float32)),
        temperature=1, top_k=3, top_p=0.6, cfg_depth=True,
    )
    np.testing.assert_allclose(probabilities, [0.6875, 0.3125, 0, 0], atol=1e-7)


@pytest.mark.parametrize("scenario", ["random", "ties", "underflow"])
@pytest.mark.parametrize("top_k", [0, 1, 2, 50, 2048, 2051])
@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.5, 1e-6])
def test_cfg_depth_distribution_matches_stable_sort(scenario, top_k, top_p):
    """分区筛选须逐位复现稳定排序的并列、下溢与累积概率边界。

    Args:
        scenario: 随机分数、密集并列或指数下溢的输入场景。
        top_k: 前若干项筛选的保留数量。
        top_p: 累积概率筛选的阈值。
    Returns:
        所有概率逐位相等时正常返回。
    """
    rng = np.random.default_rng(71)
    scenarios = {
        "random": rng.normal(0, 3, 2051).astype(np.float32),
        "ties": rng.integers(-3, 4, 2051).astype(np.float32),
        "underflow": rng.choice(np.asarray([0, -100, -104, -120, -1000], dtype=np.float32), 2051),
    }
    logits = scenarios[scenario]
    suppressed = (2048, 2049, 2050)
    scores = logits.copy()
    scores[list(suppressed)] = -np.inf
    scores /= 0.9
    expected = np.exp(scores - scores.max())
    expected /= expected.sum()
    if 0 < top_k < scores.size:
        keep = np.argsort(-expected, kind="stable")[:top_k]
        filtered = np.zeros_like(expected)
        filtered[keep] = expected[keep]
        expected = filtered / filtered.sum()
    if top_p < 1.0:
        order = np.argsort(-expected, kind="stable")
        cumulative = np.cumsum(expected[order])
        remove = np.concatenate(([False], cumulative[:-1] > top_p))
        expected[order[remove]] = 0.0
        expected /= expected.sum()
    actual = sampling_probabilities(
        logits, temperature=0.9, top_k=top_k, top_p=top_p, suppressed=suppressed, cfg_depth=True,
    )
    np.testing.assert_array_equal(actual, expected)


def test_repetition_penalty_uses_unique_history_and_original_logit_sign():
    """同一历史编号只惩罚一次，正负分数按官方公式分别处理。"""
    scores = np.asarray([4, -2, 1], dtype=np.float32)
    probabilities = sampling_probabilities(
        scores, temperature=1, top_k=0, top_p=1, history=[0, 0, 1], repetition_penalty=2,
    )
    expected = np.exp(np.asarray([2, -4, 1], dtype=np.float32) - 2)
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_array_equal(scores, [4, -2, 1])


def test_greedy_sampling_keeps_penalty_and_suppression_without_temperature_filters():
    """贪心路径仍应用重复惩罚与保留符号屏蔽。"""
    token = sample_token(
        np.asarray([5, 4, 100], dtype=np.float32), np.random.default_rng(3), do_sample=False,
        temperature=0.001, top_k=1, top_p=0.01, history=[0], repetition_penalty=2, suppressed=(2,),
    )
    assert token == 1


def test_greedy_argmax_does_not_lose_small_score_differences_in_softmax():
    """贪心直接比较分数，避免概率舍入把不同分数误变成平局。"""
    token = sample_token(
        np.asarray([0, 1e-8], dtype=np.float32), np.random.default_rng(3), do_sample=False,
        temperature=1, top_k=0, top_p=1,
    )
    assert token == 1
