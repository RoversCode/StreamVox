"""保留 Breeze 官方筛选顺序的采样运算。"""

from __future__ import annotations

import numpy as np


def sampling_probabilities(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    history: list[int] | None = None,
    repetition_penalty: float = 1.0,
    suppressed: tuple[int, ...] = (),
    cfg_depth: bool = False,
) -> np.ndarray:
    """按重复惩罚、屏蔽、温度、前若干项、累积概率的顺序计算分布。

    Args:
        logits: 一个码本的原始分数。
        temperature: 正温度。
        top_k: 保留项数；零表示关闭。
        top_p: 保留的累积概率。
        history: 仅包含已经生成的第一码本历史。
        repetition_penalty: 官方按符号处理的重复惩罚系数。
        suppressed: 本层禁止生成的保留符号。
        cfg_depth: 是否使用官方条件引导深度分支的概率筛选顺序。
    Returns:
        与输入等长的概率数组。
    """
    scores = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    if history and repetition_penalty != 1.0:
        indices = np.unique(history)
        previous = scores[indices]
        scores[indices] = np.where(previous > 0, previous / repetition_penalty, previous * repetition_penalty)
    scores[list(suppressed)] = -np.inf
    scores /= temperature
    if cfg_depth:
        # 官方条件引导深度路径先计算概率，再精确保留若干项并重新归一化。
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        if top_k > 0 and top_k < scores.size:
            threshold = np.partition(probabilities, probabilities.size - top_k)[-top_k]
            keep = probabilities > threshold
            # 边界并列按原编号补齐，保持稳定降序排序选中的精确集合。
            tied = np.flatnonzero(probabilities == threshold)
            keep[tied[:top_k - np.count_nonzero(keep)]] = True
            filtered = np.zeros_like(probabilities)
            filtered[keep] = probabilities[keep]
            probabilities = filtered / filtered.sum()
        if top_p < 1.0:
            order = np.argsort(-probabilities, kind="stable")
            cumulative = np.cumsum(probabilities[order])
            remove = np.concatenate(([False], cumulative[:-1] > top_p))
            probabilities[order[remove]] = 0.0
            probabilities /= probabilities.sum()
        return probabilities
    if top_k > 0 and top_k < scores.size:
        threshold = np.partition(scores, scores.size - top_k)[-top_k]
        scores[scores < threshold] = -np.inf
    if top_p < 1.0:
        order = np.argsort(scores, kind="stable")
        ordered = scores[order]
        weights = np.exp(ordered - ordered.max())
        cumulative = np.cumsum(weights / weights.sum())
        # 普通生成遵循固定 Transformers 版本：从小概率端剔除累计不超过补集的项。
        remove = cumulative <= (1.0 - top_p)
        remove[-1] = False
        scores[order[remove]] = -np.inf
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()


def sample_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    do_sample: bool,
    **parameters: object,
) -> int:
    """从官方概率分布采样，或在关闭采样时选取最大分数。

    Args:
        logits: 一个码本的原始分数。
        rng: 当前请求独占的随机数生成器。
        do_sample: 是否使用随机采样。
        parameters: 分布参数。
    Returns:
        被选中的符号编号。
    """
    if not do_sample:
        # 官方贪心路径只应用重复惩罚和保留符号屏蔽，不应用温度与概率裁剪。
        scores = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
        history = parameters.get("history")
        penalty = parameters.get("repetition_penalty", 1.0)
        if history and penalty != 1.0:
            indices = np.unique(history)
            scores[indices] = np.where(scores[indices] > 0, scores[indices] / penalty, scores[indices] * penalty)
        scores[list(parameters.get("suppressed", ()))] = -np.inf
        if not np.isfinite(scores).any() or np.isnan(scores).any() or np.isposinf(scores).any():
            raise RuntimeError("Breeze 采样分数包含非有限值，请检查模型数值。")
        return int(np.argmax(scores))
    probabilities = sampling_probabilities(logits, **parameters)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Breeze 采样分布包含非有限值，请检查模型数值。")
    # 使用双精度归一化满足 NumPy 对概率和的校验，不改变筛选集合。
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(probabilities.size, p=probabilities))


def guided_logits(positive: np.ndarray, negative: np.ndarray | None, scale: float) -> np.ndarray:
    """按官方单条件引导公式组合两个分支。

    Args:
        positive: 带指令分支的分数。
        negative: 移除指令后的分支分数。
        scale: 条件引导强度。
    Returns:
        合并后的分数。
    """
    if negative is None:
        return positive
    return negative + scale * (positive - negative)
