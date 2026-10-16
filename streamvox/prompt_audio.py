from __future__ import annotations

import hashlib

import numpy as np

from .models.base import PromptReference


def build_prompt_reference(
    *,
    audio: np.ndarray,
    sample_rate: int,
    index: int,
) -> PromptReference:
    """
    构造当前模型目标采样率下的 prompt 参考音频。

    参数:
        audio: 已由模型 adapter 读取并重采样后的音频数组。
        sample_rate: 当前模型实际消费的采样率。
        index: 多参考音频中的位置序号。

    返回:
        返回只包含音频、哈希和序号的 PromptReference。
    """
    normalized = normalize_prompt_audio(audio)
    return PromptReference(
        audio=normalized,
        audio_sha256=compute_prompt_audio_sha256(normalized, sample_rate),
        index=int(index),
    )


def normalize_prompt_audio(audio: np.ndarray) -> np.ndarray:
    """
    将 prompt 音频规整为一维 float32 单声道数组。

    参数:
        audio: 模型 adapter 产出的音频数组。

    返回:
        返回连续内存布局的一维 float32 音频。
    """
    array = np.asarray(audio, dtype=np.float32)
    # prompt 资料层只保存模型可直接消费的一维波形，避免不同读取库的多声道布局泄漏到公开 API。
    if array.ndim == 2:
        # 常见音频库会分别产出 channel/time 或 time/channel，这里按较短且不超过 8 的维度识别声道轴。
        if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
            array = np.mean(array, axis=0, dtype=np.float32)
        elif array.shape[1] <= 8 and array.shape[1] < array.shape[0]:
            array = np.mean(array, axis=1, dtype=np.float32)
        else:
            array = np.mean(array, axis=1, dtype=np.float32)
    elif array.ndim != 1:
        array = array.reshape(-1)
    return np.ascontiguousarray(array, dtype=np.float32)


def compute_prompt_audio_sha256(audio: np.ndarray, sample_rate: int) -> str:
    """
    计算目标采样率参考音频的稳定哈希。

    参数:
        audio: 已规整的一维 float32 音频。
        sample_rate: 当前模型实际消费的采样率。

    返回:
        返回十六进制 sha256 字符串。
    """
    normalized = np.ascontiguousarray(audio, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(int(sample_rate)).encode("ascii"))
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()
