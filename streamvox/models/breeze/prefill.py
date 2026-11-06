"""复用参考前缀的原生缓存，并隔离每个请求的目标后缀。"""

from __future__ import annotations

import hashlib

import numpy as np

from .llama import LlamaDecoder


class BackbonePrefill:
    """管理单分支参考前缀缓存及所有请求的收尾。

    Args:
        decoder: 由运行时独占的原生主干解码器。
    Returns:
        只驻留最近一个参考前缀的缓存管理器。
    """

    def __init__(self, decoder: LlamaDecoder) -> None:
        self.decoder = decoder
        self._reference: tuple[int, bytes] | None = None
        self._request_reference: tuple[int, bytes] | None = None
        self._active = False

    def reserve(self, n_ctx: int) -> None:
        """扩容原生上下文，并让旧上下文的前缀记录失效。

        Args:
            n_ctx: 本次请求需要的每分支缓存容量。
        Returns:
            原生容量准备完成后返回。
        """
        if n_ctx > self.decoder.n_ctx:
            self._reference = None
        self.decoder.reserve(n_ctx)

    def _fingerprint(self, reference: np.ndarray) -> tuple[int, bytes] | None:
        """计算实际驻留前缀的长度与统一浮点内容摘要。

        Args:
            reference: 单分支参考嵌入，形状为批次、长度、隐藏维度。
        Returns:
            对齐后的前缀长度与内容摘要；没有可缓存位置时返回空值。
        """
        if reference.ndim != 3 or reference.shape[0] != 1 or reference.shape[2] == 0:
            raise ValueError("主干参考嵌入必须为单分支且隐藏维度非空的三维数组。")
        # 保留原有两词微批次边界，奇数参考的末词随目标一起重新计算。
        alignment = 2 if self.decoder.backend == "cuda" else 1
        length = reference.shape[1] - reference.shape[1] % alignment
        if length == 0:
            return None
        prefix = np.ascontiguousarray(reference[:, :length], dtype=np.float32)
        return length, hashlib.sha256(prefix.tobytes()).digest()

    def warm(self, reference: np.ndarray) -> None:
        """在目标请求开始前预填充参考，复用已经驻留的相同前缀。

        Args:
            reference: 单分支完整参考嵌入，容量已由调用方准备。
        Returns:
            可缓存前缀预填充完成后返回；空前缀会清空现有缓存。
        """
        if self._active:
            raise RuntimeError("主干上一请求尚未完成缓存收尾。")
        reference = np.asarray(reference, dtype=np.float32)
        fingerprint = self._fingerprint(reference)
        if fingerprint is not None and fingerprint == self._reference:
            return
        self._reference = None
        self.decoder.reset()
        if fingerprint is None:
            return
        try:
            self.decoder.decode(
                embeddings=reference[:, :fingerprint[0]], positions=[0], sequence_ids=[0], output=False,
            )
        except BaseException:
            self.decoder.reset()
            raise
        self._reference = fingerprint

    def decode(self, prompts: list[np.ndarray], reference_length: int) -> list[dict]:
        """单分支复用参考，其他请求执行完整预填充。

        Args:
            prompts: 各分支完整的参考与目标嵌入。
            reference_length: 所有分支共有的参考长度，零表示没有参考。
        Returns:
            各分支最后位置的原生分数与隐藏状态。
        """
        if self._active:
            raise RuntimeError("主干上一请求尚未完成缓存收尾。")
        if not prompts or not 0 <= reference_length < min(value.shape[1] for value in prompts):
            raise ValueError("主干参考边界必须在非空目标之前。")
        reference = None
        # 多分支共享会改变长参考的物理归约布局，仅缓存已通过数值验收的单分支。
        if len(prompts) == 1:
            reference = self._fingerprint(prompts[0][:, :reference_length])
        self._request_reference = reference
        self._active = True
        reused = reference is not None and reference == self._reference
        if not reused:
            self._reference = None
            self.decoder.reset()
        start = reference[0] if reused else 0
        return [self.decoder.decode(
            embeddings=value[:, start:], positions=[start], sequence_ids=[sequence],
        ) for sequence, value in enumerate(prompts)]

    def finish(self, succeeded: bool) -> None:
        """正常结束只保留参考，取消或异常则清空全部分支。

        Args:
            succeeded: 是否已完整完成请求并输出全部音频。
        Returns:
            本次目标后缀和不再使用的分支清理完成后返回。
        """
        if not self._active:
            return
        self._active = False
        reference, self._request_reference = self._request_reference, None
        if not succeeded or reference is None:
            self._reference = None
            self.decoder.reset()
            return
        try:
            self.decoder.remove_sequence(0, reference[0])
        except BaseException:
            self._reference = None
            self.decoder.reset()
            raise
        self._reference = reference
