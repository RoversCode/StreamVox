"""用单个后台线程将 Breeze 声码器与下一帧码本生成重叠执行。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Mapping, Sequence

import numpy as np

from .ort import OrtGraph, as_numpy


class StreamingCodec:
    """顺序推进声码器状态，并最多保留一个尚未收割的帧。

    Args:
        decoder: 提供 run 方法的声码器图，生命周期由调用方管理。
        state_specs: 包含 name 与 shape 的完整状态规格。
        executor: 运行时持有的固定单线程执行器，跨请求复用图捕获上下文。
    Returns:
        由调用方单线程按帧顺序使用的异步声码器。
    """

    def __init__(self, decoder: OrtGraph, state_specs: Sequence[Mapping], executor: ThreadPoolExecutor) -> None:
        self.decoder = decoder
        self.state_specs = tuple(
            {"name": item["name"], "shape": tuple(item["shape"])}
            for item in state_specs
        )
        self._executor = executor
        self._pending: Future[np.ndarray] | None = None
        self._states: dict[str, object] | None = None
        self._failure: BaseException | None = None
        self._closed = False

    def push(self, frame: Sequence[int] | np.ndarray, position: int) -> np.ndarray | None:
        """收割上一帧音频，再提交当前帧的独立输入副本。

        Args:
            frame: 当前帧的十六码本编号。
            position: 当前帧在本次请求中的位置。
        Returns:
            上一帧的独立主机音频；首次提交时为空。
        """
        previous = self.finish()
        codes = np.array(frame, dtype=np.int64, order="C", copy=True).reshape(1, 1, 16)
        positions = np.asarray([position], dtype=np.int64)
        self._pending = self._executor.submit(self._run_frame, codes, positions)
        return previous

    def finish(self) -> np.ndarray | None:
        """等待并取走最后提交的帧，将解码异常传递给调用方。

        Returns:
            尚未收割的最后一帧音频；没有待处理帧时为空。
        """
        if self._closed:
            raise RuntimeError("流式声码器已经关闭。")
        if self._failure is not None:
            raise RuntimeError("流式声码器上一帧执行失败，须为新请求创建实例。") from self._failure
        pending = self._pending
        if pending is None:
            return None
        try:
            return pending.result()
        except BaseException as error:
            # 状态可能已被后端部分写入，失败后拒绝继续推进该请求。
            self._failure = error
            raise
        finally:
            # 主线程等待时被中断，后台任务仍可能在写状态，关闭时必须继续等待它。
            if pending.done():
                self._pending = None

    def _run_frame(self, codes: np.ndarray, position: np.ndarray) -> np.ndarray:
        """在唯一后台线程内初始化状态并连续执行一帧。

        Args:
            codes: 本次提交独占的整数码本数组。
            position: 本次提交独占的帧位置数组。
        Returns:
            拥有独立存储的主机单声道浮点音频。
        """
        states = self._states
        if states is None:
            states = {item["name"]: np.zeros(item["shape"], dtype=np.float32)
                      for item in self.state_specs}
        output = self.decoder.run({"codes": codes, "position": position, **states})
        next_states = {item["name"]: output[f"next_{item['name']}"] for item in self.state_specs}
        audio = as_numpy(output["audio"])
        # 固定绑定会在后续帧复用输出地址，返回前必须保留独立的主机副本。
        result = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        self._states = next_states
        return result

    def close(self) -> None:
        """等待本请求的在途任务，丢弃未收割输出且允许重复关闭。"""
        if self._closed:
            return
        self._closed = True
        # 执行器由运行时跨请求复用；取消只等待本请求，避免遮盖原始异常。
        if self._pending is not None:
            wait((self._pending,))
        self._pending = None
        self._states = None
