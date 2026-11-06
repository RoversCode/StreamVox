"""验证异步声码器的帧顺序、独立存储与请求清理。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread, current_thread

import numpy as np
import pytest

from streamvox.models.breeze.streaming import StreamingCodec


STATE_SPECS = [
    {"name": "carry", "shape": [1]},
    {"name": "history", "shape": [1, 2]},
]


@pytest.fixture
def executor():
    """提供请求间共享的单线程执行器，并在测试结束后关闭线程。

    Yields:
        测试中各个请求共同借用的执行器。
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-breeze-codec")
    worker = pool.submit(current_thread).result(timeout=5)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)
        assert not worker.is_alive()


class FakeGraph:
    """在处理器中复用音频缓冲，并记录每帧接收的完整状态。

    Args:
        gate: 执行前等待的事件，用于确定性控制在途帧。
        fail_position: 需要模拟执行失败的位置。
        audio_dtype: 假图音频缓冲的浮点类型。
    """

    def __init__(self, *, gate=None, fail_position=None, audio_dtype=np.float32):
        self.gate = gate
        self.fail_position = fail_position
        self.started = Event()
        self.finished = Event()
        self.audio = np.zeros((1, 3), dtype=audio_dtype)
        self.calls = []
        self.workers = set()

    def run(self, feeds):
        """根据上一帧状态推进输出，并复用同一音频数组。

        Args:
            feeds: 码本、位置与两份完整状态。
        Returns:
            音频及用于下一帧的状态映射。
        """
        self.workers.add(current_thread())
        self.started.set()
        try:
            if self.gate is not None:
                assert self.gate.wait(5), "假图没有收到继续执行信号。"
            codes, position = feeds["codes"], feeds["position"]
            assert codes.shape == (1, 1, 16) and codes.dtype == np.int64
            assert position.shape == (1,) and position.dtype == np.int64
            for item in STATE_SPECS:
                state = feeds[item["name"]]
                assert state.shape == tuple(item["shape"]) and state.dtype == np.float32
            self.calls.append({name: value.copy() for name, value in feeds.items()})
            index = int(position[0])
            if index == self.fail_position:
                raise ValueError("模拟声码器失败。")
            value = float(feeds["carry"][0]) + float(codes.sum()) + index
            self.audio[:] = [value, index, float(feeds["history"].sum())]
            return {
                "audio": self.audio,
                "next_carry": np.asarray([value], dtype=np.float32),
                "next_history": feeds["history"] + np.asarray([[index, 1]], dtype=np.float32),
            }
        finally:
            self.finished.set()


def test_frames_preserve_input_and_audio_copies_and_advance_states(executor: ThreadPoolExecutor) -> None:
    """连续提交只返回前一帧，输入与共享输出均保留独立副本。"""
    gate = Event()
    graph = FakeGraph(gate=gate)
    codec = StreamingCodec(graph, STATE_SPECS, executor)
    frame = np.arange(16, dtype=np.int64)
    try:
        assert codec.finish() is None
        assert codec.push(frame, 0) is None
        assert graph.started.wait(5)
        # 后台尚未读取输入时修改调用方数组，检查提交时已经完成复制。
        frame[:] = -99
        gate.set()
        first = codec.push(np.ones(16, dtype=np.int64), 1)
        second = codec.push(np.full(16, 2, dtype=np.int64), 2)
        third = codec.finish()
        assert codec.finish() is None
        np.testing.assert_array_equal(first, [120, 0, 0])
        np.testing.assert_array_equal(second, [137, 1, 1])
        np.testing.assert_array_equal(third, [171, 2, 3])
        for audio in (first, second, third):
            assert audio.dtype == np.float32 and audio.shape == (3,)
            assert not np.shares_memory(audio, graph.audio)
        np.testing.assert_array_equal(graph.calls[0]["codes"].reshape(-1), np.arange(16))
        np.testing.assert_array_equal(graph.calls[1]["carry"], [120])
        np.testing.assert_array_equal(graph.calls[1]["history"], [[0, 1]])
        np.testing.assert_array_equal(graph.calls[2]["history"], [[1, 2]])
        assert len(graph.workers) == 1 and current_thread() not in graph.workers
    finally:
        gate.set()
        codec.close()
    assert all(worker.is_alive() for worker in graph.workers)


def test_new_request_reusing_graph_starts_with_zero_states(executor: ThreadPoolExecutor) -> None:
    """新请求在同一工作线程从全部零状态开始，并返回主机单精度音频。"""
    graph = FakeGraph(audio_dtype=np.float64)
    first_request = StreamingCodec(graph, STATE_SPECS, executor)
    try:
        first_request.push(np.ones(16, dtype=np.int64), 0)
        np.testing.assert_array_equal(first_request.finish(), [16, 0, 0])
    finally:
        first_request.close()
    assert len(graph.workers) == 1
    assert all(worker.is_alive() for worker in graph.workers)
    second_request = StreamingCodec(graph, STATE_SPECS, executor)
    try:
        assert second_request.push(np.zeros(16, dtype=np.int64), 0) is None
        audio = second_request.finish()
        np.testing.assert_array_equal(audio, [0, 0, 0])
        assert audio.dtype == np.float32
        for item in STATE_SPECS:
            np.testing.assert_array_equal(
                graph.calls[-1][item["name"]], np.zeros(item["shape"], dtype=np.float32),
            )
    finally:
        second_request.close()
    # 请求清理保留原工作线程，避免重新建立线程本地的图捕获上下文。
    assert len(graph.workers) == 1
    assert all(worker.is_alive() for worker in graph.workers)


@pytest.mark.parametrize("consume", ["push", "finish"])
def test_worker_failure_propagates_and_prevents_further_frames(
    consume: str, executor: ThreadPoolExecutor,
) -> None:
    """两个收割入口均传播后台异常，失败请求不能继续推进状态。"""
    graph = FakeGraph(fail_position=0)
    codec = StreamingCodec(graph, STATE_SPECS, executor)
    try:
        assert codec.push(np.zeros(16, dtype=np.int64), 0) is None
        with pytest.raises(ValueError, match="模拟声码器失败"):
            if consume == "push":
                codec.push(np.ones(16, dtype=np.int64), 1)
            else:
                codec.finish()
        with pytest.raises(RuntimeError, match="上一帧执行失败"):
            codec.push(np.ones(16, dtype=np.int64), 2)
        assert len(graph.calls) == 1
    finally:
        codec.close()
        codec.close()
    assert all(worker.is_alive() for worker in graph.workers)


def test_close_waits_for_inflight_frame_and_rejects_later_submissions(executor: ThreadPoolExecutor) -> None:
    """关闭期间等待本请求在途帧，保留共享线程并拒绝该请求重新提交。"""
    gate = Event()
    graph = FakeGraph(gate=gate)
    codec = StreamingCodec(graph, STATE_SPECS, executor)
    close_started, close_finished = Event(), Event()
    errors = []

    def close_codec():
        """记录关闭调用的开始、结束和异常，便于检查等待行为。"""
        close_started.set()
        try:
            codec.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_finished.set()

    closer = Thread(target=close_codec)
    try:
        codec.push(np.zeros(16, dtype=np.int64), 0)
        assert graph.started.wait(5)
        closer.start()
        assert close_started.wait(5)
        assert not close_finished.wait(0.05)
    finally:
        gate.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        codec.close()
    assert close_finished.is_set() and not closer.is_alive()
    assert errors == [] and graph.finished.is_set()
    assert all(worker.is_alive() for worker in graph.workers)
    with pytest.raises(RuntimeError, match="已经关闭"):
        codec.push(np.zeros(16, dtype=np.int64), 1)


def test_cancellation_cleanup_preserves_original_exception(executor: ThreadPoolExecutor) -> None:
    """请求取消时等待后台失败完成，清理不覆盖调用方原有异常。"""
    graph = FakeGraph(fail_position=0)
    codec = StreamingCodec(graph, STATE_SPECS, executor)
    with pytest.raises(LookupError, match="请求已取消"):
        try:
            codec.push(np.zeros(16, dtype=np.int64), 0)
            raise LookupError("请求已取消。")
        finally:
            codec.close()
    codec.close()
    assert graph.finished.is_set()
    assert all(worker.is_alive() for worker in graph.workers)


def test_interrupted_result_wait_still_waits_for_inflight_frame(executor, monkeypatch) -> None:
    """等待结果被中断后，关闭仍须等待后台状态写入完成。"""
    gate, closed = Event(), Event()
    graph = FakeGraph(gate=gate)
    codec = StreamingCodec(graph, STATE_SPECS, executor)
    codec.push(np.zeros(16, dtype=np.int64), 0)
    assert graph.started.wait(5)

    def interrupted_wait():
        """模拟主线程等待未来结果时收到中断信号。"""
        raise KeyboardInterrupt()

    def close_codec():
        """在独立线程中观察请求关闭是否等待在途帧。"""
        codec.close()
        closed.set()

    monkeypatch.setattr(codec._pending, "result", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        codec.finish()
    closer = Thread(target=close_codec)
    closer.start()
    try:
        assert not closed.wait(0.05)
    finally:
        gate.set()
        closer.join(5)
    assert closed.is_set() and graph.finished.is_set() and not closer.is_alive()
