"""用处理器替身验证声码器异常后的流排空及缓冲复用边界。"""

from __future__ import annotations

from threading import get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from streamvox.models.breeze.codec import CudaStreamingDecoder
from streamvox.models.breeze.ort import OrtGraph


class FakeStream:
    """保留尚未完成的复制，只有同步成功才清空队列。"""

    def __init__(self):
        self.pending = []
        self.copy_count = 0
        self.sync_count = 0
        self.fail_copy_at = None
        self.copy_error = None

    def activate(self):
        """模拟选择设备，无需初始化 CUDA。"""

    def copy(self, destination, source, size, kind):
        """记录待完成传输，并在指定调用处注入异常。

        Args:
            destination: 目标地址。
            source: 来源地址。
            size: 传输字节数。
            kind: 复制方向。
        Returns:
            成功入队后正常返回。
        """
        self.copy_count += 1
        if self.copy_count == self.fail_copy_at:
            raise self.copy_error
        self.pending.append((destination, source, size, kind))

    def synchronize(self):
        """模拟等待全部传输完成，随后才能覆写锁页缓冲。"""
        self.sync_count += 1
        self.pending.clear()


def make_decoder():
    """直接准备单个状态的运行替身，绕过真实会话及设备初始化。

    Returns:
        最小解码器对象和完整主机初始输入。
    """
    decoder = CudaStreamingDecoder.__new__(CudaStreamingDecoder)
    decoder._closed = decoder._failed = False
    decoder._initialized = decoder._warmed_up = True
    decoder._thread_id = get_ident()
    decoder._stream = FakeStream()
    decoder._state_specs = [{"name": "state", "shape": [1]}]
    decoder._feed_names = {"codes", "position", "state"}
    value = SimpleNamespace(data_ptr=lambda: 17)
    decoder._input_states = {"state": value}
    decoder._output_states = {"state": value}
    decoder._outputs = {"next_state": value}
    decoder._codes = decoder._position = decoder._audio = value
    decoder._host_states = {"state": np.zeros(1, dtype=np.float32)}
    decoder._host_codes = np.zeros((1, 1, 16), dtype=np.int64)
    decoder._host_position = np.zeros(1, dtype=np.int64)
    decoder._host_audio = np.zeros((1, 4), dtype=np.float32)
    decoder._state_copies = [(17, 17, 4)]
    decoder._binding = decoder._run_options = object()
    decoder.session = SimpleNamespace(run_with_iobinding=Mock())
    feeds = {"codes": np.ones((1, 1, 16), dtype=np.int64), "position": np.zeros(1, dtype=np.int64),
             "state": np.ones(1, dtype=np.float32)}
    return decoder, feeds


@pytest.mark.parametrize("failure_point", ["copy", "session"])
def test_run_failure_drains_pending_copies_before_allowing_fresh_request(failure_point):
    """入队异常必须先排空流，之后只能从完整主机状态重新开始。

    Args:
        failure_point: 在后续输入复制或会话执行处注入异常。
    Returns:
        原异常保留、队列排空及新请求重置均正确时正常返回。
    """
    decoder, feeds = make_decoder()
    original_error = RuntimeError("injected enqueue failure")
    if failure_point == "copy":
        decoder._stream.fail_copy_at = 3
        decoder._stream.copy_error = original_error
    else:
        decoder.session.run_with_iobinding.side_effect = original_error
    with pytest.raises(RuntimeError) as caught:
        decoder.run(feeds)
    assert caught.value is original_error
    assert decoder._stream.sync_count == 1
    assert decoder._stream.pending == []
    assert decoder._initialized is False
    with pytest.raises(ValueError, match="完整主机初始状态"):
        decoder.run({**feeds, "state": decoder._output_states["state"]})
    sync_count_before_reset = decoder._stream.sync_count
    decoder._stream.fail_copy_at = None
    decoder.session.run_with_iobinding.side_effect = None
    decoder.run(feeds)
    assert decoder._stream.pending == []
    assert decoder._stream.sync_count == sync_count_before_reset + 1
    assert decoder._initialized is True


@pytest.mark.parametrize("failure_point", ["drain", "final_sync"])
def test_unfinished_stream_preserves_original_error_and_rejects_buffer_reuse(failure_point):
    """无法确认流已完成时保留原错误，并拒绝改写所有主机缓冲。

    Args:
        failure_point: 在异常排空或正常末尾同步处注入失败。
    Returns:
        原错误、补充说明及不可复用状态均正确时正常返回。
    """
    decoder, feeds = make_decoder()
    original_error = RuntimeError("injected original failure")
    cleanup_error = RuntimeError("injected drain failure")
    if failure_point == "drain":
        decoder._stream.fail_copy_at = 3
        decoder._stream.copy_error = original_error
        decoder._stream.synchronize = Mock(side_effect=[cleanup_error, None])
    else:
        decoder._stream.synchronize = Mock(side_effect=[original_error, None])
    with pytest.raises(RuntimeError) as caught:
        decoder.run(feeds)
    assert caught.value is original_error
    if failure_point == "drain":
        assert any("injected drain failure" in note for note in original_error.__notes__)
    decoder._stream.synchronize.assert_called_once()
    assert decoder._stream.pending
    assert decoder._initialized is False
    assert decoder._failed is True
    host_codes = decoder._host_codes.copy()
    copy_count = decoder._stream.copy_count
    with pytest.raises(RuntimeError, match="无法复用"):
        decoder.run({**feeds, "codes": np.full((1, 1, 16), 99, dtype=np.int64)})
    with pytest.raises(RuntimeError, match="无法复用"):
        decoder.warmup()
    np.testing.assert_array_equal(decoder._host_codes, host_codes)
    assert decoder._stream.copy_count == copy_count


def test_close_releases_metadata_owned_session_before_external_stream():
    """先释放持有会话的输入输出元数据，再销毁外部流。

    Returns:
        会话析构先于流关闭，且重复关闭不重复释放时正常返回。
    """
    actions = []

    class SessionResource:
        """用析构事件模拟元数据持有的原生会话。"""

        def __del__(self):
            """记录最后一个会话引用被释放的时刻。"""
            actions.append("session")

    decoder, _ = make_decoder()
    decoder.session = SessionResource()
    decoder.inputs = [SimpleNamespace(owner=decoder.session)]
    decoder.outputs = [SimpleNamespace(owner=decoder.session)]
    decoder._stream.close = Mock(side_effect=lambda: actions.append("stream"))
    synchronize = decoder._stream.synchronize = Mock()
    close_stream = decoder._stream.close
    decoder.close()
    decoder.close()
    assert actions == ["session", "stream"]
    assert decoder.inputs == decoder.outputs == []
    assert decoder.session is decoder._binding is decoder._stream is None
    synchronize.assert_called_once()
    close_stream.assert_called_once()


def test_graph_close_releases_interface_metadata_after_partial_initialization():
    """未完成接口初始化的辅助图也能重复清理会话。

    Returns:
        部分初始化对象关闭后没有会话或接口元数据引用时正常返回。
    """
    graph = OrtGraph.__new__(OrtGraph)
    graph.session = object()
    graph.close()
    graph.close()
    assert graph.session is None
    assert graph.inputs == graph.outputs == []
