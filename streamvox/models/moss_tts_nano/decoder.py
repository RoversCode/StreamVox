from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import onnxruntime as ort

from ...utils.log import logger
from .common import (
    MossOnnxBundle,
    merge_audio_channels_to_mono,
    resolve_stream_decode_frame_budget,
)
from .schema.record import MossChunkTiming, MossTiming


@dataclass(slots=True)
class _MossFrameEvent:
    """
    MOSS TTS frame 生成线程传给 codec 解码线程的普通帧事件。

    参数:
        step_index: 当前帧在本段文本内的生成序号。
        frame: 当前生成出的完整 audio code frame。
        frame_generation_time: 当前帧完整生成耗时。
        local_frame_compute_time: 当前帧 local 分支计算耗时。
        global_decode_step_time: 当前帧写回 global decode 的耗时。

    返回:
        本对象只用于线程间传递数据。

    边界异常:
        本数据结构不主动抛出异常。
    """

    step_index: int
    frame: list[int]
    frame_generation_time: float = 0.0
    local_frame_compute_time: float = 0.0
    global_decode_step_time: float = 0.0


@dataclass(slots=True)
class _MossFrameFinal:
    """
    MOSS TTS frame 生成线程的正常结束事件。

    参数:
        generated_frames: 本段文本已经生成的完整 frame 列表。

    返回:
        本对象用于通知 codec 解码线程 flush 尾部 pending frame。

    边界异常:
        本数据结构不主动抛出异常。
    """

    generated_frames: list[list[int]]


@dataclass(slots=True)
class _MossFrameError:
    """
    MOSS TTS frame 生成线程的异常包装。

    参数:
        error: 后台生成线程捕获到的原始异常。

    返回:
        前台生成器收到该对象后会重新抛出 error。

    边界异常:
        本数据结构不主动抛出异常。
    """

    error: Exception


@dataclass(slots=True)
class CodecStreamingDecodeSession:
    """
    MOSS codec ONNX 增量解码会话。

    参数:
        codec_meta: codec meta 内容。
        session: `codec_decode_step.onnx` 对应的 ONNX Runtime session。

    返回:
        对象持有 codec streaming state，`run_frames` 会返回解码音频和有效长度。

    边界异常:
        ONNX 输入输出名称和 meta 不一致时会在运行阶段抛出异常。
    """

    codec_meta: dict[str, Any]
    session: ort.InferenceSession
    transformer_specs: list[dict[str, Any]] = field(init=False, default_factory=list)
    attention_specs: list[dict[str, Any]] = field(init=False, default_factory=list)
    state_feeds: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.transformer_specs = list(self.codec_meta.get("streaming_decode", {}).get("transformer_offsets", []))
        self.attention_specs = list(self.codec_meta.get("streaming_decode", {}).get("attention_caches", []))
        self.reset()

    def reset(self) -> None:
        """
        重置 codec 增量解码缓存。

        参数:
            本函数无显式参数。

        返回:
            无返回值。

        边界异常:
            meta 中 shape 非法时 numpy 创建数组会自然抛出异常。
        """

        self.state_feeds = {}
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = np.zeros(tuple(spec["offset_shape"]), dtype=np.int32)
            self.state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_values_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_positions_input_name"])] = np.full(
                tuple(spec["positions_shape"]),
                -1,
                dtype=np.int32,
            )

    def run_frames(self, frame_rows: list[list[int]]) -> tuple[np.ndarray, int] | None:
        """
        增量解码若干 audio token frame。

        参数:
            frame_rows: 若干帧音频 token，每帧包含 `n_vq` 个 codebook token。

        返回:
            返回 codec 输出 audio 张量和有效采样长度；没有 frame 时返回 None。

        边界异常:
            ONNX 执行失败或 meta 不匹配时抛出异常。
        """

        if not frame_rows:
            return None
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        frame_count = len(frame_rows)
        audio_codes = np.zeros((1, frame_count, num_quantizers), dtype=np.int32)
        for frame_index, frame_row in enumerate(frame_rows):
            for channel_index in range(num_quantizers):
                audio_codes[0, frame_index, channel_index] = int(
                    frame_row[channel_index] if channel_index < len(frame_row) else 0
                )
        feeds: dict[str, np.ndarray] = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([frame_count], dtype=np.int32),
        }
        feeds.update(self.state_feeds)
        outputs = self.session.run(None, feeds)
        output_names = [output.name for output in self.session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = named_outputs[str(spec["output_name"])]
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = named_outputs[str(spec["offset_output_name"])]
            self.state_feeds[str(spec["cached_keys_input_name"])] = named_outputs[str(spec["cached_keys_output_name"])]
            self.state_feeds[str(spec["cached_values_input_name"])] = named_outputs[str(spec["cached_values_output_name"])]
            self.state_feeds[str(spec["cached_positions_input_name"])] = named_outputs[str(spec["cached_positions_output_name"])]
        return named_outputs["audio"], int(named_outputs["audio_lengths"].reshape(-1)[0])


class MossAudioDecoder:
    """
    MOSS codec 解码器与并行流水线调度器。

    参数:
        bundle: 已加载的 MOSS bundle 元信息。
        codec_decode_session: full codec decode ONNX session。
        codec_decode_step_session: streaming codec decode ONNX session。

    返回:
        初始化后持有 full decode、streaming decode 和并行流水线逻辑。

    边界异常:
        codec 图或 meta 不匹配时会在运行阶段抛出异常。
    """

    def __init__(
        self,
        *,
        bundle: MossOnnxBundle,
        codec_decode_step_session: ort.InferenceSession,
    ) -> None:
        self.bundle = bundle
        self.codec_meta = bundle.codec_meta
        self.codec_streaming_session = CodecStreamingDecodeSession(
            codec_meta=bundle.codec_meta,
            session=codec_decode_step_session,
        )

    @property
    def sample_rate(self) -> int:
        """
        返回 codec 输出采样率。

        参数:
            本属性无参数。

        返回:
            返回整数 Hz 采样率。

        边界异常:
            codec meta 缺失字段时会自然抛出 KeyError。
        """

        return int(self.codec_meta["codec_config"]["sample_rate"])

    def iter_synthesize_mono(
        self,
        *,
        request_rows: dict[str, list[list[int]]],
        frame_iterator_factory: Callable[[dict[str, list[list[int]]]], Iterator[dict[str, Any]]],
        timing: MossTiming | None = None,
    ) -> Iterator[np.ndarray]:
        """
        将 MOSS audio frame 增量解码为一维 PCM 音频块。

        参数:
            request_rows: 当前文本段的 TTS prefill 输入。
            frame_iterator_factory: 生成 frame 事件的迭代器工厂。
            timing: 可选性能统计对象；传入后会记录每个音频 chunk 的耗时明细。

        返回:
            逐块 yield float32 mono PCM。

        边界异常:
            TTS 生成或 codec step decode 失败时抛出异常。
        """

        frame_queue: queue.Queue[object] = queue.Queue(maxsize=16)
        stop_event = threading.Event()
        frame_worker = threading.Thread(
            target=self._run_frame_generation_worker,
            kwargs={
                "request_rows": request_rows,
                "frame_iterator_factory": frame_iterator_factory,
                "output_queue": frame_queue,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        frame_worker.start()

        pending_decode_events: list[_MossFrameEvent] = []
        emitted_samples_total = 0
        first_audio_emitted_at: float | None = None
        emitted_chunk_index = len(timing.chunk_timings) if timing is not None else 0
        self.codec_streaming_session.reset()

        def decode_pending_frames(force: bool) -> list[np.ndarray]:
            nonlocal emitted_samples_total, first_audio_emitted_at
            nonlocal emitted_chunk_index
            pending_count = len(pending_decode_events)
            if pending_count <= 0:
                return []
            sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
            decode_budget = resolve_stream_decode_frame_budget(
                emitted_samples_total,
                sample_rate,
                first_audio_emitted_at,
            )
            if not force and pending_count < max(1, decode_budget):
                return []

            frame_budget = pending_count if force else min(pending_count, max(1, decode_budget))
            frame_events = pending_decode_events[:frame_budget]
            del pending_decode_events[:frame_budget]
            frame_chunk = [list(item.frame) for item in frame_events]
            decode_start_time = time.perf_counter() if timing is not None else 0.0
            decoded = self.codec_streaming_session.run_frames(frame_chunk)
            codec_decode_time = time.perf_counter() - decode_start_time if timing is not None else 0.0
            if decoded is None:
                return []
            audio, audio_length = decoded
            if audio_length <= 0:
                return []
            waveform_chunk = merge_audio_channels_to_mono(audio[0, :, :audio_length])
            if waveform_chunk.size <= 0:
                return []
            if first_audio_emitted_at is None:
                first_audio_emitted_at = __import__("time").perf_counter()
            emitted_samples_total += int(waveform_chunk.shape[0])
            if timing is not None:
                emitted_chunk_index += 1
                timing.chunk_timings.append(
                    MossChunkTiming(
                        chunk_index=int(emitted_chunk_index),
                        frame_count=len(frame_events),
                        frame_generation_time=sum(item.frame_generation_time for item in frame_events),
                        local_frame_compute_time=sum(item.local_frame_compute_time for item in frame_events),
                        global_decode_step_time=sum(item.global_decode_step_time for item in frame_events),
                        codec_decode_time=float(codec_decode_time),
                        audio_samples=int(waveform_chunk.shape[0]),
                    )
                )
            return [waveform_chunk]

        try:
            while True:
                item = frame_queue.get()
                if isinstance(item, _MossFrameError):
                    raise item.error
                if isinstance(item, _MossFrameEvent):
                    pending_decode_events.append(item)
                    for waveform in decode_pending_frames(False):
                        yield waveform
                    continue
                if isinstance(item, _MossFrameFinal):
                    for waveform in decode_pending_frames(True):
                        yield waveform
                    break
        finally:
            stop_event.set()
            self.codec_streaming_session.reset()
            frame_worker.join(timeout=1.0)

    def _run_frame_generation_worker(
        self,
        *,
        request_rows: dict[str, list[list[int]]],
        frame_iterator_factory: Callable[[dict[str, list[list[int]]]], Iterator[dict[str, Any]]],
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        """
        在后台线程中执行 MOSS TTS frame 生成。

        参数:
            request_rows: 当前文本段的 TTS prefill 输入。
            frame_iterator_factory: 生成 frame 事件的迭代器工厂。
            output_queue: 线程间输出队列。
            stop_event: 前台生成器提前结束时设置的停止信号。

        返回:
            本函数无返回值；所有结果都通过 `output_queue` 传给前台。

        边界异常:
            本函数内部捕获异常并包装成 `_MossFrameError` 放入队列。
        """

        try:
            generated_frames: list[list[int]] = []
            for event in frame_iterator_factory(request_rows):
                if stop_event.is_set():
                    return
                if event["type"] == "frame":
                    frame = [int(token) for token in event["frame"]]
                    generated_frames = event["generated_frames"]
                    if not self._put_frame_queue_item(
                        output_queue=output_queue,
                        item=_MossFrameEvent(
                            step_index=int(event["step_index"]),
                            frame=frame,
                            frame_generation_time=float(event.get("frame_generation_time", 0.0)),
                            local_frame_compute_time=float(event.get("local_frame_compute_time", 0.0)),
                            global_decode_step_time=float(event.get("global_decode_step_time", 0.0)),
                        ),
                        stop_event=stop_event,
                    ):
                        return
                    continue
                generated_frames = event["generated_frames"]
                self._put_frame_queue_item(
                    output_queue=output_queue,
                    item=_MossFrameFinal(generated_frames=generated_frames),
                    stop_event=stop_event,
                )
                return
            self._put_frame_queue_item(
                output_queue=output_queue,
                item=_MossFrameFinal(generated_frames=generated_frames),
                stop_event=stop_event,
            )
        except Exception as exc:
            self._put_frame_queue_item(
                output_queue=output_queue,
                item=_MossFrameError(exc),
                stop_event=stop_event,
            )

    def _put_frame_queue_item(
        self,
        *,
        output_queue: queue.Queue,
        item: object,
        stop_event: threading.Event,
    ) -> bool:
        """
        可中断地向 frame 队列写入事件。

        参数:
            output_queue: TTS 生成线程和 codec 解码线程之间的队列。
            item: 待写入的事件对象。
            stop_event: 前台提前结束时设置的停止信号。

        返回:
            写入成功返回 True；检测到 `stop_event` 后返回 False。

        边界异常:
            队列写入过程中的非满队列异常会自然抛出。
        """

        while not stop_event.is_set():
            try:
                output_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def warmup(self, *, build_minimal_frames: Callable[[], list[list[int]]]) -> None:
        """
        预热 codec streaming decode。

        参数:
            build_minimal_frames: 返回最小合法 audio frame 列表的回调。

        返回:
            无返回值。

        边界异常:
            streaming codec 解码图执行失败时抛出异常。
        """

        warmup_frames = build_minimal_frames()
        self.codec_streaming_session.reset()
        self.codec_streaming_session.run_frames(warmup_frames[:1])
        self.codec_streaming_session.reset()

    def shutdown(self) -> None:
        """
        释放 decoder 持有的 ONNX session 引用。

        参数:
            本函数无显式参数。

        返回:
            无返回值；重复调用安全返回。

        边界异常:
            资源释放由 Python 对象析构负责，本函数不主动抛出异常。
        """

        self.codec_streaming_session = None


__all__ = ["CodecStreamingDecodeSession", "MossAudioDecoder"]
