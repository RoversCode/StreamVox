from __future__ import annotations

from dataclasses import dataclass, field

from ....utils.log import logger
from ....utils.performance import format_perf_value


@dataclass(slots=True)
class MossChunkTiming:
    """
    MOSS-TTS-Nano 单个音频 chunk 的耗时明细。

    参数:
        chunk_index: 当前 chunk 在一次 synthesize 调用内的顺序编号。
        frame_count: 该 chunk 解码前累计消费的 audio frame 数量。
        frame_generation_time: 该 chunk 对应 audio frame 的总生成耗时。
        local_frame_compute_time: 该 chunk 内 local 分支生成 frame 的总耗时。
        global_decode_step_time: 该 chunk 内每帧写回 global decode 的总耗时。
        codec_decode_time: 该 chunk 的 codec streaming decode 耗时。
        audio_samples: 该 chunk 输出的单声道采样点数量。

    返回:
        本类只作为统计容器，不直接参与推理逻辑。

    边界异常:
        本类不主动抛出异常。
    """

    chunk_index: int
    frame_count: int = 0
    frame_generation_time: float = 0.0
    local_frame_compute_time: float = 0.0
    global_decode_step_time: float = 0.0
    codec_decode_time: float = 0.0
    audio_samples: int = 0

    @property
    def total_time(self) -> float:
        """
        返回当前 chunk 的可观测总耗时。

        参数:
            本属性无参数。

        返回:
            返回 frame 生成与 codec 解码两部分耗时之和。

        边界异常:
            本属性不主动抛出异常。
        """

        return self.frame_generation_time + self.codec_decode_time


@dataclass(slots=True)
class MossTiming:
    """
    MOSS-TTS-Nano 一次 synthesize 调用的性能统计。

    参数:
        preprocess_time: 文本预处理耗时。
        request_build_times: 每个文本 segment 构造 request rows 的耗时。
        prefill_times: 每个文本 segment 的 TTS prefill 耗时。
        chunk_timings: 每个输出音频 chunk 的耗时明细。
        total_frames: 全部输出 audio frame 数量。
        total_segments: 文本预处理后实际执行的 segment 数。

    返回:
        本类只作为统计容器，不直接返回业务数据。

    边界异常:
        本类不主动抛出异常。
    """

    preprocess_time: float = 0.0
    request_build_times: list[float] = field(default_factory=list)
    prefill_times: list[float] = field(default_factory=list)
    chunk_timings: list[MossChunkTiming] = field(default_factory=list)
    total_frames: int = 0
    total_segments: int = 0

    @property
    def total_request_build_time(self) -> float:
        """
        返回所有 segment 构造 request rows 的总耗时。

        参数:
            本属性无参数。

        返回:
            返回 request_build_times 的求和结果。

        边界异常:
            本属性不主动抛出异常。
        """

        return sum(self.request_build_times)

    @property
    def total_prefill_time(self) -> float:
        """
        返回所有 segment 的 prefill 总耗时。

        参数:
            本属性无参数。

        返回:
            返回 prefill_times 的求和结果。

        边界异常:
            本属性不主动抛出异常。
        """

        return sum(self.prefill_times)

    @property
    def total_frame_generation_time(self) -> float:
        """
        返回全部 chunk 的 frame 生成总耗时。

        参数:
            本属性无参数。

        返回:
            返回各 chunk 的 frame_generation_time 总和。

        边界异常:
            本属性不主动抛出异常。
        """

        return sum(chunk.frame_generation_time for chunk in self.chunk_timings)

    @property
    def total_local_frame_compute_time(self) -> float:
        """
        返回全部 chunk 的 local frame 计算总耗时。

        参数:
            本属性无参数。

        返回:
            返回各 chunk 的 local_frame_compute_time 总和。

        边界异常:
            本属性不主动抛出异常。
        """

        return sum(chunk.local_frame_compute_time for chunk in self.chunk_timings)

    @property
    def total_global_decode_step_time(self) -> float:
        """
        返回全部 chunk 的 global decode step 总耗时。

        参数:
            本属性无参数。

        返回:
            返回各 chunk 的 global_decode_step_time 总和。

        边界异常:
            本属性不主动抛出异常。
        """

        return sum(chunk.global_decode_step_time for chunk in self.chunk_timings)

    @property
    def total_codec_decode_time(self) -> float:
        """
        返回全部 chunk 的 codec decode 总耗时。

        参数:
            本属性无参数。

        返回:
            返回各 chunk 的 codec_decode_time 总和。

        边界异常:
            本属性不主动抛出异常。
        """

        return sum(chunk.codec_decode_time for chunk in self.chunk_timings)

    @property
    def total_inference_time(self) -> float:
        """
        返回一次调用可观测到的总耗时。

        参数:
            本属性无参数。

        返回:
            返回预处理、request 构造、prefill、frame 生成和 codec 解码之和。

        边界异常:
            本属性不主动抛出异常。
        """

        return (
            self.preprocess_time
            + self.total_request_build_time
            + self.total_prefill_time
            + self.total_frame_generation_time
            + self.total_codec_decode_time
        )

    @property
    def first_chunk(self) -> MossChunkTiming | None:
        """
        返回首个输出 chunk 的耗时明细。

        参数:
            本属性无参数。

        返回:
            有 chunk 时返回第一个 MossChunkTiming，否则返回 None。

        边界异常:
            本属性不主动抛出异常。
        """

        return self.chunk_timings[0] if self.chunk_timings else None

    @property
    def first_chunk_request_build_time(self) -> float:
        """
        返回首个 chunk 所属首段 request 构造耗时。

        参数:
            本属性无参数。

        返回:
            request_build_times 首项；为空时返回 0。

        边界异常:
            本属性不主动抛出异常。
        """

        return self.request_build_times[0] if self.request_build_times else 0.0

    @property
    def first_chunk_prefill_time(self) -> float:
        """
        返回首个 chunk 所属首段 prefill 耗时。

        参数:
            本属性无参数。

        返回:
            prefill_times 首项；为空时返回 0。

        边界异常:
            本属性不主动抛出异常。
        """

        return self.prefill_times[0] if self.prefill_times else 0.0

    @property
    def first_chunk_latency(self) -> float:
        """
        返回首个音频 chunk 的全链路延迟。

        参数:
            本属性无参数。

        返回:
            返回预处理、首段 request 构造、首段 prefill、首个 chunk 的 frame 生成和 codec 解码耗时之和。

        边界异常:
            本属性不主动抛出异常。
        """

        first_chunk = self.first_chunk
        if first_chunk is None:
            return 0.0
        return (
            self.preprocess_time
            + self.first_chunk_request_build_time
            + self.first_chunk_prefill_time
            + first_chunk.frame_generation_time
            + first_chunk.codec_decode_time
        )

    @property
    def avg_chunk_generation_time(self) -> float:
        """
        返回平均每个 chunk 的 frame 生成耗时。

        参数:
            本属性无参数。

        返回:
            返回所有 chunk 的平均 frame_generation_time；没有 chunk 时返回 0。

        边界异常:
            本属性不主动抛出异常。
        """

        if not self.chunk_timings:
            return 0.0
        return self.total_frame_generation_time / len(self.chunk_timings)

    @property
    def avg_chunk_decode_time(self) -> float:
        """
        返回平均每个 chunk 的 codec 解码耗时。

        参数:
            本属性无参数。

        返回:
            返回所有 chunk 的平均 codec_decode_time；没有 chunk 时返回 0。

        边界异常:
            本属性不主动抛出异常。
        """

        if not self.chunk_timings:
            return 0.0
        return self.total_codec_decode_time / len(self.chunk_timings)

    @property
    def avg_chunk_total_time(self) -> float:
        """
        返回平均每个 chunk 的总耗时。

        参数:
            本属性无参数。

        返回:
            返回所有 chunk 的平均 total_time；没有 chunk 时返回 0。

        边界异常:
            本属性不主动抛出异常。
        """

        if not self.chunk_timings:
            return 0.0
        return sum(chunk.total_time for chunk in self.chunk_timings) / len(self.chunk_timings)


def log_moss_performance(timing: MossTiming) -> None:
    """
    输出 MOSS-TTS-Nano 一次 synthesize 调用的性能日志。

    参数:
        timing: 当前调用累积出的性能统计。

    返回:
        无返回值；日志通过项目 logger 输出。

    边界异常:
        本函数不主动抛出异常。
    """

    # 每个音频 chunk 单独打印，便于观察首包与后续 chunk 的稳定性差异。
    for chunk in timing.chunk_timings:
        logger.info(
            "Moss Nano chunk %s time: total=%ss, frame_generation=%ss, local_frame=%ss, global_decode_step=%ss, codec_decode=%ss, frames=%s, samples=%s",
            chunk.chunk_index,
            format_perf_value(chunk.total_time),
            format_perf_value(chunk.frame_generation_time),
            format_perf_value(chunk.local_frame_compute_time),
            format_perf_value(chunk.global_decode_step_time),
            format_perf_value(chunk.codec_decode_time),
            chunk.frame_count,
            chunk.audio_samples,
        )

    logger.info("Moss Nano total inference time: %ss", format_perf_value(timing.total_inference_time))
    logger.info("Moss Nano preprocess time: %ss", format_perf_value(timing.preprocess_time))
    logger.info("Moss Nano request build time: %ss", format_perf_value(timing.total_request_build_time))
    logger.info("Moss Nano prefill time: %ss", format_perf_value(timing.total_prefill_time))
    logger.info("Moss Nano frame generation time: %ss", format_perf_value(timing.total_frame_generation_time))
    logger.info("Moss Nano local frame compute time: %ss", format_perf_value(timing.total_local_frame_compute_time))
    logger.info("Moss Nano global decode step time: %ss", format_perf_value(timing.total_global_decode_step_time))
    logger.info("Moss Nano codec decode time: %ss", format_perf_value(timing.total_codec_decode_time))
    logger.info("Moss Nano total segments: %s", timing.total_segments)
    logger.info("Moss Nano total frames: %s", timing.total_frames)

    first_chunk = timing.first_chunk
    if first_chunk is not None:
        logger.info(
            "Moss Nano first chunk latency: total=%ss, preprocess=%ss, request_build=%ss, prefill=%ss, frame_generation=%ss, local_frame=%ss, global_decode_step=%ss, codec_decode=%ss, frames=%s, samples=%s",
            format_perf_value(timing.first_chunk_latency),
            format_perf_value(timing.preprocess_time),
            format_perf_value(timing.first_chunk_request_build_time),
            format_perf_value(timing.first_chunk_prefill_time),
            format_perf_value(first_chunk.frame_generation_time),
            format_perf_value(first_chunk.local_frame_compute_time),
            format_perf_value(first_chunk.global_decode_step_time),
            format_perf_value(first_chunk.codec_decode_time),
            first_chunk.frame_count,
            first_chunk.audio_samples,
        )

    logger.info("Moss Nano avg chunk generation time: %ss", format_perf_value(timing.avg_chunk_generation_time))
    logger.info("Moss Nano avg chunk decode time: %ss", format_perf_value(timing.avg_chunk_decode_time))
    logger.info("Moss Nano avg chunk total time: %ss", format_perf_value(timing.avg_chunk_total_time))
