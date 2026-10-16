from __future__ import annotations

from dataclasses import dataclass, field

from ....utils.log import logger
from ....utils.performance import format_perf_value


@dataclass(slots=True)
class S2ProTiming:
    """
    S2-Pro 合成链路性能统计。

    参数:
        preprocess_time: 文本预处理耗时。
        prompt_restore_time: 从 prompt 资产恢复 slow cache 的耗时。
        slow_prefill_times: 每个 batch 追加 user/assistant 前缀的 slow decode 耗时。
        slow_step_times: 每帧 acoustic code 写回 slow context 的耗时。
        fast_generate_times: 每帧 fast codebook 展开的耗时。
        semantic_sample_times: 每次 semantic token 采样耗时。
        chunk_gen_times: 每个 decoder chunk 送出前的攒码耗时。
        decoder_compute_times: decoder worker 返回的每个音频 chunk 计算耗时。
        total_steps: 已生成的 acoustic 帧数。

    返回:
        该类仅作为统计容器使用，不直接返回业务数据。

    异常:
        本类不主动抛出异常。
    """

    preprocess_time: float = 0.0
    prompt_restore_time: float = 0.0
    slow_prefill_times: list[float] = field(default_factory=list)
    slow_step_times: list[float] = field(default_factory=list)
    fast_generate_times: list[float] = field(default_factory=list)
    semantic_sample_times: list[float] = field(default_factory=list)
    chunk_gen_times: list[float] = field(default_factory=list)
    decoder_compute_times: list[float] = field(default_factory=list)
    total_steps: int = 0

    @property
    def slow_prefill_time(self) -> float:
        """所有 batch 前缀追加到 slow context 的总耗时。"""
        return sum(self.slow_prefill_times)

    @property
    def slow_step_time(self) -> float:
        """所有 acoustic 帧写回 slow context 的总耗时。"""
        return sum(self.slow_step_times)

    @property
    def fast_generate_time(self) -> float:
        """所有 fast codebook 展开的总耗时。"""
        return sum(self.fast_generate_times)

    @property
    def semantic_sample_time(self) -> float:
        """所有 semantic token 采样的总耗时。"""
        return sum(self.semantic_sample_times)

    @property
    def total_chunk_gen_time(self) -> float:
        """所有 decoder chunk 送出前攒码的总耗时。"""
        return sum(self.chunk_gen_times)

    @property
    def total_decoder_time(self) -> float:
        """所有 decoder worker 音频计算的总耗时。"""
        return sum(self.decoder_compute_times)

    @property
    def total_inference_time(self) -> float:
        """合成链路内可观测阶段的总耗时。"""
        return (
            self.preprocess_time
            + self.prompt_restore_time
            + self.total_chunk_gen_time
            + self.total_decoder_time
        )

    @property
    def first_decode_latency(self) -> float:
        """首个 decoder 音频 chunk 的计算耗时。"""
        return self.decoder_compute_times[0] if self.decoder_compute_times else 0.0

    @property
    def first_chunk_latency(self) -> float:
        """首个 decoder chunk 送出前的攒码耗时。"""
        return self.chunk_gen_times[0] if self.chunk_gen_times else 0.0

    @property
    def first_audio_latency(self) -> float:
        """首音延迟，按 prompt 恢复、首包攒码和首包解码计算。"""
        if self.chunk_gen_times and self.decoder_compute_times:
            return self.prompt_restore_time + self.first_chunk_latency + self.first_decode_latency
        return 0.0

    @property
    def avg_chunk_gen_time(self) -> float:
        """平均每个音频 chunk 的攒码耗时。"""
        return self.total_chunk_gen_time / len(self.chunk_gen_times) if self.chunk_gen_times else 0.0

    @property
    def avg_decoder_time(self) -> float:
        """平均每个音频 chunk 的 decoder 计算耗时。"""
        return self.total_decoder_time / len(self.decoder_compute_times) if self.decoder_compute_times else 0.0

    @property
    def avg_chunk_rtf(self) -> float:
        """沿用 qwen3 口径，近似把每个 chunk 的攒码和解码耗时相加作为平均 RTF。"""
        return self.avg_chunk_gen_time + self.avg_decoder_time


def log_s2_pro_performance(timing: S2ProTiming) -> None:
    """
    输出 S2-Pro 合成链路性能日志。

    参数:
        timing: 当前 synthesize 调用累积的性能统计。

    返回:
        无返回值；日志通过项目 logger 输出。

    异常:
        本函数不主动抛出异常。
    """
    # 每个 AUDIO chunk 输出一行明细，便于和 qwen3 一样观察首包、尾包和中间 chunk 的波动。
    for idx in range(min(len(timing.chunk_gen_times), len(timing.decoder_compute_times))):
        chunk_total = timing.chunk_gen_times[idx] + timing.decoder_compute_times[idx]
        msg = f"S2-Pro chunk {idx} generation time: {format_perf_value(chunk_total)}"
        msg += f", generate code time: {format_perf_value(timing.chunk_gen_times[idx])}"
        msg += f", decode time: {format_perf_value(timing.decoder_compute_times[idx])}"
        logger.info(msg)

    # 汇总日志拆分 slow/fast/decoder，有助于判断瓶颈在 GGUF 生成还是 ONNX codec 解码。
    logger.info(f"S2-Pro total inference time: {format_perf_value(timing.total_inference_time)}")
    logger.info(f"S2-Pro text preprocess time: {format_perf_value(timing.preprocess_time)}")
    logger.info(f"S2-Pro prompt restore time: {format_perf_value(timing.prompt_restore_time)}")
    logger.info(f"S2-Pro slow prefill time: {format_perf_value(timing.slow_prefill_time)}")
    logger.info(f"S2-Pro slow step time: {format_perf_value(timing.slow_step_time)}")
    logger.info(f"S2-Pro fast code generation time: {format_perf_value(timing.fast_generate_time)}")
    logger.info(f"S2-Pro semantic sample time: {format_perf_value(timing.semantic_sample_time)}")
    logger.info(f"S2-Pro decoder time: {format_perf_value(timing.total_decoder_time)}")
    logger.info(f"S2-Pro generated acoustic frames: {timing.total_steps}")
    logger.info(f"S2-Pro first chunk audio generation time: {format_perf_value(timing.first_audio_latency)}, generate code time: {format_perf_value(timing.first_chunk_latency)}, decode time: {format_perf_value(timing.first_decode_latency)}")
    logger.info(f"S2-Pro avg chunk code generation time: {format_perf_value(timing.avg_chunk_gen_time)}")
    logger.info(f"S2-Pro avg chunk decoder time: {format_perf_value(timing.avg_decoder_time)}")
    logger.info(f"S2-Pro avg chunk RTF: {format_perf_value(timing.avg_chunk_rtf)}")
