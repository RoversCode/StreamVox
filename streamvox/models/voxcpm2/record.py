from __future__ import annotations

from dataclasses import dataclass, field


# 关键常量：当前 voxcpm2 每个 chunk 对应 160ms 音频，因此 RTF 需要除以 0.16 秒。
VOXCPM2_CHUNK_AUDIO_SECONDS = 0.16


@dataclass(slots=True)
class _VoxCPM2Timing:
    """
    voxcpm2 流式合成各阶段耗时统计。

    参数:
        prefill_time: 当前 synthesize 调用内所有分段 prefill 的累计耗时。
        chunk_gen_times: 每个 decoder chunk 对应的 latent patch 生成耗时列表。
        decoder_compute_times: 每个 decoder chunk 在 AudioVAE worker 内部的解码耗时列表。

    返回:
        本类只承载耗时统计，不直接参与推理控制。

    边界异常:
        本数据结构本身不抛异常；调用方负责保证写入值是秒级浮点数。
    """

    # 关键变量：prefill_time 记录首包前同步准备阶段的累计耗时，作用域仅限单次 synthesize 调用。
    prefill_time: float = 0.0
    # 关键变量：chunk_gen_times 按 chunk 顺序记录完整生成耗时，
    # 口径包含“当前 patch 生成”以及“为下一轮自回归准备状态”的回灌耗时。
    chunk_gen_times: list[float] = field(default_factory=list)
    # 关键变量：decoder_compute_times 按 chunk 顺序记录 decoder worker 计算耗时。
    decoder_compute_times: list[float] = field(default_factory=list)

    @property
    def first_decode_latency(self) -> float:
        """
        返回首个音频 chunk 在 decoder worker 内部的解码耗时。

        参数:
            本属性不接收参数。

        返回:
            首个 decoder 计算耗时；没有记录时返回 0.0。

        边界异常:
            空列表时不抛异常，统一返回 0.0，避免性能日志阶段反向打断主流程。
        """

        return self.decoder_compute_times[0] if self.decoder_compute_times else 0.0

    @property
    def first_chunk_latency(self) -> float:
        """
        返回首个 latent patch 攒完并提交前的墙钟耗时。

        参数:
            本属性不接收参数。

        返回:
            首个 patch 生成耗时；没有记录时返回 0.0。

        边界异常:
            空列表时返回 0.0，避免读取首项时报下标异常。
        """

        return self.chunk_gen_times[0] if self.chunk_gen_times else 0.0

    @property
    def first_audio_latency(self) -> float:
        """
        返回首个 chunk 从开始推理到音频回包完成的近似总耗时。

        参数:
            本属性不接收参数。

        返回:
            prefill 耗时、首个 patch 生成耗时和首个 decoder 耗时之和。

        边界异常:
            任一列表为空时返回 0.0，表示当前统计还不足以组成完整首包链路。
        """

        if self.chunk_gen_times and self.decoder_compute_times:
            return self.prefill_time + self.first_chunk_latency + self.first_decode_latency
        return 0.0

    @property
    def avg_chunk_gen_time(self) -> float:
        """
        返回平均每个 latent patch 的生成耗时。

        参数:
            本属性不接收参数。

        返回:
            完整 chunk 生成耗时平均值；没有记录时返回 0.0。

        边界异常:
            空列表时返回 0.0，避免除零错误。
        """

        return sum(self.chunk_gen_times) / len(self.chunk_gen_times) if self.chunk_gen_times else 0.0

    @property
    def avg_decoder_time(self) -> float:
        """
        返回平均每个音频 chunk 的 decoder 解码耗时。

        参数:
            本属性不接收参数。

        返回:
            decoder 计算耗时平均值；没有记录时返回 0.0。

        边界异常:
            空列表时返回 0.0，避免除零错误。
        """

        return sum(self.decoder_compute_times) / len(self.decoder_compute_times) if self.decoder_compute_times else 0.0

    @property
    def total_chunk_gen_time(self) -> float:
        """
        返回当前请求内所有 latent patch 的累计生成耗时。

        参数:
            本属性不接收参数。

        返回:
            完整 chunk 生成耗时总和。

        边界异常:
            空列表时自然返回 0.0。
        """

        return sum(self.chunk_gen_times)

    @property
    def total_decoder_time(self) -> float:
        """
        返回当前请求内所有音频 chunk 的累计 decoder 解码耗时。

        参数:
            本属性不接收参数。

        返回:
            decoder 计算耗时总和。

        边界异常:
            空列表时自然返回 0.0。
        """

        return sum(self.decoder_compute_times)

    @property
    def total_inference_time(self) -> float:
        """
        返回当前请求的总耗时。

        参数:
            本属性不接收参数。

        返回:
            统计口径为 prefill + patch 生成 + decoder 解码。

        边界异常:
            本属性不抛异常；缺省值下自然回落到已有统计项求和结果。
        """

        return self.prefill_time + self.total_chunk_gen_time + self.total_decoder_time

    @property
    def avg_chunk_rtf(self) -> float:
        """
        近似返回平均 chunk RTF。

        参数:
            本属性不接收参数。

        返回:
            返回“平均每个 chunk 的总耗时 / 单个 chunk 对应音频时长”的比值。

        边界异常:
            没有 chunk 记录时返回 0.0。
        """

        # 这里必须把“平均 chunk 总耗时”除以当前 chunk 的真实音频时长，
        # 否则得到的只是秒数，不是标准 RTF。
        avg_chunk_total_time = self.avg_chunk_gen_time + self.avg_decoder_time
        if avg_chunk_total_time <= 0:
            return 0.0
        return avg_chunk_total_time / VOXCPM2_CHUNK_AUDIO_SECONDS

    def merge_from(self, other: "_VoxCPM2Timing") -> None:
        """
        把单段文本的统计合并进整次 synthesize 的总统计。

        参数:
            other: 需要被累加进来的另一份 voxcpm2 耗时统计。

        返回:
            无返回值；当前对象会被原地累加。

        边界异常:
            调用方应保证 other 为同结构对象；本函数不额外做运行时类型校验，避免热路径重复分支。
        """

        # 这里先累加 prefill，再拼接 chunk 列表，保证多段文本最终日志仍保持时间维度完整。
        self.prefill_time += other.prefill_time
        self.chunk_gen_times.extend(other.chunk_gen_times)
        self.decoder_compute_times.extend(other.decoder_compute_times)
