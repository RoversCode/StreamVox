#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   record.py
@Time    :   2026/03/07 09:39:24
@Author  :   RoversCode
@Version :   1.0
@Desc    :   记录耗时
'''

# here put the import lib
from typing import List
from dataclasses import dataclass, field


@dataclass
class Timing:
    """推理各阶段耗时统计"""
    # 基础耗时 (支持列表记录，用于分析分段耗时)
    build_prompt_time: float = 0.0                            # 仅单次
    prefill_time: float = 0.0                           # 仅单次
    chunk_gen_times:        List[float] = field(default_factory=list)
    decoder_compute_times:  List[float] = field(default_factory=list)
    
    total_steps: int = 0

    @property
    def first_decode_latency(self) -> float:
        """从第一块音频消息发给子进程到渲染完成的时间"""
        return self.decoder_compute_times[0] if self.decoder_compute_times else 0.0

    @property
    def first_chunk_latency(self) -> float:
        """即 chunk_gen_times[0]，从推理开始到第一个 chunk 攒码完成的绝对耗时"""
        return self.chunk_gen_times[0] if self.chunk_gen_times else 0.0

    @property
    def first_audio_latency(self) -> float:
        """全链路首音延迟 = 攒码完成耗时 + 首包解码渲染耗时"""
        if self.chunk_gen_times and self.decoder_compute_times:
            return self.first_chunk_latency + self.first_decode_latency + self.prefill_time
        return 0.0
    
    @property
    def avg_chunk_gen_time(self) -> float:
        return sum(self.chunk_gen_times) / len(self.chunk_gen_times) if self.chunk_gen_times else 0.0
    
    @property
    def avg_decoder_time(self) -> float:
        return sum(self.decoder_compute_times) / len(self.decoder_compute_times) if self.decoder_compute_times else 0.0

    @property
    def total_chunk_gen_time(self) -> float:
        return sum(self.chunk_gen_times)
    
    @property
    def total_decoder_time(self) -> float:
        return sum(self.decoder_compute_times)
    
    @property
    def total_build_prompt_time(self) -> float:
        """ prompt构建耗时 """
        return self.build_prompt_time

    @property
    def total_inference_time(self) -> float:
        """全链路总耗时（含解码渲染）"""
        return (self.build_prompt_time + self.prefill_time + self.total_chunk_gen_time + self.total_decoder_time)

    def merge_from(self, other: "Timing") -> None:
        """把分段统计累加到总统计里，避免多段文本时只看到最后一段的数据"""
        self.build_prompt_time += other.build_prompt_time
        self.prefill_time += other.prefill_time
        self.chunk_gen_times.extend(other.chunk_gen_times)
        self.decoder_compute_times.extend(other.decoder_compute_times)
        self.total_steps += other.total_steps

    @property
    def avg_chunk_rtf(self) -> float:
        """
        业务口径里每个 chunk 近似视为 1 秒音频，
        因此单 chunk 的平均 RTF 可以近似为：
        平均 chunk 码生成耗时 + 平均 chunk 解码耗时。
        """
        return self.avg_chunk_gen_time + self.avg_decoder_time
