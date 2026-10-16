#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   item.py
@Time    :   2026/03/06 17:28:13
@Author  :   RoversCode
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union
from .decoder_protocol import DecoderState, DecoderResponse
from .record import Timing




@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    prompt_code: np.ndarray               # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    prompt_spk_embedding: np.ndarray                 # (D,)
    icl_mode: bool
    prompt_text: Optional[str] = None
    state: Optional[DecoderState] = None
    
    
@dataclass
class PromptData:
    """包装构建好的 Prompt Embedding 数据"""
    embd: np.ndarray # (1, seq, D) - 进入 Talker 的初始 Prompt
    text: str
    # self.text_ids = text_ids  # 目标文本的 ID
    spk_emb: np.ndarray
    trailing_text_embd: Optional[np.ndarray]
    state: Optional[DecoderState] = None
    compile_time: float = 0


@dataclass
class DecodeResult:
    """DecoderProxy.decode 的返回对象"""
    responses: List[DecoderResponse] = field(default_factory=list)

    @property
    def audio(self) -> Optional[np.ndarray]:
        """拼接后的完整音频"""
        if not self.responses:
            return None
        pcms = [r.audio for r in self.responses if r.audio is not None]
        return np.concatenate(pcms) if pcms else None

    @property
    def total_compute_time(self) -> float:
        """子进程总计算耗时"""
        return sum(self.chunk_compute_times)

    @property
    def chunk_compute_times(self) -> List[float]:
        """每块详细耗时"""
        return [r.compute_time for r in self.responses if r.msg_type == "AUDIO"]

    @property
    def first_response_time(self) -> float:
        """第一个响应返回的绝对时间点"""
        valid = [r for r in self.responses if r.msg_type == "AUDIO"]
        return valid[0].recv_time if valid else 0.0
    
    @property
    def final_state(self) -> Optional["DecoderState"]:
        """从响应列表中提取最终状态"""
        for r in reversed(self.responses):
            if r.state is not None:
                return r.state
        return None


@dataclass
class LoopOutput:
    """推理内核循环的输出封装"""
    all_codes: List[List[int]]       # 所有生成的 Codec IDs
    summed_embeds: List[np.ndarray]  # 叠加特征序列
    timing: Timing                   # 性能统计对象
    decode_result: Optional[DecodeResult] = None # 解码器完整返回对象