#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   prompt_builder.py
@Time    :   2026/03/06 17:36:51
@Author  :   RoversCode
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
'''

# here put the import lib
import time
import numpy as np
from typing import Optional, List
from .schema.constant import SPECIAL_TOKENS
from .schema.item import VoiceClonePromptItem, PromptData
from ...utils.log import logger

        
class PromptBuilder:
    def __init__(self, tokenizer, assets):
        self.tokenizer = tokenizer
        self.assets = assets
        self.special_tokens = SPECIAL_TOKENS

    def _get_ids(self, text: str) -> List[int]:
        """内聚 tokenizer 调度"""
        res = self.tokenizer.encode(text)
        if hasattr(res, "ids"):
            return res.ids
        return res
    
    def build_design_prompt(self, text: str, instruct: str, lang_id: int, stream: bool=False) -> PromptData:
        """[音色设计入口]"""
        return self._build_core(text, lang_id=lang_id, instruct=instruct, stream=stream)
    
    def build_custom_prompt(self, text: str, spk_id: str, lang_id: int, instruct: str, stream: bool=False) -> PromptData:
        """[精品音色入口]"""
        return self._build_core(text, lang_id=lang_id, spk_id=spk_id, instruct=instruct, stream=stream)

    def build_clone_prompt(self, text: str, lang_id: int, stream: bool, prompt_item: VoiceClonePromptItem) -> PromptData:
        """[声音克隆入口]"""
        return self._build_core(
            text,
            lang_id=lang_id,
            spk_emb=prompt_item.prompt_spk_embedding,
            ref_text=prompt_item.prompt_text,
            codes=prompt_item.prompt_code,
            icl=prompt_item.icl_mode,
            stream=stream
        )

    # ===================================================================
    # 统一核心构造器
    # ===================================================================

    def _build_core(
        self,
        text: str,
        lang_id: Optional[int] = None,
        spk_emb: Optional[np.ndarray] = None,
        spk_id: Optional[int] = None,
        instruct: Optional[str] = None,
        ref_text: Optional[str] = None,
        codes: Optional[np.ndarray] = None,
        icl: bool = False,
        stream: bool = False,
    ) -> PromptData:
        r"""
        [统一 Prompt 构造器]

        通过参数组合控制三种模式:

        ┌─────────────────────────────────────────────────────────────────┐
        │ 模式 A: 非流式 (codes=None, icl=False)                         │
        │   CustomVoice / VoiceDesign                                    │
        │   所有文本 + codec_pad 一次性放入 Prompt, trailing=None         │
        │                                                                │
        │ 模式 B: 流式 (codes=None, icl=True)                            │
        │   仅首个文本 token 进入 Prompt, 其余 + tts_eos 进 trailing      │
        │                                                                │
        │ 模式 C: ICL (codes=array)                                      │
        │   VoiceClone: 文本与参考音频按位融合, 超出部分进 trailing        │
        └─────────────────────────────────────────────────────────────────┘

        通用 Prefix 结构 (所有模式共享):
            [指令块]              (可选)
            <|im_start|>assistant\n
            tts_pad + think/nothink
            tts_pad + think_bos
           (tts_pad + lang_id)    (可选)
            tts_pad + think_eos
           (tts_pad + spk_emb)    (可选)
            tts_bos + codec_pad
        """
        
        t_start = time.time()
        
        if codes is None and icl:
            logger.error("❌ icl模式下，code不能为空")
            return None
        
        # 1. 文本 ID 构造
        if ref_text is not None and icl:
            text_ids = self._get_ids(ref_text + text)
        else:
            text_ids = self._get_ids(text)


        # 2. 前缀构造
        prefix = []
        text_pad_embedding = self.assets.tts_pad  # (t)
        hidden_dim = text_pad_embedding.shape[0]
        
        # 2a. 指令块
        if instruct:  # design模型
            ins_ids = self._get_ids(f"<|im_start|>user\n{instruct}<|im_end|>\n")
            for tid in ins_ids:
                prefix.append(self.assets.text_table[tid])
                
        # 2b. 角色头
        role_ids = self._get_ids("<|im_start|>assistant\n")
        for tid in role_ids:
            prefix.append(self.assets.text_table[tid])
            
        # 2c. 语言
        if lang_id is not None and lang_id in range(2048, 2147):
            prefill_ids = [self.special_tokens["THINK"], self.special_tokens["THINK_BOS"], lang_id, self.special_tokens["THINK_EOS"]]
        else:
            prefill_ids = [self.special_tokens["NOTHINK"], self.special_tokens["THINK_BOS"], self.special_tokens["THINK_EOS"]]
        
        for tid in prefill_ids:
            prefix.append(text_pad_embedding + self.assets.emb_tables[0][tid])

        # 2d. 说话人
        cur_spk_emb = None
        if spk_emb is not None:
            cur_spk_emb = spk_emb
            prefix.append(text_pad_embedding + spk_emb)  # shape (T)
        elif spk_id is not None:
            cur_spk_emb = self.assets.emb_tables[0][spk_id]
            prefix.append(text_pad_embedding + cur_spk_emb)
            
            
        # 2e. TTS_BOS + codec_pad
        prefix.append(
            self.assets.text_table[self.special_tokens["TTS_BOS"]] + self.assets.emb_tables[0][self.special_tokens["PAD"]]
        )
        
        # 3. Body 构造
        '''
        Custom Voice、 Design Voice 
        '''
        if stream:
            if icl: 
                text_pool_ids = text_ids + [self.special_tokens["TTS_EOS"]]
                text_embedding = self.assets.text_table[text_pool_ids]  #  text + tts_eos

                audio_vectors = [self.assets.emb_tables[0][self.special_tokens["BOS"]]]  #  codec_bos
                for t in range(codes.shape[0]):
                    step_sum = np.zeros(hidden_dim, dtype=np.float32)
                    for q in range(16):
                        step_sum += self.assets.emb_tables[q][codes[t, q]]
                    audio_vectors.append(step_sum)
                code_embedding = np.array(audio_vectors)
                
                t_len, a_len = len(text_pool_ids), len(code_embedding)
        
                if t_len > a_len:  # 文本长度 大于 音频长度
                    body = text_embedding[:a_len] + code_embedding
                    trailing = text_embedding[a_len:]
                else:
                    pad_seq = np.tile(text_pad_embedding, (a_len - t_len, 1))
                    text_padded = np.vstack([text_embedding, pad_seq])  # vtack堆叠
                    body = text_padded + code_embedding
                    trailing = None

            else:
                # Custom Voice voice design base xvec_only
                first_text_emb = self.assets.text_table[text_ids[0]]
                codec_bos_emb = self.assets.emb_tables[0][self.special_tokens["BOS"]]
                body = (first_text_emb + codec_bos_emb).reshape(1, -1)

                # trailing: 剩余文本 + tts_eos (纯文本嵌入, decode_step 中与音频融合)
                tts_eos_emb = self.assets.text_table[self.special_tokens["TTS_EOS"]]
                if len(text_ids) > 1:
                    remaining = self.assets.text_table[text_ids[1:]] if len(text_ids) > 2 else self.assets.text_table[text_ids[1]][np.newaxis, :]
                    trailing = np.vstack([remaining, tts_eos_emb])  # (1, t)
                else:
                    trailing = tts_eos_emb[np.newaxis, :]  # (1, t)
            
        else:
            # text_embedding + codec_pad
            text_embedding = np.vstack([self.assets.text_table[text_ids], self.assets.text_table[self.special_tokens["TTS_EOS"]]])   # ndarray
            text_embedding = text_embedding + self.assets.emb_tables[0][self.special_tokens["PAD"]] # (1, t)
            if icl:  # icl 下的非流式
                # # text_embedding + codec_pad
                # text_embedding = self.assets.text_table[text_ids] + [self.assets.text_table[self.special_tokens["TTS_EOS"]]]   # ndarray
                # text_embedding = text_embedding + self.assets.emb_tables[0][self.special_tokens["PAD"]] # (1, t)
                
                # text_pad  + codec_embedding
                audio_vectors = [self.assets.emb_tables[0][self.special_tokens["BOS"]]]  #  codec embedding
                for t in range(codes.shape[0]):  # codec -> [t, 16]
                    step_sum = np.zeros(hidden_dim, dtype=np.float32)
                    for q in range(16):
                        step_sum += self.assets.emb_tables[q][codes[t, q]]
                    audio_vectors.append(step_sum)
                
                code_embedding = np.array(audio_vectors)
                code_embedding = code_embedding + text_pad_embedding
                
                body = np.vstack([text_embedding, code_embedding]) # vstack 自动在第0行堆叠, 会自动处理1维和2维数组的堆叠
                trailing = None
                
            else:
                # Custom Voice voice design base xvec_only
                bos_codec = text_pad_embedding + self.assets.emb_tables[0][self.special_tokens["BOS"]] # (t)
                body = np.vstack([text_embedding, bos_codec])
                trailing = None
        

        # 拼接
        initial_prompt = np.vstack([np.array(prefix), body])[np.newaxis, :, :]  # (1, t, dim)
        trailing = trailing[np.newaxis, :, :] if trailing is not None else None

        return PromptData(
            embd=initial_prompt,
            text=text,
            # text_ids=target_text_ids,
            spk_emb=cur_spk_emb,
            trailing_text_embd=trailing,
            compile_time=time.time() - t_start,
        )
