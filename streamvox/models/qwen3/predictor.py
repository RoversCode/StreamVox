#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predictor.py
@Time    :   2026/03/07 14:35:40
@Author  :   RoversCode
@Version :   1.0
@Desc    :   None
"""

# here put the import lib
import numpy as np
from . import llama


class Predictor:
    """
    封装工匠模型 (Predictor) 的推理行为。

    负责单帧 Codec 详细码 (Q1-Q15) 的阶梯式推理。
    """

    def __init__(self, model, context, batch, assets):
        self.model = model
        self.ctx = context
        self.batch = batch
        self.assets = assets

    def predict_frame(
        self,
        master_hidden: np.ndarray,
        code_0: int,
        sampler=None,
        do_sample=True,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
    ) -> tuple:
        """
        为给定的 Master 隐层状态和第一个码，生成完整的 16 个码。
        """
        # 1. 准备输入：Master Hidden + Code_0 Embedding
        proj = self.assets.proj
        if proj is not None:
            # 1.7B 模式：从 2048 投影到 1024
            m_h_1024 = master_hidden @ proj["weight"].T + proj["bias"]
        else:
            # 0.6B 模式：本身就是 1024
            m_h_1024 = master_hidden

        step_codes = [code_0]
        # 获取原始维度 Embedding (0.6B 是 1024, 1.7B 是 2048)
        step_embeds_raw = [self.assets.get_codec_embedding(0, code_0).copy()]

        c_in = np.stack(
            [m_h_1024, self.assets.get_codec_embedding_1024(0, code_0)], axis=0
        )

        # 2. 清理工匠的记忆
        self.ctx.clear_kv_cache()

        # 3. Prefill 工匠
        self.batch.set_embd(c_in, pos=0, seq_id=0)
        self.ctx.decode(self.batch)

        # 使用外部传入的采样器，或者退化为临时创建
        local_sampler = False
        if sampler is None:
            local_sampler = True
            if do_sample:
                sampler = llama.LlamaSampler(
                    temperature=temperature, top_p=top_p, top_k=top_k
                )
            else:
                sampler = llama.LlamaSampler(temperature=0)

        try:
            # 4. 阶梯式生成 Q1 -> Q15
            # 4. 阶梯式生成 Q1 -> Q15
            for cs in range(1, 16):
                # 利用原生采样器的 range limit 功能直接在全量 Logits 上采样
                # 偏移量算法：(cs-1) * 2048
                start_offset = (cs - 1) * 2048
                end_offset = cs * 2048

                # sample 返回的是 absolute token id
                # 我们需要通过 limit_start/end 限制采样范围
                token_id = sampler.sample(
                    self.ctx, limit_start=start_offset, limit_end=end_offset
                )

                # 将 absolute token id 转换为相对 code
                c = token_id - start_offset

                step_codes.append(c)
                step_embeds_raw.append(self.assets.get_codec_embedding(cs, c).copy())

                if cs < 15:
                    emb_1024 = self.assets.get_codec_embedding_1024(cs, c)
                    if emb_1024.ndim == 1:
                        emb_1024 = emb_1024.reshape(1, -1)

                    self.batch.set_embd(emb_1024, pos=cs + 1, seq_id=0)
                    self.ctx.decode(self.batch)

        finally:
            if local_sampler:
                sampler.free()

        return step_codes, step_embeds_raw
