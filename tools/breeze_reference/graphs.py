"""保留 Breeze 原始权重与公式的显式缓存导出图。

第三方来源：https://github.com/breezeblue-ai/breeze-tts ，
提交 daa6cb2efe5471f09369229f322d20bb956cb3c3；
其中 Breeze 解码器继承 Sesame 与 Hugging Face 的 Apache-2.0 实现。
本模块复用其层与归一化权重，仅把缓存容器改成 ONNX 张量接口。
"""

from __future__ import annotations

import torch
from torch import nn

from .normalization import rms_norm


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    """将旋转位置编码的两个半区交换并改变符号。

    Args:
        value: 最后一维由两个等长半区组成的张量。

    Returns:
        与输入形状相同的旋转张量。
    """
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _attention(
    attention: nn.Module,
    hidden: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    mask: torch.Tensor,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按上游公式计算注意力，并在设备内拼接键值缓存。

    Args:
        attention: 持有原始投影与可选查询、键归一化的注意力层。
        hidden: 当前输入，形状为批次、长度、隐藏维度。
        cosine: 当前绝对位置的旋转余弦。
        sine: 当前绝对位置的旋转正弦。
        mask: 显式因果加法掩码。
        past_key: 历史键，形状为批次、键值头数、历史长度、头维度。
        past_value: 历史值，与历史键形状相同。

    Returns:
        注意力输出以及包含当前输入的新键、新值。
    """
    batch, length = hidden.shape[:2]
    dimension = attention.head_dim
    query = attention.q_proj(hidden).reshape(batch, length, -1, dimension)
    key = attention.k_proj(hidden).reshape(batch, length, -1, dimension)
    value = attention.v_proj(hidden).reshape(batch, length, -1, dimension)
    # Qwen3 的查询、键归一化属于模型公式，不能作为推理优化省略。
    if hasattr(attention, "q_norm"):
        query = rms_norm(attention.q_norm, query)
        key = rms_norm(attention.k_norm, key)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    query = query * cosine + _rotate_half(query) * sine
    key = key * cosine + _rotate_half(key) * sine
    key = torch.cat((past_key, key), dim=2)
    value = torch.cat((past_value, value), dim=2)
    groups = attention.num_key_value_groups
    # 分组查询通过广播共享键和值，避免预填充时物化重复的完整历史缓存。
    key_heads = attention.k_proj.out_features // dimension
    query = query.reshape(batch, key_heads, groups, length, dimension)
    scores = torch.matmul(query, key.unsqueeze(2).transpose(-2, -1))
    # Python 浮点标量使用 FP32 运算后恢复 BF16，不能先把缩放常量舍入为 BF16。
    scores = (scores.float() * attention.scaling).to(scores.dtype)
    # 先显式提升输入，避免旧导出器把 dtype 参数误实现为 Softmax 之后的转换。
    probabilities = torch.softmax((scores + mask.unsqueeze(2)).float(), dim=-1).to(query.dtype)
    output = torch.matmul(probabilities, value.unsqueeze(2))
    output = output.reshape(batch, key_heads * groups, length, dimension)
    output = output.transpose(1, 2).reshape(batch, length, -1)
    return attention.o_proj(output), key, value


def _decode(
    model: nn.Module, hidden: torch.Tensor, past: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """执行支持空缓存、多词预填充和单词续写的因果解码。

    Args:
        model: 原始解码器，提供层、最终归一化和旋转位置编码。
        hidden: 本次输入的嵌入。
        past: 按层交替排列的历史键和值。

    Returns:
        完整当前隐藏状态，以及按层交替排列的新键和值。
    """
    previous = past[0].shape[2]
    length = hidden.shape[1]
    positions = torch.arange(previous, previous + length, device=hidden.device)
    cosine, sine = model.rotary_emb(hidden, positions.unsqueeze(0))
    key_positions = torch.arange(previous + length, device=hidden.device)
    allowed = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[None, None]
    present: list[torch.Tensor] = []
    for index, layer in enumerate(model.layers):
        normalized = rms_norm(layer.input_layernorm, hidden)
        attention, key, value = _attention(
            layer.self_attn,
            normalized,
            cosine,
            sine,
            mask,
            past[index * 2],
            past[index * 2 + 1],
        )
        hidden = hidden + attention
        normalized = rms_norm(layer.post_attention_layernorm, hidden)
        if hidden.dtype == torch.bfloat16:
            # 上游融合 SiLU 在 FP32 内部计算后只舍入一次；直接导出会增加 BF16 中间舍入。
            gate = layer.mlp.gate_proj(normalized)
            activation = torch.nn.functional.silu(gate.float()).to(gate.dtype)
            mlp_output = layer.mlp.down_proj(activation * layer.mlp.up_proj(normalized))
        else:
            mlp_output = layer.mlp(normalized)
        hidden = hidden + mlp_output
        present.extend((key, value))
    return (rms_norm(model.norm, hidden), *present)


class TextEncoderGraph(nn.Module):
    """保留上游 T5Gemma2 双向局部注意力的文本投影图。

    Args:
        encoder: 上游文本编码器，注意力实现必须设置为 eager。
        projection: 原始文本到主干空间的线性投影。
    """

    def __init__(self, encoder: nn.Module, projection: nn.Module):
        super().__init__()
        self.text_encoder = encoder
        self.text_encoder_proj = projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """将带填充掩码的文本编号转换为主干嵌入。

        Args:
            input_ids: 整数文本编号，形状为批次、长度。
            attention_mask: 整数有效词掩码，与编号形状相同。

        Returns:
            浮点嵌入，形状为批次、长度、主干隐藏维度。
        """
        result = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return self.text_encoder_proj(result.last_hidden_state)


class AudioEmbeddingGraph(nn.Module):
    """复用原始码本偏移与求和的音频嵌入图。

    Args:
        embedding: 原始 Breeze 音频嵌入组件。
    """

    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """将一批完整音频帧转换为主干嵌入。

        Args:
            codes: 形状为批次、帧数、码本数的整数编号。

        Returns:
            形状为批次、帧数、主干隐藏维度的浮点嵌入。
        """
        return self.embedding(codes)


class BackboneGraph(nn.Module):
    """输出下一帧首码本分数与显式缓存的主干图。

    Args:
        backbone: 保留原始层、归一化和位置编码的主干。
        head: 包含独立结束类别的原始输出投影。
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone_model = backbone
        self.lm_head = head

    def forward(self, embeddings: torch.Tensor, *past: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """执行本次输入并只投影最后一个位置。

        Args:
            embeddings: 本次文本或音频嵌入。
            past: 按层交替排列的历史键和值；初始历史长度为零。

        Returns:
            首码本分数、最后隐藏状态，以及每层的新键和值。
        """
        # 文本与音频图保持 FP32 接口，仅在主干入口恢复检查点的计算精度。
        dtype = self.lm_head.weight.dtype
        hidden, *present = _decode(self.backbone_model, embeddings.to(dtype), past)
        final = hidden[:, -1, :]
        return (self.lm_head(final).float(), final, *present)


class DepthGraph(nn.Module):
    """按帧独立重置缓存的残余码本解码图。

    Args:
        decoder: 带位置专用输出权重的原始深度解码器。
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.depth_decoder = decoder

    def forward(
        self,
        token_ids: torch.Tensor,
        backbone_hidden: torch.Tensor,
        position: torch.Tensor,
        *past: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """处理帧内一个位置并返回残余码本分数。

        Args:
            token_ids: 本次前一枚码本编号，形状为批次、一。
            backbone_hidden: 主干最后状态，仅在位置零注入。
            position: 长度为一的整数张量，取值零到码本数减一。
            past: 当前帧内按层交替排列的历史键和值。

        Returns:
            下一残余码本分数以及各层的新键和值；位置零分数不用采样。
        """
        model = self.depth_decoder.model
        codebook_index = torch.clamp(position - 1, min=0)
        embedded = model.embed_tokens(token_ids + codebook_index * model.vocab_size)
        condition = backbone_hidden
        if model.backbone_hidden_state_projector is not None:
            condition = model.backbone_hidden_state_projector(condition)
        embedded = torch.where((position == 0).reshape(1, 1, 1), condition[:, None], embedded)
        hidden, *present = _decode(model, model.inputs_embeds_projector(embedded), past)
        weight = self.depth_decoder.codebooks_head.weight[codebook_index[0]]
        logits = torch.matmul(hidden[:, -1, :], weight)
        return (logits.float(), *present)
