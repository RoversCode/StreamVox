"""按 Breeze 官方分段边界组装文本、参考音频与指令。"""

from __future__ import annotations

import numpy as np

from .ort import as_numpy
from .prompt import BreezePromptAsset


# 只复用最近一段参考转录的编码，避免角色切换使处理器内存持续增长。
REFERENCE_CACHE_LIMIT_BYTES = 16 * 1024 * 1024


def _tokenize_segment(tokenizer: object, text: str) -> list[int]:
    """按官方特殊符号边界解码并重新编码单个文本段。

    Args:
        tokenizer: 原检查点的分词器。
        text: 已添加说话人和可选指令标记的文本。
    Returns:
        保留官方空格和特殊符号边界的编号。
    """
    rendered = tokenizer.decode(tokenizer.encode(text, add_special_tokens=True).ids, skip_special_tokens=False)
    return tokenizer.encode(rendered, add_special_tokens=False).ids


def _encode_reference(
    text_encoder: object,
    ids: np.ndarray,
    reference_cache: dict[tuple[int, ...], np.ndarray] | None,
) -> np.ndarray:
    """只复用最近一段参考转录，并限制缓存持有的数组大小。

    Args:
        text_encoder: 文本编码图。
        ids: 单段参考转录编号。
        reference_cache: 当前运行时的有界参考转录缓存。
    Returns:
        参考转录的独立编码结果。
    """
    key = tuple(ids[0].tolist())
    if reference_cache is not None and key in reference_cache:
        return reference_cache[key]
    result = text_encoder.run({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    embeddings = as_numpy(result["embeddings"])
    if reference_cache is None:
        return embeddings
    reference_cache.clear()
    if embeddings.nbytes <= REFERENCE_CACHE_LIMIT_BYTES:
        reference_cache[key] = embeddings
    return embeddings


def build_reference_prefix(
    tokenizer: object,
    text_encoder: object,
    audio_embedding: object,
    *,
    prompt: BreezePromptAsset,
    hidden_size: int,
    eos_code: int,
    num_codebooks: int,
    reference_cache: dict[tuple[int, ...], np.ndarray] | None = None,
) -> np.ndarray | None:
    """组装可在声音准备阶段提前计算的完整参考前缀。

    Args:
        tokenizer: 原检查点的分词器。
        text_encoder: 原模型的文本编码图。
        audio_embedding: 音频码本嵌入图。
        prompt: 包含完整转录和音频编码的参考资产。
        hidden_size: 主干模型的隐藏维度。
        eos_code: 参考音频结束帧的码本符号。
        num_codebooks: 每帧码本数。
        reference_cache: 当前运行时最近一段参考文本的编码缓存。
    Returns:
        单精度参考转录和含结束帧音频的嵌入，无参考音频时为空值。
    """
    if not prompt.audio_codes.size:
        return None
    reference_ids = np.asarray([_tokenize_segment(tokenizer, f"[S0]{prompt.prompt_text}")], dtype=np.int64)
    reference_embeddings = _encode_reference(text_encoder, reference_ids, reference_cache)
    codes = np.concatenate((prompt.audio_codes, np.full((1, num_codebooks), eos_code, dtype=np.int64)))
    audio_embeddings = as_numpy(audio_embedding.run({"codes": codes[None]})["embeddings"])
    prefix = np.concatenate((reference_embeddings, audio_embeddings), axis=1).astype(np.float32, copy=False)
    if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[2] != hidden_size:
        raise ValueError("Breeze 文本或音频嵌入维度与主干不匹配。")
    return prefix


def build_prompt_embeddings(
    tokenizer: object,
    text_encoder: object,
    audio_embedding: object,
    *,
    text: str,
    prompt: BreezePromptAsset,
    instructions: list[str],
    hidden_size: int,
    eos_code: int,
    num_codebooks: int,
    reference_cache: dict[tuple[int, ...], np.ndarray] | None = None,
) -> tuple[list[np.ndarray], int]:
    """共享请求内参考前缀，并将各目标文本作为独立批次行编码。

    Args:
        tokenizer: 原检查点的分词器。
        text_encoder: 原模型的文本编码图。
        audio_embedding: 音频码本嵌入图。
        text: 原始目标文本。
        prompt: 可选的参考音频资产。
        instructions: 按分支排列的自然语言指令，空字符串表示无指令。
        hidden_size: 主干模型的隐藏维度。
        eos_code: 参考音频结束帧的码本符号。
        num_codebooks: 每帧码本数。
        reference_cache: 当前运行时最近一段参考文本的编码缓存。
    Returns:
        各分支完整嵌入，以及参考转录和含结束帧音频的总长度。
    """
    if not instructions:
        raise ValueError("Breeze 提示需要至少一个文本分支。")
    reference_prefix = build_reference_prefix(
        tokenizer, text_encoder, audio_embedding, prompt=prompt, hidden_size=hidden_size,
        eos_code=eos_code, num_codebooks=num_codebooks, reference_cache=reference_cache,
    )
    reference_parts = () if reference_prefix is None else (reference_prefix,)
    reference_length = 0 if reference_prefix is None else reference_prefix.shape[1]

    target_ids = []
    for instruction in instructions:
        target = f"[S0]{text}"
        if instruction:
            target = f"[S0]<ins_bos>{instruction}<ins_eos>{text}"
        target_ids.append(_tokenize_segment(tokenizer, target))
    lengths = [len(ids) for ids in target_ids]
    input_ids = np.zeros((len(target_ids), max(lengths)), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids)
    # 各分支独立从零开始，右侧填充仅屏蔽无效键，避免双向注意力跨段串扰。
    for index, ids in enumerate(target_ids):
        input_ids[index, : lengths[index]] = ids
        attention_mask[index, : lengths[index]] = 1
    result = text_encoder.run({"input_ids": input_ids, "attention_mask": attention_mask})
    targets = as_numpy(result["embeddings"])
    branches = []
    for index, length in enumerate(lengths):
        target_embeddings = targets[index : index + 1, :length]
        embeddings = np.concatenate((*reference_parts, target_embeddings), axis=1).astype(np.float32, copy=False)
        if embeddings.shape[2] != hidden_size:
            raise ValueError("Breeze 文本或音频嵌入维度与主干不匹配。")
        branches.append(embeddings)
    return branches, reference_length
