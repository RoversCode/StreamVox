from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from .tokenizer import InferenceTokenizer, SEMANTIC_TOKEN_TEMPLATE


EOS_TOKEN = "<|endoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
MODALITY_VOICE_TOKEN = "<|voice|>"


@dataclass
class PromptSegment:
    kind: str
    text: str | None = None
    codes: np.ndarray | None = None


@dataclass
class HistoryTurn:
    user_text: str
    assistant_codes: np.ndarray


@dataclass
class BuiltPrompt:
    values: np.ndarray
    segments: list[PromptSegment] = field(default_factory=list)


def split_text_by_speaker(text: str) -> list[str]:
    # 将整段多说话人文本切成 turn，保留每个 turn 开头的 <|speaker:X|> 标签。
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)
    turns: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turns.append((part + parts[i + 1]).strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1
    return turns


def build_text_values(
    tokenizer: InferenceTokenizer,
    num_codebooks: int,
    text: str,
) -> np.ndarray:
    """
    将普通文本 token 转成 S2-Pro slow runtime 使用的 values 矩阵。

    参数:
        tokenizer: S2-Pro 推理 tokenizer。
        num_codebooks: codec codebook 数量。
        text: 需要写入 slow cache 的普通文本。

    返回:
        返回形状为 (num_codebooks + 1, T) 的 values 矩阵。

    异常:
        tokenizer 编码失败时会透出底层异常。
    """
    # 普通文本只占第 0 行主 token 流，codebook 行全部填 0，与 build_turn_prompt 的文本段一致。
    tokens = np.asarray(tokenizer.encode(text), dtype=np.int64)
    values = np.zeros((num_codebooks + 1, len(tokens)), dtype=np.int64)
    values[0] = tokens
    return values


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 5, max_bytes: int = 300
) -> list[str]:
    # 对齐官方 generate_long 的思路：
    # 既限制一个 batch 内说话人数，也限制 UTF-8 字节数。
    batches: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))
        if current and (len(current) >= max_speakers or current_bytes + turn_bytes > max_bytes):
            batches.append("\n".join(current))
            current = [turn]
            current_bytes = turn_bytes
        else:
            current.append(turn)
            current_bytes += turn_bytes
    if current:
        batches.append("\n".join(current))
    return batches


def normalize_prompt_inputs(
    prompt_text: list[str] | None,
    prompt_tokens: list[np.ndarray] | None,
) -> tuple[list[str] | None, list[np.ndarray] | None]:
    # 参考文本没有 speaker tag 时，这里补成 <|speaker:i|>，保证文本与参考 token 对齐。
    if not prompt_text or not prompt_tokens:
        return None, None
    tagged: list[str] = []
    for i, text in enumerate(prompt_text):
        if re.search(r"<\|speaker:\d+\|>", text):
            tagged.append(text)
        else:
            tagged.append(f"<|speaker:{i}|>{text}")
    return tagged, prompt_tokens


def build_batches(text: str, chunk_length: int) -> list[str]:
    turns = split_text_by_speaker(text)
    if not turns:
        return [text]
    return group_turns_into_batches(turns, max_speakers=5, max_bytes=chunk_length)


def build_prompt(
    tokenizer: InferenceTokenizer,
    num_codebooks: int,
    batch_text: str,
    history: list[HistoryTurn],
    prompt_text: list[str] | None = None,
    prompt_tokens: list[np.ndarray] | None = None,
) -> BuiltPrompt:
    # 这里不照搬官方 Conversation / ContentSequence 的复杂抽象，
    # 只围绕当前推理场景组织最小 prompt 结构：
    # 1. system 里的参考文本 / 参考 speech token
    # 2. 历史 user / assistant speech
    # 3. 当前 batch 的 user + 待生成 assistant 前缀
    prompt_text, prompt_tokens = normalize_prompt_inputs(prompt_text, prompt_tokens)

    segments: list[PromptSegment] = []
    if prompt_text and prompt_tokens:  # prompt部分的序列构建
        segments.append(
            PromptSegment(
                kind="system_prefix",
                text=f"{IM_START_TOKEN}system\n",
            )
        )
        segments.append(
            PromptSegment(
                kind="system_text_intro",
                text="convert the provided text to speech reference to the following:\n\nText:\n",
            )
        )
        segments.append(
            PromptSegment(
                kind="system_reference_text",
                text="\n".join(prompt_text),
            )
        )
        segments.append(
            PromptSegment(
                kind="system_speech_intro",
                text="\n\nSpeech:\n",
            )
        )
        merged = np.concatenate(prompt_tokens, axis=1)
        segments.append(PromptSegment(kind="system_vq", codes=merged))
        segments.append(PromptSegment(kind="system_end", text=IM_END_TOKEN + "\n")) # prompt部分拼接结束
    else:
        segments.append(
            PromptSegment(
                kind="system_text",
                text=f"{IM_START_TOKEN}system\nconvert the provided text to speech{IM_END_TOKEN}\n",
            )
        )

    for turn in history:  # 拼接之前的历史上下文
        segments.append(
            PromptSegment(
                kind="history_user",
                text=f"{IM_START_TOKEN}user\n{turn.user_text}{IM_END_TOKEN}\n",
            )
        )
        segments.append(
            PromptSegment(
                kind="history_assistant_prefix",
                text=f"{IM_START_TOKEN}assistant\n{MODALITY_VOICE_TOKEN}",
            )
        )
        segments.append(PromptSegment(kind="history_assistant_vq", codes=turn.assistant_codes))
        segments.append(PromptSegment(kind="history_assistant_end", text=IM_END_TOKEN + "\n"))

    # 本次要生成的speech token的文本
    segments.append(
        PromptSegment(
            kind="user_text",
            text=f"{IM_START_TOKEN}user\n{batch_text}{IM_END_TOKEN}\n",
        )
    )
    segments.append(
        PromptSegment(
            kind="assistant_prefix",
            text=f"{IM_START_TOKEN}assistant\n{MODALITY_VOICE_TOKEN}",
        )
    )

    row0: list[int] = []
    codebooks: list[np.ndarray] = []
    for segment in segments:
        if segment.text is not None:
            # 普通文本段只写入主 token 流，10 层 codebook 全补 0。
            tokens = np.asarray(tokenizer.encode(segment.text), dtype=np.int64)
            row0.extend(tokens.tolist())
            codebooks.append(np.zeros((num_codebooks, len(tokens)), dtype=np.int64))
            continue
        if segment.codes is None:
            continue
        codes = np.asarray(segment.codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != num_codebooks:
            raise ValueError(f"Expected codes shape ({num_codebooks}, T), got {codes.shape}")
        # speech token 段在主流里用 semantic token，占位到 values[0]；
        # 其余 10 层 acoustic codebook 写到 values[1:]。
        semantic_tokens = codes[0] + tokenizer.semantic_begin_id
        row0.extend(semantic_tokens.tolist())
        codebooks.append(codes)

    # 最终 prompt 形状固定为 (num_codebooks + 1, T)：
    # 第 0 行是主 token 流，第 1~10 行是 acoustic codebook。
    values = np.zeros((num_codebooks + 1, len(row0)), dtype=np.int64)
    values[0] = np.asarray(row0, dtype=np.int64)
    if codebooks:
        values[1:] = np.concatenate(codebooks, axis=1) # values -> [11, len(row0)]
    return BuiltPrompt(values=values, segments=segments)


def build_role_prefix_prompt(
    tokenizer: InferenceTokenizer,
    num_codebooks: int,
    prompt_text: list[str],
    prompt_tokens: list[np.ndarray],
) -> BuiltPrompt:
    '''
        "<|im_start|>" 【role (system） 】+ '\n' + 【modality_token】 +  "convert the provided text to speech reference to the following:\n\nText:\n"
        + 【 <|speaker: x|> + ref text】  + 【\n\nSpeech:\n'】 + 【speech token】
        + "<|im_end|>"+ '\n' 
        
    '''
    # prompt_text添加了 <|speaker: x|>
    normalized_text, normalized_tokens = normalize_prompt_inputs(prompt_text, prompt_tokens)
    if not normalized_text or not normalized_tokens:
        raise ValueError("prompt_text and prompt_tokens are required")
    segments = [
        PromptSegment(
            kind="system_prefix",
            text=f"{IM_START_TOKEN}system\n",
        ),
        PromptSegment(
            kind="system_text_intro",
            text="convert the provided text to speech reference to the following:\n\nText:\n",
        ),
        PromptSegment(
            kind="system_reference_text",
            text="\n".join(normalized_text),
        ),
        PromptSegment(
            kind="system_speech_intro",
            text="\n\nSpeech:\n",
        ),
        PromptSegment(
            kind="system_vq",
            codes=np.concatenate(normalized_tokens, axis=1),
        ),
        PromptSegment(
            kind="system_end",
            text=IM_END_TOKEN + "\n",
        ),
    ]
    row0: list[int] = []
    codebooks: list[np.ndarray] = []
    for segment in segments:
        if segment.text is not None:
            tokens = np.asarray(tokenizer.encode(segment.text), dtype=np.int64)
            row0.extend(tokens.tolist())
            codebooks.append(np.zeros((num_codebooks, len(tokens)), dtype=np.int64))
            continue
        if segment.codes is None:
            continue
        codes = np.asarray(segment.codes, dtype=np.int64)
        semantic_tokens = codes[0] + tokenizer.semantic_begin_id
        row0.extend(semantic_tokens.tolist())
        codebooks.append(codes)

    values = np.zeros((num_codebooks + 1, len(row0)), dtype=np.int64)
    values[0] = np.asarray(row0, dtype=np.int64)
    if codebooks:
        values[1:] = np.concatenate(codebooks, axis=1)
    return BuiltPrompt(values=values, segments=segments)


def build_turn_prompt(
    tokenizer: InferenceTokenizer,
    num_codebooks: int,
    batch_text: str,
) -> BuiltPrompt:
    segments = [
        PromptSegment(
            kind="user_text",
            text=f"{IM_START_TOKEN}user\n{batch_text}{IM_END_TOKEN}\n",
        ),
        PromptSegment(
            kind="assistant_prefix",
            text=f"{IM_START_TOKEN}assistant\n{MODALITY_VOICE_TOKEN}",
        ),
    ]
    row0: list[int] = []
    codebooks: list[np.ndarray] = []
    for segment in segments:
        tokens = np.asarray(tokenizer.encode(segment.text or ""), dtype=np.int64)
        row0.extend(tokens.tolist())
        codebooks.append(np.zeros((num_codebooks, len(tokens)), dtype=np.int64))

    values = np.zeros((num_codebooks + 1, len(row0)), dtype=np.int64)
    values[0] = np.asarray(row0, dtype=np.int64)
    if codebooks:
        values[1:] = np.concatenate(codebooks, axis=1)
    return BuiltPrompt(values=values, segments=segments)
