from __future__ import annotations

import re
from typing import Any

from ....utils.log import logger
from ...text_processing import (
    TextProcessResult,
    TextPreprocessConfig,
    clean_text_by_allowed_pattern,
    remove_escape_chars,
    clean_text as _clean_common_text,
    segment_text as _segment_common_text,
)


# S2-Pro speaker 标签正则：只接受官方形态 <|speaker:数字|>，避免把其它 <|...|> 控制 token 误判为说话人。
SPEAKER_TAG_RE = re.compile(r"<\|speaker:(\d+)\|>")

# S2-Pro 官方单次对话 batch 最多包含 5 个说话人；文本层负责在进入 runtime 前完成该约束。
S2_PRO_MAX_SPEAKERS_PER_BATCH = 5


# S2-Pro 默认句级切分符：覆盖中日韩标点、西文标点、阿拉伯问号等常见多语句末符号。
DEFAULT_SENTENCE_DELIMITERS: tuple[str, ...] = (
    "。",
    "！",
    "？",
    ".",
    "!",
    "?",
    "؟",
    "।",
    "\n",
)

# S2-Pro 默认子句切分符：长句超过上限时使用这些停顿符继续拆分，尽量保留自然语气边界。
DEFAULT_CLAUSE_DELIMITERS: tuple[str, ...] = (
    "，",
    "、",
    "；",
    "：",
    ",",
    ";",
    ":",
    "؛",
    "،",
)

# S2-Pro 默认特殊 token 包裹符：方括号用于自由风格描述，<|...|> 保持历史兼容。
DEFAULT_SPECIAL_TOKEN_WRAPPERS: tuple[tuple[str, str], ...] = (("[", "]"), ("<|", "|>"))

# S2-Pro 清理模式下的允许字符规则：
# - 非 ASCII 可见字符宽松保留，覆盖模型支持的大量语言文字；
# - ASCII 范围只保留字母、数字、空白和明确有朗读意义的标点；
# - <|...|> 和 [...] 等特殊 token 会先被保护，不受这里的 ASCII 符号限制。
S2_PRO_ALLOWED_TEXT_PATTERN = (
    r"(?:"
    r"[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f\u2060\ufeff!-/:-@\[-`{-~]"
    r"|[A-Za-z0-9\s.,!?;:'\"\\\-…]"
    r")"
)

# S2-Pro 文本预处理默认配置：这里声明模型能力和分段规则，实际清洗由本文件的 S2-Pro 专用逻辑负责。
S2_PRO_TEXT_PREPROCESS_CONFIG = TextPreprocessConfig(
    special_token_wrappers=DEFAULT_SPECIAL_TOKEN_WRAPPERS,
    sentence_delimiters=DEFAULT_SENTENCE_DELIMITERS,
    clause_delimiters=DEFAULT_CLAUSE_DELIMITERS,
)


def preprocess_text(
    text: str,
    tokenizer,
    language: str | None = None,
    max_length: int = 450,
    min_length: int = 40,
    config: TextPreprocessConfig | None = None,
    remove_meaningless_chars: bool = False,
) -> TextProcessResult:
    """
    S2-Pro 专用文本预处理入口。

    参数:
        text: 用户输入的待合成文本。
        language: 用户显式传入的语言标识；文本层不做自动检测。
        max_length: 单次 S2-Pro batch 的字节预算；默认 300，对齐官方单次 speech token 推理限制。
        min_length: 单段最小合并长度，用于减少过短碎片。
        tokenizer: 可选 tokenizer，用于更贴近模型上下文的长度度量。
        config: 可选模型级预处理配置；为空时使用 S2-Pro 默认配置。
        remove_meaningless_chars: 是否按 S2-Pro 允许字符正则清除无语义字符。

    返回:
        TextProcessResult，segments 字段直接包含 S2-Pro runtime 要消费的最终 batch。

    异常:
        当 max_length/min_length 非法时抛出 ValueError。
    """
    # 配置为空时使用 S2-Pro 内置能力声明，确保 runtime 和独立调用行为一致。
    active_config = config or S2_PRO_TEXT_PREPROCESS_CONFIG

    # 分段上限必须为正数，否则后续强制切分会无法向前推进。
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    # 最小长度不能为负数；负数没有业务意义，也会让短段合并判断失真。
    if min_length < 0:
        raise ValueError("min_length must be greater than or equal to 0")

    # 空文本不进入清洗和分段，直接返回空结果并记录可诊断日志。
    if not text or not text.strip():
        logger.warning("S2-Pro 输入文本为空")
        return TextProcessResult(
            segments=[],
            language=language or "auto",
            original_text=text or "",
            cleaned_text="",
        )

    # 保留原文用于结果追踪，避免调用方在报错或日志中丢失用户输入上下文。
    original_text = text

    # language 完全由调用方提供；文本层不做检测、不做语言专属处理。
    resolved_language = language or "auto"
    # 清除各种转义字符
    text = remove_escape_chars(text)

    # 根据入口开关决定是否清理无语义字符；默认保留原文，避免宽语言模型被过度清洗。
    if remove_meaningless_chars:
        # 开启清理时使用 S2-Pro 自己的允许字符正则，并保护方括号自由描述 token。
        cleaned_text = clean_text_by_allowed_pattern(
            text=text,
            allowed_pattern=S2_PRO_ALLOWED_TEXT_PATTERN,
            special_tokens=active_config.special_tokens,
            special_token_wrappers=_resolve_special_token_wrappers(active_config),
        )
    else:
        # 关闭清理时只去掉首尾空白，最大限度保留用户输入给 S2-Pro 自己处理。
        cleaned_text = text.strip()

    # 清洗后为空说明用户输入只有控制字符或不可见内容，继续推理没有意义。
    if not cleaned_text:
        logger.error("S2-Pro 有效文本为空")
        return TextProcessResult(
            segments=[],
            language=resolved_language,
            original_text=original_text,
            cleaned_text="",
        )

    # S2-Pro 官方约束是 300 bytes；这里保留按当前文本平均字节数折算出的预算，再交给 tokenizer 级逻辑使用。
    byte_aware_max_length = _resolve_byte_aware_max_length(cleaned_text, max_length)

    # 清洗后的文本如果没有 speaker 标签，按单说话人路径处理，并在每个最终 batch 前补 <|speaker:0|>。
    if SPEAKER_TAG_RE.search(cleaned_text) is None:
        # TODO: 后续可以判定当前的prompt是否是多说话人，是的话，才warning。
        logger.warning("S2-Pro 输入文本未包含 speaker 标签，已按单说话人 <|speaker:0|> 处理")
        segments = _build_single_speaker_batches(
            text=cleaned_text,
            max_length=byte_aware_max_length,
            min_length=min(min_length, byte_aware_max_length),
            tokenizer=tokenizer,
            config=active_config,
        )
    else:
        # 清洗后的文本包含 speaker 标签时，先按 speaker 粒度解析，再只拆分单个 speaker 的正文。
        segments = _build_multi_speaker_batches(
            text=cleaned_text,
            max_length=byte_aware_max_length,
            min_length=min(min_length, byte_aware_max_length),
            tokenizer=tokenizer,
            config=active_config,
        )

    # 记录最终 batch，便于排查长文本被切分后每批实际送入模型的内容。
    logger.info(f"S2-Pro 文本预处理完成: 语言={resolved_language}, batch数={len(segments)}")

    # 返回统一结果结构，保持与 qwen3 和通用 text_processing 的调用方兼容；S2-Pro 的 segments 即最终 batch。
    return TextProcessResult(
        segments=segments,
        language=resolved_language,
        original_text=original_text,
        cleaned_text=cleaned_text,
    )


def _resolve_byte_aware_max_length(text: str, max_length: int) -> int:
    """
    将 S2-Pro 的字节预算折算为当前文本的 token 级预算估算值。

    参数:
        text: 已清洗文本，用于估算当前输入的平均 UTF-8 字节密度。
        max_length: 用户或默认配置传入的单批字节预算，默认 300。

    返回:
        返回后续 tokenizer 级分段逻辑使用的预算估算值，至少为 1。

    异常:
        本函数不主动抛出异常；max_length 合法性由 preprocess_text 入口校验。
    """
    # 空文本在入口已被拦截；这里仍做防御，避免独立调用时除零。
    if not text:
        return max(1, max_length)

    # 计算当前文本每个 Unicode 字符平均占用的 UTF-8 字节数，用它保守估算 300 bytes 对应的预算。
    average_bytes_per_char = len(text.encode("utf-8")) / max(1, len(text))

    # 平均值至少按 1 处理，避免 ASCII 文本因为浮点异常被放大到超过字节预算。
    normalized_average = max(1.0, average_bytes_per_char)

    # 向下取整可以保守满足官方限制；至少保留 1，确保分段逻辑可以向前推进。
    
    return min(300, max(1, int(max_length / normalized_average)))


def _build_single_speaker_batches(
    text: str,
    max_length: int,
    min_length: int,
    tokenizer: Any,
    config: TextPreprocessConfig,
) -> list[str]:
    """
    构建 S2-Pro 单说话人 batch。

    参数:
        text: 不含 speaker 标签的清洗后正文。
        max_length: 已按平均字节数折算出的 token 级分段上限。
        min_length: token 级短段合并阈值，不能超过 max_length。
        tokenizer: 必传的 S2-Pro tokenizer，用于计算 speaker 标签和正文的真实 token 数。
        config: S2-Pro 文本预处理配置。

    返回:
        返回每段都带 <|speaker:0|> 前缀的 batch 列表。

    异常:
        通用分段器异常会自然透出。
    """
    # 单说话人路径固定使用 speaker 0；标签会进入最终 batch，因此正文 token 预算需要预留标签 token 数。
    speaker_tag = "<|speaker:0|>"
    speaker_tag_length = _encoded_length(speaker_tag, tokenizer)
    body_max_length = max(1, max_length - speaker_tag_length)

    # 当前正文没有超过剩余 token 预算时不做分段，避免短文本被无意义拆碎。
    if _encoded_length(text, tokenizer) <= body_max_length:
        body_segments = [text]
    else:
        # 只有正文 token 数超过预算时，才直接调用 segment_text 按 tokenizer 长度做分段。
        body_segments = segment_text(
            text=text,
            tokenizer=tokenizer,
            max_length=body_max_length,
            min_length=min(min_length, body_max_length),
            sentence_delimiters=config.sentence_delimiters,
            clause_delimiters=config.clause_delimiters,
            special_tokens=config.special_tokens,
            special_token_wrappers=_resolve_special_token_wrappers(config),
        )

    # 每个分段都补上 speaker 0，让任意长文本跨批次生成时仍保留说话人身份。
    return [f"{speaker_tag}{segment}" for segment in body_segments]


def _build_multi_speaker_batches(
    text: str,
    max_length: int,
    min_length: int,
    tokenizer: Any,
    config: TextPreprocessConfig,
) -> list[str]:
    """
    构建 S2-Pro 多说话人 batch。

    参数:
        text: 包含 speaker 标签的清洗后文本。
        max_length: 已按平均字节数折算出的 token 级分段上限。
        min_length: token 级短段合并阈值，不能超过 max_length。
        tokenizer: 必传的 S2-Pro tokenizer，用于计算每个 speaker 正文和标签的真实 token 数。
        config: S2-Pro 文本预处理配置。

    返回:
        返回按 speaker 粒度拆分并重新合并后的 batch 列表。

    异常:
        通用分段器异常会自然透出。
    """
    # 先把全文解析成按顺序排列的 (speaker标签, 正文) 列表，后续只允许拆正文，不拆 speaker 标签。
    speaker_turns = _split_text_into_speaker_turns(text)

    # tagged_segments 保存已经完成正文分段的最小 speaker 片段，元素形如 ("<|speaker:0|>", "正文")。
    tagged_segments: list[tuple[str, str]] = []

    # 遍历每个 speaker turn；只有当前 speaker 正文超过上限时才在该 speaker 内部继续切分。
    for speaker_tag, body_text in speaker_turns:
        # 空正文没有生成意义，跳过以避免构造只有 speaker 标签的 batch。
        if not body_text.strip():
            continue

        # speaker 标签会进入最终 batch，因此当前 speaker 正文 token 预算需要预留标签 token 数。
        speaker_tag_length = _encoded_length(speaker_tag, tokenizer)
        body_max_length = max(1, max_length - speaker_tag_length)

        # 当前 speaker 正文未超过剩余 token 预算时保持 speaker 粒度，不进入分段器。
        if _encoded_length(body_text, tokenizer) <= body_max_length:
            body_segments = [body_text]
        else:
            # 只有当前 speaker 正文 token 数超过预算时，才直接调用 segment_text 做内部切分。
            body_segments = segment_text(
                text=body_text,
                tokenizer=tokenizer,
                max_length=body_max_length,
                min_length=min(min_length, body_max_length),
                sentence_delimiters=config.sentence_delimiters,
                clause_delimiters=config.clause_delimiters,
                special_tokens=config.special_tokens,
                special_token_wrappers=_resolve_special_token_wrappers(config),
            )

        # 将每个正文片段和 speaker 标签成对保留，方便最后按顺序合并。
        tagged_segments.extend((speaker_tag, segment) for segment in body_segments)

    # 将 speaker 片段按顺序合并成尽量接近 max_length 的 batch，同时保留单批最多 5 个说话人的官方限制。
    return _merge_tagged_segments_into_batches(tagged_segments, max_length=max_length, tokenizer=tokenizer)


def _split_text_into_speaker_turns(text: str) -> list[tuple[str, str]]:
    """
    将包含 speaker 标签的文本解析成有序 speaker turn。

    参数:
        text: 至少包含一个 <|speaker:X|> 标签的文本。

    返回:
        返回 (speaker标签, 正文) 列表；标签前游离文本按 speaker 0 处理。

    异常:
        本函数不主动抛出异常；不完整控制 token 会作为普通正文保留。
    """
    # 使用 finditer 保留标签位置，便于准确截取每个标签后面的正文范围。
    matches = list(SPEAKER_TAG_RE.finditer(text))
    turns: list[tuple[str, str]] = []

    # 如果第一个 speaker 标签前存在正文，说明用户只给部分文本标注了 speaker；这段按 speaker 0 兜底。
    prefix = text[: matches[0].start()].strip() if matches else text.strip()
    if prefix:
        logger.warning("S2-Pro 多说话人文本存在未标注 speaker 的前置文本，已按 <|speaker:0|> 处理")
        turns.append(("<|speaker:0|>", prefix))

    # 按输入顺序遍历 speaker 标签，把当前标签到下一个标签之间的正文归属给当前 speaker。
    for index, match in enumerate(matches):
        speaker_tag = match.group(0)
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body_text = text[content_start:content_end].strip()

        # 空 turn 只代表用户写了标签但没有可朗读内容，跳过可避免无效 batch。
        if not body_text:
            continue

        # 保留原始 speaker 标签文本，避免重新格式化影响用户给定的说话人编号。
        turns.append((speaker_tag, body_text))

    return turns


def _merge_tagged_segments_into_batches(
    tagged_segments: list[tuple[str, str]],
    max_length: int,
    tokenizer: Any,
) -> list[str]:
    """
    将带 speaker 标签的正文片段合并为 S2-Pro batch。

    参数:
        tagged_segments: 已按正文上限切好的 (speaker标签, 正文) 列表。
        max_length: batch 内文本 tokenizer 编码长度上限。
        tokenizer: 必传的 S2-Pro tokenizer，用于合批时计算 batch 的真实 token 数。

    返回:
        返回最终 batch 列表，每个 batch 保留输入顺序并尽量逼近 max_length。

    异常:
        本函数不主动抛出异常。
    """
    batches: list[str] = []
    current_parts: list[str] = []
    current_text_length = 0
    current_speakers: set[str] = set()
    # 记录当前 batch 内最后一个片段的 speaker；连续相同 speaker 时后续片段不重复写标签。
    current_last_speaker: str | None = None

    # 顺序遍历最小 speaker 片段，尽量把多个片段合并进同一 batch。
    for speaker_tag, body_text in tagged_segments:
        # 如果当前 batch 里上一段就是同一个 speaker，则当前片段只追加正文，避免连续重复 speaker 标签干扰模型。
        part_text = body_text if current_last_speaker == speaker_tag else f"{speaker_tag}{body_text}"

        # part_text 是最终会送入模型的单个片段，合批预算按 tokenizer 编码长度计算。
        part_length = _encoded_length(part_text, tokenizer)
        next_speakers = set(current_speakers)
        next_speakers.add(speaker_tag)

        # 当前 batch 已有内容且继续加入会超过 token 上限或 5 speaker 上限时，先提交当前 batch。
        if current_parts and (
            current_text_length + part_length > max_length
            or len(next_speakers) > S2_PRO_MAX_SPEAKERS_PER_BATCH
        ):
            batches.append("\n".join(current_parts))
            current_parts = []
            current_text_length = 0
            current_speakers = set()
            current_last_speaker = None
            next_speakers = {speaker_tag}
            # 新 batch 的第一段必须重新写 speaker 标签，确保跨 batch 后说话人身份不丢失。
            part_text = f"{speaker_tag}{body_text}"
            part_length = _encoded_length(part_text, tokenizer)

        # 将当前片段加入 batch；同 speaker 连续片段可能只含正文，新 batch 第一段一定含 speaker 标签。
        current_parts.append(part_text)
        current_text_length += part_length
        current_speakers = next_speakers
        current_last_speaker = speaker_tag

    # 收尾提交最后一个 batch，避免末尾正文丢失。
    if current_parts:
        batches.append("\n".join(current_parts))

    return batches


def _encoded_length(text: str, tokenizer: Any) -> int:
    """
    计算文本在 S2-Pro tokenizer 下的编码长度。

    参数:
        text: 需要度量的文本。
        tokenizer: S2-Pro tokenizer，必须提供 encode 方法。

    返回:
        返回 tokenizer.encode(text) 的 token 数。

    异常:
        tokenizer 为空时抛出 ValueError；tokenizer.encode 失败时透出底层异常。
    """
    # S2-Pro 的 max_length/min_length 语义是 tokenizer 编码后的 token 数，不能用字符数替代。
    if tokenizer is None:
        raise ValueError("s2-pro preprocess_text requires tokenizer for token-length batching")

    # 直接使用 tokenizer.encode 的结果长度，确保分段判断和模型实际上下文消耗一致。
    return len(tokenizer.encode(text))


def clean_text(
    text: str,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = DEFAULT_SPECIAL_TOKEN_WRAPPERS,
) -> str:
    """
    清洗 S2-Pro 输入文本，同时保护特殊 token。

    参数:
        text: 原始输入文本。
        special_tokens: 需要按字面值保护的固定 token。
        special_token_wrappers: 需要按包裹符保护的自由描述 token。

    返回:
        返回清洗后的文本；方括号特殊 token 会原样保留。

    异常:
        本函数不主动抛出异常；异常输入会通过标准字符串处理自然暴露。
    """
    # S2-Pro 的清理策略现在复用通用工具；本函数只负责固定默认特殊 token 保护规则。
    return _clean_common_text(
        text=text,
        special_tokens=special_tokens,
        special_token_wrappers=special_token_wrappers,
    )


def segment_text(
    text: str,
    tokenizer,
    max_length: int = 300,
    min_length: int = 20,
    sentence_delimiters: tuple[str, ...] | None = None,
    clause_delimiters: tuple[str, ...] | None = None,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = DEFAULT_SPECIAL_TOKEN_WRAPPERS,
) -> list[str]:
    """
    按句级优先、子句兜底、强制切分兜底的顺序拆分 S2-Pro 文本。

    参数:
        text: 已清洗的输入文本。
        max_length: 单段最大长度。
        min_length: 单段最小合并长度。
        tokenizer: tokenizer，用于 token 级长度度量。
        sentence_delimiters: 可选句级切分符。
        clause_delimiters: 可选子句切分符。
        special_tokens: 需要保护的固定 token。
        special_token_wrappers: 需要保护的包裹型 token。

    返回:
        返回非空文本段列表，每段长度尽量不超过 max_length。

    异常:
        本函数不主动抛出异常；max_length 合法性由 preprocess_text 入口校验。
    """
    # S2-Pro 的分段策略复用通用工具；这里传入 S2-Pro 的多语标点和特殊 token 保护规则。
    return _segment_common_text(
        text=text,
        max_length=max_length,
        min_length=min_length,
        tokenizer=tokenizer,
        sentence_delimiters=sentence_delimiters or DEFAULT_SENTENCE_DELIMITERS,
        clause_delimiters=clause_delimiters or DEFAULT_CLAUSE_DELIMITERS,
        special_tokens=special_tokens,
        special_token_wrappers=special_token_wrappers,
    )


def _resolve_special_token_wrappers(config: TextPreprocessConfig) -> tuple[tuple[str, str], ...]:
    """
    合并配置中的特殊 token 包裹符和 S2-Pro 默认包裹符。

    参数:
        config: 当前文本预处理配置。

    返回:
        去重后的包裹符列表。

    异常:
        本函数不主动抛出异常；空包裹符会被忽略。
    """
    # 使用列表保持顺序，先尊重外部配置，再补充 S2-Pro 必需的方括号规则。
    wrappers: list[tuple[str, str]] = []

    # 遍历外部配置与默认配置，确保用户配置不会意外关闭方括号特殊 token 保护。
    for open_token, close_token in (*config.special_token_wrappers, *DEFAULT_SPECIAL_TOKEN_WRAPPERS):
        # 空包裹符没有匹配意义，跳过以避免生成异常正则。
        if not open_token or not close_token:
            continue

        # 已存在的包裹符不重复加入，保持正则分支数量最小。
        if (open_token, close_token) not in wrappers:
            wrappers.append((open_token, close_token))

    # 返回不可变元组，避免调用方意外修改内部规则。
    return tuple(wrappers)
