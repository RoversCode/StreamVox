from __future__ import annotations

from typing import Any

from ...text_processing import (
    TextProcessResult,
    TextPreprocessConfig,
    clean_text_by_allowed_pattern,
    remove_escape_chars,
    segment_text,
)


# MOSS-TTS-Nano 使用普通自然文本输入，不需要引入参考项目里的 WeText 或强语义归一化。
# 这里的配置只声明 StreamVox 公共分段器需要知道的句读边界。
MOSS_TTS_NANO_TEXT_PREPROCESS_CONFIG = TextPreprocessConfig(
    sentence_delimiters=("。", "！", "？", ".", "!", "?", "\n"),
    clause_delimiters=("，", "、", "；", "：", ",", ";", ":"),
)

# MOSS-TTS-Nano 的轻量清理规则：
# - 保留常见多语言可见字符、数字、空白和朗读标点；
# - 删除控制字符、零宽字符和明显没有朗读意义的 ASCII 符号；
# - 不做数字、日期、金额等语义展开，符合 StreamVox 当前文本处理边界。
MOSS_TTS_NANO_ALLOWED_TEXT_PATTERN = (
    r"(?:"
    r"[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f\u2060\ufeff!-/:-@\[-`{-~]"
    r"|[A-Za-z0-9\s.,!?;:'\"\\\-…]"
    r")"
)


def preprocess_text(
    text: str,
    tokenizer: Any,
    language: str | None = None,
    max_length: int = 80,
    min_length: int = 12,
    config: TextPreprocessConfig | None = None,
    remove_meaningless_chars: bool = False,
) -> TextProcessResult:
    """
    MOSS-TTS-Nano 专用文本预处理入口。

    参数:
        text: 用户输入的待合成文本。
        tokenizer: SentencePiece tokenizer；用于按真实文本 token 数做分段预算。
        language: 调用方显式语言标识；当前函数不做语言检测，仅透传。
        max_length: 单段最大 token 预算，默认使用 MOSS runtime 的 80 token。
        min_length: 短段合并下限，用于减少过碎的 ONNX 推理调用。
        config: 可选模型级分段配置；为空时使用 MOSS 默认配置。
        remove_meaningless_chars: 是否执行轻量允许字符过滤。

    返回:
        返回 TextProcessResult；segments 字段可直接交给 runtime 逐段合成。

    边界异常:
        max_length 小于等于 0 或 min_length 小于 0 时抛出 ValueError；空文本返回空分段。
    """

    # 分段上限必须为正数，否则强制切分逻辑无法保证向前推进。
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    # 最小合并长度不能为负数；负数会让短段合并判断失去业务意义。
    if min_length < 0:
        raise ValueError("min_length must be greater than or equal to 0")

    # 空文本无需进入 tokenizer；直接返回空结果，由 runtime 统一抛出更贴近合成语义的错误。
    if not text or not text.strip():
        return TextProcessResult(
            segments=[],
            language=language or "auto",
            original_text=text or "",
            cleaned_text="",
        )

    # 保留原文，方便后续日志或错误信息追踪用户真实输入。
    original_text = text

    # MOSS 文本层不做语言检测；language 只作为调用方传入的追踪字段。
    resolved_language = language or "auto"

    # 无论是否开启允许字符过滤，都先移除控制字符，避免不可见字符进入 SentencePiece。
    text = remove_escape_chars(text)

    # 开启清理时只执行“无意义字符”过滤；不做数字、日期、金额等文本归一化。
    active_config = config or MOSS_TTS_NANO_TEXT_PREPROCESS_CONFIG
    if remove_meaningless_chars:
        cleaned_text = clean_text_by_allowed_pattern(
            text=text,
            allowed_pattern=MOSS_TTS_NANO_ALLOWED_TEXT_PATTERN,
            special_tokens=active_config.special_tokens,
            special_token_wrappers=active_config.special_token_wrappers,
        )
    else:
        cleaned_text = text.strip()

    # 清理后为空说明输入没有可合成正文，直接返回空分段，避免后端收到非法 prompt。
    if not cleaned_text:
        return TextProcessResult(
            segments=[],
            language=resolved_language,
            original_text=original_text,
            cleaned_text="",
        )

    # 使用 StreamVox 公共分段器；长度度量交给 MOSS 的 SentencePiece tokenizer。
    segments = segment_text(
        text=cleaned_text,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min(min_length, max_length),
        sentence_delimiters=active_config.sentence_delimiters,
        clause_delimiters=active_config.clause_delimiters,
        special_tokens=active_config.special_tokens,
        special_token_wrappers=active_config.special_token_wrappers,
    )

    # 返回统一结果对象，让 runtime 与其它模型保持相同调用形状。
    return TextProcessResult(
        segments=segments,
        language=resolved_language,
        original_text=original_text,
        cleaned_text=cleaned_text,
    )
