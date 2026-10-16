from __future__ import annotations

from typing import Any

from ...text_processing import (
    TextProcessResult,
    TextPreprocessConfig,
    clean_text_by_allowed_pattern,
    remove_escape_chars,
    segment_text,
)

# VoxCPM2-GGUF 需要原样保护一组语气/拟声特殊 token。
# 这些 token 不能被允许字符过滤拆散，也不能在分段时从中间切开，
# 否则会破坏 tokenizer 对特殊发音控制片段的识别。
VOXCPM2_SPECIAL_TOKENS: tuple[str, ...] = (
    "[laughing]",
    "[sigh]",
    "[Uhm]",
    "[Shh]",
    "[Question-ah]",
    "[Question-ei]",
    "[Question-en]",
    "[Question-oh]",
    "[Surprise-wa]",
    "[Surprise-yo]",
    "[Dissatisfaction-hnn]",
)

# 这个模型的文本层职责非常收敛：
# 1. 统一移除控制字符与转义字符
# 2. 按模型支持语言集合保留有效字符，并原样保护特殊 token
# 3. 使用通用 segment_text 做长文本分段
VOXCPM2_DEFAULT_SENTENCE_DELIMITERS: tuple[str, ...] = (
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

# 多语种长句二次切分时，同时覆盖中日韩、西文和阿拉伯语常见停顿符。
VOXCPM2_DEFAULT_CLAUSE_DELIMITERS: tuple[str, ...] = (
    "，",
    "、",
    "；",
    "：",
    ",",
    ";",
    ":",
    "،",
    "؛",
)

VOXCPM2_TEXT_PREPROCESS_CONFIG = TextPreprocessConfig(
    special_tokens=VOXCPM2_SPECIAL_TOKENS,
    sentence_delimiters=VOXCPM2_DEFAULT_SENTENCE_DELIMITERS,
    clause_delimiters=VOXCPM2_DEFAULT_CLAUSE_DELIMITERS,
)

# VoxCPM2-GGUF 支持语言覆盖：
# - 亚洲：中文、日语、韩语、印地语、泰语、越南语、印尼语、马来语、他加禄语、高棉语、老挝语、缅甸语
# - 欧洲与美洲：英语、法语、德语、西班牙语、意大利语、俄语、葡萄牙语、荷兰语、波兰语、瑞典语、丹麦语、芬兰语、挪威语、希腊语
# - 中东与非洲：阿拉伯语、土耳其语、希伯来语、斯瓦希里语
# 这个正则只表达“允许保留哪些字符”，不做语言检测。
VOXCPM2_ALLOWED_TEXT_PATTERN = (
    r"[\u4e00-\u9fff\u3400-\u4dbf"
    r"\u3040-\u30ff\u31f0-\u31ff\uff65-\uff9f"
    r"\uac00-\ud7af\u1100-\u11ff\u3130-\u318f"
    r"\u0900-\u097f"
    r"\u0e00-\u0e7f"
    r"\u1780-\u17ff"
    r"\u0e80-\u0eff"
    r"\u1000-\u109f\uaa60-\uaa7f"
    r"\u0400-\u04ff"
    r"\u0370-\u03ff"
    r"\u0590-\u05ff"
    r"\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
    r"A-Za-z"
    r"À-ÿ\u0100-\u024f\u1e00-\u1eff"
    r"0-9\s"
    r".,!?;:'\"\\-…"
    r"。，！？；：、—“”‘’"
    r"؟،؛।]"
)


def preprocess_text(
    text: str,
    tokenizer: Any,
    language: str | None = None,
    max_length: int = 300,
    min_length: int = 20,
    config: TextPreprocessConfig | None = None,
    remove_meaningless_chars: bool = False,
) -> TextProcessResult:
    """
    VoxCPM2-GGUF 专用文本预处理入口。

    参数:
        text: 用户输入的待合成文本。
        tokenizer: 模型 tokenizer；用于让 `segment_text` 按真实上下文长度切分。
        language: 调用方显式传入的语言标识；VoxCPM2 文本层不做自动检测。
        max_length: 单段最大长度阈值。
        min_length: 短段合并下限。
        config: 可选模型级预处理配置；为空时使用 VoxCPM2 默认配置。
        remove_meaningless_chars: 是否按 VoxCPM2 允许字符规则清理无语义字符。

    返回:
        返回统一的 `TextProcessResult`，其中 `segments` 可直接送入 runtime 逐段合成。

    边界异常:
        当 `max_length/min_length` 非法时抛出 `ValueError`；
        当输入文本为空或清洗后为空时返回空分段结果。
    """

    # 配置为空时使用模型内置规则，确保 runtime 和独立调用脚本行为一致。
    active_config = config or VOXCPM2_TEXT_PREPROCESS_CONFIG

    # 分段上限必须为正数，否则长文本切分无法向前推进。
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    # 最小合并长度不能为负数；负数没有业务意义，也会让短段合并逻辑失真。
    if min_length < 0:
        raise ValueError("min_length must be greater than or equal to 0")

    # 空文本直接返回空结果，避免后续 tokenizer 和分段器走无意义路径。
    if not text or not text.strip():
        return TextProcessResult(
            segments=[],
            language=language or "auto",
            original_text=text or "",
            cleaned_text="",
        )

    # 保留原文用于日志和错误定位，避免清洗后丢失调用上下文。
    original_text = text

    # VoxCPM2 不做语言检测；文本层只透传调用方给出的 language。
    resolved_language = language or "auto"

    # 无论是否开启允许字符过滤，都先统一移除控制字符与转义字符，避免无效字符进入 tokenizer。
    text = remove_escape_chars(text)

    # 开启清理时，只保留模型支持语言集合中的正文、数字、空白和常用朗读标点。
    if remove_meaningless_chars:
        cleaned_text = clean_text_by_allowed_pattern(
            text=text,
            allowed_pattern=VOXCPM2_ALLOWED_TEXT_PATTERN,
            special_tokens=active_config.special_tokens,
            special_token_wrappers=active_config.special_token_wrappers,
        )
    else:
        # 关闭清理时只去除首尾空白，最大限度保留用户原始输入。
        cleaned_text = text.strip()

    # 清洗后为空说明文本不包含任何可合成内容，直接返回空分段交给上游统一报错。
    if not cleaned_text:
        return TextProcessResult(
            segments=[],
            language=resolved_language,
            original_text=original_text,
            cleaned_text="",
        )

    # 这里只复用通用分段器；特殊 token 保护规则已经在模型配置里声明，
    # 因此这里不再额外分叉专用分段逻辑。
    segments = segment_text(
        text=cleaned_text,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        sentence_delimiters=active_config.sentence_delimiters,
        clause_delimiters=active_config.clause_delimiters,
        special_tokens=active_config.special_tokens,
        special_token_wrappers=active_config.special_token_wrappers,
    )

    # 返回统一结果对象，保持与 qwen3/s2-pro 的调用形状一致。
    return TextProcessResult(
        segments=segments,
        language=resolved_language,
        original_text=original_text,
        cleaned_text=cleaned_text,
    )
