from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# 通用句级分隔符：覆盖 StreamVox 当前模型共用的常见句末停顿，模型可以在自己的配置中覆盖。
DEFAULT_SENTENCE_DELIMITERS: tuple[str, ...] = ("。", "！", "？", ".", "!", "?", "\n")

# 通用子句级分隔符：当句子超过长度上限时使用这些停顿符继续拆分。
DEFAULT_CLAUSE_DELIMITERS: tuple[str, ...] = ("，", "、", "；", "：", ",", ";", ":")

# 通用重复标点压缩集合：只处理明确表达同一停顿或语气的连续重复符号。
REPEATED_PUNCTUATION_PATTERN = r"([。.！!？?，,；;：:、…؟؛،।])\1+"

# NFKC 会把这些中文/全角朗读标点改写为半角符号；TTS 文本清理必须保留用户原始停顿形态。
PRESERVED_NFKC_FULLWIDTH_PUNCTUATION: frozenset[str] = frozenset("，：！？；")

# 通用转义字符别名表：同时支持调用方传入实际字符 "\n"，也支持传入字面量 "\\n"。
ESCAPE_CHAR_ALIASES: dict[str, str] = {
    "\\0": "\0",
    "\\a": "\a",
    "\\b": "\b",
    "\\f": "\f",
    "\\n": "\n",
    "\\r": "\r",
    "\\t": "\t",
    "\\v": "\v",
}

# 默认移除的控制字符范围：覆盖常见转义字符和 ASCII DEL，保留策略由 remove_escape_chars 的参数决定。
DEFAULT_REMOVABLE_ESCAPE_CHARS: frozenset[str] = frozenset(
    chr(codepoint) for codepoint in (*range(0x00, 0x20), 0x7F)
)


@dataclass(frozen=True, slots=True)
class TextPreprocessConfig:
    """
    文本工具配置容器。

    注意：
    - 这里不表达模型支持语言；
    - 这里不做语言检测；
    - 这里只承载公共清理和分段函数需要的规则参数。
    """

    special_tokens: tuple[str, ...] = ()
    special_token_wrappers: tuple[tuple[str, str], ...] = ()
    sentence_delimiters: tuple[str, ...] | None = None
    clause_delimiters: tuple[str, ...] | None = None


@dataclass(slots=True)
class TextProcessResult:
    """文本预处理结果容器"""

    segments: list[str] = field(default_factory=list)
    language: str = "auto"
    original_text: str = ""
    cleaned_text: str = ""


def remove_escape_chars(
    text: str,
    keep_escape_chars: Iterable[str] | None = None,
) -> str:
    """
    清除文本中的通用转义/控制字符，并允许调用方保留指定字符。

    参数:
        text: 需要清理的原始文本。
        keep_escape_chars: 允许保留的转义字符列表；既可以传实际字符如 "\n"，也可以传字面量如 "\\n"。

    返回:
        返回移除未保留转义字符后的文本。

    异常:
        当 text 不是字符串时抛出 TypeError。
    """
    # 入口类型必须明确，否则后续逐字符遍历会把非字符串容器误当文本处理。
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # 将保留列表统一规范化成实际字符集合，避免调用方在 "\n" 和 "\\n" 两种写法之间踩坑。
    keep_chars = _normalize_escape_keep_chars(keep_escape_chars)

    # cleaned_chars 按原始顺序收集允许保留的字符，保证清理操作只删除字符，不重排文本。
    cleaned_chars: list[str] = []

    # 逐字符扫描文本；控制字符且不在保留列表时删除，其它正常字符原样保留。
    for char in text:
        # 当前字符属于通用转义/控制字符，且调用方没有显式保留时，跳过该字符。
        if char in DEFAULT_REMOVABLE_ESCAPE_CHARS and char not in keep_chars:
            continue

        # 普通字符或被保留的转义字符都进入输出缓冲。
        cleaned_chars.append(char)

    # 合并清理后的字符，返回与输入同类型的普通字符串。
    return "".join(cleaned_chars)


def _normalize_escape_keep_chars(keep_escape_chars: Iterable[str] | None) -> set[str]:
    """
    规范化需要保留的转义字符列表。

    参数:
        keep_escape_chars: 调用方传入的保留列表，元素可以是实际字符或反斜杠字面量。

    返回:
        返回实际字符组成的集合，用于 remove_escape_chars 快速判断。

    异常:
        当保留列表中的元素不是字符串时抛出 TypeError。
    """
    # 未提供保留列表时返回空集合，表示默认删除所有通用转义/控制字符。
    if keep_escape_chars is None:
        return set()

    # normalized 保存最终需要保留的实际字符，集合结构能避免重复项带来的额外判断。
    normalized: set[str] = set()

    # 遍历调用方提供的每一个保留项，逐项转换成实际字符。
    for item in keep_escape_chars:
        # 保留项必须是字符串；其它类型无法表达明确的转义字符语义。
        if not isinstance(item, str):
            raise TypeError("keep_escape_chars must contain only strings")

        # 常见写法 "\\n"、"\\t" 等先映射成对应实际字符。
        if item in ESCAPE_CHAR_ALIASES:
            normalized.add(ESCAPE_CHAR_ALIASES[item])
            continue

        # 调用方直接传入实际字符时，将其中每个字符都加入保留集合，兼容 "\r\n" 这类组合写法。
        normalized.update(item)

    # 返回规范化后的保留集合，供主清理函数使用。
    return normalized


def _normalize_text_preserving_fullwidth_punctuation(text: str) -> str:
    """
    执行 Unicode NFKC 规范化，同时保留中文/全角朗读标点。

    参数:
        text: 需要规范化的普通正文片段，不应包含已经被上游拆出的特殊 token。

    返回:
        返回规范化后的文本；数字、拉丁兼容字符等仍会被 NFKC 收敛，中文/全角朗读标点保持原样。

    异常:
        本函数不主动抛出异常；Unicode 规范化异常会由 Python 标准库自然暴露。
    """
    # 逐字符处理是为了在保留标点原貌的同时，继续保留 NFKC 对全角数字等兼容字符的收敛收益。
    normalized_chars: list[str] = []

    for char in text:
        # 中文/全角朗读标点直接参与 TTS 韵律表达，不能被 NFKC 改写成半角标点。
        if char in PRESERVED_NFKC_FULLWIDTH_PUNCTUATION:
            normalized_chars.append(char)
            continue

        normalized_chars.append(unicodedata.normalize("NFKC", char))

    return "".join(normalized_chars)


def clean_text(
    text: str,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = (),
) -> str:
    """
    清理模型无关的无意义字符，同时保护模型声明的特殊 token。

    参数:
        text: 原始输入文本。
        special_tokens: 需要原样保护的固定 token。
        special_token_wrappers: 需要原样保护的包裹型 token。

    返回:
        返回清理后的文本；正常 Unicode 正文会被保留，控制字符和不可见噪声会被删除。

    异常:
        本函数不主动抛出异常；正则构造异常会由 Python 标准库自然暴露。
    """
    # 先按特殊 token 切开文本，确保后续 Unicode 归一化不会改写模型控制 token。
    protected_parts = _split_with_special_tokens(text, special_tokens, special_token_wrappers)

    # cleaned_parts 按原文顺序保存清洗后的片段，最后再合并成完整文本。
    cleaned_parts: list[str] = []

    # 遍历每一个片段；特殊 token 原样保留，普通文本只做模型无关清理。
    for part, is_special in protected_parts:
        # 特殊 token 是模型协议的一部分，不能被通用清理改写。
        if is_special:
            cleaned_parts.append(part)
            continue

        # 普通正文保留所有正常 Unicode 字符，只删除控制字符、零宽字符等无意义噪声。
        cleaned_parts.append(_clean_regular_text(part))

    # 合并所有片段，恢复用户输入的原始顺序。
    cleaned = "".join(cleaned_parts)

    # 压缩横向空白，避免复制粘贴产生的大量空格消耗上下文。
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 压缩过多空行，保留最多一个空行作为自然段落停顿。
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 压缩重复标点，避免极端重复符号给 TTS 模型带来不稳定韵律。
    cleaned = re.sub(REPEATED_PUNCTUATION_PATTERN, r"\1", cleaned)

    # 去掉首尾空白，避免空白段进入后续分段。
    return cleaned.strip()


def clean_text_by_allowed_pattern(
    text: str,
    allowed_pattern: str | re.Pattern[str],
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = (),
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
    collapse_repeated_punctuation: bool = True,
) -> str:
    """
    按模型传入的允许字符正则清理文本，同时保护特殊 token。

    参数:
        text: 原始输入文本。
        allowed_pattern: 模型定义的“单个允许字符”正则；每个普通字符都必须能 fullmatch 该正则才会保留。
        special_tokens: 需要原样保护的固定 token。
        special_token_wrappers: 需要原样保护的包裹型 token。
        normalize_unicode: 是否对普通文本片段做 NFKC 规范化。
        collapse_whitespace: 是否压缩普通空白，避免清理后产生过多空格。
        collapse_repeated_punctuation: 是否压缩连续重复停顿符号。

    返回:
        返回清理后的文本；特殊 token 会按原始内容保留，不参与 allowed_pattern 过滤。

    异常:
        allowed_pattern 非法时会由 re.compile 自然抛出 re.error。
    """
    # allowed_re 是模型自己的允许字符规则；通用层只执行规则，不解释语言含义。
    allowed_re = re.compile(allowed_pattern) if isinstance(allowed_pattern, str) else allowed_pattern

    # 先把特殊 token 拆出来，防止模型协议 token 被逐字符过滤破坏。
    protected_parts = _split_with_special_tokens(text, special_tokens, special_token_wrappers)

    # cleaned_parts 按原文顺序保存清理后的普通片段和原样保护的特殊片段。
    cleaned_parts: list[str] = []

    # 遍历普通片段和特殊片段，分别执行过滤或原样保留。
    for part, is_special in protected_parts:
        # 特殊 token 是模型协议或风格控制内容，必须绕过 allow 正则。
        if is_special:
            cleaned_parts.append(part)
            continue

        # 普通文本可选做 Unicode 规范化；是否开启由模型策略决定。
        # regular_text = unicodedata.normalize("NFKC", part) if normalize_unicode else part
        regular_text = _normalize_text_preserving_fullwidth_punctuation(part) if normalize_unicode else part

        # 统一换行和制表符，让模型正则只需要处理稳定的空白形态。
        regular_text = regular_text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

        # 逐字符保留允许字符；这里不内置任何语言判断，完全服从模型传入的正则。
        cleaned_parts.append("".join(char for char in regular_text if allowed_re.fullmatch(char)))

    # 合并所有片段，恢复文本顺序。
    cleaned = "".join(cleaned_parts)

    # 模型允许时压缩横向空白，减少清理后多空格进入 tokenizer。
    if collapse_whitespace:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 模型允许时压缩重复标点，避免极端重复符号影响韵律稳定性。
    if collapse_repeated_punctuation:
        cleaned = re.sub(REPEATED_PUNCTUATION_PATTERN, r"\1", cleaned)

    # 去掉首尾空白，避免空段进入分段逻辑。
    return cleaned.strip()


def segment_text(
    text: str,
    tokenizer: Any,
    max_length: int = 300,
    min_length: int = 20,
    sentence_delimiters: tuple[str, ...] | None = None,
    clause_delimiters: tuple[str, ...] | None = None,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    """
    按句级优先分段，超长句再降级到子句级，再必要时强制截断。

    参数:
        text: 已清理文本。
        max_length: 单段最大长度。
        min_length: 单段最小合并长度。
        tokenizer: tokenizer；存在时按 token 数度量长度。
        sentence_delimiters: 模型可选覆盖的句级切分符。
        clause_delimiters: 模型可选覆盖的子句级切分符。
        special_tokens: 模型需要保护的固定 token，分段时不会在 token 内部切开。
        special_token_wrappers: 模型需要保护的包裹型 token，分段时不会在包裹内容内部切开。

    返回:
        返回非空文本段列表。

    异常:
        本函数不主动抛出异常；tokenizer.encode 异常会自然暴露。

    关于多语言标点：
    - 拉丁语系和俄语通常以 `. ! ?` 结束句子，以 `, ; :` 做子句停顿；
    - 中文/日文更常见 `。！？` 与 `，、；：`；
    - 韩文现代文本既会使用西文 `. , ! ?`，也常混入全角标点。
    - 未来模型可能支持非常多语言，因此默认规则保持宽松，模型需要特殊协议时在模型自己的 preprocess 里覆盖。
    """
    # 根据 tokenizer 或字符长度创建度量函数，统一后续长度判断。
    measure = _make_measure(tokenizer)

    # 解析句级/子句级分隔符；模型配置优先，缺省时使用通用多语标点。
    sentence_pattern, clause_pattern = _resolve_segmentation_patterns(
        sentence_delimiters=sentence_delimiters,
        clause_delimiters=clause_delimiters,
    )

    # 预先定位特殊 token 的保护范围，避免 [surprised!] 内部的标点被误判为句子边界。
    protected_ranges = _find_special_token_ranges(
        text=text,
        special_tokens=special_tokens,
        special_token_wrappers=special_token_wrappers,
    )

    # 先按句级边界切分，尽量让每个生成单元保持自然语义完整。
    sentences = _split_by_pattern(text, sentence_pattern, protected_ranges=protected_ranges)

    # segments 存储已经确定的输出段，current 存储当前正在累计的段。
    segments: list[str] = []
    current = ""

    # 遍历句子，短句合并，超长句降级到子句或强制切分。
    for sentence in sentences:
        # sentence_length 决定当前句子是否需要继续拆分。
        sentence_length = measure(sentence)

        # 单句超过上限时，先提交已有缓冲，再对子句继续拆。
        if sentence_length > max_length:
            # 已累计内容不能和超长句混在一起处理，否则会改变前文边界。
            if current:
                segments.append(current)
                current = ""

            # 子句拆分用于减少硬切，尽量在逗号、分号等自然停顿处断开。
            clauses = _split_by_pattern(
                sentence,
                clause_pattern,
                protected_ranges=_find_special_token_ranges(
                    text=sentence,
                    special_tokens=special_tokens,
                    special_token_wrappers=special_token_wrappers,
                ),
            )
            clause_buffer = ""

            # 遍历子句，按长度上限合并成尽量少的段。
            for clause in clauses:
                # 子句仍然超长时，只能进入强制切分兜底。
                if measure(clause) > max_length:
                    # 先提交已有子句缓冲，避免强制切分影响前面的自然片段。
                    if clause_buffer:
                        segments.append(clause_buffer)
                        clause_buffer = ""
                    segments.extend(
                        _force_split(
                            clause,
                            max_length,
                            measure,
                            tokenizer,
                            special_tokens=special_tokens,
                            special_token_wrappers=special_token_wrappers,
                        )
                    )
                # 当前缓冲加上新子句会超长时，先提交缓冲再开启新缓冲。
                elif clause_buffer and measure(clause_buffer + clause) > max_length:
                    segments.append(clause_buffer)
                    clause_buffer = clause
                # 当前子句可以并入缓冲，继续等待后续子句。
                else:
                    clause_buffer += clause

            # 子句循环结束后，提交最后剩余的缓冲。
            if clause_buffer:
                segments.append(clause_buffer)
            continue

        # 当前累计段加上新句会超长时，先提交当前段。
        if current and measure(current + sentence) > max_length:
            segments.append(current)
            current = sentence
        # 新句不会超长时，合并到当前段，减少碎片。
        else:
            current += sentence

    # 遍历结束后提交最后一个累计段。
    if current:
        segments.append(current)

    # 对过短段做二次合并，减少 runtime 的碎片化调用。
    merged_segments = _merge_short_segments(segments, min_length, max_length, measure)

    # 最终只返回非空段，避免空白文本进入模型。
    return [segment.strip() for segment in merged_segments if segment.strip()]


def _clean_regular_text(text: str) -> str:
    """
    清理普通正文片段中的模型无关噪声。

    参数:
        text: 不属于特殊 token 的普通文本片段。

    返回:
        返回清理后的普通正文；合法 Unicode 字符会被保留。

    异常:
        本函数不主动抛出异常；Unicode 规范化异常会自然暴露。
    """
    # 使用 NFKC 统一全角/半角和兼容字符，减少 tokenizer 面对等价字符时的碎片化。
    # [旧逻辑已废弃 - 原因]: 直接对整段执行 NFKC 会把中文/全角朗读标点改写成半角符号，破坏用户输入的韵律形态。
    # normalized = unicodedata.normalize("NFKC", text)
    normalized = _normalize_text_preserving_fullwidth_punctuation(text)

    # 统一换行和制表符，让后续空白处理面对稳定格式。
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # 删除 ASCII 控制字符，但保留换行，因为换行可以作为段落边界。
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

    # 删除零宽字符和不可见分隔符，避免网页复制文本带入不可见噪声。
    normalized = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff]", "", normalized)

    # 通用层不再按语言白名单删除字符，避免多语言模型误删合法正文。
    return normalized


def _split_with_special_tokens(
    text: str,
    special_tokens: tuple[str, ...],
    special_token_wrappers: tuple[tuple[str, str], ...],
) -> list[tuple[str, bool]]:
    patterns: list[str] = []

    unique_tokens = sorted({token for token in special_tokens if token}, key=len, reverse=True)
    patterns.extend(re.escape(token) for token in unique_tokens)

    for open_token, close_token in special_token_wrappers:
        if not open_token or not close_token:
            continue
        # 为什么这里用非贪婪匹配：
        # - 目标是保护「一段被包裹的特殊描述」，例如 [laugh]、<cry>；
        # - 不希望一个开符号把整行后续内容都吞进去，所以限制不跨换行，并尽早闭合。
        patterns.append(f"{re.escape(open_token)}[^\\n]*?{re.escape(close_token)}")

    if not patterns:
        return [(text, False)]

    pattern = re.compile("|".join(patterns))
    parts: list[tuple[str, bool]] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > cursor:
            parts.append((text[cursor:start], False))
        parts.append((match.group(0), True))
        cursor = end
    if cursor < len(text):
        parts.append((text[cursor:], False))
    return parts


def _find_special_token_ranges(
    text: str,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = (),
) -> list[tuple[int, int]]:
    """
    找出文本中特殊 token 的字符范围。

    参数:
        text: 需要扫描的文本。
        special_tokens: 需要保护的固定 token。
        special_token_wrappers: 需要保护的包裹型 token。

    返回:
        返回半开区间列表，形如 (start, end)，end 不包含在范围内。

    异常:
        本函数不主动抛出异常；非法正则构造会由 re.compile 自然暴露。
    """
    # patterns 收集所有需要保护的正则分支，固定 token 和包裹 token 使用同一套扫描流程。
    patterns: list[str] = []

    # 固定 token 按长度倒序匹配，避免短 token 抢先命中长 token 的前缀。
    unique_tokens = sorted({token for token in special_tokens if token}, key=len, reverse=True)
    patterns.extend(re.escape(token) for token in unique_tokens)

    # 包裹型 token 使用非贪婪匹配，并限制不跨换行，避免一个开符号吞掉整段文本。
    for open_token, close_token in special_token_wrappers:
        # 空包裹符没有匹配意义，跳过以避免正则异常。
        if not open_token or not close_token:
            continue

        # 包裹内容整体作为受保护范围，后续句级/强制切分都不能切到内部。
        patterns.append(f"{re.escape(open_token)}[^\\n]*?{re.escape(close_token)}")

    # 没有任何特殊 token 规则时返回空范围，调用方按普通文本处理。
    if not patterns:
        return []

    # 编译统一匹配器并返回所有保护区间。
    pattern = re.compile("|".join(patterns))
    return [match.span() for match in pattern.finditer(text)]


def _position_inside_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    """
    判断字符位置是否落在保护范围内。

    参数:
        position: 待判断的字符位置。
        ranges: 半开保护区间列表。

    返回:
        如果位置处于任意保护范围内部则返回 True，否则返回 False。

    异常:
        本函数不主动抛出异常。
    """
    # 遍历保护范围，任意一个范围命中即可判定当前位置不可作为切分点。
    for start, end in ranges:
        # 半开区间判断和 Python 字符串切片边界保持一致。
        if start <= position < end:
            return True

    # 所有范围都未命中，说明当前位置可以按普通正文处理。
    return False


def _adjust_split_pos_for_special_tokens(
    text: str,
    split_pos: int,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = (),
) -> int:
    """
    调整强制切分位置，避免切开特殊 token。

    参数:
        text: 当前待切分文本。
        split_pos: 初步计算出的切分位置。
        special_tokens: 固定 token 保护规则。
        special_token_wrappers: 包裹型 token 保护规则。

    返回:
        调整后的切分位置；必要时允许单个特殊 token 独立超过 max_length。

    异常:
        本函数不主动抛出异常。
    """
    # 先找出当前文本中的保护范围，用于判断硬切点是否落在 token 内部。
    protected_ranges = _find_special_token_ranges(
        text=text,
        special_tokens=special_tokens,
        special_token_wrappers=special_token_wrappers,
    )

    # 遍历保护范围，找到包含切分点的特殊 token。
    for start, end in protected_ranges:
        # 切分点不在当前 token 内部时继续检查下一个范围。
        if not (start < split_pos < end):
            continue

        # 如果 token 前面已有正文，优先在 token 前切，保证 token 完整进入下一段。
        if start > 0 and text[:start].strip():
            return start

        # 如果 token 位于段首，只能扩展到 token 结尾，避免把 token 切碎。
        return end

    # 初始位置没有切开特殊 token，原样返回即可。
    return split_pos


def _resolve_segmentation_patterns(
    sentence_delimiters: tuple[str, ...] | None,
    clause_delimiters: tuple[str, ...] | None,
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """
    解析通用分段正则。

    参数:
        sentence_delimiters: 模型传入的句级分隔符；为空时使用通用默认值。
        clause_delimiters: 模型传入的子句级分隔符；为空时使用通用默认值。

    返回:
        返回句级正则和子句级正则。

    异常:
        本函数不主动抛出异常；非法正则构造会自然暴露。
    """
    # 没有模型覆盖时使用宽松的通用多语分隔符，而不是从语言注册表推导。
    if sentence_delimiters is None:
        sentence_delimiters = DEFAULT_SENTENCE_DELIMITERS

    # 没有模型覆盖时使用通用子句分隔符，模型协议特殊时应在模型自己的 preprocess 中控制。
    if clause_delimiters is None:
        clause_delimiters = DEFAULT_CLAUSE_DELIMITERS

    # 编译为保留分隔符的切分正则。
    return _compile_delimiter_pattern(sentence_delimiters), _compile_delimiter_pattern(clause_delimiters)


def _compile_delimiter_pattern(delimiters: tuple[str, ...]) -> re.Pattern[str]:
    escaped = sorted((re.escape(item) for item in delimiters if item), key=len, reverse=True)
    if not escaped:
        return re.compile(r"($^)")
    return re.compile(f"({'|'.join(escaped)})")


def _split_by_pattern(
    text: str,
    pattern: re.Pattern[str],
    protected_ranges: list[tuple[int, int]] | None = None,
) -> list[str]:
    """
    使用保留分隔符的正则切分文本，并避开特殊 token 内部。

    参数:
        text: 待切分文本。
        pattern: 由 _compile_delimiter_pattern 生成的正则。
        protected_ranges: 不允许作为切分点的半开字符范围。

    返回:
        切分后的非空片段列表，分隔符会保留在前一个片段末尾。

    异常:
        本函数不主动抛出异常。
    """
    # 未提供保护范围时使用空列表，保持旧调用路径行为。
    protected_ranges = protected_ranges or []

    # segments 存储保留分隔符后的切分结果。
    segments: list[str] = []

    # cursor 记录当前段起点，每遇到一个合法分隔符就切出一段。
    cursor = 0

    # 遍历所有分隔符命中，只接受不在保护区间内部的切分点。
    for match in pattern.finditer(text):
        # 特殊 token 内部的标点属于 token 内容，不是正文停顿边界。
        if _position_inside_ranges(match.start(), protected_ranges):
            continue

        # 分隔符保留在当前段末尾，避免丢失语气符号。
        end = match.end()

        # 当前切分点前方有非空文本时才生成段，避免连续分隔符产生空段。
        if text[cursor:end].strip():
            segments.append(text[cursor:end])

        # 游标推进到分隔符之后，下一段从这里开始。
        cursor = end

    # 扫描结束后，尾部无分隔符文本也要保留。
    if text[cursor:].strip():
        segments.append(text[cursor:])

    # 去掉纯空白段，避免后续 runtime 处理空文本。
    return [segment for segment in segments if segment.strip()]


def _make_measure(tokenizer: Any = None) -> Callable[[str], int]:
    """有 tokenizer 时按 token 数量度量，否则退化成字符数。"""
    if tokenizer is not None:
        def _measure(text: str) -> int:
            encoded = tokenizer.encode(text)
            return len(encoded.ids if hasattr(encoded, "ids") else encoded)

        return _measure
    return len


def _find_token_split_pos(text: str, max_tokens: int, tokenizer: Any) -> int:
    encoding = tokenizer.encode(text)
    ids = encoding.ids if hasattr(encoding, "ids") else encoding
    if len(ids) <= max_tokens:
        return len(text)

    offsets = getattr(encoding, "offsets", None)
    if not offsets or max_tokens >= len(offsets):
        return len(text)
    return max(1, offsets[max_tokens][0])


def _force_split(
    text: str,
    max_length: int,
    measure: Callable[[str], int],
    tokenizer: Any = None,
    special_tokens: tuple[str, ...] = (),
    special_token_wrappers: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    """兜底拆分时尽量回退到自然断点，避免在中间生硬截断。"""
    break_chars = " .,;:，。；：、!?！？؟؛،।\n"
    chunks: list[str] = []
    remaining = text.strip()

    while remaining and measure(remaining) > max_length:
        split_pos = _find_token_split_pos(remaining, max_length, tokenizer) if tokenizer is not None else max_length
        best = _adjust_split_pos_for_special_tokens(
            text=remaining,
            split_pos=min(split_pos, len(remaining)),
            special_tokens=special_tokens,
            special_token_wrappers=special_token_wrappers,
        )
        for index in range(best, max(0, best - 30), -1):
            if index < len(remaining) and remaining[index] in break_chars:
                best = _adjust_split_pos_for_special_tokens(
                    text=remaining,
                    split_pos=index + 1,
                    special_tokens=special_tokens,
                    special_token_wrappers=special_token_wrappers,
                )
                break

        chunk = remaining[:best].strip()
        if not chunk:
            chunk = remaining[:split_pos].strip()
            best = max(1, split_pos)
        chunks.append(chunk)
        remaining = remaining[best:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _merge_short_segments(
    segments: list[str],
    min_length: int,
    max_length: int,
    measure: Callable[[str], int],
) -> list[str]:
    """短段优先向后并，后并不下再尝试向前并。"""
    if not segments:
        return segments

    result = list(segments)
    changed = True
    while changed:
        changed = False
        merged: list[str] = []
        index = 0
        while index < len(result):
            current = result[index]
            if measure(current) < min_length:
                if index + 1 < len(result) and measure(current + result[index + 1]) <= max_length:
                    merged.append(current + result[index + 1])
                    index += 2
                    changed = True
                    continue
                if merged and measure(merged[-1] + current) <= max_length:
                    merged[-1] = merged[-1] + current
                    index += 1
                    changed = True
                    continue
            merged.append(current)
            index += 1
        result = merged

    return result
