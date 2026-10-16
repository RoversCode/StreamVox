#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   preprocess_text.py
@Time    :   2026/03/11
@Author  :   RoversCode
@Version :   1.0
@Desc    :   qwen3 文本预处理兼容入口

说明:
  1. 共享预处理逻辑已经提升到 streamvox.models.text_processing
  2. qwen3 这里只保留模型能力声明与兼容导出，避免外部调用方断裂
"""

from typing import Any, Optional

from ...text_processing import (
    TextProcessResult,
    TextPreprocessConfig,
    clean_text_by_allowed_pattern,
    segment_text,
)


CLONE_LANGUAGE_MAP = {
    "auto": "auto",
    "中文": "chinese",
    "英语": "english",
    "德语": "german",
    "意大利语": "italian",
    "葡萄牙语": "portuguese",
    "西班牙语": "spanish",
    "日语": "japanese",
    "韩语": "korean",
    "法语": "french",
    "俄语": "russian",
}

QWEN3_TEXT_PREPROCESS_CONFIG = TextPreprocessConfig(
    # special_tokens=(
    #     "<|im_start|>",
    #     "<|im_end|>",
    #     "<laughter>",
    #     "<laugh>",
    #     "<cry>",
    #     "<cough>",
    #     "<sigh>",
    #     "<breath>",
    # ),
    # special_token_wrappers=(
    #     ("<", ">"),
    #     ("[", "]"),
    # ),
)

# Qwen3 清理模式下的允许字符规则：覆盖当前声明的中英日韩俄和欧洲拉丁语系，并保留常用数字、空白和朗读标点。
QWEN3_ALLOWED_TEXT_PATTERN = (
    r"[\u4e00-\u9fff\u3400-\u4dbf"
    r"\u3040-\u30ff\u31f0-\u31ff\uff65-\uff9f"
    r"\uac00-\ud7af\u1100-\u11ff\u3130-\u318f"
    r"\u0400-\u04ff"
    r"A-Za-zÀ-ÿ\u0152\u0153\u0178\u1e00-\u1eff"
    r"0-9\s"
    r".,!?;:'\"\\-…"
    r"。，！？；：、—“”‘’]"
)


def preprocess_text(
    text: str,
    language: Optional[str] = None,
    max_length: int = 150,
    min_length: int = 20,
    tokenizer: Any = None,
    config: TextPreprocessConfig | None = None,
    remove_meaningless_chars: bool = False,
) -> TextProcessResult:
    """兼容旧调用签名，默认走 qwen3 自己的能力声明。"""
    # active_config 保存当前模型预处理配置，允许 runtime 注入覆盖配置。
    active_config = config or QWEN3_TEXT_PREPROCESS_CONFIG

    # 空文本直接返回空结果，避免后续分段生成空 prompt。
    if not text or not text.strip():
        return TextProcessResult(segments=[], language=language or "auto", original_text=text or "", cleaned_text="")

    # Qwen3 不再做任何语言检测；language 完全由调用方提供，下游自行决定如何映射。
    resolved_language = language or "auto"

    # 根据入口开关决定是否调用模型自己的允许字符正则清理。
    if remove_meaningless_chars:
        cleaned_text = clean_text_by_allowed_pattern(
            text=text,
            allowed_pattern=QWEN3_ALLOWED_TEXT_PATTERN,
            special_tokens=active_config.special_tokens,
            special_token_wrappers=active_config.special_token_wrappers,
        )
    else:
        cleaned_text = text.strip()

    # 清理后为空说明用户输入没有可合成内容，返回空分段交给 runtime 报错。
    if not cleaned_text:
        return TextProcessResult(segments=[], language=resolved_language, original_text=text, cleaned_text="")

    # 分段只使用公共工具；模型自己提供长度阈值、tokenizer、标点和特殊 token 规则。
    segments = segment_text(
        text=cleaned_text,
        max_length=max_length,
        min_length=min_length,
        tokenizer=tokenizer,
        sentence_delimiters=active_config.sentence_delimiters,
        clause_delimiters=active_config.clause_delimiters,
        special_tokens=active_config.special_tokens,
        special_token_wrappers=active_config.special_token_wrappers,
    )

    # 返回统一结果对象，保持 runtime 调用形状稳定。
    return TextProcessResult(
        segments=segments,
        language=resolved_language,
        original_text=text,
        cleaned_text=cleaned_text,
    )
