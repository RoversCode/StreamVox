#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   constant.py
@Time    :   2026/03/06 17:24:25
@Author  :   RoversCode
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
from ....utils.log import logger


# 说话人 ID 映射 (Verified from official config.json)
SPEAKER_MAP = {
    "vivian": 3065, 
    "serena": 3066, 
    "uncle_fu": 3010, 
    "ryan": 3061,
    "aiden": 2861, 
    "ono_anna": 2873, 
    "sohee": 2864, 
    "eric": 2875, 
    "dylan": 2878
}

# 语言 ID 映射
LANGUAGE_MAP = {
    "english": 2050, 
    "german": 2053, 
    "spanish": 2054, 
    "chinese": 2055, 
    "japanese": 2058, 
    "french": 2061, 
    "sichuan_dialect": 2062, 
    "korean": 2064,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
    "beijing_dialect": 2074, 
}

# 官方流程协议标签
SPECIAL_TOKENS = {
    "PAD": 2148, 
    "BOS": 2149, 
    "EOS": 2150, 
    "TTS_BOS": 151672, 
    "TTS_EOS": 151673,
    "THINK": 2154, 
    "NOTHINK": 2155, 
    "THINK_BOS": 2156, 
    "THINK_EOS": 2157
}


def map_speaker(spk) -> int:
    """将说话人名称或 ID 映射为官方数值 ID (2800-3071)"""
    if isinstance(spk, int):
        # 即使是数字，也要验证区间
        if 2800 <= spk <= 3071:
            return spk
        logger.warning(f"⚠️ 非法的 Speaker ID: {spk}, 必须在 2800-3071 之间。回退到默认对齐 Vivian (3065)")
        return None

    return SPEAKER_MAP.get(str(spk).lower(), None)

def map_language(lang) -> int:
    """将语言名称或 ID 映射为官方数值 ID (2048-2147)"""
    if isinstance(lang, int):
        if 2048 <= lang <= 2147:
            return lang
        logger.warning(f"⚠️ 非法的 Language ID: {lang}, 必须在 2048-2147 之间。回退到 Chinese (2055)")
        return None
        
    return LANGUAGE_MAP.get(str(lang).lower(), None)
