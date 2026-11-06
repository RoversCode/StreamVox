"""定义可独立缓存的 Breeze2 参考资产。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..base import BasePromptAsset


@dataclass(slots=True)
class BreezePromptAsset(BasePromptAsset):
    """保存完整参考转录和逐帧音频编码。

    Args:
        prompt_text: 与参考音频完整对应的转录；无参考模式使用空值。
        audio_codes: 按时间、码本排列的整数编码，空数组表示无参考音频。

    Returns:
        可由缓存序列化并直接传给运行时的提示资产。
    """

    prompt_text: str | None = None
    audio_codes: np.ndarray = field(default_factory=lambda: np.empty((0, 16), dtype=np.int64))
