"""集中管理模型和参考音频缓存的本地目录。"""

from __future__ import annotations

import os
from pathlib import Path


def get_streamvox_home() -> Path:
    """解析并创建当前用户的 SDK 数据目录。

    Returns:
        优先使用 STREAMVOX_HOME，否则使用当前平台的用户数据目录。
    """
    configured_home = os.environ.get("STREAMVOX_HOME")
    if configured_home:
        # 显式目录允许容器或桌面应用将模型与参考音频缓存保存在稳定位置。
        root = Path(configured_home).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "StreamVox"
    else:
        root = Path.home() / ".streamvox"

    root.mkdir(parents=True, exist_ok=True)
    return root


def get_default_model_root() -> Path:
    """解析并创建默认模型下载目录。

    Returns:
        SDK 数据目录下的 ckpts 目录。
    """
    model_dir = get_streamvox_home() / "ckpts"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
