from __future__ import annotations

import importlib.metadata
from pathlib import Path
import re


SDK_VERSION = "2.0.0"


def _read_source_project_version() -> str | None:
    """
    读取源码树中的项目版本。

    返回:
        如果当前运行位置能找到仓库根目录的 pyproject.toml，则返回其中的项目版本；否则返回 None。
    """
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    text = pyproject_path.read_text(encoding="utf-8")
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?:^\[|\Z)", text)
    if project_match is None:
        return None

    version_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project_match.group(1))
    if version_match is None:
        return None
    return version_match.group(1)


def resolve_sdk_version() -> str:
    """
    解析当前 SDK 版本。

    返回:
        源码优先读取项目版本，其余安装形式读取包元数据；缺少元数据时使用发行版本。
    """
    # 开发环境可能残留旧安装元数据，源码树存在时必须以 pyproject.toml 为准。
    source_version = _read_source_project_version()
    if source_version is not None:
        return source_version

    # wheel 安装后通常没有源码 pyproject.toml，此时包元数据就是发布版本的可信来源。
    try:
        return importlib.metadata.version("streamvox")
    except importlib.metadata.PackageNotFoundError:
        return SDK_VERSION


__version__ = resolve_sdk_version()
