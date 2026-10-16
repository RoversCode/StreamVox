from __future__ import annotations

import importlib.metadata
from unittest.mock import patch

import streamvox
from streamvox.version import SDK_VERSION, resolve_sdk_version


def test_source_project_version_overrides_stale_installed_metadata():
    """源码开发时应采用当前项目版本，避免旧安装元数据污染结果。"""
    with (
        patch("streamvox.version._read_source_project_version", return_value="2.0.0"),
        patch("importlib.metadata.version", return_value="1.2.10") as metadata,
    ):
        assert resolve_sdk_version() == "2.0.0"

    metadata.assert_not_called()


def test_installed_package_uses_distribution_metadata():
    """安装包应从标准发行元数据读取实际安装版本。"""
    with (
        patch("streamvox.version._read_source_project_version", return_value=None),
        patch("importlib.metadata.version", return_value="9.8.7") as metadata,
    ):
        assert resolve_sdk_version() == "9.8.7"

    metadata.assert_called_once_with("streamvox")


def test_missing_distribution_metadata_uses_release_version():
    """独立拷贝源码且缺少安装元数据时仍提供明确版本。"""
    with (
        patch("streamvox.version._read_source_project_version", return_value=None),
        patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError),
    ):
        assert resolve_sdk_version() == SDK_VERSION == "2.0.0"


def test_package_exports_current_project_version():
    """公开入口应导出与发行声明一致的版本。"""
    assert streamvox.__version__ == resolve_sdk_version() == SDK_VERSION == "2.0.0"
