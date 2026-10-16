"""验证原生库目录隔离和进程内句柄生命周期。"""

from unittest.mock import patch

import pytest

from streamvox_runtime import native_loader


def test_discovery_ignores_external_interpreters(tmp_path):
    """用户包和别的解释器不能补齐当前发行环境的缺库。"""
    own = tmp_path / "venv"
    external = tmp_path / "other"
    for root in (own, external):
        (root / "Lib/site-packages/nvidia/cudnn/bin").mkdir(parents=True)
    with patch.object(native_loader.sys, "prefix", str(own)), patch.object(
        native_loader.site, "getsitepackages", return_value=[str(root / "Lib/site-packages") for root in (own, external)]
    ), patch.dict(native_loader.os.environ, {"STREAMVOX_NVIDIA_LIB_DIRS": ""}):
        assert native_loader.nvidia_library_directories(consumer="onnx_cuda") == (own / "Lib/site-packages/nvidia/cudnn/bin",)


def test_required_runtime_fails_without_managed_libraries():
    """CUDA 不能在未安装发行依赖时借系统环境继续运行。"""
    with patch.object(native_loader, "nvidia_library_directories", return_value=()):
        with pytest.raises(RuntimeError, match="未找到"):
            native_loader.initialize_nvidia_runtime(consumer="llama", required=True)
        assert native_loader.initialize_nvidia_runtime(consumer="llama", ) == ()


def test_conflicting_onnxruntime_packages_fail():
    """pip 安装多个互相覆盖的发行包时必须在推理前报错。"""
    with patch.object(native_loader.importlib.metadata, "version", return_value="1.21.1"):
        with pytest.raises(RuntimeError, match="多个 ONNX Runtime"):
            native_loader.ensure_single_onnxruntime_distribution()


@pytest.mark.skipif(native_loader.os.name != "nt", reason="Windows 搜索句柄")
def test_directory_handles_are_retained_once(tmp_path):
    """重复初始化只注册一次目录，句柄一直由模块持有。"""
    handle = object()
    with patch.object(native_loader, "nvidia_library_directories", return_value=(tmp_path,)), patch.object(
        native_loader.os, "add_dll_directory", return_value=handle
    ) as add_directory, patch.object(native_loader, "_loaded", set()), patch.object(
        native_loader, "_dll_handles", []
    ), patch.dict(native_loader.os.environ):
        native_loader.initialize_nvidia_runtime(consumer="llama", )
        native_loader.initialize_nvidia_runtime(consumer="llama", )
        add_directory.assert_called_once_with(str(tmp_path))
        assert native_loader._dll_handles == [handle]
