"""验证开源 Windows 发行的标签与后端依赖边界。"""

from pathlib import Path
import runpy
import tomllib

from streamvox.build_matrix import list_targets


def test_release_is_readable_source_without_python_extension_abi():
    """原生 DLL 不应伪装成需要 CPython 扩展接口的模块。"""
    targets = list_targets()
    assert len(targets) == 1
    assert targets[0].wheel_filename == "streamvox-2.0.0-cp312-none-win_amd64.whl"
    assert targets[0].sdist_filename == "streamvox-2.0.0.tar.gz"


def test_llama_cuda_extra_has_only_required_libraries():
    """安装 llama CUDA 不能顺带安装 ONNX GPU 或 cuDNN。"""
    project_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert extras["llama-cuda"] == ["nvidia-cuda-runtime-cu12==12.4.99", "nvidia-cublas-cu12==12.4.2.65"]
    assert "embedded-nearmate" not in extras
    assert project["license"] == "Apache-2.0"
    assert project["requires-python"] == ">=3.12,<3.13"


def test_build_backend_writes_the_same_platform_tag(monkeypatch):
    """普通源码包重建也必须得到同一 Windows 无扩展接口标签。"""
    import setuptools
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    command = namespace["StreamVoxBdistWheel"](setuptools.Distribution())
    command.finalize_options()
    assert command.root_is_pure is False
    assert command.get_tag() == ("cp312", "none", "win_amd64")
