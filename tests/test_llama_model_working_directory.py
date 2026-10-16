"""验证三套模型绑定在加载成功或失败后都恢复调用方工作目录。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(params=("qwen3", "s2_pro", "voxcpm2"))
def native_binding(request, tmp_path, monkeypatch):
    """隔离工作目录和原生接口，避免测试加载实际动态库。

    Args:
        request: 当前参数化模型绑定。
        tmp_path: 用例独占的文件目录。
        monkeypatch: 模块属性、环境与工作目录的恢复工具。
    Returns:
        绑定模块、调用方目录、动态库目录和可观察的加载状态。
    """
    binding = importlib.import_module(f"streamvox.models.{request.param}.llama")
    package = tmp_path / "package"
    lib_dir = package / "bin"
    lib_dir.mkdir(parents=True)
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    (caller_dir / "model.gguf").write_bytes(b"fixture")
    monkeypatch.chdir(caller_dir)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setattr(binding, "__file__", str(package / "models" / request.param / "llama.py"))
    monkeypatch.setattr(binding.os, "add_dll_directory", lambda _path: None, raising=False)
    state = SimpleNamespace(result=object(), params=SimpleNamespace(), calls=[])

    def initialize(*, requested_backend):
        """确认初始化仍在库目录中执行，并保留调用方选择的后端。"""
        assert Path.cwd() == lib_dir
        assert requested_backend == "cuda"

    def load_from_file(model_path, params):
        """记录传入原生层的路径和参数，不接触实际动态库。"""
        assert Path.cwd() == lib_dir
        assert (Path.cwd() / model_path.decode("utf-8")).resolve() == caller_dir / "model.gguf"
        state.calls.append(params)
        return state.result

    monkeypatch.setattr(binding, "init_llama_lib", initialize)
    monkeypatch.setattr(binding, "llama_model_default_params", lambda: state.params, raising=False)
    monkeypatch.setattr(binding, "llama_model_load_from_file", load_from_file, raising=False)
    if hasattr(binding, "_build_model_kv_overrides"):
        monkeypatch.setattr(binding, "_build_model_kv_overrides", lambda _path: None)
    return SimpleNamespace(module=binding, caller_dir=caller_dir, lib_dir=lib_dir, state=state)


@pytest.mark.parametrize("failure_stage", ("init_llama_lib", "llama_model_default_params", "llama_model_load_from_file"))
def test_loading_exception_restores_callers_directory(native_binding, monkeypatch, failure_stage):
    """初始化、参数创建或原生加载抛错时必须恢复调用方目录。"""
    error = RuntimeError("模拟模型加载失败")

    def fail(*_args, **_kwargs):
        """在切换目录后的指定阶段抛出原始异常。"""
        assert Path.cwd() == native_binding.lib_dir
        raise error

    monkeypatch.setattr(native_binding.module, failure_stage, fail)

    with pytest.raises(RuntimeError) as caught:
        native_binding.module.load_model("model.gguf", backend="cuda")

    assert caught.value is error
    assert Path.cwd() == native_binding.caller_dir


@pytest.mark.parametrize("has_model", (True, False))
def test_model_result_restores_callers_directory(native_binding, has_model):
    """加载成功和原生层返回空指针时都保持既有结果并恢复目录。"""
    if not has_model:
        native_binding.state.result = None

    result = native_binding.module.load_model(
        "model.gguf", n_gpu_layers=5, main_gpu=2, split_mode=0, backend="cuda"
    )

    assert result is native_binding.state.result
    assert Path.cwd() == native_binding.caller_dir
    assert len(native_binding.state.calls) == 1
    params = native_binding.state.calls[0]
    assert params.n_gpu_layers == 5
    assert params.main_gpu == 2
    assert params.split_mode == 0


def test_dll_directory_failure_restores_callers_directory(native_binding, monkeypatch):
    """注册动态库搜索目录失败时同样不能改变调用方目录。"""
    def fail_directory_registration(_path):
        """模拟系统拒绝注册动态库搜索目录。"""
        raise OSError("模拟目录注册失败")

    monkeypatch.setattr(native_binding.module.os, "add_dll_directory", fail_directory_registration)

    with pytest.raises(OSError, match="模拟目录注册失败"):
        native_binding.module.load_model("model.gguf", backend="cuda")

    assert Path.cwd() == native_binding.caller_dir
