"""验证无需激活的本地推理、音频输出和流式资源生命周期。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from urllib import request

import numpy as np
import pytest

from streamvox import TTSEngine
from streamvox.models.base import BasePromptAsset, ExecutionPlan
from streamvox.models.registry import ModelRegistry
from streamvox.paths import get_default_model_root, get_streamvox_home


@pytest.fixture
def local_runtime(tmp_path, monkeypatch):
    """使用真实本地目录解析和可控模型分块创建离线引擎。

    Args:
        tmp_path: 当前用例的隔离文件目录。
        monkeypatch: 环境与联网入口的隔离工具。
    Returns:
        引擎、用户目录和用于观察分块与关闭行为的状态。
    """
    home = tmp_path / "home"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(
        json.dumps({"model": "fixture", "assets": []}), encoding="utf-8"
    )
    monkeypatch.setenv("STREAMVOX_HOME", str(home))

    def reject_network(*_args, **_kwargs):
        """本地推理用例发现任何联网尝试时立即失败。"""
        pytest.fail("本地模型初始化和流式合成不应发起网络请求")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(request, "urlopen", reject_network)
    state = SimpleNamespace(chunks=[], closed=0, started=0, error=None)
    runtime = SimpleNamespace(
        sample_rate=24000,
        runtime_info=SimpleNamespace(model="fixture"),
        shutdown=lambda: None,
    )

    def synthesize(*_args, **_kwargs):
        """逐块产出测试音频，并记录底层生成器是否已释放。"""
        state.started += 1
        try:
            yield from state.chunks
            if state.error is not None:
                raise state.error
        finally:
            state.closed += 1

    adapter = SimpleNamespace(
        model_name="fixture",
        get_hardware_profile=lambda: None,
        resolve_execution_plan=lambda device, **_kwargs: ExecutionPlan(raw_device=device, targets={}),
        load_runtime=lambda _request: runtime,
        get_text_preprocess_config=lambda: None,
        make_default_prompt=lambda _runtime: BasePromptAsset(
            role_name="default", model="fixture", sample_rate=24000
        ),
        normalize_synthesize_kwargs=lambda **kwargs: kwargs,
        synthesize=synthesize,
    )
    engine = TTSEngine(str(model_dir), device="cpu", registry=ModelRegistry({"fixture": adapter}))
    yield SimpleNamespace(engine=engine, state=state, home=home)
    engine.shutdown()


def test_local_engine_streams_offline_without_activation_files(local_runtime):
    """离线合成只创建正常缓存目录，不产生设备或令牌缓存。"""
    local_runtime.state.chunks = [np.array([0.25, -0.25], dtype=np.float32)]
    chunks = list(local_runtime.engine.stream("离线语音"))

    np.testing.assert_array_equal(chunks[0], local_runtime.state.chunks[0])
    assert (local_runtime.home / "prompts" / "fixture").is_dir()
    assert not (local_runtime.home / "device_id").exists()
    assert not (local_runtime.home / "online_tokens").exists()


def test_local_engine_ignores_existing_activation_files(local_runtime, monkeypatch):
    """旧安装留下的设备和令牌文件不能影响新的离线推理。"""
    legacy_files = [local_runtime.home / "device_id", local_runtime.home / "online_tokens" / "fixture.json"]
    for path in legacy_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("无效的历史缓存", encoding="utf-8")
    original_read_text = Path.read_text

    def read_without_activation(path, *args, **kwargs):
        """读取历史缓存意味着仍存在未清理的激活逻辑。"""
        if path in legacy_files:
            pytest.fail("新引擎不能读取历史激活缓存")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_without_activation)
    engine = TTSEngine(
        str(local_runtime.engine.model_dir), device="cpu", registry=local_runtime.engine.registry
    )
    local_runtime.state.chunks = [np.ones(2, dtype=np.float32)]
    try:
        assert len(list(engine.stream("忽略历史缓存"))) == 1
    finally:
        engine.shutdown()

    assert all(original_read_text(path, encoding="utf-8") == "无效的历史缓存" for path in legacy_files)


def test_stream_preserves_chunk_contents_shape_and_order(local_runtime):
    """原始浮点音频不应被加噪、裁剪或重新分块。"""
    originals = [
        np.array([0.0, -0.0, 1.25, -1.25], dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.array([[0.1, 0.2], [-0.3, -0.4]], dtype=np.float32),
    ]
    expected = [(chunk.shape, chunk.tobytes()) for chunk in originals]
    local_runtime.state.chunks = originals

    actual = list(local_runtime.engine.stream("保留模型输出"))

    assert len(actual) == len(expected)
    for chunk, source, (shape, payload) in zip(actual, originals, expected):
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.float32
        assert chunk.shape == shape
        assert chunk.tobytes() == payload
        assert source.tobytes() == payload
    assert local_runtime.state.closed == 1


def test_stream_keeps_existing_float32_conversion(local_runtime):
    """适配器的其他浮点类型仍统一转换为公开的单精度数组。"""
    source = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    local_runtime.state.chunks = [source]
    stream = local_runtime.engine.stream("类型转换")

    chunk = next(stream)
    stream.close()

    assert chunk.dtype == np.float32
    assert chunk.tobytes() == np.asarray(source, dtype=np.float32).tobytes()
    assert source.dtype == np.float64


def test_closing_stream_closes_underlying_generator(local_runtime):
    """调用方提前结束输出时，应立即释放底层模型会话。"""
    local_runtime.state.chunks = [np.ones(2, dtype=np.float32)] * 3
    stream = local_runtime.engine.stream("提前停止")
    next(stream)

    stream.close()
    stream.close()

    assert local_runtime.state.closed == 1


def test_model_failure_closes_generator_and_propagates(local_runtime):
    """模型生成失败时保留异常，同时执行会话释放。"""
    local_runtime.state.chunks = [np.ones(2, dtype=np.float32)]
    local_runtime.state.error = RuntimeError("模型生成失败")
    stream = local_runtime.engine.stream("模型异常")
    next(stream)

    with pytest.raises(RuntimeError, match="模型生成失败"):
        next(stream)

    assert local_runtime.state.closed == 1


def test_backend_check_failure_closes_generator(local_runtime, monkeypatch):
    """输出前发现后端错误时，也应关闭已启动的模型生成器。"""
    local_runtime.state.chunks = [np.ones(2, dtype=np.float32)]

    def fail_backend_check():
        """模拟实际后端与调用方要求不一致。"""
        raise RuntimeError("后端不一致")

    monkeypatch.setattr(local_runtime.engine, "_verify_requested_backends", fail_backend_check)

    with pytest.raises(RuntimeError, match="后端不一致"):
        next(local_runtime.engine.stream("后端异常"))

    assert local_runtime.state.closed == 1


def test_unstarted_stream_does_not_open_model_session(local_runtime):
    """未消费的流不应提前占用模型会话资源。"""
    stream = local_runtime.engine.stream("未开始")

    stream.close()

    assert local_runtime.state.started == 0
    assert local_runtime.state.closed == 0


@pytest.mark.parametrize("arguments", [{"license_key": "obsolete"}, {"embedded_nearmate": object()}])
def test_removed_activation_parameters_are_rejected(arguments):
    """旧激活参数应直接报错，避免留下行为不明确的兼容入口。"""
    with pytest.raises(TypeError):
        TTSEngine("fixture", **arguments)


def test_old_second_positional_argument_is_rejected():
    """旧调用中的第二位置参数不能被解释成设备选择。"""
    with pytest.raises(TypeError):
        TTSEngine("fixture", "obsolete")

    parameters = inspect.signature(TTSEngine).parameters
    assert parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        value.kind is inspect.Parameter.KEYWORD_ONLY
        for name, value in parameters.items()
        if name != "model"
    )


def test_custom_home_keeps_model_and_prompt_directories(tmp_path, monkeypatch):
    """路径模块迁移后仍遵循既有环境变量和缓存目录布局。"""
    home = tmp_path / "custom" / "home"
    monkeypatch.setenv("STREAMVOX_HOME", str(home))

    assert get_streamvox_home() == home
    assert get_default_model_root() == home / "ckpts"
    assert (home / "ckpts").is_dir()


def test_windows_home_uses_appdata(tmp_path, monkeypatch):
    """未设置自定义目录时继续使用 Windows 用户应用数据目录。"""
    monkeypatch.delenv("STREAMVOX_HOME", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert get_streamvox_home() == tmp_path / "StreamVox"


def test_benchmark_cli_no_longer_accepts_activation_options():
    """基准工具应直接运行，并明确拒绝已移除的激活选项。"""
    script = Path(__file__).resolve().parents[1] / "tools" / "benchmark_streamvox_models.py"
    # Windows 默认管道编码随系统区域变化，子进程显式使用 UTF-8 保持测试可复现。
    help_result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--license-key" not in help_result.stdout

    old_option = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), "--license-key", "obsolete"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert old_option.returncode == 2
    assert "unrecognized arguments" in old_option.stderr
