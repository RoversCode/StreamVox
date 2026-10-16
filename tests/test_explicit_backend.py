"""验证明确选择的执行后端不会静默回退。"""

from unittest.mock import Mock, patch

import pytest

from streamvox.models.base import ExecutionTarget
from streamvox.models.onnx_backend import resolve_explicit_onnx_backend, verify_session_provider
from streamvox.models.llama_backend import select_llama_backend


def test_directml_selection_survives_cuda_availability():
    """设备默认策略不能覆盖调用方明确选中的 DirectML。"""
    target = ExecutionTarget('onnx', 'gpu', 2, 'directml')
    with patch('onnxruntime.get_available_providers', return_value=['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']):
        assert resolve_explicit_onnx_backend(target) == ('dml', 2)


def test_unavailable_explicit_provider_fails():
    """只装 CPU 的环境不能把明确的 DirectML 请求当成功。"""
    with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
        with pytest.raises(RuntimeError, match='不可用'):
            resolve_explicit_onnx_backend(ExecutionTarget('onnx', 'gpu', 0, 'directml'))


def test_session_cpu_fallback_fails():
    """注册了 GPU 并不代表会话真的在 GPU 上初始化。"""
    session = Mock()
    session.get_providers.return_value = ['CPUExecutionProvider']
    with pytest.raises(RuntimeError, match='回退'):
        verify_session_provider(session, ['DmlExecutionProvider', 'CPUExecutionProvider'])
    session.disable_fallback.assert_not_called()


def test_explicit_llama_cpu_is_independent_of_gpu_request():
    """ONNX 使用 GPU 时也允许 llama 单独选 CPU。"""
    result = select_llama_backend(mode='gpu', device_index=0, model_label='test', explicit_backend='cpu', packaged_backends=('cuda', 'vulkan'))
    assert result.backend == 'cpu'
