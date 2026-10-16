"""验证两种 NVIDIA 使用方不会互相拉入多余原生库。"""

import ctypes
from unittest.mock import patch

import pytest

from streamvox_runtime import native_loader


@pytest.mark.skipif(native_loader.os.name != 'nt', reason='Windows 原生库加载')
def test_llama_loads_only_three_required_libraries(tmp_path):
    """cuDNN 已存在时 llama 也不能加载它。"""
    names = ('cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll')
    for name in (*names, 'cudnn64_9.dll'):
        (tmp_path / name).write_bytes(b'fake')
    with patch.object(native_loader, 'nvidia_library_directories', return_value=(tmp_path,)), patch.object(native_loader, '_loaded', set()), patch.object(native_loader, '_dll_handles', []), patch.object(native_loader.os, 'add_dll_directory'), patch.object(ctypes, 'WinDLL') as load, patch('streamvox_runtime.windows_libraries.verify_loaded_library_path'):
        native_loader.initialize_nvidia_runtime(consumer='llama', required=True)
        native_loader.initialize_nvidia_runtime(consumer='llama', required=True)
    assert [call.args[0] for call in load.call_args_list] == [str(tmp_path / name) for name in names]


def test_unknown_consumer_is_rejected():
    """拼错使用方时必须尽早失败，不能退回完整 CUDA 加载。"""
    with pytest.raises(ValueError, match='使用方'):
        native_loader.nvidia_library_directories(consumer='unknown')


def test_llama_discovery_excludes_onnx_packages(tmp_path):
    """当前解释器有 cuDNN 也不将它纳入 llama 搜索目录。"""
    root = tmp_path / 'venv'
    packages = root / 'Lib/site-packages'
    for name in ('cuda_runtime', 'cublas', 'cudnn', 'cufft'):
        (packages / 'nvidia' / name / 'bin').mkdir(parents=True)
    with patch.object(native_loader.sys, 'prefix', str(root)), patch.object(native_loader.site, 'getsitepackages', return_value=[str(packages)]), patch.dict(native_loader.os.environ, {'STREAMVOX_NVIDIA_LIB_DIRS': ''}):
        dirs = native_loader.nvidia_library_directories(consumer='llama')
    assert {directory.parent.name for directory in dirs} == {'cuda_runtime', 'cublas'}
