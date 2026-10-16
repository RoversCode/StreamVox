"""显式加载随发行包提供的 NVIDIA 动态库。

运行期只检查产品配置目录和当前解释器 site-packages 下的 nvidia 包，绝不依赖
PyTorch、用户目录或系统 CUDA 安装。每个进程（包括 multiprocessing 子进程）都应
在创建 ONNX Runtime/llama session 前调用 ``initialize_nvidia_runtime``。
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import site
import sys
import threading
from pathlib import Path
from typing import Literal

_lock = threading.RLock()
_dll_handles: list[object] = []
_loaded: set[str] = set()

NvidiaConsumer = Literal["llama", "onnx_cuda"]

_NVIDIA_PACKAGES = {
    "llama": ("cuda_runtime", "cublas"),
    "onnx_cuda": ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc", "cufft", "curand"),
}
_WINDOWS_LIBRARIES = {
    "llama": ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"),
    "onnx_cuda": (
        "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll",
        "cufft64_11.dll", "curand64_10.dll", "nvrtc-builtins64_124.dll", "nvrtc64_120_0.dll",
        "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
        "cudnn_graph64_9.dll", "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll", "cudnn_heuristic64_9.dll",
    ),
}
_LINUX_LIBRARIES = {
    "llama": ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"),
    "onnx_cuda": (
        "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12",
        "libcufft.so.11", "libcurand.so.10", "libnvrtc.so.12", "libcudnn.so.9",
    ),
}


def ensure_single_onnxruntime_distribution() -> None:
    """确保环境中只存在一种 ONNX Runtime 发行包。

    Returns:
        环境合法时返回；检测到多个发行包时抛出 RuntimeError。
    """
    installed = []
    for name in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        installed.append(name)
    if len(installed) > 1:
        raise RuntimeError(f"发行环境不能同时安装多个 ONNX Runtime 包：{', '.join(installed)}")


def nvidia_library_directories(*, consumer: NvidiaConsumer) -> tuple[Path, ...]:
    """查找发行配置和当前解释器 NVIDIA wheel 的动态库目录。

    Args:
        consumer: llama 只发现其两种运行库；onnx_cuda 发现完整 ONNX CUDA 依赖。
    Returns:
        去重后存在的目录；不搜索用户级包或系统 CUDA。
    """
    if consumer not in _NVIDIA_PACKAGES:
        raise ValueError(f"未知 NVIDIA 运行库使用方：{consumer!r}")
    configured = os.environ.get("STREAMVOX_NVIDIA_LIB_DIRS", "")
    roots = [Path(item).expanduser().resolve() for item in configured.split(os.pathsep) if item]
    for site_root in site.getsitepackages():
        root = Path(site_root).resolve()
        if not root.is_relative_to(Path(sys.prefix).resolve()):
            continue
        roots.extend(root / "nvidia" / package / folder
                     for package in _NVIDIA_PACKAGES[consumer] for folder in ("bin", "lib"))
    result: list[Path] = []
    for path in roots:
        if path.is_dir() and path not in result:
            result.append(path)
    return tuple(result)


def initialize_nvidia_runtime(*, consumer: NvidiaConsumer, required: bool = False) -> tuple[Path, ...]:
    """把 NVIDIA 运行库加入当前进程并按依赖顺序预加载。

    Args:
        consumer: 明确区分 llama 与 ONNX CUDA，避免 llama 顺带加载 cuDNN 等大库。
        required: 为 True 且没有可用目录或关键库时抛出异常；CPU/DirectML 模式应传 False。
    Returns:
        实际加入搜索路径的目录元组。
    """
    directories = nvidia_library_directories(consumer=consumer)
    if not directories:
        if required:
            raise RuntimeError("未找到发行包提供的 NVIDIA 运行库目录")
        return directories
    with _lock:
        if os.name == "nt":
            current_path = os.environ.get("PATH", "").split(os.pathsep)
            for directory in directories:
                text = str(directory)
                if text not in current_path:
                    current_path.insert(0, text)
                if hasattr(os, "add_dll_directory") and text not in _loaded:
                    _dll_handles.append(os.add_dll_directory(text))
                _loaded.add(text)
            os.environ["PATH"] = os.pathsep.join(current_path)
        elif sys.platform.startswith("linux"):
            current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
            os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([str(item) for item in directories] + current))
        # 只有明确使用 CUDA 时才加载；CPU 和 DirectML 不应承担 CUDA 初始化成本。
        if not required:
            return directories
        names = (_WINDOWS_LIBRARIES if os.name == "nt" else _LINUX_LIBRARIES)[consumer]
        for name in names:
            candidates = [directory / name for directory in directories if (directory / name).is_file()]
            if len(candidates) != 1:
                raise RuntimeError(f"NVIDIA 运行库 {name} 必须有唯一来源，实际找到 {len(candidates)} 份")
            path = str(candidates[0])
            if path in _loaded:
                continue
            # 使用绝对路径，缺失依赖直接报错，避免静默借用另一套 CUDA 继续推理。
            if os.name == "nt":
                from .windows_libraries import verify_loaded_library_path

                verify_loaded_library_path(candidates[0])
                # 只允许本 DLL 的目录和 Windows 系统目录解析二级依赖。
                library = ctypes.WinDLL(path, winmode=0x00000100 | 0x00000800)
                verify_loaded_library_path(candidates[0])
            else:
                library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            _dll_handles.append(library)
            _loaded.add(path)
    return directories
