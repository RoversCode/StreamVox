from __future__ import annotations

import os
import shutil
import subprocess
import sys
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..utils.log import logger


LLAMA_GPU_BACKEND_FILENAMES: dict[str, tuple[str, ...]] = {
    "cuda": ("libggml-cuda.so", "ggml-cuda.dll", "libggml-cuda.dylib"),
    "vulkan": ("libggml-vulkan.so", "ggml-vulkan.dll", "libggml-vulkan.dylib"),
    "metal": ("libggml-metal.dylib",),
}
LLAMA_GPU_BACKEND_PRIORITY: tuple[str, ...] = ("cuda", "vulkan", "metal")


@dataclass(frozen=True, slots=True)
class LlamaBackendResolution:
    """
    llama.cpp GGUF 后端解析结果。

    参数:
        backend: 最终实际使用的 backend，取值为 cpu/cuda/vulkan/metal。
        gpu_index: GPU backend 对应的设备索引；CPU 时为 None。
        packaged_backends: 当前 wheel/bin 内实际存在的 GPU backend 集合。
        reason: 面向日志与诊断的解析原因。

    返回:
        本数据结构只作为解析函数返回值，不主动执行任何 I/O。

    异常:
        本数据结构本身不抛出异常。
    """

    backend: str
    gpu_index: int | None
    packaged_backends: tuple[str, ...]
    reason: str


def default_llama_bin_root() -> Path:
    """
    返回 StreamVox 包内 llama.cpp 动态库目录。

    参数:
        无显式参数；路径由当前模块位置推导。

    返回:
        返回 `streamvox/bin` 的绝对路径。

    异常:
        本函数不访问文件系统，不主动抛出异常。
    """

    return Path(__file__).resolve().parents[1] / "bin"


def packaged_gpu_backends(bin_root: Path | None = None) -> tuple[str, ...]:
    """
    扫描当前包内实际打进去的 llama.cpp GPU backend。

    参数:
        bin_root: 可选的 bin 目录；为空时使用 `streamvox/bin`。

    返回:
        按优先级返回已打包 GPU backend 元组，例如 `("cuda", "vulkan")`。

    异常:
        本函数只做文件存在性检查，不主动抛出异常；非法路径会自然表现为空集合。
    """

    root = default_llama_bin_root() if bin_root is None else Path(bin_root)
    packaged: list[str] = []
    for backend in LLAMA_GPU_BACKEND_PRIORITY:
        filenames = LLAMA_GPU_BACKEND_FILENAMES[backend]
        if any((root / filename).exists() for filename in filenames):
            packaged.append(backend)
    return tuple(packaged)


def first_packaged_gpu_backend(bin_root: Path | None = None) -> str | None:
    """
    返回当前包内优先级最高的单个 GPU backend。

    参数:
        bin_root: 可选的 bin 目录；为空时使用 `streamvox/bin`。

    返回:
        返回第一个已打包 backend；没有 GPU backend 时返回 None。

    异常:
        本函数不主动抛出异常。
    """

    backends = packaged_gpu_backends(bin_root)
    if not backends:
        return None
    return backends[0]


def _has_linux_nvidia_gpu() -> bool:
    """
    通过 Linux 设备节点判断是否存在 NVIDIA 显卡/驱动。

    参数:
        无显式参数。

    返回:
        只要 `/proc/driver/nvidia/gpus` 或 `/dev/nvidia*` 暴露设备，就返回 True。

    异常:
        文件系统访问失败时按未检测到 NVIDIA 处理，不向外抛出异常。
    """

    try:
        proc_root = Path("/proc/driver/nvidia/gpus")
        if proc_root.exists() and any(proc_root.iterdir()):
            return True
        return any(Path("/dev").glob("nvidia[0-9]*"))
    except Exception:
        return False


def _nvidia_smi_detects_gpu() -> bool:
    """
    使用 nvidia-smi 做轻量 NVIDIA 探测。

    参数:
        无显式参数。

    返回:
        命令存在且能列出至少一块 GPU 时返回 True。

    异常:
        命令缺失、超时或驱动异常均吞掉并返回 False，避免设备解析阻断 runtime 初始化。
    """

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return False
    return bool(result.stdout.strip())


def _has_cuda_driver() -> bool:
    """
    判断当前系统是否具备 CUDA driver/NVIDIA 设备信号。

    参数:
        无显式参数。

    返回:
        检测到 NVIDIA driver 或 NVIDIA GPU 时返回 True。

    异常:
        探测异常会被降级为 False。
    """

    if os.name == "nt":
        # 只看 cudart 会把“装了 CUDA runtime 但没有 NVIDIA driver/显卡”的机器误判成 CUDA 可用。
        return find_library("nvcuda") is not None or _nvidia_smi_detects_gpu()
    return (
        _has_linux_nvidia_gpu()
        or find_library("cuda") is not None
        or Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1").exists()
        or _nvidia_smi_detects_gpu()
    )


def _has_vulkan_runtime() -> bool:
    """
    判断当前系统是否具备 Vulkan loader。

    参数:
        无显式参数。

    返回:
        找到系统 Vulkan loader 时返回 True。

    异常:
        探测异常会被降级为 False。
    """

    try:
        if os.name == "nt":
            return find_library("vulkan-1") is not None
        if sys.platform == "darwin":
            return find_library("vulkan") is not None
        return find_library("vulkan") is not None or Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists()
    except Exception:
        return False


def _has_linux_render_device() -> bool:
    """
    判断 Linux 是否存在可供 Vulkan 使用的 render device。

    参数:
        无显式参数。

    返回:
        检测到 DRM render node 或 NVIDIA 设备时返回 True。

    异常:
        探测异常会被降级为 False。
    """

    try:
        return any(Path("/dev/dri").glob("renderD*")) or _has_linux_nvidia_gpu()
    except Exception:
        return False


def llama_backend_available(backend: str) -> bool:
    """
    判断指定 llama.cpp GPU backend 在当前机器是否具备基础运行条件。

    参数:
        backend: 待判断的 backend 名称，支持 cuda/vulkan/metal。

    返回:
        当前宿主机具备该 backend 的基础运行时条件时返回 True。

    异常:
        未知 backend 不抛异常，统一返回 False，便于调用方做 CPU 降级。
    """

    normalized = str(backend).strip().lower()
    if normalized == "cuda":
        return _has_cuda_driver()
    if normalized == "vulkan":
        if os.name == "nt":
            return _has_vulkan_runtime()
        return _has_vulkan_runtime() and _has_linux_render_device()
    if normalized == "metal":
        return sys.platform == "darwin"
    return False


def _ordered_unique_backends(backends: Iterable[str]) -> tuple[str, ...]:
    """
    将 backend 集合清洗成已知优先级顺序。

    参数:
        backends: 待清洗的 backend 名称序列。

    返回:
        返回去重、归一化且按 CUDA/Vulkan/Metal 优先级排列的 backend 元组。

    异常:
        本函数不主动抛出异常；未知值会被忽略。
    """

    requested = {str(backend).strip().lower() for backend in backends}
    return tuple(backend for backend in LLAMA_GPU_BACKEND_PRIORITY if backend in requested)


def normalize_explicit_llama_backend(explicit_backend: str | None) -> str | None:
    """
    规范化用户显式指定的 llama.cpp backend。

    参数:
        explicit_backend: 用户传入的 backend 名称；为空时表示自动选择。

    返回:
        返回规整后的 backend 名称，或 None。

    异常:
        当 backend 不是字符串、为空白或不在已知集合中时抛出异常。
    """

    if explicit_backend is None:
        return None
    if not isinstance(explicit_backend, str):
        raise TypeError("explicit llama backend must be a string or None")
    normalized = explicit_backend.strip().lower()
    if not normalized:
        raise ValueError("explicit llama backend must not be empty")
    if normalized not in (*LLAMA_GPU_BACKEND_PRIORITY, "cpu"):
        choices = ", ".join(LLAMA_GPU_BACKEND_PRIORITY)
        raise ValueError(f"explicit llama backend must be one of: {choices}")
    return normalized


def _first_available_gpu_backend(
    packaged_backends: tuple[str, ...],
    availability_checker: Callable[[str], bool],
) -> str | None:
    """
    返回按优先级扫描后的首个可用 GPU backend。

    参数:
        packaged_backends: 当前 wheel 内打包的 GPU backend 列表。
        availability_checker: backend 可用性判断函数。

    返回:
        返回首个可用 backend；没有可用 GPU backend 时返回 None。

    异常:
        本函数不主动抛出异常。
    """

    for backend in packaged_backends:
        if availability_checker(backend):
            return backend
    return None


def select_llama_backend(
    *,
    mode: str,
    device_index: int | None,
    model_label: str,
    packaged_backends: Iterable[str] | None = None,
    availability_checker: Callable[[str], bool] | None = None,
    explicit_backend: str | None = None,
) -> LlamaBackendResolution:
    """
    按统一规则选择 llama.cpp GGUF backend。

    参数:
        mode: adapter 解析出的执行模式，支持 auto/cpu/gpu。
        device_index: 用户指定的 GPU 索引；为空时 GPU backend 默认使用 0。
        model_label: 日志中展示的模型族名称，例如 qwen3/s2-pro/voxcpm2-gguf。
        packaged_backends: 可选的包内 backend 集合；为空时自动扫描 `streamvox/bin`。
        availability_checker: 可选的可用性判断函数，测试中可注入以隔离真实硬件。
        explicit_backend: 用户显式指定的 llama.cpp backend；为空时按自动策略选择。

    返回:
        返回包含最终 backend、GPU 索引、包内 backend 和原因的解析结果。

    异常:
        mode 不属于 auto/cpu/gpu 时抛出 ValueError。
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unsupported llama target mode: {mode}")

    packaged = _ordered_unique_backends(packaged_gpu_backends() if packaged_backends is None else packaged_backends)
    checker = llama_backend_available if availability_checker is None else availability_checker
    normalized_explicit = normalize_explicit_llama_backend(explicit_backend)

    if normalized_mode == "cpu" or normalized_explicit == "cpu":
        resolution = LlamaBackendResolution(
            backend="cpu",
            gpu_index=None,
            packaged_backends=packaged,
            reason=(
                "user requested CPU"
                if normalized_explicit is None
                else f"user requested CPU; ignored explicit llama_backend={normalized_explicit}"
            ),
        )
        _log_resolution(
            model_label=model_label,
            mode=normalized_mode,
            resolution=resolution,
            warning=normalized_explicit is not None,
        )
        return resolution

    if normalized_explicit is not None:
        if normalized_explicit not in packaged:
            raise ValueError(
                f"requested llama_backend={normalized_explicit!r} for model {model_label!r} is not packaged in this build"
            )
        if normalized_explicit == "cuda" and not checker("cuda"):
            auto_backend = _first_available_gpu_backend(packaged, checker)
            if auto_backend is not None:
                resolution = LlamaBackendResolution(
                    backend=auto_backend,
                    gpu_index=device_index if device_index is not None else 0,
                    packaged_backends=packaged,
                    reason=(
                        f"ignored explicit cuda request because CUDA is unavailable on this host; "
                        f"defaulted to {auto_backend}"
                    ),
                )
                _log_resolution(model_label=model_label, mode=normalized_mode, resolution=resolution, warning=True)
                return resolution
            resolution = LlamaBackendResolution(
                backend="cpu",
                gpu_index=None,
                packaged_backends=packaged,
                reason="ignored explicit cuda request because CUDA is unavailable on this host; defaulted to cpu",
            )
            _log_resolution(model_label=model_label, mode=normalized_mode, resolution=resolution, warning=True)
            return resolution
        if not checker(normalized_explicit):
            raise ValueError(
                f"requested llama_backend={normalized_explicit!r} for model {model_label!r} is not available on this host"
            )
        resolution = LlamaBackendResolution(
            backend=normalized_explicit,
            gpu_index=device_index if device_index is not None else 0,
            packaged_backends=packaged,
            reason=f"user explicitly requested llama_backend={normalized_explicit}",
        )
        _log_resolution(model_label=model_label, mode=normalized_mode, resolution=resolution, warning=False)
        return resolution

    for backend in packaged:
        if checker(backend):
            resolution = LlamaBackendResolution(
                backend=backend,
                gpu_index=device_index if device_index is not None else 0,
                packaged_backends=packaged,
                reason=f"{backend} backend is packaged and available",
            )
            _log_resolution(model_label=model_label, mode=normalized_mode, resolution=resolution, warning=False)
            return resolution

    reason = "no packaged GPU backend" if not packaged else "packaged GPU backends are not available on this host"
    resolution = LlamaBackendResolution(
        backend="cpu",
        gpu_index=None,
        packaged_backends=packaged,
        reason=reason,
    )
    _log_resolution(model_label=model_label, mode=normalized_mode, resolution=resolution, warning=True)
    return resolution


def _log_resolution(
    *,
    model_label: str,
    mode: str,
    resolution: LlamaBackendResolution,
    warning: bool,
) -> None:
    """
    打印统一 GGUF backend 解析日志。

    参数:
        model_label: 当前模型族名称。
        mode: 用户请求模式。
        resolution: 已解析完成的 backend 结果。
        warning: 是否按降级告警输出。

    返回:
        本函数无返回值。

    异常:
        日志系统异常由调用方环境处理，本函数不额外捕获。
    """

    packaged_label = ",".join(resolution.packaged_backends) if resolution.packaged_backends else "none"
    gpu_index_label = "none" if resolution.gpu_index is None else str(resolution.gpu_index)
    message = (
        "[GGUF后端] 模型=%s 请求模式=%s 打包后端=%s 最终后端=%s GPU索引=%s 原因=%s"
    )
    args = (
        model_label,
        mode,
        packaged_label,
        resolution.backend,
        gpu_index_label,
        resolution.reason,
    )
    if warning:
        logger.warning("[设备回退][重要] " + message, *args)
        return
    logger.info(message, *args)
