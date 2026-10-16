from __future__ import annotations

from dataclasses import dataclass
import os
import platform


@dataclass(slots=True)
class RuntimeEnvironment:
    os_name: str
    machine: str
    python_version: str
    preferred_backend: str


def detect_runtime_environment(device: str = "auto") -> RuntimeEnvironment:
    '''
        检测运行环境
    '''
    os_name = platform.system().lower()
    machine = platform.machine().lower()
    python_version = platform.python_version()
    preferred_backend = resolve_backend(device=device, os_name=os_name)  # 运行后端
    return RuntimeEnvironment(
        os_name=os_name,
        machine=machine,
        python_version=python_version,
        preferred_backend=preferred_backend,
    )


def resolve_backend(device: str, os_name: str | None = None) -> str:
    normalized = device.lower()
    if normalized == "gpu":
        return "cuda"
    if normalized == "directml":
        return "dml"
    if normalized != "auto":
        return normalized

    resolved_os = (os_name or platform.system()).lower()
    if resolved_os.startswith("win"):
        return "dml"
    if resolved_os == "linux":
        if _cuda_is_likely_available():
            return "cuda"
        return "cpu"
    return "cpu"


def _cuda_is_likely_available() -> bool:
    markers = (
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    )
    for marker in markers:
        if marker and marker.lower() not in {"", "none", "void"}:
            return True
    return False
