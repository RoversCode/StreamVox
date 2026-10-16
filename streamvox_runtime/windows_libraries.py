"""校验 Windows 已加载 DLL 的真实来源，避免复用开发环境的 CUDA。"""

from __future__ import annotations

import ctypes
from pathlib import Path


def verify_loaded_library_path(expected: Path) -> None:
    """拒绝已经从其他目录加载的同名 DLL。

    Args:
        expected: 本次发行环境中的 DLL 绝对路径。
    Returns:
        未加载或来源一致时返回；来源冲突时抛出异常。
    """
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint32
    handle = kernel.GetModuleHandleW(expected.name)
    if not handle:
        return
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
    if not length or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), f"无法确认 {expected.name} 的加载来源")
    actual = Path(buffer.value).resolve()
    if actual != expected.resolve():
        raise RuntimeError(f"{expected.name} 已从其他环境加载：{actual}；预期：{expected}")
