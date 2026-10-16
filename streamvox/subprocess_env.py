from __future__ import annotations

"""
跨 Python 虚拟环境启动子进程时使用的环境变量整理工具。

这个模块的核心职责，是在“父进程解释器版本 A 调起子进程解释器版本 B”时，
移除会污染子解释器启动过程的环境变量，并补齐目标虚拟环境自己的 PATH 上下文。
当前发布链路里最典型的场景是：
1. 用 `.venv310` 启动 `streamvox.release_matrix`；
2. 再由它去调用 `.venv312` / `.venv313` 的 `python.exe`；
3. 如果父进程保留了 `PYTHONHOME`、`PYTHONPATH`、`VIRTUAL_ENV` 等变量，
   子解释器可能会错误加载到父版本运行时，导致版本探测、Nuitka 或 build 直接失败。
"""

import os
from pathlib import Path


_PYTHON_ENV_BLOCKLIST: tuple[str, ...] = (
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
)


def _guess_virtual_env_root(python_executable: Path) -> Path | None:
    """
    根据目标解释器路径反推出其虚拟环境根目录。

    入参：
    - python_executable: 目标 Python 可执行文件路径，典型值为 `.../Scripts/python.exe`
      或 `.../bin/python`。

    返回：
    - 若路径符合常见虚拟环境结构，则返回虚拟环境根目录；
    - 否则返回 `None`，调用方继续只做最小环境净化。
    """
    parent_dir = python_executable.parent
    if parent_dir.name.lower() in {"scripts", "bin"}:
        return parent_dir.parent
    return None


def build_python_subprocess_env(
    *,
    python_executable: str | Path,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    构建适用于目标 Python 解释器的干净子进程环境。

    核心职责：
    - 清除父解释器遗留的 Python 运行时变量，避免跨版本污染；
    - 把目标解释器所在目录前置到 PATH，确保子进程优先加载自己的 DLL / 脚本；
    - 在能够识别虚拟环境根目录时，恢复目标环境对应的 `VIRTUAL_ENV`；
    - 支持叠加额外业务环境变量，例如 wheel tag 注入。

    入参：
    - python_executable: 即将用于启动子进程的目标解释器路径。
    - extra_env: 调用方希望额外写入的环境变量。

    返回：
    - 可以直接传给 `subprocess.run(..., env=...)` 的环境变量字典。
    """
    sanitized_env = os.environ.copy()
    for variable_name in _PYTHON_ENV_BLOCKLIST:
        sanitized_env.pop(variable_name, None)

    python_path = Path(python_executable).expanduser()
    if python_path.exists():
        python_path = python_path.resolve()

    path_entries = [str(python_path.parent)]
    virtual_env_root = _guess_virtual_env_root(python_path)
    if virtual_env_root is not None:
        sanitized_env["VIRTUAL_ENV"] = str(virtual_env_root)
        path_entries.append(str(virtual_env_root))

    current_path = sanitized_env.get("PATH", "")
    if current_path:
        path_entries.append(current_path)
    sanitized_env["PATH"] = os.pathsep.join(path_entries)

    if extra_env:
        sanitized_env.update(extra_env)
    return sanitized_env
