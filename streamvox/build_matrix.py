"""集中定义开源 SDK 的平台、解释器与产物名称。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .version import __version__ as PACKAGE_VERSION


PACKAGE_NAME = "streamvox"


@dataclass(frozen=True, slots=True)
class BuildTarget:
    """描述携带三种原生后端且保留 Python 源码的 Windows 发行目标。

    Args:
        os_name: 目标系统名称。
        python_version: 已验收的解释器主次版本。
        python_tag: wheel 解释器标签。
        abi_tag: wheel 的 Python 扩展接口标签。
        platform_tag: wheel 平台标签。
    """

    os_name: str
    python_version: str
    python_tag: str
    abi_tag: str
    platform_tag: str

    @property
    def slug(self) -> str:
        """返回隔离当前平台与解释器的目录名。"""
        return f"{self.os_name}-py{self.python_version.replace('.', '')}"

    @property
    def wheel_filename(self) -> str:
        """返回唯一开源 wheel 的文件名。"""
        return f"{PACKAGE_NAME}-{PACKAGE_VERSION}-{self.python_tag}-{self.abi_tag}-{self.platform_tag}.whl"

    @property
    def sdist_filename(self) -> str:
        """返回能够重新构建 wheel 的源码发行文件名。"""
        return f"{PACKAGE_NAME}-{PACKAGE_VERSION}.tar.gz"


SUPPORTED_BUILD_TARGETS = (BuildTarget("windows", "3.12", "cp312", "none", "win_amd64"),)


def list_targets(os_name: str | None = None) -> list[BuildTarget]:
    """按系统筛选已经验收的发行目标。

    Args:
        os_name: 可选的目标系统。
    Returns:
        匹配的发行目标。
    """
    return [target for target in SUPPORTED_BUILD_TARGETS if os_name is None or target.os_name == os_name]


def find_target(os_name: str, python_version: str) -> BuildTarget:
    """拒绝尚未验收的平台或解释器组合。

    Args:
        os_name: 目标系统。
        python_version: 解释器主次版本。
    Returns:
        匹配的发行目标。
    """
    for target in SUPPORTED_BUILD_TARGETS:
        if (target.os_name, target.python_version) == (os_name, python_version):
            return target
    raise ValueError(f"不支持的发行目标：{os_name} / {python_version}")


def build_output_dir(project_root: str | Path, target: BuildTarget) -> Path:
    """返回按平台和版本隔离的构建目录。

    Args:
        project_root: 源码根目录。
        target: 已验收的发行目标。
    Returns:
        当前版本的构建目录。
    """
    return Path(project_root) / "dist" / target.slug / PACKAGE_VERSION


def release_plan(project_root: Path, python_executable: str, target: BuildTarget, source_commit: str) -> dict:
    """生成只供展示、不会执行构建的发行计划。

    Args:
        project_root: 源码根目录。
        python_executable: 构建解释器路径。
        target: 发行目标。
        source_commit: 公共源码仓库的完整提交号。
    Returns:
        预期产物路径和可直接执行的构建命令。
    """
    output = build_output_dir(project_root, target) / "artifacts"
    command = [python_executable, "-m", "streamvox.package_release", "--project-root", str(project_root),
               "--python", python_executable, "--source-commit", source_commit]
    return {"target": target.slug, "expected_wheel": str(output / target.wheel_filename),
            "expected_sdist": str(output / target.sdist_filename), "build_command": subprocess.list2cmdline(command)}
