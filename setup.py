"""为包含原生 DLL、保留 Python 源码的 Windows 包写入准确标签。"""

from __future__ import annotations

import struct
import sys
import platform

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class StreamVoxBdistWheel(bdist_wheel):
    """将已验收的 Windows CPython 3.12 运行环境写入 wheel。"""

    def finalize_options(self) -> None:
        """禁止把携带 Windows DLL 的发行包标记为通用平台。"""
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        """校验构建宿主，并声明不存在 Python 扩展接口依赖。

        Returns:
            解释器、扩展接口和平台标签。
        """
        if sys.platform != "win32" or sys.implementation.name != "cpython":
            raise RuntimeError("发行 wheel 只支持 Windows CPython。")
        if (sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8
                or platform.machine().lower() not in {"amd64", "x86_64"}):
            raise RuntimeError("发行 wheel 必须使用 64 位 CPython 3.12 构建。")
        return "cp312", "none", "win_amd64"


setup(cmdclass={"bdist_wheel": StreamVoxBdistWheel})
