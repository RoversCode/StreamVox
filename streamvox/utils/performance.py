from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from .log import logger


def format_perf_value(value: float) -> str:
    """
    将性能指标数值格式化为 6 位有效数字。

    参数:
        value: 需要输出到性能日志中的数值，通常是秒级耗时或 RTF。

    返回:
        返回最多保留 6 位有效数字的字符串，便于日志既稳定又不过度冗长。

    异常:
        本函数不主动抛出异常；传入非数值对象时由 float 格式化逻辑抛出 TypeError 或 ValueError。
    """
    # 使用通用有效数字格式，而不是固定小数位，避免极小耗时被格式化成 0.000000。
    return f"{value:.6g}"


@contextmanager
def timed_block(message: str = "") -> Iterator[None]:
    """with 块结束时打印耗时；message 为耗时信息前的自定义前缀。"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        prefix = f"{message} " if message else ""
        logger.info(f"{prefix}耗时: {format_perf_value(elapsed)}s")
