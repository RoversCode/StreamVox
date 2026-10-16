#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   log.py
@Author  :   RoversCode (Optimized)
@Desc    :   通用日志配置模块 (即插即用)
'''

import os
import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

@dataclass
class LoggerSettings:
    console_enabled: bool = True
    file_enabled: bool = False
    level: int = logging.INFO
    log_dir: str = "logs"
    log_file: str = "latest.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _settings_from_env() -> LoggerSettings:
    return LoggerSettings(
        console_enabled=_env_enabled("STREAMVOX_LOG_CONSOLE", True),
        file_enabled=_env_enabled("STREAMVOX_LOG_FILE", False),
        level=getattr(logging, os.getenv("STREAMVOX_LOG_LEVEL", "INFO").upper(), logging.INFO),
        log_dir=os.getenv("STREAMVOX_LOG_DIR", "logs"),
        log_file=os.getenv("STREAMVOX_LOG_FILE_NAME", "latest.log"),
    )


def _setup_logger(settings: LoggerSettings | None = None):
    settings = settings or _settings_from_env()
    # 1. 自动创建日志目录，防止 FileNotFoundError
    log_dir = settings.log_dir
    if settings.file_enabled and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, settings.log_file)

    # 2. 获取根 logger
    logger = logging.getLogger("AppLogger") # 给 logger 起个名字，避免污染 root logger
    
    # 防止在某些环境中（如 Jupyter）重复添加 handler 导致日志重复打印
    if logger.handlers:
        logger.handlers.clear()

    # 设置最低捕获级别
    logger.setLevel(settings.level)

    # 3. 定义统一的日志格式
    # 格式：时间 - 模块名:行号 - 级别 - 信息
    log_format = logging.Formatter(
        "%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
    )

    # 4. 创建控制台处理器 (输出 INFO 及以上级别的日志)
    if settings.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(settings.level)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # 5. 创建文件处理器 (输出 DEBUG 及以上级别的所有日志，且按大小自动切割)
    # maxBytes=10MB, backupCount=5 表示最多保留 5 个 10MB 的备份文件
    # 指定 encoding="utf-8" 防止中文乱码
    if settings.file_enabled:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(settings.level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def configure_logger(
    *,
    console_enabled: bool | None = None,
    file_enabled: bool | None = None,
    level: int | None = None,
    log_dir: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """运行时重配日志开关，方便测试和正式环境按需关闭控制台/文件输出。"""
    base = _settings_from_env()
    settings = LoggerSettings(
        console_enabled=base.console_enabled if console_enabled is None else console_enabled,
        file_enabled=base.file_enabled if file_enabled is None else file_enabled,
        level=base.level if level is None else level,
        log_dir=base.log_dir if log_dir is None else log_dir,
        log_file=base.log_file if log_file is None else log_file,
        max_bytes=base.max_bytes,
        backup_count=base.backup_count,
    )
    global logger
    logger = _setup_logger(settings)
    return logger

# 实例化 logger，外部只需 import 这个对象
logger = _setup_logger()
