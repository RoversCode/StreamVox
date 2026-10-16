from __future__ import annotations

from .builtin_models import register_default_models
from .registry import default_registry


# 模块导入时立即完成默认模型注册，外层只需要拿 registry 使用即可。
register_default_models()


__all__ = ["register_default_models", "default_registry"]
