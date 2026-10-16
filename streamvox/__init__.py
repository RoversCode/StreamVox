from .runtime_compat import ensure_openmp_runtime_compat

# 中文说明：
# 包入口必须尽早做原生运行时兼容处理，保证用户直接 `from streamvox import TTSEngine`
# 时，就已经把 Windows 下最容易炸进程的 OpenMP 冲突兜住。
ensure_openmp_runtime_compat(source="streamvox.__init__")

from .engine import TTSEngine
from .models.base import PromptData, PromptReference
from .version import __version__

__all__ = ["TTSEngine", "PromptData", "PromptReference", "__version__"]
