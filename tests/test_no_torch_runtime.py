"""保证运行包的所有 Python 模块不再直接依赖 Torch。"""

import ast
from pathlib import Path


def test_runtime_has_no_torch_imports():
    """逐模块检查导入，避免仅测到某个模型恰好未导入 Torch。"""
    root = Path(__file__).resolve().parents[1]
    forbidden = {"torch", "torchaudio", "torchvision"}
    violations = []
    for package in ("streamvox", "streamvox_runtime"):
        for path in (root / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(name.split(".")[0] in forbidden for name in names):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations
