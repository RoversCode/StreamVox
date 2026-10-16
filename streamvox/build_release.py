"""展示开源 SDK 的构建计划，不创建产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .build_matrix import find_target, release_plan
from .release_artifacts import validate_source_commit


def build_parser() -> argparse.ArgumentParser:
    """定义构建计划所需的显式输入。

    Returns:
        命令行解析器。
    """
    parser = argparse.ArgumentParser(description="展示 StreamVox 开源构建计划。")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--os", default="windows", choices=["windows"])
    parser.add_argument("--python-version", default="3.12", choices=["3.12"])
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    """输出经过版本和提交号校验的计划。"""
    args = build_parser().parse_args()
    plan = release_plan(args.project_root.resolve(), args.python, find_target(args.os, args.python_version),
                        validate_source_commit(args.source_commit))
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    for key, value in plan.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
