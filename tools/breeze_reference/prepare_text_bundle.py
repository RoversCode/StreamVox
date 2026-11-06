"""从已验收的 Breeze bundle 派生支持显卡文本主干的独立 bundle。"""

from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path
import shutil
import stat

from streamvox.model_bundle import sha256_file, validate_local_bundle, write_local_manifest

from .text_encoder_split import split_text_encoder


def prepare_text_bundle(source: Path, output: Path) -> Path:
    """保留原始资产，派生三个文本资产并生成独立清单。

    Args:
        source: 已带完整校验清单的原始 Breeze 模型目录。
        output: 尚不存在的独立输出目录。
    Returns:
        新 bundle 的完整清单路径。
    """
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("快速 bundle 必须位于原模型目录之外。")
    validated = validate_local_bundle(source, "breeze-tts-2-gguf", True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if any(item["path"] == "ckpt/text_encoder_trunk.onnx" for item in manifest["assets"]):
        raise ValueError("源 bundle 已包含快速文本资产，请直接使用。")
    source_report = json.loads((source / "ckpt/export-report.json").read_text(encoding="utf-8"))
    if source_report.get("tiny") or source_report.get("generation", {}).get("validated") is not True:
        raise ValueError("快速 bundle 必须派生自已完成原生数值验收的真实模型。")
    output.mkdir(parents=True, exist_ok=False)
    owned_output = output.lstat()
    try:
        assets = []
        runtime_assets = {path.resolve() for path in validated.asset_paths}
        for item in manifest["assets"]:
            original = (source / item["path"]).resolve(strict=True)
            if not original.is_relative_to(source) or not original.is_file():
                raise ValueError("源资产必须是 bundle 内的普通文件。")
            if original not in runtime_assets and sha256_file(original) != item["sha256"]:
                raise ValueError(f"源许可或归属文件摘要不符：{item['path']}")
            destination = output / original.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 元数据单独复制以便增补新资产，大权重优先链接，文件系统不支持时再复制。
            if original.suffix == ".json":
                shutil.copy2(original, destination)
                assets.append(destination)
                continue
            try:
                destination.hardlink_to(original)
            except OSError as error:
                if (
                    error.errno not in {errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
                    and getattr(error, "winerror", None) not in {1, 17, 50}
                ):
                    raise
                shutil.copy2(original, destination)
            assets.append(destination)
        created = split_text_encoder(output / "ckpt")
        report_path = output / "ckpt/export-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["text_encoder_split"] = {"validated_structure": True, "precision": "float32", "lookup_backend": "cpu", "trunk_backend": "cuda"}
        for path in created:
            report["files"][path.relative_to(output / "ckpt").as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return write_local_manifest(
            "breeze-tts-2-gguf", output, assets + list(created),
            sources=manifest["sources"], adapter=manifest["adapter"],
        )
    except BaseException as error:
        try:
            current = output.lstat()
            # 只清理由本次创建且仍保持原身份的目录，避免路径替换误删源模型或用户目录。
            if (
                output.resolve(strict=True) != output or output == source
                or output.is_relative_to(source) or source.is_relative_to(output)
                or output.is_symlink() or output.is_junction()
                or not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != (owned_output.st_dev, owned_output.st_ino)
            ):
                raise RuntimeError("输出目录路径或身份已变化，已停止半成品清理。")
            shutil.rmtree(output)
        except BaseException as cleanup_error:
            error.add_note(f"快速 bundle 半成品清理失败：{cleanup_error}")
        raise


def main() -> None:
    """读取本地模型及目标目录，完成可选快速模式资产准备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare_text_bundle(arguments.source, arguments.output))


if __name__ == "__main__":
    main()
