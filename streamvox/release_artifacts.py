"""生成并核对公开源码提交对应的发行附件与校验清单。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile

from .native_assets import native_manifest


PUBLIC_REPOSITORY = "RoversCode/StreamVox"


def validate_source_commit(source_commit: str) -> str:
    """只接受能够唯一追溯公共源码的完整提交号。

    Args:
        source_commit: 公共仓库提交号。
    Returns:
        规范化的小写完整提交号。
    """
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError("source_commit 必须是公共源码仓库的 40 位提交号。")
    return source_commit.lower()


def sha256_file(path: str | Path) -> str:
    """按块计算大文件摘要，避免把全部原生库读入内存。

    Args:
        path: 文件路径。
    Returns:
        十六进制 SHA256。
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_checksums(dist_root: str | Path) -> list[dict]:
    """只收集最终目录中的 wheel 与源码发行包。

    Args:
        dist_root: 独立发行目录。
    Returns:
        文件名、字节数与摘要组成的清单。
    """
    root = Path(dist_root)
    files = sorted([*root.glob("*.whl"), *root.glob("*.tar.gz")])
    return [{"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files if path.is_file()]


def format_sha256sums(entries: list[dict]) -> str:
    """输出标准校验工具可读取的摘要文本。

    Args:
        entries: 已计算的文件清单。
    Returns:
        SHA256SUMS 文件内容。
    """
    return "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)


def write_release_artifacts(dist_root: Path, version: str, source_commit: str,
                            source_manifest: dict[str, str]) -> tuple[Path, Path]:
    """在同一轮校验结果上生成带公共源码来源的两份清单。

    Args:
        dist_root: 含最终构建产物的目录。
        version: SDK 发行版本。
        source_commit: 公共源码仓库完整提交号。
        source_manifest: 从该提交导出的源码、构建配置与许可摘要。
    Returns:
        校验文件与元数据文件的路径。
    """
    commit = validate_source_commit(source_commit)
    if not source_manifest:
        raise ValueError("发行来源必须包含实际源码摘要。")
    entries = build_checksums(dist_root)
    if not entries:
        raise ValueError("发行目录没有 wheel 或源码发行包。")
    metadata = {"package": "streamvox", "version": version, "channel": "stable",
                "source_repository": f"https://github.com/{PUBLIC_REPOSITORY}", "source_commit": commit,
                "native_assets": native_manifest(),
                "source_files": source_manifest,
                "generated_at": datetime.now(timezone.utc).isoformat(), "artifacts": entries}
    sums_path = dist_root / "SHA256SUMS.txt"
    metadata_path = dist_root / "release-metadata.json"
    # 校验材料也是发行证据，禁止覆盖已生成的同名文件。
    with sums_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(format_sha256sums(entries))
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return sums_path, metadata_path


def verify_release_artifacts(dist_root: Path, version: str, source_commit: str) -> list[Path]:
    """发布前重新计算摘要并拒绝被改动或混入的附件。

    Args:
        dist_root: 待发布目录。
        version: 预期 SDK 版本。
        source_commit: 公共源码仓库完整提交号。
    Returns:
        经过校验的全部上传文件。
    """
    metadata_path = dist_root / "release-metadata.json"
    sums_path = dist_root / "SHA256SUMS.txt"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = build_checksums(dist_root)
    expected = {f"streamvox-{version}-cp312-none-win_amd64.whl", f"streamvox-{version}.tar.gz"}
    if {entry["path"] for entry in entries} != expected:
        raise ValueError("发行目录必须只包含本次版本的唯一 wheel 与源码发行包。")
    if (metadata.get("version"), metadata.get("source_commit"), metadata.get("source_repository")) != (
        version, validate_source_commit(source_commit), f"https://github.com/{PUBLIC_REPOSITORY}"
    ):
        raise ValueError("发行元数据的版本或公共源码来源不匹配。")
    if (metadata.get("artifacts") != entries or metadata.get("native_assets") != native_manifest()
            or sums_path.read_text(encoding="utf-8") != format_sha256sums(entries)):
        raise ValueError("发行附件与校验清单不一致。")
    sources = metadata.get("source_files")
    if not isinstance(sources, dict) or not sources or any(
        not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in sources.items()
    ):
        raise ValueError("发行元数据缺少有效的实际源码摘要。")
    archived_sources = {}
    with tarfile.open(dist_root / f"streamvox-{version}.tar.gz", "r:gz") as archive:
        for member in archive:
            relative = member.name.removeprefix(f"streamvox-{version}/")
            if relative not in sources:
                continue
            if not member.isfile() or relative in archived_sources:
                raise ValueError("源码发行包含重复文件或非法来源链接。")
            with archive.extractfile(member) as handle:
                archived_sources[relative] = hashlib.file_digest(handle, "sha256").hexdigest()
    if archived_sources != sources:
        raise ValueError("源码发行与公开提交的源码摘要不一致。")
    files = [dist_root / entry["path"] for entry in entries] + [sums_path, metadata_path]
    if {path.name for path in dist_root.iterdir()} != {path.name for path in files}:
        raise ValueError("发行目录包含清单以外的文件或目录。")
    return files


def main() -> None:
    """为已有本地产物生成不可覆盖的发行清单。"""
    parser = argparse.ArgumentParser(description="生成带公开源码提交的发行校验清单。")
    parser.add_argument("--dist-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    for path in write_release_artifacts(args.dist_root, args.version, args.source_commit, manifest):
        print(path)


if __name__ == "__main__":
    main()
