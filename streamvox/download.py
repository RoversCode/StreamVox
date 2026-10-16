from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .manifests import DEFAULT_MODELSCOPE_REPOS
from .model_bundle import (
    is_runtime_manifest_asset,
    load_local_manifest,
    write_local_manifest,
)


@dataclass(slots=True)
class DownloadResult:
    model: str
    target_dir: Path
    downloaded_files: list[Path]


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """描述一次模型下载任务的当前阶段和进度。"""

    model: str
    phase: str
    completed_bytes: int
    total_bytes: int | None
    current_file: str | None = None
    files_completed: int = 0
    files_total: int | None = None

    @property
    def fraction(self) -> float | None:
        """返回 0 到 1 之间的进度；远端下载大小未知时返回 None。"""
        if self.total_bytes is None or self.total_bytes <= 0:
            return None
        return min(1.0, max(0.0, self.completed_bytes / self.total_bytes))


class DownloadCancelled(RuntimeError):
    """用户取消模型下载时抛出的异常。"""


ProgressCallback = Callable[[DownloadProgress], None]


def download_model(
    model: str,
    source: str = "modelscope",
    target: str = "./ckpts",
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> DownloadResult:
    """
    下载 StreamVox 模型 bundle。

    参数:
        model: 需要下载的模型名称。
        source: 下载来源；当前仅支持 modelscope。
        target: 模型保存根目录，最终会落到 target/model 目录。
        progress_callback: 每个下载阶段发生变化时接收 DownloadProgress 的回调。
        cancel_event: 可选的线程事件；被设置后会在安全边界取消任务并保留下载缓存。
        cache_dir: 持久下载缓存根目录；省略时使用 target/.streamvox-download-cache。
        revision: 生产下载需要使用完整 Git 提交哈希，与清单哈希一起传入。
        expected_manifest_sha256: 产品模型目录记录的原始远端清单 SHA256。

    返回:
        返回下载结果，包含模型名、目标目录和 manifest 声明的资产路径。

    异常:
        当下载源不支持、manifest 非法、资产缺失或 sha256 不匹配时抛出异常。
    """
    # 当前下载入口只支持 ModelScope；保留 source 参数是为了维持 CLI/API 的扩展空间。
    if source != "modelscope":
        raise ValueError(f"unsupported download source: {source}")
    try:
        repo_id = DEFAULT_MODELSCOPE_REPOS[model]
    except KeyError as exc:
        raise ValueError(f"no default modelscope repo configured for model: {model}") from exc

    target_root = Path(target).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / model

    # 缓存目录必须稳定存在，应用重启后 ModelScope 才能继续使用已完成的分片。
    download_cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else target_root / ".streamvox-download-cache"
    )
    download_cache_dir = download_cache_root / modelscope_safe_name(model)
    download_cache_dir.mkdir(parents=True, exist_ok=True)

    _check_cancelled(cancel_event)
    _emit_progress(
        progress_callback,
        DownloadProgress(model=model, phase="download", completed_bytes=0, total_bytes=None),
    )

    if (revision is None) != (expected_manifest_sha256 is None):
        raise ValueError("固定版本下载必须同时提供 revision 和 expected_manifest_sha256")
    if revision is not None:
        from .pinned_download import download_pinned_snapshot

        def report(completed: int, total: int, current: str, count: int, all_files: int) -> None:
            """把固定版本下载的字节进度转换为公开 SDK 事件。"""
            _emit_progress(progress_callback, DownloadProgress(
                model=model, phase="download", completed_bytes=completed, total_bytes=total,
                current_file=current, files_completed=count, files_total=all_files,
            ))

        downloaded_root = download_pinned_snapshot(
            repository=repo_id, revision=revision, expected_manifest_sha256=expected_manifest_sha256,
            cache_dir=download_cache_dir, report=report,
            check_cancelled=lambda: _check_cancelled(cancel_event),
        )
    else:
        # 开发者仍可手工抓取最新快照；产品下载入口必须传递预先固定的版本与清单哈希。
        snapshot_download = _resolve_modelscope_snapshot_download()
        downloaded_root = Path(snapshot_download(repo_id=repo_id, cache_dir=str(download_cache_dir))).resolve()
    _check_cancelled(cancel_event)

    # 先写入唯一的隐藏目录。校验全部通过前，当前可用 bundle 不会被触碰。
    publish_dir = target_root / f".{modelscope_safe_name(model)}.publish-{uuid4().hex}"
    try:
        _copy_downloaded_repo(
            downloaded_root=downloaded_root,
            target_dir=publish_dir,
            model=model,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        manifest = load_manifest(model_dir=publish_dir, model=model, source="modelscope")
        downloaded_files = _resolve_manifest_assets(
            target_dir=publish_dir,
            manifest=manifest,
            model=model,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if revision is None:
            write_local_manifest(
                model=model,
                root_dir=publish_dir,
                asset_paths=downloaded_files,
                sources=_merge_manifest_sources(manifest=manifest, source="modelscope"),
                adapter=manifest.get("adapter", model),
            )
        _check_cancelled(cancel_event)
        _emit_progress(
            progress_callback,
            DownloadProgress(model=model, phase="publish", completed_bytes=1, total_bytes=1),
        )
        _atomic_publish(source_dir=publish_dir, target_dir=target_dir)
        _emit_progress(
            progress_callback,
            DownloadProgress(model=model, phase="complete", completed_bytes=1, total_bytes=1),
        )
        published_files = [
            target_dir / path.relative_to(publish_dir) for path in downloaded_files
        ]
        return DownloadResult(
            model=model,
            target_dir=target_dir,
            downloaded_files=published_files,
        )
    finally:
        # 取消或校验失败时保留 ModelScope 缓存，但清理尚未发布的半成品目录。
        if publish_dir.exists():
            shutil.rmtree(publish_dir, ignore_errors=True)


def load_manifest(model_dir: str | Path, model: str, source: str) -> dict[str, Any]:
    manifest = load_local_manifest(model_dir)
    manifest_model = manifest.get("model")
    if not isinstance(manifest_model, str) or not manifest_model.strip():
        raise ValueError("manifest is missing a valid model field")
    if manifest_model != model:
        raise ValueError("manifest model mismatch")
    # 远端模型仓库里的 manifest 可能只声明 sources=["local"]，它表达的是“这个目录可作为本地 bundle 加载”。
    # 自动下载链路的 source="modelscope" 表达的是“这次搬运资产的下载通道”，两者不是同一个业务概念。
    # 因此这里允许 ModelScope 下载到只声明 local 的 bundle，但仍然拒绝完全未知的来源，避免错误仓库被误收。
    if not _manifest_supports_source(manifest=manifest, source=source):
        raise ValueError(f"manifest does not support source: {source}")
    for asset in manifest.get("assets", []):
        path = asset.get("path")
        sha256 = asset.get("sha256")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("manifest asset is missing a valid path")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"manifest asset {path!r} is missing a valid sha256")
    return manifest


def _manifest_supports_source(*, manifest: dict[str, Any], source: str) -> bool:
    """
    判断远端 manifest 是否可以被当前下载通道接受。

    参数:
        manifest: 已从模型目录读取的 manifest 字典。
        source: 当前下载入口使用的来源名称，例如 modelscope。

    返回:
        当 manifest 显式声明当前来源，或 ModelScope 下载到本地 bundle 语义兼容时返回 True。

    异常:
        本函数不主动抛出异常；非法 sources 会被视为不支持，调用方负责生成用户可读错误。
    """
    # sources 是 manifest 对外声明的可用来源；必须先规范成字符串集合，避免非字符串字段绕过判断。
    manifest_sources = {
        item
        for item in manifest.get("sources", [])
        if isinstance(item, str) and item.strip()
    }

    # 如果 manifest 已经显式声明当前下载来源，说明远端清单和下载通道完全匹配。
    if source in manifest_sources:
        return True

    # ModelScope 快照下载后会被 SDK 复制成标准本地 bundle；远端 sources=["local"] 在这里是可接受的。
    # 其它来源组合保持严格拒绝，避免把不属于当前下载链路的资产误标为已验证。
    return source == "modelscope" and "local" in manifest_sources


def _merge_manifest_sources(*, manifest: dict[str, Any], source: str) -> list[str]:
    """
    生成写入本地缓存 manifest 的 sources 字段。

    参数:
        manifest: 远端仓库随模型资产提供的 manifest 字典。
        source: 当前实际使用的下载通道。

    返回:
        返回去重排序后的来源列表，既保留远端声明，也记录本次真实下载通道。

    异常:
        本函数不主动抛出异常；非法 sources 项会被忽略，保证本地 manifest 始终是可序列化的字符串列表。
    """
    # 本地缓存 manifest 需要同时表达“它现在是一个 local bundle”和“它来自哪个下载通道”，便于后续排查来源。
    sources = {
        item
        for item in manifest.get("sources", [])
        if isinstance(item, str) and item.strip()
    }
    sources.add(source)
    sources.add("local")
    return sorted(sources)


def _verify_sha256(path: Path, expected_sha256: str) -> None:
    # 大模型权重可能达到数 GB，分块读取避免一次性占满内存。
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"sha256 mismatch for {path.name}")


def _resolve_modelscope_snapshot_download():
    # 兼容 ModelScope SDK 的两个常见导入入口，避免因为安装版本差异导致自动下载失效。
    try:
        from modelscope import snapshot_download
    except (ImportError, ModuleNotFoundError):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            raise ImportError(
                "modelscope SDK is required for automatic modelscope downloads"
            ) from exc
    return snapshot_download


def _copy_downloaded_repo(
    *,
    downloaded_root: Path,
    target_dir: Path,
    model: str,
    progress_callback: ProgressCallback | None,
    cancel_event: Any | None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_files = [path for path in sorted(downloaded_root.rglob("*")) if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in source_files)
    completed_bytes = 0
    for index, source_path in enumerate(source_files, start=1):
        _check_cancelled(cancel_event)
        relative_path = source_path.relative_to(downloaded_root)
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 每次重新下载都用 SDK 快照覆盖目标文件，避免默认目录里残留旧资产。
        shutil.copy2(source_path, destination)
        completed_bytes += source_path.stat().st_size
        _emit_progress(
            progress_callback,
            DownloadProgress(
                model=model,
                phase="copy",
                completed_bytes=completed_bytes,
                total_bytes=total_bytes,
                current_file=str(relative_path).replace("\\", "/"),
                files_completed=index,
                files_total=len(source_files),
            ),
        )


def _resolve_manifest_assets(
    *,
    target_dir: Path,
    manifest: dict[str, Any],
    model: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> list[Path]:
    runtime_assets = [
        asset for asset in manifest.get("assets", []) if is_runtime_manifest_asset(asset["path"])
    ]
    total_bytes = sum((target_dir / asset["path"]).stat().st_size for asset in runtime_assets if (target_dir / asset["path"]).exists())
    completed_bytes = 0
    assets: list[Path] = []
    for index, asset in enumerate(manifest.get("assets", []), start=1):
        _check_cancelled(cancel_event)
        # manifest 来自已下载的 repo，它只负责声明“应该校验哪些文件、sha256 是什么”。
        # 真正的下载动作统一由 ModelScope SDK 负责，避免出现“清单和下载源分离”导致的漂移。
        # 这里只校验运行资产；下载快照已保留法律文件，许可文档变化不参与推理清单校验。
        if not is_runtime_manifest_asset(asset["path"]):
            continue
        relative_path = Path(asset["path"])
        local_path = target_dir / relative_path
        if not local_path.exists():
            raise FileNotFoundError(f"missing manifest asset after modelscope download: {local_path}")
        _verify_sha256(local_path, asset["sha256"])
        assets.append(local_path)
        completed_bytes += local_path.stat().st_size
        if model is not None:
            _emit_progress(
                progress_callback,
                DownloadProgress(
                    model=model,
                    phase="verify",
                    completed_bytes=completed_bytes,
                    total_bytes=total_bytes,
                    current_file=str(relative_path).replace("\\", "/"),
                    files_completed=len(assets),
                    files_total=len(runtime_assets),
                ),
            )
    return assets


def _check_cancelled(cancel_event: Any | None) -> None:
    """在不会留下半个正式 bundle 的安全边界检查取消请求。"""
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled("model download cancelled")


def _emit_progress(callback: ProgressCallback | None, progress: DownloadProgress) -> None:
    if callback is not None:
        callback(progress)


def _atomic_publish(*, source_dir: Path, target_dir: Path) -> None:
    """将已校验的隐藏目录切换成正式目录，失败时尽量恢复旧目录。"""
    backup_dir = target_dir.with_name(f".{target_dir.name}.previous-{uuid4().hex}")
    moved_existing = False
    try:
        if target_dir.exists():
            os.replace(target_dir, backup_dir)
            moved_existing = True
        os.replace(source_dir, target_dir)
    except Exception:
        if moved_existing and not target_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, target_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


def modelscope_safe_name(model: str) -> str:
    return model.replace("/", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download StreamVox model assets.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="modelscope")
    parser.add_argument("--target", default="./ckpts")
    parser.add_argument("--revision")
    parser.add_argument("--expected-manifest-sha256")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = download_model(
        model=args.model,
        source=args.source,
        target=args.target,
        revision=args.revision,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(f"downloaded {len(result.downloaded_files)} files into {result.target_dir}")


if __name__ == "__main__":
    # .venv310/bin/python -m streamvox.download --model qwen3-tts-clone-1.7b-gguf --target tests/ckpts
    main()
