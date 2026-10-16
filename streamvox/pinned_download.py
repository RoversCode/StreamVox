"""按不可变提交和预期清单下载模型，缓存保留到下一次重试。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from .model_bundle import is_runtime_manifest_asset


LEGAL_ASSET_NAMES = {"LICENSE", "LICENSE.MD", "LICENSE.TXT", "NOTICE", "NOTICE.MD", "NOTICE.TXT", "COPYING", "COPYING.TXT"}


def download_pinned_snapshot(
    *, repository: str, revision: str, expected_manifest_sha256: str,
    cache_dir: Path, report: Callable[[int, int, str, int, int], None],
    check_cancelled: Callable[[], None],
) -> Path:
    """下载指定提交中的模型及许可文件，分别核对运行清单和提交文件哈希。

    Args:
        repository: ModelScope 仓库全名。
        revision: 完整的四十位 Git 提交哈希。
        expected_manifest_sha256: 产品目录中预先登记的原始清单哈希。
        cache_dir: 可以跨重启保留的下载缓存根目录。
        report: 已完成字节、总字节、当前文件、完成文件数、总文件数回调。
        check_cancelled: 检查取消事件的函数，被取消时应抛出异常。
    Returns:
        已验证的固定版本快照目录。
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision 必须是完整的不可变提交哈希")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("expected_manifest_sha256 必须是有效 SHA256")
    snapshot = cache_dir / revision / expected_manifest_sha256
    snapshot.mkdir(parents=True, exist_ok=True)
    endpoint = f"https://modelscope.cn/api/v1/models/{repository}/repo"
    check_cancelled()
    with requests.Session() as session:
        response = session.get(endpoint, params={"Revision": revision, "FilePath": "manifest.json"}, timeout=(15, 60))
        response.raise_for_status()
        manifest_bytes = response.content
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
            raise ValueError("固定版本的 manifest.json 与产品目录哈希不匹配")
        manifest = json.loads(manifest_bytes)
        response = session.get(endpoint + "/files", params={"Revision": revision, "Recursive": "True"}, timeout=(15, 60))
        response.raise_for_status()
        remote_files = {item["Path"]: item for item in response.json()["Data"]["Files"] if item["Type"] == "blob"}
        assets: list[dict[str, Any]] = []
        for asset in manifest.get("assets", []):
            path = PurePosixPath(asset["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in asset["path"] or ":" in asset["path"]:
                raise ValueError(f"模型清单路径越界：{asset['path']}")
            if not is_runtime_manifest_asset(str(path)):
                continue
            remote = remote_files.get(str(path))
            if remote is None or remote["Sha256"] != asset["sha256"]:
                raise ValueError(f"固定提交中的文件与清单不一致：{path}")
            assets.append({"path": str(path), "sha256": asset["sha256"], "size": int(remote["Size"])})
        if not assets:
            raise ValueError("固定版本清单没有运行期资产")

        # 法律文件不参与启动校验，但分发模型时必须保留同一提交中的许可和署名。
        for remote_path, remote in sorted(remote_files.items()):
            path = PurePosixPath(remote_path)
            if path.name.upper() not in LEGAL_ASSET_NAMES:
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in remote_path or ":" in remote_path:
                raise ValueError(f"模型许可文件路径越界：{remote_path}")
            digest = remote.get("Sha256", "")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"固定提交中的许可文件缺少有效哈希：{remote_path}")
            assets.append({"path": str(path), "sha256": digest, "size": int(remote["Size"])})
        total = sum(asset["size"] for asset in assets)
        completed = 0
        for index, asset in enumerate(assets):
            check_cancelled()
            target = snapshot / asset["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == asset["size"] and _file_sha256(target) == asset["sha256"]:
                completed += asset["size"]
                report(completed, total, asset["path"], index + 1, len(assets))
                continue
            partial = target.with_name(target.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= asset["size"]:
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(endpoint, params={"Revision": revision, "FilePath": asset["path"]}, headers=headers, stream=True, timeout=(15, 60)) as stream:
                stream.raise_for_status()
                # 服务端不支持断点续传时从头写入，绝不把完整文件追加到旧分片后。
                if offset and (stream.status_code != 206 or not stream.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    offset = 0
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in stream.iter_content(chunk_size=1024 * 1024):
                        check_cancelled()
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset > asset["size"]:
                            raise ValueError(f"下载文件超过清单大小：{asset['path']}")
                        report(completed + offset, total, asset["path"], index, len(assets))
            if offset != asset["size"] or _file_sha256(partial) != asset["sha256"]:
                raise ValueError(f"下载文件校验失败：{asset['path']}")
            os.replace(partial, target)
            completed += asset["size"]
            report(completed, total, asset["path"], index + 1, len(assets))
        (snapshot / "manifest.json").write_bytes(manifest_bytes)
    return snapshot


def _file_sha256(path: Path) -> str:
    """分块校验缓存文件，避免大模型一次读入内存。

    Args:
        path: 需要校验的本地文件。
    Returns:
        文件内容的 SHA256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
