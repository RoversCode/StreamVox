"""验证原生资产来源校验、同名合并与公开 OpenMP 替换规则。"""

import hashlib
import io
from pathlib import Path
import tarfile
from zipfile import ZipFile

import pytest

from streamvox import native_assets


def make_archives(tmp_path, monkeypatch, *, conflicting=False):
    """构造包含受限 OpenMP 的最小双后端资产。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 测试替换工具。
        conflicting: 是否让第二份归档破坏公共库。
    Returns:
        双后端归档和公开 OpenMP 归档路径。
    """
    contents = {"llama.dll": b"shared", "ggml-cuda.dll": b"cuda", "ggml-vulkan.dll": b"vulkan",
                native_assets.OPENMP_DLL: b"redistributable-llvm"}
    monkeypatch.setattr(native_assets, "NATIVE_DLL_SHA256",
                        {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()})
    archives = []
    for backend in ("cuda", "vulkan"):
        path = tmp_path / f"{backend}.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("llama.dll", b"conflict" if conflicting and backend == "vulkan" else b"shared")
            archive.writestr(f"ggml-{backend}.dll", contents[f"ggml-{backend}.dll"])
            archive.writestr(native_assets.OPENMP_DLL, b"not-redistributable-msvc")
            archive.writestr("../../unselected.exe", b"ignored")
        archives.append(path)
    openmp = tmp_path / "openmp.tar.bz2"
    with tarfile.open(openmp, "w:bz2") as archive:
        item = tarfile.TarInfo(native_assets.OPENMP_MEMBER)
        item.size = len(contents[native_assets.OPENMP_DLL])
        archive.addfile(item, io.BytesIO(contents[native_assets.OPENMP_DLL]))
    return archives, openmp


def test_archive_merge_requires_identical_shared_libraries_and_replaces_openmp(tmp_path, monkeypatch):
    """三后端共用同一核心库，且不携带微软 OpenMP。"""
    archives, openmp = make_archives(tmp_path, monkeypatch)
    destination = tmp_path / "bin"
    native_assets.merge_native_archives(archives, openmp, destination)
    native_assets.validate_native_directory(destination)
    assert (destination / native_assets.OPENMP_DLL).read_bytes() == b"redistributable-llvm"
    assert not (tmp_path / "unselected.exe").exists()


def test_archive_merge_rejects_different_shared_libraries(tmp_path, monkeypatch):
    """后端归档不能按提取顺序覆盖不同版本的公共库。"""
    archives, openmp = make_archives(tmp_path, monkeypatch, conflicting=True)
    with pytest.raises(ValueError, match="同名 DLL 冲突"):
        native_assets.merge_native_archives(archives, openmp, tmp_path / "bin")


def test_cached_archive_requires_the_pinned_sha256(tmp_path):
    """同名下载缓存也必须重新校验，不能凭文件存在就信任。"""
    archive = native_assets.NativeArchive("asset.zip", "https://example.invalid/asset.zip", "0" * 64)
    (tmp_path / archive.filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="缓存摘要不匹配"):
        native_assets.download_archive(archive, tmp_path)


def test_openmp_archive_cannot_supply_a_symbolic_link(tmp_path, monkeypatch):
    """精确文件提取仍须拒绝归档中的链接，避免替换来源被重定向。"""
    archives, openmp = make_archives(tmp_path, monkeypatch)
    with tarfile.open(openmp, "w:bz2") as archive:
        item = tarfile.TarInfo(native_assets.OPENMP_MEMBER)
        item.type = tarfile.SYMTYPE
        item.linkname = "../../outside.dll"
        archive.addfile(item)
    with pytest.raises(ValueError, match="普通 DLL"):
        native_assets.merge_native_archives(archives, openmp, tmp_path / "bin")


def test_preparation_preserves_unknown_user_files(tmp_path, monkeypatch):
    """目标目录含非发行文件时停止装配，不能递归清空用户目录。"""
    archives, openmp = make_archives(tmp_path, monkeypatch)
    fixtures = iter([*archives, openmp])
    monkeypatch.setattr(native_assets, "download_archive", lambda *_: next(fixtures))
    destination = tmp_path / "existing"
    destination.mkdir()
    personal = destination / "personal.txt"
    personal.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="未知文件"):
        native_assets.prepare_native_assets(tmp_path, destination)
    assert personal.read_text(encoding="utf-8") == "keep"
