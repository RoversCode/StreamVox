"""审计实际 wheel 与源码发行包的源码、原生依赖和许可材料。"""

from __future__ import annotations

from email.parser import Parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from zipfile import ZipFile

from .build_matrix import PACKAGE_VERSION, find_target
from .native_assets import NATIVE_DLL_SHA256, native_manifest
from .release_artifacts import sha256_file


_REMOVED_RESOURCES = {"public.pem", "license-ca.pem", "nearmate-trust.json", "nearmate-connection.json",
                      "protection-manifest.json"}


def source_files(project_root: Path) -> dict[str, str]:
    """记录本次应公开发行的全部 Python 源码及其摘要。

    Args:
        project_root: 本次构建的源码根目录。
    Returns:
        相对源码路径到内容摘要的映射。
    """
    return {path.relative_to(project_root).as_posix(): sha256_file(path)
            for package in ("streamvox", "streamvox_runtime")
            for path in sorted((project_root / package).rglob("*.py"))}


def license_files(project_root: Path) -> dict[str, str]:
    """记录 SDK 与第三方许可原文，防止打包规则遗漏声明。

    Args:
        project_root: 源码根目录。
    Returns:
        许可材料相对路径与内容摘要。
    """
    paths = [project_root / name for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")]
    paths += sorted((project_root / "third_party/licenses").glob("*.txt"))
    required = {"llama-cpp-MIT.txt", "llvm-OpenMP.txt"}
    if not required.issubset({path.name for path in paths}):
        raise ValueError("缺少 llama.cpp 或 LLVM OpenMP 的第三方许可原文。")
    return {path.relative_to(project_root).as_posix(): sha256_file(path) for path in paths}


def validate_payload_path(path: str) -> None:
    """拒绝隐藏扩展、旧授权资源、缓存以及不安全的归档路径。

    Args:
        path: 归档中的相对文件路径。
    Returns:
        无返回值；发现禁止内容立即失败。
    """
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or ":" in path:
        raise ValueError(f"发行包含不安全路径：{path}")
    if parsed.suffix.lower() in {".pyc", ".pyo", ".pyd", ".so", ".pem"} or "__pycache__" in parsed.parts:
        raise ValueError(f"开源发行包残留编译或授权文件：{path}")
    if "licensing" in parsed.parts or parsed.name in _REMOVED_RESOURCES:
        raise ValueError(f"开源发行包残留授权资源：{path}")
    if any(part in {".artifacts", ".git", ".worktrees", "streamvox_release_github"} for part in parsed.parts):
        raise ValueError(f"发行包混入内部工作区：{path}")


def _verify_payload(hashes: dict[str, str], sources: dict[str, str]) -> None:
    """核对每个运行源码与原生库，阻断缺文件或混入旧版本。

    Args:
        hashes: 归档文件摘要映射。
        sources: 本次源码快照。
    Returns:
        无返回值；源码和原生资产必须与构建输入完全一致。
    """
    for path, digest in sources.items():
        if hashes.get(path) != digest:
            raise ValueError(f"发行包缺少可读源码或源码不一致：{path}")
    actual_dlls = {path: digest for path, digest in hashes.items() if path.lower().endswith(".dll")}
    expected_dlls = {f"streamvox/bin/{name}": digest for name, digest in NATIVE_DLL_SHA256.items()}
    if actual_dlls != expected_dlls:
        raise ValueError("发行包的三后端 DLL 清单或摘要不一致。")


def audit_wheel(wheel_path: Path, project_root: Path) -> dict:
    """核查最终 wheel 中实际安装的源码、平台标签及许可材料。

    Args:
        wheel_path: 待发行 wheel。
        project_root: 对应源码快照目录。
    Returns:
        审计通过后的源码与原生库数量。
    """
    target = find_target("windows", "3.12")
    if wheel_path.name != target.wheel_filename:
        raise ValueError("wheel 文件名与本次版本或平台不一致。")
    hashes = {}
    licenses = {}
    with ZipFile(wheel_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = member.filename
            for marker in (".data/purelib/", ".data/platlib/"):
                path = path.split(marker, 1)[-1]
            validate_payload_path(path)
            if path in hashes:
                raise ValueError(f"wheel 包含重复安装路径：{path}")
            with archive.open(member) as handle:
                hashes[path] = hashlib.file_digest(handle, "sha256").hexdigest()
            if ".dist-info/licenses/" in path:
                licenses[path.split(".dist-info/licenses/", 1)[1]] = hashes[path]
        wheel_member = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata_member = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        wheel = Parser().parsestr(archive.read(wheel_member).decode("utf-8"))
        metadata = Parser().parsestr(archive.read(metadata_member).decode("utf-8"))
        if wheel.get_all("Tag") != ["cp312-none-win_amd64"] or wheel.get("Root-Is-Purelib") != "false":
            raise ValueError("wheel 元数据必须声明 Windows 平台且无 Python 扩展接口依赖。")
        if metadata.get("Version") != PACKAGE_VERSION or metadata.get("License-Expression") != "Apache-2.0":
            raise ValueError("wheel 元数据的版本或开源许可证不正确。")
        if {value.strip() for value in metadata.get("Requires-Python", "").split(",")} != {">=3.12", "<3.13"}:
            raise ValueError("wheel 必须只声明已经验收的 Python 3.12 范围。")
        if any("rovers-licensing" in value.lower() or "embedded-nearmate" in value.lower()
               for value in metadata.get_all("Requires-Dist", []) + metadata.get_all("Provides-Extra", [])):
            raise ValueError("wheel 元数据仍然包含收费授权依赖。")
        manifest_path = next(name for name in archive.namelist() if name.endswith("streamvox/bin/native-manifest.json"))
        if json.loads(archive.read(manifest_path)) != native_manifest():
            raise ValueError("wheel 原生资产来源清单不正确。")
    sources = source_files(project_root)
    _verify_payload(hashes, sources)
    for path, digest in license_files(project_root).items():
        if licenses.get(path) != digest:
            raise ValueError(f"wheel 缺少或修改了许可材料：{path}")
    return {"readable_modules": len(sources), "native_libraries": len(NATIVE_DLL_SHA256)}


def audit_sdist(sdist_path: Path, project_root: Path) -> dict:
    """保证源码发行包包含标准重建需要的代码、脚本、DLL 与许可证。

    Args:
        sdist_path: 待发行源码归档。
        project_root: 对应源码快照目录。
    Returns:
        审计通过后的源码和原生库数量。
    """
    target = find_target("windows", "3.12")
    if sdist_path.name != target.sdist_filename:
        raise ValueError("源码发行文件名与本次版本不一致。")
    hashes = {}
    manifest = None
    prefix = f"streamvox-{PACKAGE_VERSION}/"
    with tarfile.open(sdist_path, "r:gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError(f"源码发行包含未知根目录或链接：{member.name}")
            path = member.name[len(prefix):]
            validate_payload_path(path)
            if path in hashes:
                raise ValueError(f"源码发行包含重复路径：{path}")
            with archive.extractfile(member) as handle:
                if path == "streamvox/bin/native-manifest.json":
                    content = handle.read()
                    manifest = json.loads(content)
                    hashes[path] = hashlib.sha256(content).hexdigest()
                else:
                    hashes[path] = hashlib.file_digest(handle, "sha256").hexdigest()
    sources = source_files(project_root)
    _verify_payload(hashes, sources)
    required = {"pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "uv.lock"}
    if not required.issubset(hashes) or manifest != native_manifest():
        raise ValueError("源码发行缺少标准重建文件或原生来源清单。")
    for path, digest in license_files(project_root).items():
        if hashes.get(path) != digest:
            raise ValueError(f"源码发行缺少或修改了许可材料：{path}")
    return {"readable_modules": len(sources), "native_libraries": len(NATIVE_DLL_SHA256)}
