"""在干净暂存目录构建并审计可读源码 wheel 与源码发行包。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from uuid import uuid4

from .build_matrix import build_output_dir, find_target
from .native_assets import LLAMA_TAG, prepare_native_assets
from .release_artifacts import sha256_file, validate_source_commit
from .release_audit import audit_sdist, audit_wheel, license_files, source_files, validate_payload_path
from .subprocess_env import build_python_subprocess_env


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """描述本次公开发行的构建输入。

    Args:
        source_commit: 公共仓库源码提交号。
        native_cache: 固定原生归档的缓存目录。
    """

    source_commit: str
    native_cache: Path | None = None


@dataclass(frozen=True, slots=True)
class BuildResult:
    """保存经过审计的产物及其对应的源码摘要。

    Args:
        wheel: 最终运行包。
        sdist: 最终源码发行包。
        source_manifest: 来自声明提交的源码、构建配置及许可摘要。
    """

    wheel: Path
    sdist: Path
    source_manifest: dict[str, str]


def export_source_commit(project_root: Path, stage_root: Path, source_commit: str) -> dict[str, str]:
    """从干净 Git 提交安全导出源码，排除工作区未跟踪或被忽略的内容。

    Args:
        project_root: Git 源码检出目录。
        stage_root: 全新源码暂存目录。
        source_commit: 必须与当前检出一致的完整提交号。
    Returns:
        运行源码、构建配置与许可证的 SHA256 清单。
    """
    commit = validate_source_commit(source_commit)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True).stdout.decode().strip()
    if head != commit:
        raise ValueError("构建源码 HEAD 必须精确等于声明的公共源码提交。")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=project_root,
                            check=True, capture_output=True).stdout
    if status:
        raise ValueError("构建源码存在已跟踪文件的未提交修改。")
    tree = subprocess.run(["git", "ls-tree", "-rz", "--full-tree", commit], cwd=project_root,
                          check=True, capture_output=True).stdout
    tracked = set()
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, encoded_path = entry.split(b"\t", 1)
        mode, object_type, _ = header.split(b" ")
        path = encoded_path.decode("utf-8")
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            raise ValueError(f"公开源码不能包含符号链接或子模块：{path}")
        validate_payload_path(path)
        if path.lower().endswith(".dll"):
            raise ValueError("公开 Git 提交不能携带 DLL，原生资产应从固定公开归档恢复。")
        tracked.add(path)
    archived = subprocess.run(["git", "archive", "--format=tar", commit], cwd=project_root,
                              check=True, capture_output=True).stdout
    extracted = set()
    with tarfile.open(fileobj=io.BytesIO(archived), mode="r:") as archive:
        for member in archive:
            if member.isdir():
                continue
            validate_payload_path(member.name)
            if not member.isfile() or member.name not in tracked or member.name in extracted:
                raise ValueError(f"Git 源码归档包含未知文件、链接或重复路径：{member.name}")
            target = stage_root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                output.write(source.read())
            extracted.add(member.name)
    if extracted != tracked:
        raise ValueError("Git 归档遗漏已跟踪源码，不能生成完整的公开发行。")
    manifest = {**source_files(stage_root), **license_files(stage_root)}
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "uv.lock"):
        manifest[name] = sha256_file(stage_root / name)
    return dict(sorted(manifest.items()))


def validate_build_environment(python_executable: Path, project_root: Path) -> None:
    """提前核对解释器平台和构建依赖，避免产生错误标签。

    Args:
        python_executable: 本次构建使用的解释器。
        project_root: 执行检查的工作目录。
    Returns:
        无返回值；环境不符合发行契约时抛错。
    """
    code = ("import sys,struct,platform,json,build,wheel,importlib.metadata as m; "
            "print(json.dumps({'implementation':sys.implementation.name,'version':list(sys.version_info[:2]),"
            "'platform':sys.platform,'bits':struct.calcsize('P')*8,'machine':platform.machine().lower(),"
            "'setuptools':m.version('setuptools')}))")
    result = subprocess.run([str(python_executable), "-c", code], cwd=project_root, check=True,
                            capture_output=True, text=True, encoding="utf-8",
                            env=build_python_subprocess_env(python_executable=python_executable))
    details = json.loads(result.stdout)
    if (details["implementation"], details["version"], details["platform"], details["bits"]) != (
        "cpython", [3, 12], "win32", 64
    ):
        raise RuntimeError("发行构建必须使用 Windows x64 CPython 3.12。")
    if details["machine"] not in {"amd64", "x86_64"}:
        raise RuntimeError("发行构建必须使用 x64 解释器，不能使用 ARM64。")
    if int(details["setuptools"].split(".", 1)[0]) < 77:
        raise RuntimeError("发行构建需要 setuptools>=77，以准确打包开源许可证。")


def build_release(project_root: Path, python_executable: str, config: BuildConfig) -> BuildResult:
    """从干净暂存区生成新产物，禁止复用或覆盖任何已有发行文件。

    Args:
        project_root: 当前源码根目录。
        python_executable: 用于打包的解释器。
        config: 公共源码提交与原生缓存配置。
    Returns:
        审计通过的产物及对应源码摘要。
    """
    project_root = project_root.resolve()
    validate_source_commit(config.source_commit)
    interpreter = Path(python_executable).expanduser()
    if not interpreter.is_absolute():
        interpreter = project_root / interpreter
    interpreter = interpreter.absolute()
    validate_build_environment(interpreter, project_root)
    target = find_target("windows", "3.12")
    output_root = build_output_dir(project_root, target)
    artifacts = output_root / "artifacts"
    if artifacts.exists():
        raise FileExistsError(f"当前版本已有构建产物，禁止复用或覆盖：{artifacts}")
    run_root = output_root / "r" / uuid4().hex[:8]
    stage = run_root / "staging"
    stage.mkdir(parents=True)
    source_manifest = export_source_commit(project_root, stage, config.source_commit)
    cache = config.native_cache or project_root / ".artifacts/native" / LLAMA_TAG
    if not cache.is_absolute():
        cache = project_root / cache
    prepare_native_assets(cache, stage / "streamvox/bin")
    pending_artifacts = run_root / "artifacts"
    subprocess.run([str(interpreter), "-m", "build", "--sdist", "--wheel", "--no-isolation", "--outdir", str(pending_artifacts)],
                   cwd=stage, check=True, env=build_python_subprocess_env(python_executable=interpreter))
    wheel = pending_artifacts / target.wheel_filename
    sdist = pending_artifacts / target.sdist_filename
    audit_wheel(wheel, stage)
    audit_sdist(sdist, stage)
    # 只有整轮构建和审计成功后才公布产物目录，失败尝试不会污染正式产物位置。
    pending_artifacts.rename(artifacts)
    return BuildResult(artifacts / wheel.name, artifacts / sdist.name, source_manifest)


def build_parser() -> argparse.ArgumentParser:
    """定义源码发行构建的显式参数。

    Returns:
        命令行解析器。
    """
    parser = argparse.ArgumentParser(description="构建并审计 StreamVox 开源 wheel 与源码发行包。")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--native-cache", type=Path)
    return parser


def main() -> None:
    """构建本地产物，不执行任何发布。"""
    args = build_parser().parse_args()
    result = build_release(args.project_root, args.python, BuildConfig(args.source_commit, args.native_cache))
    for path in (result.wheel, result.sdist):
        print(path)


if __name__ == "__main__":
    main()
