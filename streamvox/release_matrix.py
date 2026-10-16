"""汇总开源发行产物，并只向指定公共仓库发布已验证提交的标签。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from .build_matrix import PACKAGE_VERSION, find_target
from .package_release import BuildConfig, build_release, export_source_commit
from .release_artifacts import PUBLIC_REPOSITORY, validate_source_commit, verify_release_artifacts, write_release_artifacts
from .release_audit import audit_sdist, audit_wheel


@dataclass(frozen=True, slots=True)
class MatrixBuildConfig:
    """描述一次完整的本地发行汇总。

    Args:
        project_root: 源码根目录。
        python_executable: Windows CPython 3.12 解释器。
        release_dir: 全新发行汇总目录。
        source_commit: 公共源码仓库完整提交号。
        native_cache: 固定原生资产缓存目录。
    """

    project_root: Path
    python_executable: str
    release_dir: Path
    source_commit: str
    native_cache: Path | None = None


@dataclass(frozen=True, slots=True)
class GitHubReleaseConfig:
    """描述已经明确目标仓库与源码提交的发布操作。

    Args:
        tag: 必须预先推送到公共仓库的版本标签。
        repo: 明确指定的公共发行仓库。
        source_commit: 标签必须指向的完整提交号。
        title: 可选发行标题。
        notes_file: 可选说明文件。
        draft: 是否创建草稿。
        prerelease: 是否标记预发布。
    """

    tag: str
    repo: str
    source_commit: str
    title: str | None = None
    notes_file: Path | None = None
    draft: bool = False
    prerelease: bool = False


def build_release_matrix(config: MatrixBuildConfig) -> list[Path]:
    """从全新构建结果生成发行目录，禁止旧产物复用和覆盖。

    Args:
        config: 本地源码与发行目录配置。
    Returns:
        已核对的 wheel、源码包和校验材料。
    """
    validate_source_commit(config.source_commit)
    if config.release_dir.exists() and any(config.release_dir.iterdir()):
        raise FileExistsError(f"发行目录必须为空，禁止覆盖：{config.release_dir}")
    result = build_release(config.project_root, config.python_executable,
                           BuildConfig(config.source_commit, config.native_cache))
    config.release_dir.mkdir(parents=True, exist_ok=True)
    for source in (result.wheel, result.sdist):
        destination = config.release_dir / source.name
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    write_release_artifacts(config.release_dir, PACKAGE_VERSION, config.source_commit, result.source_manifest)
    return verify_release_artifacts(config.release_dir, PACKAGE_VERSION, config.source_commit)


def _github_json(arguments: list[str], project_root: Path) -> dict:
    """读取明确目标的 GitHub 元数据，并保留失败诊断。

    Args:
        arguments: gh 的只读命令参数。
        project_root: 命令工作目录。
    Returns:
        GitHub 返回的 JSON 对象。
    """
    result = subprocess.run(["gh", *arguments], cwd=project_root, check=True, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    return json.loads(result.stdout)


def validate_github_release(config: GitHubReleaseConfig, project_root: Path) -> None:
    """验证公共仓库、远端标签及不可覆盖的发布契约。

    Args:
        config: 明确的发布目标。
        project_root: 命令工作目录。
    Returns:
        无返回值；标签错误、仓库不公开或已有发行都会失败。
    """
    source_commit = validate_source_commit(config.source_commit)
    if config.repo.casefold() != PUBLIC_REPOSITORY.casefold():
        raise ValueError(f"此发行流程只允许发布到公共仓库 {PUBLIC_REPOSITORY}。")
    if config.tag != f"v{PACKAGE_VERSION}":
        raise ValueError("发布标签必须与本次 SDK 版本完全一致。")
    repo = _github_json(["repo", "view", config.repo, "--json", "visibility,viewerPermission"], project_root)
    if repo["visibility"] != "PUBLIC" or repo["viewerPermission"] not in {"ADMIN", "MAINTAIN", "WRITE"}:
        raise ValueError("发行仓库必须公开，且当前账号必须有写入权限。")
    ref = _github_json(["api", f"repos/{config.repo}/git/ref/tags/{config.tag}"], project_root)["object"]
    # 注解标签需要解析到最终提交，不能拿标签对象本身的摘要冒充源码提交。
    for _ in range(5):
        if ref["type"] == "commit":
            break
        if ref["type"] != "tag":
            raise ValueError("远端标签没有指向源码提交。")
        ref = _github_json(["api", f"repos/{config.repo}/git/tags/{ref['sha']}"], project_root)["object"]
    if ref["type"] != "commit" or ref["sha"] != source_commit:
        raise ValueError("远端标签没有精确指向本次公共源码提交。")
    existing = subprocess.run(["gh", "api", f"repos/{config.repo}/releases/tags/{config.tag}"],
                              cwd=project_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if existing.returncode == 0:
        raise FileExistsError("该版本已经存在 GitHub Release，禁止覆盖历史附件。")
    if "HTTP 404" not in existing.stderr:
        raise RuntimeError(f"无法确认 GitHub Release 是否存在：{existing.stderr.strip()}")


def publish_github_release(config: GitHubReleaseConfig, *, project_root: Path, release_dir: Path) -> None:
    """把已校验附件上传到精确源码标签，永不隐式创建标签或覆盖旧附件。

    Args:
        config: 公共仓库与标签配置。
        project_root: 命令工作目录。
        release_dir: 已审计的发行目录。
    Returns:
        无返回值。
    """
    files = verify_release_artifacts(release_dir, PACKAGE_VERSION, config.source_commit)
    validate_github_release(config, project_root)
    if config.notes_file is not None:
        notes = config.notes_file.read_text(encoding="utf-8")
    else:
        notes = (f"StreamVox {PACKAGE_VERSION}\n\nWindows x64 / CPython 3.12\n"
                 f"Source: https://github.com/{config.repo}/commit/{config.source_commit}\n")
    with TemporaryDirectory(prefix="streamvox-release-notes-") as work:
        snapshot = Path(work) / "source"
        snapshot.mkdir()
        expected_sources = export_source_commit(project_root, snapshot, config.source_commit)
        metadata = json.loads((release_dir / "release-metadata.json").read_text(encoding="utf-8"))
        if metadata["source_files"] != expected_sources:
            raise ValueError("待发布产物的源码摘要与实际公共源码提交不一致。")
        target = find_target("windows", "3.12")
        # 整包校验和只证明附件未变；重新审计安装内容，才能证明 wheel 也来自同一源码提交。
        audit_wheel(release_dir / target.wheel_filename, snapshot)
        audit_sdist(release_dir / target.sdist_filename, snapshot)
        notes_file = Path(work) / "notes.md"
        notes_file.write_text(notes, encoding="utf-8")
        command = ["gh", "release", "create", config.tag, *map(str, files), "--repo", config.repo,
                   "--verify-tag", "--target", config.source_commit, "--title", config.title or f"StreamVox {PACKAGE_VERSION}",
                   "--notes-file", str(notes_file)]
        if config.draft:
            command.append("--draft")
        if config.prerelease:
            command.append("--prerelease")
        subprocess.run(command, cwd=project_root, check=True)


def build_parser() -> argparse.ArgumentParser:
    """定义本地构建和显式 GitHub 发布参数。

    Returns:
        命令行解析器。
    """
    parser = argparse.ArgumentParser(description="构建并汇总 StreamVox 开源发行产物。")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--release-dir", type=Path, default=Path("dist/release") / PACKAGE_VERSION)
    parser.add_argument("--github-tag")
    parser.add_argument("--github-repo")
    parser.add_argument("--github-title")
    parser.add_argument("--github-notes-file", type=Path)
    parser.add_argument("--github-draft", action="store_true")
    parser.add_argument("--github-prerelease", action="store_true")
    return parser


def main() -> None:
    """先构建可审计产物，仅在指定标签和公共仓库时发布。"""
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    release_dir = args.release_dir if args.release_dir.is_absolute() else project_root / args.release_dir
    github = None
    if args.github_tag:
        if not args.github_repo:
            raise SystemExit("--github-tag 必须同时指定 --github-repo RoversCode/StreamVox。")
        github = GitHubReleaseConfig(args.github_tag, args.github_repo, args.source_commit,
                                     args.github_title, args.github_notes_file, args.github_draft, args.github_prerelease)
        validate_github_release(github, project_root)
    files = build_release_matrix(MatrixBuildConfig(project_root, args.python, release_dir,
                                                   args.source_commit, args.native_cache))
    for path in files:
        print(path)
    if github is not None:
        publish_github_release(github, project_root=project_root, release_dir=release_dir)


if __name__ == "__main__":
    main()
