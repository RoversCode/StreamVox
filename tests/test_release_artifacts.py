"""验证可读源码发行审计、来源追溯与不可覆盖的发布流程。"""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from zipfile import ZipFile

import pytest

from streamvox import native_assets, package_release, release_audit, release_matrix
from streamvox.build_matrix import PACKAGE_VERSION, find_target
from streamvox.release_artifacts import PUBLIC_REPOSITORY, verify_release_artifacts, write_release_artifacts


SOURCE_COMMIT = "a" * 40


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    """构造足以检验源码、DLL 与许可证契约的最小项目。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 测试替换工具。
    Returns:
        项目根目录。
    """
    contents = {"streamvox/engine.py": b"answer = 42\n", "streamvox_runtime/__init__.py": b"",
                "streamvox/bin/llama.dll": b"native", "LICENSE": b"Apache-2.0",
                "NOTICE": b"copyright", "THIRD_PARTY_NOTICES.md": b"third-party",
                "third_party/licenses/llama-cpp-MIT.txt": b"MIT",
                "third_party/licenses/llvm-OpenMP.txt": b"LLVM",
                "pyproject.toml": b"[project]\n", "setup.py": b"", "MANIFEST.in": b"", "README.md": b"", "uv.lock": b""}
    dlls = {"llama.dll": hashlib.sha256(b"native").hexdigest()}
    monkeypatch.setattr(native_assets, "NATIVE_DLL_SHA256", dlls)
    monkeypatch.setattr(release_audit, "NATIVE_DLL_SHA256", dlls)
    contents["streamvox/bin/native-manifest.json"] = json.dumps(native_assets.native_manifest()).encode()
    for name, data in contents.items():
        target = tmp_path / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return tmp_path / "source"


def make_wheel(project_root, destination, *, omitted=None, extra=None, abi="none"):
    """按标准 wheel 布局生成审计样本。

    Args:
        project_root: 最小项目。
        destination: 样本输出目录。
        omitted: 需要模拟遗漏的文件。
        extra: 需要模拟混入的路径和内容。
        abi: 样本扩展接口标签。
    Returns:
        样本 wheel 路径。
    """
    path = destination / find_target("windows", "3.12").wheel_filename
    prefix = f"streamvox-{PACKAGE_VERSION}.dist-info"
    with ZipFile(path, "w") as archive:
        for file in project_root.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(project_root).as_posix()
            if relative == omitted:
                continue
            if relative.startswith(("streamvox/", "streamvox_runtime/")):
                archive.write(file, relative)
            if relative in release_audit.license_files(project_root):
                archive.write(file, f"{prefix}/licenses/{relative}")
        archive.writestr(f"{prefix}/WHEEL", f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-{abi}-win_amd64\n")
        archive.writestr(f"{prefix}/METADATA", f"Metadata-Version: 2.4\nName: streamvox\nVersion: {PACKAGE_VERSION}\n"
                         "Requires-Python: >=3.12,<3.13\nLicense-Expression: Apache-2.0\n")
        if extra:
            archive.writestr(*extra)
    return path


def test_wheel_preserves_actual_source_and_complete_licenses(source_tree, tmp_path):
    """最终内容与源码输入及原生来源完全一致时才允许发行。"""
    path = make_wheel(source_tree, tmp_path)
    assert release_audit.audit_wheel(path, source_tree) == {"readable_modules": 2, "native_libraries": 1}


@pytest.mark.parametrize("extra", [("streamvox/engine.pyd", b"hidden"),
                                   ("streamvox/licensing/client.py", b"paid"),
                                   ("streamvox/data/nearmate-trust.json", b"{}")])
def test_wheel_rejects_hidden_extensions_and_removed_paid_payload(source_tree, tmp_path, extra):
    """开源 wheel 不得夹带旧保护扩展或授权材料。"""
    with pytest.raises(ValueError, match="残留"):
        release_audit.audit_wheel(make_wheel(source_tree, tmp_path, extra=extra), source_tree)


def test_wheel_rejects_missing_source(source_tree, tmp_path):
    """保留引导文件不等于完整公开运行源码。"""
    with pytest.raises(ValueError, match="缺少可读源码"):
        release_audit.audit_wheel(make_wheel(source_tree, tmp_path, omitted="streamvox/engine.py"), source_tree)


def test_wheel_rejects_python_extension_abi_tag(source_tree, tmp_path):
    """移除原生 Python 扩展后不能继续使用旧接口标签。"""
    with pytest.raises(ValueError, match="扩展接口"):
        release_audit.audit_wheel(make_wheel(source_tree, tmp_path, abi="cp312"), source_tree)


def test_sdist_contains_standard_rebuild_inputs_and_native_libraries(source_tree, tmp_path):
    """源码发行包必须能用标准构建工具离线重建包含 DLL 的 wheel。"""
    path = tmp_path / find_target("windows", "3.12").sdist_filename
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source_tree, arcname=f"streamvox-{PACKAGE_VERSION}")
    assert release_audit.audit_sdist(path, source_tree)["readable_modules"] == 2


def test_release_metadata_binds_public_source_and_detects_tampering(tmp_path):
    """上传前重算校验，不能把已被更换的附件配上旧清单。"""
    target = find_target("windows", "3.12")
    (tmp_path / target.wheel_filename).write_bytes(b"wheel")
    source = tmp_path / target.sdist_filename
    content = b"source"
    with tarfile.open(source, "w:gz") as archive:
        item = tarfile.TarInfo(f"streamvox-{PACKAGE_VERSION}/streamvox/engine.py")
        item.size = len(content)
        archive.addfile(item, io.BytesIO(content))
    source_manifest = {"streamvox/engine.py": hashlib.sha256(content).hexdigest()}
    write_release_artifacts(tmp_path, PACKAGE_VERSION, SOURCE_COMMIT, source_manifest)
    assert len(verify_release_artifacts(tmp_path, PACKAGE_VERSION, SOURCE_COMMIT)) == 4
    metadata = json.loads((tmp_path / "release-metadata.json").read_text())
    assert metadata["source_repository"] == f"https://github.com/{PUBLIC_REPOSITORY}"
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="清单不一致"):
        verify_release_artifacts(tmp_path, PACKAGE_VERSION, SOURCE_COMMIT)


def test_github_publish_requires_exact_existing_tag_and_never_clobbers(tmp_path, monkeypatch):
    """发布命令必须绑定明确公共仓库、源码提交和预先存在的标签。"""
    config = release_matrix.GitHubReleaseConfig(f"v{PACKAGE_VERSION}", PUBLIC_REPOSITORY, SOURCE_COMMIT)
    monkeypatch.setattr(release_matrix, "verify_release_artifacts", lambda *_: [tmp_path / "package.whl"])
    monkeypatch.setattr(release_matrix, "audit_wheel", lambda *_: None)
    monkeypatch.setattr(release_matrix, "audit_sdist", lambda *_: None)
    source_manifest = {"streamvox/engine.py": "d" * 64}
    monkeypatch.setattr(release_matrix, "export_source_commit", lambda *_: source_manifest)
    (tmp_path / "release-metadata.json").write_text(json.dumps({"source_files": source_manifest}), encoding="utf-8")
    responses = iter([{"visibility": "PUBLIC", "viewerPermission": "WRITE"},
                      {"object": {"type": "commit", "sha": SOURCE_COMMIT}}])
    monkeypatch.setattr(release_matrix, "_github_json", lambda *_: next(responses))
    calls = []

    def run(command, **kwargs):
        """捕获上传命令，并模拟发行尚不存在。

        Args:
            command: 待执行命令。
            kwargs: 子进程选项。
        Returns:
            模拟的子进程结果。
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if command[1] == "api" else 0,
                                           stderr="gh: Not Found (HTTP 404)")

    monkeypatch.setattr(release_matrix.subprocess, "run", run)
    release_matrix.publish_github_release(config, project_root=tmp_path, release_dir=tmp_path)
    publish = calls[-1]
    assert publish[:3] == ["gh", "release", "create"]
    assert "--verify-tag" in publish
    assert publish[publish.index("--target") + 1] == SOURCE_COMMIT
    assert publish[publish.index("--repo") + 1] == PUBLIC_REPOSITORY
    assert "--clobber" not in publish


def test_github_publish_rejects_wrong_source_before_upload(tmp_path, monkeypatch):
    """远端同名标签指向旧代码时，不能上传新版本附件。"""
    config = release_matrix.GitHubReleaseConfig(f"v{PACKAGE_VERSION}", PUBLIC_REPOSITORY, SOURCE_COMMIT)
    responses = iter([{"visibility": "PUBLIC", "viewerPermission": "ADMIN"},
                      {"object": {"type": "commit", "sha": "b" * 40}}])
    monkeypatch.setattr(release_matrix, "_github_json", lambda *_: next(responses))
    with pytest.raises(ValueError, match="精确指向"):
        release_matrix.validate_github_release(config, tmp_path)


def test_github_publish_rejects_private_development_origin(tmp_path):
    """不能从当前工作目录自动推断并发布到私有开发仓库。"""
    config = release_matrix.GitHubReleaseConfig(f"v{PACKAGE_VERSION}", "RoversCode/streamvox-code", SOURCE_COMMIT)
    with pytest.raises(ValueError, match="只允许发布到公共仓库"):
        release_matrix.validate_github_release(config, tmp_path)


def mock_git_checkout(source_tree, monkeypatch, *, head=SOURCE_COMMIT, dirty=False, extra_tree=b""):
    """模拟 Git 的提交归档，避免在项目或测试中创建真实提交。

    Args:
        source_tree: 模拟工作区。
        monkeypatch: 测试替换工具。
        head: 当前检出提交号。
        dirty: 是否存在已跟踪修改。
        extra_tree: 需要模拟的特殊 Git 条目。
    Returns:
        Git 实际跟踪的路径集合。
    """
    tracked = [path for path in source_tree.rglob("*") if path.is_file()
               and "bin" not in path.relative_to(source_tree).parts]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in tracked:
            archive.add(path, arcname=path.relative_to(source_tree).as_posix(), recursive=False)
    tree = b"".join(b"100644 blob " + b"0" * 40 + b"\t" + path.relative_to(source_tree).as_posix().encode() + b"\0"
                    for path in tracked) + extra_tree

    def run(command, **kwargs):
        """按只读 Git 命令返回固定提交数据。

        Args:
            command: 待执行命令。
            kwargs: 子进程参数。
        Returns:
            模拟的 Git 输出。
        """
        output = {"rev-parse": (head + "\n").encode(), "status": b" M streamvox/engine.py\n" if dirty else b"",
                  "ls-tree": tree, "archive": buffer.getvalue()}[command[1]]
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(package_release.subprocess, "run", run)
    return {path.relative_to(source_tree).as_posix() for path in tracked}


def test_source_export_only_uses_declared_commit_and_excludes_untracked_files(source_tree, tmp_path, monkeypatch):
    """未跟踪或忽略文件不能混入声明的公开源码快照。"""
    tracked = mock_git_checkout(source_tree, monkeypatch)
    (source_tree / "streamvox/private.py").write_text("secret = True", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    manifest = package_release.export_source_commit(source_tree, stage, SOURCE_COMMIT)
    assert not (stage / "streamvox/private.py").exists()
    assert {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()} == tracked
    assert "streamvox/engine.py" in manifest


@pytest.mark.parametrize("head,dirty,message", [("b" * 40, False, "HEAD"),
                                               (SOURCE_COMMIT, True, "未提交修改")])
def test_source_export_rejects_wrong_commit_or_modified_tracked_files(source_tree, tmp_path, monkeypatch,
                                                                     head, dirty, message):
    """格式正确的 SHA 也不能替代对实际构建源码的核对。"""
    mock_git_checkout(source_tree, monkeypatch, head=head, dirty=dirty)
    with pytest.raises(ValueError, match=message):
        package_release.export_source_commit(source_tree, tmp_path / "stage", SOURCE_COMMIT)


@pytest.mark.parametrize("mode,kind,path", [(b"120000", b"blob", b"linked.py"),
                                          (b"160000", b"commit", b"submodule"),
                                          (b"100644", b"blob", b"C:/outside.py")])
def test_source_export_rejects_links_submodules_and_windows_absolute_paths(source_tree, tmp_path, monkeypatch,
                                                                         mode, kind, path):
    """源码导出不能跟随链接或写到暂存目录之外。"""
    entry = mode + b" " + kind + b" " + b"0" * 40 + b"\t" + path + b"\0"
    mock_git_checkout(source_tree, monkeypatch, extra_tree=entry)
    with pytest.raises(ValueError):
        package_release.export_source_commit(source_tree, tmp_path / "stage", SOURCE_COMMIT)


def test_existing_github_release_is_never_updated(tmp_path, monkeypatch):
    """历史发行即使标签正确也不能被重上传附件覆盖。"""
    config = release_matrix.GitHubReleaseConfig(f"v{PACKAGE_VERSION}", PUBLIC_REPOSITORY, SOURCE_COMMIT)
    responses = iter([{"visibility": "PUBLIC", "viewerPermission": "ADMIN"},
                      {"object": {"type": "commit", "sha": SOURCE_COMMIT}}])
    monkeypatch.setattr(release_matrix, "_github_json", lambda *_: next(responses))
    monkeypatch.setattr(release_matrix.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="{}"))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        release_matrix.validate_github_release(config, tmp_path)


def test_publish_rejects_replaced_wheel_even_when_sidecars_and_sdist_are_consistent(source_tree, tmp_path, monkeypatch):
    """重算整包摘要不能让另一份源码的 wheel 冒充正确源码发行包。"""
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    old_source = tmp_path / "old-source"
    shutil.copytree(source_tree, old_source)
    (old_source / "streamvox/engine.py").write_text("answer = 41\n", encoding="utf-8")
    make_wheel(old_source, release_dir)
    target = find_target("windows", "3.12")
    with tarfile.open(release_dir / target.sdist_filename, "w:gz") as archive:
        archive.add(source_tree, arcname=f"streamvox-{PACKAGE_VERSION}")
    manifest = release_audit.source_files(source_tree)
    write_release_artifacts(release_dir, PACKAGE_VERSION, SOURCE_COMMIT, manifest)
    # 先证明 sidecar 与正确 sdist 本身能够通过，随后真实 wheel 审计必须发现源码不同。
    verify_release_artifacts(release_dir, PACKAGE_VERSION, SOURCE_COMMIT)

    def export(project_root, stage, source_commit):
        """把正确提交快照交给真实产物审计。

        Args:
            project_root: 模拟公共源码目录。
            stage: 审计暂存目录。
            source_commit: 预期提交号。
        Returns:
            正确提交的源码摘要。
        """
        shutil.copytree(source_tree, stage, dirs_exist_ok=True)
        return manifest

    calls = []
    monkeypatch.setattr(release_matrix, "export_source_commit", export)
    monkeypatch.setattr(release_matrix, "validate_github_release", lambda *_: None)
    monkeypatch.setattr(release_matrix.subprocess, "run", lambda command, **kwargs: calls.append(command))
    config = release_matrix.GitHubReleaseConfig(f"v{PACKAGE_VERSION}", PUBLIC_REPOSITORY, SOURCE_COMMIT)
    with pytest.raises(ValueError, match="缺少可读源码或源码不一致"):
        release_matrix.publish_github_release(config, project_root=source_tree, release_dir=release_dir)
    assert not calls
