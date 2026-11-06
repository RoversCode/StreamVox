"""验证派生快速文本 bundle 时保留许可和原始不可变资产。"""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from streamvox.model_bundle import write_local_manifest
from tools.breeze_reference import prepare_text_bundle as bundle_preparation
from tools.breeze_reference.prepare_text_bundle import prepare_text_bundle


@pytest.fixture
def bundle_case(tmp_path, monkeypatch):
    """创建带许可的源资产，并隔离实际数学拆图。

    Args:
        tmp_path: 当前用例独占的临时目录。
        monkeypatch: 当前用例的函数替换工具。
    Returns:
        源目录、输出目录及全部原始资产内容。
    """
    source, output = tmp_path / "source", tmp_path / "fast"
    checkpoint = source / "ckpt"
    checkpoint.mkdir(parents=True)
    graph = checkpoint / "text_encoder.onnx"
    graph.write_bytes(b"immutable model graph")
    report = checkpoint / "export-report.json"
    report.write_text(json.dumps({"tiny": False, "generation": {"validated": True}, "files": {}}), encoding="utf-8")
    assets = [graph, report]
    for name in ("LICENSE", "MODEL_LICENSE", "NOTICE", "README.md"):
        path = source / name
        path.write_text("原始许可与归属 " + name, encoding="utf-8")
        assets.append(path)
    write_local_manifest("breeze-tts-2-gguf", source, assets, sources=["https://example.com/model"], adapter="BreezeTTS2GGUFAdapter")
    before = {path.relative_to(source): path.read_bytes() for path in assets}

    def split(checkpoint):
        """替代数学拆图，只观察资产收口与许可复制边界。

        Args:
            checkpoint: 本次构建拥有的检查点目录。
        Returns:
            三个派生文本资产的路径集合。
        """
        paths = {checkpoint / name for name in ("text_encoder_trunk.onnx", "text_encoder_lookup.onnx", "text_encoder_lookup.json")}
        for path in paths:
            path.write_bytes(b"derived graph metadata")
        return paths

    monkeypatch.setattr("tools.breeze_reference.prepare_text_bundle.split_text_encoder", split)
    return source, output, before


def test_prepared_bundle_preserves_legal_files_and_source_content(bundle_case):
    """运行期校验只返回模型文件，派生操作仍须保留完整许可与归属。

    Args:
        bundle_case: 源目录、输出目录及原始资产内容。
    Returns:
        许可、原始资产和派生清单均完整时正常返回。
    """
    source, output, before = bundle_case
    manifest_path = prepare_text_bundle(source, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["assets"]} == {str(path).replace("\\", "/") for path in before} | {
        "ckpt/text_encoder_trunk.onnx", "ckpt/text_encoder_lookup.onnx", "ckpt/text_encoder_lookup.json",
    }
    for path, original in before.items():
        assert (source / path).read_bytes() == original
        if path.name != "export-report.json":
            assert (output / path).read_bytes() == original
    assert manifest["sources"] == ["https://example.com/model"]


@pytest.mark.parametrize(("error_number", "windows_error"), [
    (errno.EXDEV, None), (errno.ENOTSUP, None), (errno.EOPNOTSUPP, None), (errno.ENOSYS, None),
    (errno.EINVAL, 1), (errno.EXDEV, 17), (errno.EINVAL, 50),
])
def test_prepared_bundle_copies_when_hardlinks_are_unavailable(bundle_case, monkeypatch, error_number, windows_error):
    """跨卷或文件系统不支持硬链接时，仍复制完整模型资产。

    Args:
        bundle_case: 源目录、输出目录及原始资产内容。
        monkeypatch: 当前用例的函数替换工具。
        error_number: 明确表示硬链接不可用的系统错误编号。
        windows_error: 可选的原始 Windows 系统错误编号。
    Returns:
        回退复制成功且源文件未被修改时正常返回。
    """
    source, output, before = bundle_case

    def unavailable_link(destination, target):
        """模拟目标文件系统无法创建硬链接。

        Args:
            destination: 请求创建的目标路径。
            target: 原始资产路径。
        Returns:
            始终抛出指定的系统错误。
        """
        error = OSError(error_number, "硬链接不可用", str(destination))
        if windows_error is not None:
            error.winerror = windows_error
        raise error

    monkeypatch.setattr(Path, "hardlink_to", unavailable_link)
    assert prepare_text_bundle(source, output) == output / "manifest.json"
    for path, original in before.items():
        assert (source / path).read_bytes() == original
        if path.name != "export-report.json":
            assert (output / path).read_bytes() == original
            assert not (output / path).samefile(source / path)


@pytest.mark.parametrize(("phase", "error_number"), [
    ("hardlink", errno.EACCES), ("hardlink", errno.EINVAL),
    ("split", errno.EACCES), ("manifest", errno.EACCES),
])
def test_prepared_bundle_removes_failed_output_and_allows_retry(bundle_case, monkeypatch, phase, error_number):
    """构建失败传播原始错误，清除本次半成品后允许原命令重试。

    Args:
        bundle_case: 源目录、输出目录及原始资产内容。
        monkeypatch: 当前用例的函数替换工具。
        phase: 注入失败的资产链接、拆图或清单阶段。
        error_number: 不应触发硬链接回退的真实系统错误编号。
    Returns:
        错误未被吞掉、源内容完整且随后重试成功时正常返回。
    """
    source, output, before = bundle_case
    error = OSError(error_number, "模拟真实构建失败")

    def fail(*args, **kwargs):
        """在指定构建阶段保留原始系统错误。

        Args:
            args: 被替换函数的位置参数。
            kwargs: 被替换函数的关键字参数。
        Returns:
            始终抛出当前用例的原始错误。
        """
        raise error

    target, name = {
        "hardlink": (Path, "hardlink_to"),
        "split": (bundle_preparation, "split_text_encoder"),
        "manifest": (bundle_preparation, "write_local_manifest"),
    }[phase]
    with monkeypatch.context() as patch:
        patch.setattr(target, name, fail)
        with pytest.raises(OSError) as captured:
            prepare_text_bundle(source, output)
    assert captured.value is error
    assert not output.exists()
    for path, original in before.items():
        assert (source / path).read_bytes() == original
    assert prepare_text_bundle(source, output) == output / "manifest.json"


def test_prepared_bundle_keeps_preexisting_output_directory(bundle_case):
    """拒绝已有输出目录，并保留其中的用户文件。

    Args:
        bundle_case: 源目录、输出目录及原始资产内容。
    Returns:
        已有目录及文件均保持完整时正常返回。
    """
    source, output, _ = bundle_case
    output.mkdir()
    user_file = output / "user.txt"
    user_file.write_text("用户保留内容", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_text_bundle(source, output)
    assert user_file.read_text(encoding="utf-8") == "用户保留内容"


def test_prepared_bundle_keeps_output_replaced_during_failure(bundle_case, monkeypatch):
    """同名输出目录被替换后，保留替换目录并注明清理未执行。

    Args:
        bundle_case: 源目录、输出目录及原始资产内容。
        monkeypatch: 当前用例的函数替换工具。
    Returns:
        原始构建异常、用户目录和源资产均保持完整时正常返回。
    """
    source, output, before = bundle_case
    displaced = output.with_name("displaced")
    user_file = output / "user.txt"
    error = RuntimeError("模拟输出目录被替换")

    def replace_output(checkpoint):
        """在隔离临时目录内模拟其他进程替换同名输出。

        Args:
            checkpoint: 本次构建的检查点路径。
        Returns:
            完成目录替换后抛出原始构建异常。
        """
        assert checkpoint.parent == output
        assert output.resolve().parent == source.parent.resolve()
        assert displaced.resolve().parent == source.parent.resolve()
        output.rename(displaced)
        output.mkdir()
        user_file.write_text("用户保留内容", encoding="utf-8")
        raise error

    monkeypatch.setattr(bundle_preparation, "split_text_encoder", replace_output)
    with pytest.raises(RuntimeError) as captured:
        prepare_text_bundle(source, output)
    assert captured.value is error
    assert any("身份已变化" in note for note in error.__notes__)
    assert user_file.read_text(encoding="utf-8") == "用户保留内容"
    assert displaced.is_dir()
    for path, original in before.items():
        assert (source / path).read_bytes() == original


@pytest.mark.parametrize("nested", [False, True])
def test_prepared_bundle_rejects_source_directory_replacement(tmp_path, nested):
    """输出不得覆盖或嵌入源模型目录。"""
    source = tmp_path / "source"
    source.mkdir()
    output = source / "nested" if nested else source
    with pytest.raises(ValueError, match="原模型目录之外"):
        prepare_text_bundle(source, output)
