"""验证 Breeze 按模型名下载完整资产，并复用已经发布的本地目录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from streamvox import TTSEngine, download
from streamvox.model_bundle import load_local_manifest, validate_local_bundle
from streamvox.models import default_registry
from streamvox.models.base import RuntimeInfo


MODEL = "breeze-tts-2-gguf"
REPOSITORY = "ChengHee/Breeze-TTS-2-GGUF"


@pytest.fixture
def breeze_snapshot(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """创建包含生成模型、外部权重和快速文本组件的隔离快照。

    Args:
        tmp_path: 当前用例的临时根目录。
    Returns:
        快照路径和每个运行资产的原始内容。
    """
    snapshot = tmp_path / "snapshot"
    names = [
        "config.json", "tokenizer.json", "codec_metadata.json", "export-report.json",
        "backbone.gguf", "depth.gguf", "depth_conditioner.onnx", "depth_conditioner.onnx.data",
        "text_encoder.onnx", "text_encoder.onnx.data", "audio_embedding.onnx",
        "audio_embedding.onnx.data", "audio_encoder.onnx", "audio_encoder.onnx.data",
        "decoder_step.onnx", "decoder_step.onnx.data", "text_encoder_trunk.onnx",
        "text_encoder_lookup.onnx", "text_encoder_lookup.json",
    ]
    payloads = {f"ckpt/{name}": f"测试资产 {name}".encode("utf-8") for name in names}
    for name, payload in payloads.items():
        asset = snapshot / name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(payload)
    manifest = {
        "model": MODEL,
        "adapter": "BreezeTTS2GGUFAdapter",
        "sources": ["local"],
        "assets": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot, payloads


def test_model_name_downloads_complete_breeze_bundle_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breeze_snapshot: tuple[Path, dict[str, bytes]],
) -> None:
    """模型名初始化使用正式仓库，完整校验后再次初始化复用本地目录。

    Args:
        tmp_path: 当前用例的临时根目录。
        monkeypatch: 环境和模型加载入口替换工具。
        breeze_snapshot: 包含全部类型运行资产的远端快照。
    Returns:
        仓库映射、完整资产和缓存行为符合约定时正常返回。
    """
    snapshot, payloads = breeze_snapshot
    home = tmp_path / "home"
    monkeypatch.setenv("STREAMVOX_HOME", str(home))
    calls: list[dict[str, str]] = []

    def snapshot_download(**kwargs: str) -> Path:
        """记录下载参数，并以本地快照替代网络传输。

        Args:
            kwargs: 下载器向 ModelScope 传入的参数。
        Returns:
            本用例的完整快照路径。
        """
        calls.append(kwargs)
        assert kwargs["repo_id"] == REPOSITORY
        return snapshot

    def load_runtime(request) -> SimpleNamespace:
        """验证引擎获得的完整目录，并隔离原生推理依赖。

        Args:
            request: 引擎传给适配器的运行时初始化请求。
        Returns:
            可关闭的最小运行时。
        """
        result = validate_local_bundle(request.model_dir, MODEL, request.verify_model_sha256)
        assert {path.relative_to(request.model_dir).as_posix() for path in result.asset_paths} == set(payloads)
        return SimpleNamespace(
            sample_rate=24000,
            runtime_info=RuntimeInfo(
                model=MODEL, adapter="BreezeTTS2GGUFAdapter", bundle_loaded=True,
                asset_count=len(result.asset_paths),
            ),
            shutdown=lambda: None,
        )

    monkeypatch.setattr(download, "_resolve_modelscope_snapshot_download", lambda: snapshot_download)
    adapter = default_registry.get(MODEL)
    monkeypatch.setattr(adapter, "get_hardware_profile", lambda: None)
    monkeypatch.setattr(adapter, "load_runtime", load_runtime)
    target = home / "ckpts" / MODEL

    for _ in range(2):
        engine = TTSEngine(model=MODEL, device="cpu")
        try:
            assert engine.model == MODEL
            assert engine.model_dir == target.resolve()
        finally:
            engine.shutdown()

    assert len(calls) == 1
    assert Path(calls[0]["cache_dir"]).is_dir()
    for name, payload in payloads.items():
        assert (target / name).read_bytes() == payload
    manifest = load_local_manifest(target)
    assert manifest["adapter"] == "BreezeTTS2GGUFAdapter"
    assert manifest["sources"] == ["local", "modelscope"]
    assert not list(target.parent.glob(f".{MODEL}.publish-*"))


@pytest.mark.parametrize("asset_name", ["text_encoder_lookup.json", "decoder_step.onnx.data"])
def test_breeze_download_rejects_corrupt_optional_and_external_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breeze_snapshot: tuple[Path, dict[str, bytes]],
    asset_name: str,
) -> None:
    """快速文本资产和外部权重同样必须通过摘要校验后才能发布。

    Args:
        tmp_path: 当前用例的临时根目录。
        monkeypatch: 下载入口替换工具。
        breeze_snapshot: 包含运行资产及有效原始摘要的快照。
        asset_name: 需要模拟损坏的资产文件名。
    Returns:
        损坏资产被拒绝且半成品已清理时正常返回。
    """
    snapshot, _ = breeze_snapshot
    (snapshot / "ckpt" / asset_name).write_bytes(b"corrupted")
    monkeypatch.setattr(download, "_resolve_modelscope_snapshot_download", lambda: lambda **_: snapshot)
    target_root = tmp_path / "models"

    with pytest.raises(ValueError, match="sha256 mismatch"):
        download.download_model(MODEL, target=target_root)

    assert not (target_root / MODEL).exists()
    assert not list(target_root.glob(f".{MODEL}.publish-*"))
