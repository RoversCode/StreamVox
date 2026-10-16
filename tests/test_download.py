from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event

import pytest

from streamvox import download


def _write_fake_snapshot(root: Path, *, model: str, payload: bytes) -> Path:
    snapshot = root / "snapshot"
    asset = snapshot / "ckpt" / "weights.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(payload)
    manifest = {
        "model": model,
        "sources": ["local"],
        "assets": [
            {
                "path": "ckpt/weights.bin",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot


def test_download_model_reports_stages_and_atomically_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = "fake-model"
    snapshot = _write_fake_snapshot(tmp_path / "remote", model=model, payload=b"weights")
    monkeypatch.setitem(download.DEFAULT_MODELSCOPE_REPOS, model, "fake/repo")
    monkeypatch.setattr(download, "_resolve_modelscope_snapshot_download", lambda: lambda **_: snapshot)

    events: list[download.DownloadProgress] = []
    result = download.download_model(
        model=model,
        target=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        progress_callback=events.append,
    )

    assert result.target_dir == (tmp_path / "models" / model).resolve()
    assert result.downloaded_files == [result.target_dir / "ckpt" / "weights.bin"]
    assert result.downloaded_files[0].read_bytes() == b"weights"
    assert [event.phase for event in events] == ["download", "copy", "copy", "verify", "publish", "complete"]
    assert events[1].fraction is not None
    assert events[-1].fraction == 1.0
    assert (tmp_path / "cache" / download.modelscope_safe_name(model)).exists()


def test_download_model_cancellation_keeps_existing_bundle_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = "fake-model"
    snapshot = _write_fake_snapshot(tmp_path / "remote", model=model, payload=b"new")
    monkeypatch.setitem(download.DEFAULT_MODELSCOPE_REPOS, model, "fake/repo")
    monkeypatch.setattr(download, "_resolve_modelscope_snapshot_download", lambda: lambda **_: snapshot)

    target_dir = tmp_path / "models" / model
    old_asset = target_dir / "ckpt" / "weights.bin"
    old_asset.parent.mkdir(parents=True)
    old_asset.write_bytes(b"old")

    cancel_event = Event()

    def cancel_after_copy(progress: download.DownloadProgress) -> None:
        if progress.phase == "copy":
            cancel_event.set()

    with pytest.raises(download.DownloadCancelled):
        download.download_model(
            model=model,
            target=tmp_path / "models",
            cache_dir=tmp_path / "cache",
            progress_callback=cancel_after_copy,
            cancel_event=cancel_event,
        )

    assert old_asset.read_bytes() == b"old"
    assert (tmp_path / "cache" / download.modelscope_safe_name(model)).exists()
    assert not list((tmp_path / "models").glob(f".{download.modelscope_safe_name(model)}.publish-*"))


def test_verify_sha256_reads_file_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload_path = tmp_path / "large.bin"
    payload = b"0123456789" * 300_000
    payload_path.write_bytes(payload)

    class ChunkReader:
        def __init__(self, handle):
            self.handle = handle
            self.read_sizes: list[int] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

        def read(self, size: int = -1):
            self.read_sizes.append(size)
            return self.handle.read(size)

    reader: ChunkReader | None = None
    original_open = Path.open

    def open_wrapper(path: Path, *args, **kwargs):
        nonlocal reader
        handle = original_open(path, *args, **kwargs)
        reader = ChunkReader(handle)
        return reader

    monkeypatch.setattr(Path, "open", open_wrapper)
    download._verify_sha256(payload_path, hashlib.sha256(payload).hexdigest())
    assert reader is not None
    assert 1024 * 1024 in reader.read_sizes
