"""验证固定版本下载的哈希边界、断点续传和原始清单保留。"""

import hashlib
import json

import pytest

from streamvox.pinned_download import download_pinned_snapshot


class Response:
    """提供测试所需的最小 HTTP 流式响应。"""

    def __init__(self, content=b'', payload=None, status=200, headers=None):
        self.content, self.payload, self.status_code, self.headers = content, payload, status, headers or {}

    def raise_for_status(self):
        """测试响应都是成功状态。"""

    def json(self):
        """返回测试文件树。"""
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def iter_content(self, chunk_size):
        yield self.content


def test_pinned_snapshot_resumes_and_preserves_manifest(tmp_path, monkeypatch):
    """重启后只续传缺失字节，发布清单仍与预先登记的哈希一致。"""
    payload = b'model-weights'
    sha256 = hashlib.sha256(payload).hexdigest()
    manifest = json.dumps({'model': 'fake', 'assets': [{'path': 'ckpt/weights.bin', 'sha256': sha256}]}).encode()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    revision = 'a' * 40
    partial = tmp_path / revision / manifest_hash / 'ckpt/weights.bin.part'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:4])
    requests = []

    class Session(Response):
        """确保每次请求都固定到同一个提交。"""

        def get(self, url, *, params, headers=None, **_):
            assert params['Revision'] == revision
            requests.append((url, params, headers))
            if url.endswith('/files'):
                return Response(payload={'Data': {'Files': [{'Path': 'ckpt/weights.bin', 'Sha256': sha256, 'Size': len(payload), 'Type': 'blob'}]}})
            if params['FilePath'] == 'manifest.json':
                return Response(content=manifest)
            assert headers == {'Range': 'bytes=4-'}
            return Response(content=payload[4:], status=206, headers={'Content-Range': f'bytes 4-{len(payload)-1}/{len(payload)}'})

    monkeypatch.setattr('streamvox.pinned_download.requests.Session', Session)
    progress = []
    result = download_pinned_snapshot(repository='fake/repo', revision=revision, expected_manifest_sha256=manifest_hash, cache_dir=tmp_path, report=lambda *args: progress.append(args), check_cancelled=lambda: None)
    assert (result / 'ckpt/weights.bin').read_bytes() == payload
    assert (result / 'manifest.json').read_bytes() == manifest
    assert progress[-1][:2] == (len(payload), len(payload))
    assert len(requests) == 3


def test_floating_revision_rejected_before_network(tmp_path):
    """产品下载不接受会变化的分支或标签名。"""
    with pytest.raises(ValueError, match='提交哈希'):
        download_pinned_snapshot(repository='fake/repo', revision='master', expected_manifest_sha256='a'*64, cache_dir=tmp_path, report=lambda *_: None, check_cancelled=lambda: None)


def test_manifest_mismatch_stops_before_asset_download(tmp_path, monkeypatch):
    """错误远端清单不能开始搬运任何权重。"""
    class Session(Response):
        def get(self, url, **kwargs):
            return Response(content=b'changed')
    monkeypatch.setattr('streamvox.pinned_download.requests.Session', Session)
    with pytest.raises(ValueError, match='哈希不匹配'):
        download_pinned_snapshot(repository='fake/repo', revision='a'*40, expected_manifest_sha256='b'*64, cache_dir=tmp_path, report=lambda *_: None, check_cancelled=lambda: None)


def test_pinned_snapshot_keeps_legal_files_from_the_same_commit(tmp_path, monkeypatch):
    """许可文本按固定提交校验保存，不依赖可能已过时的运行清单文档哈希。"""
    files = {'ckpt/weights.bin': b'weights', 'LICENSE': b'model license', 'NOTICE.md': b'original authors'}
    manifest = json.dumps({'model': 'fake', 'assets': [
        {'path': 'ckpt/weights.bin', 'sha256': hashlib.sha256(files['ckpt/weights.bin']).hexdigest()},
        {'path': 'LICENSE', 'sha256': '0' * 64},
    ]}).encode()
    revision = 'c' * 40

    class Session(Response):
        """固定提交的文件树提供运行资产及独立法律文件。"""

        def get(self, url, *, params, **_):
            assert params['Revision'] == revision
            if url.endswith('/files'):
                return Response(payload={'Data': {'Files': [
                    {'Path': name, 'Type': 'blob', 'Sha256': hashlib.sha256(content).hexdigest(), 'Size': len(content)}
                    for name, content in files.items()
                ]}})
            return Response(content=manifest if params['FilePath'] == 'manifest.json' else files[params['FilePath']])

    monkeypatch.setattr('streamvox.pinned_download.requests.Session', Session)
    result = download_pinned_snapshot(
        repository='fake/repo', revision=revision, expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        cache_dir=tmp_path, report=lambda *_: None, check_cancelled=lambda: None,
    )
    for name, content in files.items():
        assert (result / name).read_bytes() == content
    assert (result / 'manifest.json').read_bytes() == manifest


def test_pinned_snapshot_rejects_unsafe_legal_file_paths(tmp_path, monkeypatch):
    """远端文件列表不能借助法律文件名逃离下载缓存目录。"""
    payload = b'weights'
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.dumps({'assets': [{'path': 'ckpt/weights.bin', 'sha256': digest}]}).encode()

    class Session(Response):
        """在正常权重之外注入越界许可路径。"""

        def get(self, url, *, params, **_):
            if not url.endswith('/files'):
                return Response(content=manifest)
            return Response(payload={'Data': {'Files': [
                {'Path': 'ckpt/weights.bin', 'Type': 'blob', 'Sha256': digest, 'Size': len(payload)},
                {'Path': '../LICENSE', 'Type': 'blob', 'Sha256': digest, 'Size': len(payload)},
            ]}})

    monkeypatch.setattr('streamvox.pinned_download.requests.Session', Session)
    with pytest.raises(ValueError, match='许可文件路径越界'):
        download_pinned_snapshot(
            repository='fake/repo', revision='a'*40, expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            cache_dir=tmp_path, report=lambda *_: None, check_cancelled=lambda: None,
        )
