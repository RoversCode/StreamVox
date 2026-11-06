"""验证混合文本编码器的只读查表、输入透传和资源边界。"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streamvox.models.breeze import text


class FakeGraph:
    """观察主干输入和关闭过程，不加载模型或访问显卡。

    Args:
        path: 文本主干图路径。
        backend: 执行后端名称。
        device_id: 显卡设备编号。
        cpu_initializers: 处理器图外部张量的只读映射。
    """

    def __init__(self, path, backend, device_id=0, *, cpu_initializers=None):
        self.path, self.backend, self.device_id = path, backend, device_id
        self.cpu_initializers = cpu_initializers
        self.session = object()
        self.inputs, self.outputs = [object()], [object()]
        self.calls = []
        self.close_count = 0

    def run(self, feeds):
        """直接返回查表数组，让测试观察传给主干的原始数据。

        Args:
            feeds: 文本编号、掩码和新增查表输入。
        Returns:
            未额外转换的模拟主干输出。
        """
        self.calls.append(feeds)
        if self.backend == "cpu":
            table = self.cpu_initializers["token_embedding"]
            return {"lookup_embeddings": np.ascontiguousarray(table[feeds["input_ids"]])}
        return {"embeddings": feeds["lookup_embeddings"]}

    def close(self):
        """释放持有会话的接口记录，并记录关闭次数。"""
        self.close_count += 1
        self.inputs, self.outputs, self.session = [], [], None
        self.cpu_initializers = None


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """提供含非对齐偏移和额外尾部数据的合法单精度外部表。

    Returns:
        检查点目录、原始表、元数据和主干构造记录。
    """
    ckpt = tmp_path / "ckpt"
    weights = ckpt / "weights"
    weights.mkdir(parents=True)
    table = np.arange(21, dtype=np.float32).reshape(7, 3) / np.float32(10)
    prefix = b"header!"
    (weights / "text.bin").write_bytes(prefix + table.tobytes() + b"tail")
    (ckpt / "text_encoder_trunk.onnx").write_bytes(b"fake graph")
    (ckpt / "text_encoder_lookup.onnx").write_bytes(b"fake lookup graph")
    metadata = {
        "version": 1,
        "lookup_input_name": "lookup_embeddings",
        "embedding": {
            "initializer_name": "token_embedding",
            "location": "weights/text.bin", "offset": len(prefix),
            "length": table.nbytes, "shape": list(table.shape), "dtype": "float32",
        },
    }
    (ckpt / "text_encoder_lookup.json").write_text(json.dumps(metadata), encoding="utf-8")
    graphs = []

    def create_graph(path, backend, device_id=0, *, cpu_initializers=None):
        """记录运行时要求的主干路径和后端。

        Args:
            path: 待加载的主干图。
            backend: 运行时请求的后端。
            device_id: 运行时请求的设备编号。
            cpu_initializers: 待注入处理器图的外部张量。
        Returns:
            可观察的主干替身。
        """
        graph = FakeGraph(path, backend, device_id, cpu_initializers=cpu_initializers)
        graphs.append(graph)
        return graph

    monkeypatch.setattr(text, "OrtGraph", create_graph)
    return SimpleNamespace(ckpt=ckpt, table=table, metadata=metadata, graphs=graphs)


def test_lookup_preserves_raw_float32_and_passes_original_eoi_inputs(checkpoint):
    """原编号和掩码继续进入主干，特殊符号和填充公式由原图处理。"""
    encoder = text.HybridTextEncoder(checkpoint.ckpt, device_id=2)
    try:
        ids = np.array([[5, 0, -1], [2, 3, 1]], dtype=np.int64)
        mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        output = encoder.run(feeds)
        lookup_graph, graph = checkpoint.graphs
        assert lookup_graph.path == checkpoint.ckpt / "text_encoder_lookup.onnx"
        assert lookup_graph.backend == "cpu"
        assert lookup_graph.cpu_initializers["token_embedding"] is encoder._table
        assert lookup_graph.calls[0]["input_ids"] is ids
        assert set(lookup_graph.calls[0]) == {"input_ids"}
        assert graph.path == checkpoint.ckpt / "text_encoder_trunk.onnx"
        assert graph.backend == encoder.backend == "cuda"
        assert graph.device_id == encoder.device_id == 2
        assert encoder.session is graph.session
        assert graph.calls[0]["input_ids"] is ids
        assert graph.calls[0]["attention_mask"] is mask
        assert set(feeds) == {"input_ids", "attention_mask"}
        lookup = graph.calls[0]["lookup_embeddings"]
        np.testing.assert_array_equal(lookup, checkpoint.table[ids])
        assert lookup.dtype == np.float32 and lookup.flags.c_contiguous
        assert output["embeddings"] is lookup
        assert encoder._table.mode == "r" and not encoder._table.flags.writeable
    finally:
        encoder.close()


def test_close_releases_session_metadata_and_mapping(checkpoint, monkeypatch):
    """重复关闭安全，接口记录和映射不再持有运行资源。"""
    original = (checkpoint.ckpt / "weights/text.bin").read_bytes()
    encoder = text.HybridTextEncoder(checkpoint.ckpt)
    mapping = encoder._table._mmap
    closed = []
    for graph in checkpoint.graphs:
        close_graph = graph.close

        def record_close(graph=graph, close_graph=close_graph):
            """记录关闭顺序，并确认图仍持有的映射保持有效。"""
            assert not mapping.closed
            closed.append(graph.backend)
            close_graph()

        monkeypatch.setattr(graph, "close", record_close)
    encoder.close()
    encoder.close()
    assert encoder.session is None and encoder._table is None and encoder._lookup is None
    assert encoder._lookup_input_name is None
    assert mapping.closed
    assert closed == ["cuda", "cpu"]
    for graph in checkpoint.graphs:
        assert graph.close_count == 1
        assert not graph.inputs and not graph.outputs and graph.session is None
        assert graph.cpu_initializers is None
    assert (checkpoint.ckpt / "weights/text.bin").read_bytes() == original
    with pytest.raises(RuntimeError, match="已关闭"):
        encoder.run({"input_ids": np.array([[1]]), "attention_mask": np.array([[1]])})


@pytest.mark.parametrize("failed_backend", ["cpu", "cuda"])
def test_failed_graph_construction_closes_the_mapping(checkpoint, monkeypatch, failed_backend):
    """任一图加载失败时已经创建的会话和只读映射都必须释放。"""
    mappings = []
    create_mapping = np.memmap
    create_graph = text.OrtGraph

    def record_mapping(*args, **kwargs):
        """保留映射对象以观察失败路径中的关闭结果。

        Args:
            args: 原映射的位置参数。
            kwargs: 原映射的关键字参数。
        Returns:
            原始只读映射对象。
        """
        mapping = create_mapping(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    def fail_graph(path, backend, device_id=0, **kwargs):
        """模拟处理器查表或显卡主干构造失败。

        Args:
            path: 当前图路径。
            backend: 当前图后端。
            device_id: 当前设备编号。
            kwargs: 图构造关键字参数。
        """
        if backend == failed_backend:
            raise RuntimeError("模拟主干加载失败")
        return create_graph(path, backend, device_id, **kwargs)

    monkeypatch.setattr(text.np, "memmap", record_mapping)
    monkeypatch.setattr(text, "OrtGraph", fail_graph)
    with pytest.raises(RuntimeError, match="模拟主干加载失败"):
        text.HybridTextEncoder(checkpoint.ckpt)
    assert len(mappings) == 1 and mappings[0]._mmap.closed
    assert all(graph.close_count == 1 for graph in checkpoint.graphs)


@pytest.mark.parametrize("field, value", [
    ("dtype", "float16"), ("dtype", "<f4"),
    ("shape", [7]), ("shape", [7, 0]), ("shape", [7, -3]),
    ("shape", [True, 3]), ("shape", ["7", 3]),
    ("offset", -1), ("offset", 1.5), ("offset", True),
    ("length", 83), ("length", 85), ("length", True),
    ("initializer_name", ""), ("initializer_name", None), ("initializer_name", 1),
])
def test_invalid_embedding_metadata_is_rejected_before_loading_graph(checkpoint, field, value):
    """非法类型、形状和长度不能触发主干构造。"""
    checkpoint.metadata["embedding"][field] = value
    (checkpoint.ckpt / "text_encoder_lookup.json").write_text(json.dumps(checkpoint.metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        text.HybridTextEncoder(checkpoint.ckpt)
    assert not checkpoint.graphs


@pytest.mark.parametrize("field, value", [
    ("version", 2), ("version", True), ("embedding", []),
    ("lookup_input_name", ""), ("lookup_input_name", 1),
    ("lookup_input_name", "input_ids"), ("lookup_input_name", "attention_mask"),
])
def test_invalid_lookup_metadata_is_rejected_before_loading_graph(checkpoint, field, value):
    """未知格式和会覆盖原始输入的名称不能进入运行时。"""
    checkpoint.metadata[field] = value
    (checkpoint.ckpt / "text_encoder_lookup.json").write_text(json.dumps(checkpoint.metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        text.HybridTextEncoder(checkpoint.ckpt)
    assert not checkpoint.graphs


@pytest.mark.parametrize("offset, shape", [(96, [7, 3]), (8, [8, 3]), (7, [2**63, 3])])
def test_mapping_must_fit_inside_the_external_file(checkpoint, offset, shape):
    """偏移及表大小组合不能越过实际文件尾部。"""
    checkpoint.metadata["embedding"].update(offset=offset, shape=shape, length=shape[0] * shape[1] * 4)
    (checkpoint.ckpt / "text_encoder_lookup.json").write_text(json.dumps(checkpoint.metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="文件边界"):
        text.HybridTextEncoder(checkpoint.ckpt)
    assert not checkpoint.graphs


@pytest.mark.parametrize("location", ["../outside.bin", "/outside.bin", r"C:\outside.bin", r"C:outside.bin", "weights"])
def test_embedding_location_must_be_a_relative_checkpoint_file(checkpoint, location):
    """绝对路径、目录和越界相对路径均在加载主干前被拒绝。"""
    (checkpoint.ckpt.parent / "outside.bin").write_bytes(checkpoint.table.tobytes())
    checkpoint.metadata["embedding"]["location"] = location
    (checkpoint.ckpt / "text_encoder_lookup.json").write_text(json.dumps(checkpoint.metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        text.HybridTextEncoder(checkpoint.ckpt)
    assert not checkpoint.graphs


@pytest.mark.parametrize("filename", ["weights/text.bin", "text_encoder_trunk.onnx", "text_encoder_lookup.onnx", "text_encoder_lookup.json"])
def test_external_symlink_is_rejected(checkpoint, filename):
    """表、主干和元数据都不能借助符号链接读取检查点外部文件。"""
    path = checkpoint.ckpt / filename
    external = checkpoint.ckpt.parent / Path(filename).name
    external.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(external)
    except OSError as error:
        pytest.skip(f"当前平台不能创建符号链接：{error}")
    with pytest.raises(ValueError, match="检查点内部"):
        text.HybridTextEncoder(checkpoint.ckpt)
    assert not checkpoint.graphs


def test_runtime_does_not_import_onnx(checkpoint, monkeypatch):
    """常规运行时构造只消费元数据，不依赖离线图处理包。"""
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        """阻止离线依赖进入运行时导入路径。

        Args:
            name: 请求导入的模块名。
            args: 其余导入位置参数。
            kwargs: 其余导入关键字参数。
        Returns:
            原始导入机制返回的模块。
        """
        if name == "onnx" or name.startswith("onnx."):
            raise AssertionError("运行时导入了离线依赖")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    encoder = text.HybridTextEncoder(checkpoint.ckpt)
    encoder.close()


def test_cpu_ortvalue_shares_the_readonly_embedding_mapping(checkpoint):
    """外部单精度表注入处理器图时使用同一块只读映射内存。"""
    ort = pytest.importorskip("onnxruntime")
    table = np.memmap(checkpoint.ckpt / "weights/text.bin", dtype="<f4", mode="r", offset=7, shape=(7, 3))
    try:
        value = ort.OrtValue.ortvalue_from_numpy(table)
        viewed = value.numpy()
        assert np.shares_memory(table, viewed)
        assert value.data_ptr() == table.ctypes.data
        assert not table.flags.writeable
        np.testing.assert_array_equal(viewed, checkpoint.table)
        del viewed, value
    finally:
        table._mmap.close()
