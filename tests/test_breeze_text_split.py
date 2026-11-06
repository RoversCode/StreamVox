"""用微型真实图验证单精度查表拆分及原始资产边界。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from tools.breeze_reference.text_encoder_split import split_text_encoder
from tools.breeze_reference import text_encoder_split


def _write_checkpoint(
    directory: Path, *, offset: int = 16, include_offset: bool = True, include_length: bool = True,
) -> tuple[Path, np.ndarray]:
    """构造带外部查表、结束符号替换和可观察填充掩码的微型图。

    Args:
        directory: 隔离的测试目录。
        offset: 嵌入表之前保留的字节数。
        include_offset: 是否显式声明偏移。
        include_length: 是否显式声明长度。
    Returns:
        检查点目录及原始单精度嵌入表。
    """
    ckpt = directory / "ckpt"
    weights = ckpt / "weights" / "text.bin"
    weights.parent.mkdir(parents=True)
    table = np.arange(24, dtype=np.float32).reshape(8, 3) / 7
    projection = np.array([[1, 2], [-3, 4], [5, -6]], dtype=np.float32)
    weights.write_bytes(b"x" * offset + table.tobytes() + projection.tobytes() + b"tail")
    embedding = onnx.numpy_helper.from_array(table, name="token_embedding")
    onnx.external_data_helper.set_external_data(
        embedding, location="weights/text.bin",
        offset=offset if include_offset else None, length=table.nbytes if include_length else None,
    )
    embedding.ClearField("raw_data")
    initializers = [embedding] + [onnx.numpy_helper.from_array(value, name=name) for name, value in (
        ("scale", np.array(1.25, dtype=np.float32)),
        ("eoi_id", np.array(6, dtype=np.int64)),
        ("eoi_embedding", np.array([-2, 3, 4], dtype=np.float32)),
        ("axes", np.array([2], dtype=np.int64)),
        ("projection", projection),
    )]
    onnx.external_data_helper.set_external_data(
        initializers[-1], location="weights/text.bin", offset=offset + table.nbytes, length=projection.nbytes,
    )
    initializers[-1].ClearField("raw_data")
    nodes = [
        onnx.helper.make_node("Gather", ["token_embedding", "input_ids"], ["lookup"], axis=0),
        onnx.helper.make_node("Mul", ["lookup", "scale"], ["scaled"]),
        onnx.helper.make_node("Equal", ["input_ids", "eoi_id"], ["is_eoi"]),
        onnx.helper.make_node("Unsqueeze", ["is_eoi", "axes"], ["is_eoi_3d"]),
        onnx.helper.make_node("Where", ["is_eoi_3d", "eoi_embedding", "scaled"], ["selected"]),
        onnx.helper.make_node("MatMul", ["selected", "projection"], ["projected"]),
        onnx.helper.make_node("Cast", ["attention_mask"], ["float_mask"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Unsqueeze", ["float_mask", "axes"], ["mask_3d"]),
        onnx.helper.make_node("Mul", ["projected", "mask_3d"], ["embeddings"]),
    ]
    graph = onnx.helper.make_graph(nodes, "tiny_text_encoder", [
        onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["batch", "sequence"]),
        onnx.helper.make_tensor_value_info("attention_mask", onnx.TensorProto.INT64, ["batch", "sequence"]),
    ], [onnx.helper.make_tensor_value_info("embeddings", onnx.TensorProto.FLOAT, ["batch", "sequence", 2])], initializers)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10)
    (ckpt / "text_encoder.onnx").write_bytes(model.SerializeToString())
    return ckpt, table


def _hashes(ckpt: Path) -> dict[Path, str]:
    """记录目录内所有文件的内容摘要。

    Args:
        ckpt: 待检查的模型目录。
    Returns:
        文件相对路径到内容摘要的映射。
    """
    return {path.relative_to(ckpt): hashlib.sha256(path.read_bytes()).hexdigest() for path in ckpt.rglob("*") if path.is_file()}


@pytest.mark.parametrize("offset, include_offset, include_length", [
    (16, True, True), (16, True, False), (0, False, False), (7, True, True),
])
def test_split_preserves_formula_structure_and_original_assets(
    tmp_path: Path, offset: int, include_offset: bool, include_length: bool,
) -> None:
    """拆图在负编号、结束符号和两行填充输入上保持逐元素相等。

    Args:
        tmp_path: 隔离的测试目录。
        offset: 外部嵌入表之前的字节数。
        include_offset: 是否显式声明外部偏移。
        include_length: 是否显式声明外部长度。
    Returns:
        无返回值。
    """
    ckpt, table = _write_checkpoint(tmp_path, offset=offset, include_offset=include_offset, include_length=include_length)
    original_hashes = _hashes(ckpt)
    source = onnx.load_model(ckpt / "text_encoder.onnx", load_external_data=False)
    files = split_text_encoder(ckpt)
    assert files == {ckpt / "text_encoder_lookup.onnx", ckpt / "text_encoder_trunk.onnx", ckpt / "text_encoder_lookup.json"}
    metadata = json.loads((ckpt / "text_encoder_lookup.json").read_text(encoding="utf-8"))
    assert metadata == {
        "version": 1, "lookup_input_name": "lookup",
        "embedding": {
            "location": "weights/text.bin", "offset": offset, "length": 96, "shape": [8, 3],
            "dtype": "float32", "initializer_name": "token_embedding",
        },
    }
    trunk = onnx.load_model(ckpt / "text_encoder_trunk.onnx", load_external_data=False)
    assert [node.SerializeToString() for node in trunk.graph.node] == [node.SerializeToString() for node in source.graph.node[1:]]
    assert [item.SerializeToString() for item in trunk.graph.initializer] == [item.SerializeToString() for item in source.graph.initializer[1:]]
    assert [item.SerializeToString() for item in trunk.graph.input[:2]] == [item.SerializeToString() for item in source.graph.input]
    lookup_input = trunk.graph.input[2]
    assert lookup_input.name == "lookup"
    assert lookup_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert [item.dim_param or item.dim_value for item in lookup_input.type.tensor_type.shape.dim] == ["batch", "sequence", 3]
    lookup_graph = onnx.load_model(ckpt / "text_encoder_lookup.onnx", load_external_data=False)
    assert [node.SerializeToString() for node in lookup_graph.graph.node] == [source.graph.node[0].SerializeToString()]
    assert [item.SerializeToString() for item in lookup_graph.graph.initializer] == [source.graph.initializer[0].SerializeToString()]
    assert [item.SerializeToString() for item in lookup_graph.graph.input] == [source.graph.input[0].SerializeToString()]
    assert lookup_graph.graph.output[0].SerializeToString() == lookup_input.SerializeToString()

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    original = ort.InferenceSession(str(ckpt / "text_encoder.onnx"), options, providers=["CPUExecutionProvider"])
    split = ort.InferenceSession(str(ckpt / "text_encoder_trunk.onnx"), options, providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids": np.array([[0, 6, -1, 3], [2, -2, 7, 6]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int64),
    }
    mapped = np.memmap(ckpt / metadata["embedding"]["location"], dtype=np.float32, mode="r", offset=offset, shape=(8, 3))
    lookup_session = lookup_options = external_value = None
    try:
        external_value = ort.OrtValue.ortvalue_from_numpy(mapped, "cpu", 0)
        assert not mapped.flags.writeable
        assert np.shares_memory(external_value.numpy(), mapped)
        assert external_value.data_ptr() == mapped.__array_interface__["data"][0]
        lookup_options = ort.SessionOptions()
        lookup_options.intra_op_num_threads = lookup_options.inter_op_num_threads = 1
        lookup_options.add_external_initializers([metadata["embedding"]["initializer_name"]], [external_value])
        lookup_session = ort.InferenceSession(
            str(ckpt / "text_encoder_lookup.onnx"), lookup_options, providers=["CPUExecutionProvider"],
        )
        lookup = lookup_session.run(None, {"input_ids": feeds["input_ids"]})[0]
        expected = np.where(
            feeds["input_ids"][..., None] == 6, np.array([-2, 3, 4], np.float32),
            table[feeds["input_ids"]] * 1.25,
        )
        expected = (expected @ np.array([[1, 2], [-3, 4], [5, -6]], np.float32)) * feeds["attention_mask"][..., None]
        np.testing.assert_array_equal(original.run(None, feeds)[0], expected)
        np.testing.assert_array_equal(split.run(None, {**feeds, metadata["lookup_input_name"]: lookup})[0], expected)
    finally:
        lookup_session = lookup_options = external_value = None
        mapped._mmap.close()
    del original, split

    after_first_split = _hashes(ckpt)
    assert all(after_first_split[path] == digest for path, digest in original_hashes.items())
    assert set(after_first_split) - set(original_hashes) == {path.relative_to(ckpt) for path in files}
    assert split_text_encoder(ckpt) == files
    assert _hashes(ckpt) == after_first_split


@pytest.mark.parametrize("invalid, message", [
    ("first_node", "Gather"), ("axis", "axis=0"), ("dtype", "float32"),
    ("shape", "二维"), ("inline", "外部存储"), ("multiple_users", "只能由首个"),
    ("embedding_input", "图输入或输出"), ("missing_reference", "原始文本编码图"),
    ("negative_offset", "非负整数"), ("invalid_offset", "非负整数"),
    ("underscored_offset", "非负整数"), ("unicode_offset", "非负整数"), ("fractional_offset", "非负整数"),
    ("negative_length", "非负整数"), ("length", "单精度字节数"),
    ("underscored_length", "非负整数"), ("unicode_length", "非负整数"), ("fractional_length", "非负整数"),
    ("offset_boundary", "文件边界"), ("implicit_length_boundary", "文件边界"),
    ("escape", "检查点内"), ("absolute", "相对路径"), ("other_external_escape", "检查点内"),
    ("attribute_external_escape", "检查点内"),
    ("windows_absolute", "相对路径"), ("windows_relative_drive", "相对路径"), ("windows_rooted", "相对路径"),
])
def test_split_rejects_invalid_sources_without_writing_assets(tmp_path: Path, invalid: str, message: str) -> None:
    """非法源图或外部文件区间必须在生成候选文件前明确失败。

    Args:
        tmp_path: 隔离的测试目录。
        invalid: 需要破坏的单个源图约束。
        message: 对应的可读错误说明。
    Returns:
        无返回值。
    """
    ckpt, _ = _write_checkpoint(tmp_path)
    source_path = ckpt / "text_encoder.onnx"
    model = onnx.load_model(source_path, load_external_data=False)
    embedding = model.graph.initializer[0]
    external = {item.key: item for item in embedding.external_data}
    if invalid == "first_node":
        model.graph.node[0].op_type = "Identity"
    elif invalid == "axis":
        model.graph.node[0].attribute[0].i = 1
    elif invalid == "dtype":
        embedding.data_type = onnx.TensorProto.FLOAT16
    elif invalid == "shape":
        del embedding.dims[:]
        embedding.dims.append(24)
    elif invalid == "inline":
        embedding.data_location = onnx.TensorProto.DEFAULT
    elif invalid == "multiple_users":
        model.graph.node.append(onnx.helper.make_node("Identity", [embedding.name], ["unused_embedding"]))
    elif invalid == "embedding_input":
        model.graph.input.append(onnx.helper.make_tensor_value_info(embedding.name, onnx.TensorProto.FLOAT, [8, 3]))
    elif invalid == "missing_reference":
        model.graph.node[1].input[0] = "undefined_value"
    elif invalid in (
        "negative_offset", "invalid_offset", "offset_boundary", "implicit_length_boundary",
        "underscored_offset", "unicode_offset", "fractional_offset",
    ):
        external["offset"].value = {
            "negative_offset": "-1", "invalid_offset": "no", "offset_boundary": "999",
            "implicit_length_boundary": "64", "underscored_offset": "1_6",
            "unicode_offset": "１６", "fractional_offset": "16.0",
        }[invalid]
        if invalid == "implicit_length_boundary":
            embedding.external_data.remove(external["length"])
    elif invalid in ("negative_length", "length", "underscored_length", "unicode_length", "fractional_length"):
        external["length"].value = {
            "negative_length": "-1", "length": "4", "underscored_length": "9_6",
            "unicode_length": "９６", "fractional_length": "96.0",
        }[invalid]
    elif invalid == "escape":
        external["location"].value = "../outside.bin"
    elif invalid == "absolute":
        external["location"].value = str(ckpt / "weights" / "text.bin")
    elif invalid in ("windows_absolute", "windows_relative_drive", "windows_rooted"):
        external["location"].value = {
            "windows_absolute": r"C:\weights\text.bin", "windows_relative_drive": r"C:weights\text.bin",
            "windows_rooted": r"\weights\text.bin",
        }[invalid]
    elif invalid == "other_external_escape":
        scale = model.graph.initializer[1]
        onnx.external_data_helper.set_external_data(scale, location="../outside.bin", offset=0, length=4)
        scale.ClearField("raw_data")
    elif invalid == "attribute_external_escape":
        value = onnx.numpy_helper.from_array(np.array(1, dtype=np.float32))
        onnx.external_data_helper.set_external_data(value, location="../outside.bin", offset=0, length=4)
        value.ClearField("raw_data")
        model.graph.node.append(onnx.helper.make_node("Constant", [], ["unused_external"], value=value))
    source_path.write_bytes(model.SerializeToString())
    before = _hashes(ckpt)
    with pytest.raises(ValueError, match=message):
        split_text_encoder(ckpt)
    assert _hashes(ckpt) == before


@pytest.mark.parametrize("filename", ["text_encoder_lookup.onnx", "text_encoder_trunk.onnx", "text_encoder_lookup.json"])
def test_split_rejects_output_collisions_with_source_weights(tmp_path: Path, filename: str) -> None:
    """外部权重与产物同名时不得覆盖任何源文件。

    Args:
        tmp_path: 隔离的测试目录。
        filename: 与原权重文件冲突的固定产物名称。
    Returns:
        无返回值。
    """
    ckpt, _ = _write_checkpoint(tmp_path)
    (ckpt / "weights" / "text.bin").rename(ckpt / filename)
    source_path = ckpt / "text_encoder.onnx"
    model = onnx.load_model(source_path, load_external_data=False)
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                item.value = filename
    source_path.write_bytes(model.SerializeToString())
    before = _hashes(ckpt)
    with pytest.raises(ValueError, match="输出路径.*冲突"):
        split_text_encoder(ckpt)
    assert _hashes(ckpt) == before


def test_split_keeps_external_data_unloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """源图和写出后的校验都只读取元数据，不载入外部张量。

    Args:
        tmp_path: 隔离的测试目录。
        monkeypatch: 外部数据加载入口的隔离工具。
    Returns:
        无返回值。
    """
    ckpt, _ = _write_checkpoint(tmp_path)

    def reject_loading(*args: object, **kwargs: object) -> None:
        """阻止元数据转换流程提前读取外部权重。

        Args:
            args: 外部数据加载的位置参数。
            kwargs: 外部数据加载的关键字参数。
        Returns:
            无返回值。
        """
        pytest.fail("拆图不应载入外部权重")

    monkeypatch.setattr(onnx, "load_external_data_for_model", reject_loading)
    assert len(split_text_encoder(ckpt)) == 3


@pytest.mark.parametrize("failed_check", [2, 3])
def test_failed_candidate_validation_removes_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_check: int,
) -> None:
    """候选校验失败必须清理暂存文件，并保留原始目录内容。

    Args:
        tmp_path: 隔离的测试目录。
        monkeypatch: 校验入口的隔离工具。
        failed_check: 源图之后查表图或主干图对应的校验轮次。
    Returns:
        无返回值。
    """
    ckpt, _ = _write_checkpoint(tmp_path)
    before = _hashes(ckpt)
    check_model = onnx.checker.check_model
    check_count = 0

    def reject_candidate(path: str) -> None:
        """让此前的图正常通过，模拟指定候选图的校验失败。

        Args:
            path: 当前正在验证的模型文件。
        Returns:
            当前图通过校验时无返回值。
        """
        nonlocal check_count
        check_count += 1
        if check_count == failed_check:
            raise onnx.checker.ValidationError("候选校验失败")
        check_model(path)

    monkeypatch.setattr(onnx.checker, "check_model", reject_candidate)
    with pytest.raises(onnx.checker.ValidationError, match="候选校验失败"):
        split_text_encoder(ckpt)
    assert _hashes(ckpt) == before


def test_tool_import_keeps_onnx_optional() -> None:
    """导入离线工具本身不应加载可选的 ONNX 导出依赖。

    Returns:
        无返回值。
    """
    program = "import sys; from tools.breeze_reference import text_encoder_split; assert 'onnx' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", program], cwd=Path(__file__).resolve().parents[1], check=True)


@pytest.mark.parametrize("existing_assets", [False, True])
@pytest.mark.parametrize("failed_filename", ["text_encoder_lookup.onnx", "text_encoder_trunk.onnx", "text_encoder_lookup.json"])
def test_publish_failure_restores_all_original_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_assets: bool, failed_filename: str,
) -> None:
    """任一产物发布失败时撤回所有新文件，或恢复整组历史产物。

    Args:
        tmp_path: 隔离的测试目录。
        monkeypatch: 原子替换入口的隔离工具。
        existing_assets: 是否先生成一组需要保留的历史产物。
        failed_filename: 本次模拟发布失败的目标产物。
    Returns:
        无返回值。
    """
    ckpt, _ = _write_checkpoint(tmp_path)
    if existing_assets:
        split_text_encoder(ckpt)
    source_path = ckpt / "text_encoder.onnx"
    model = onnx.load_model(source_path, load_external_data=False)
    model.graph.node[0].output[0] = "updated_lookup"
    model.graph.node[1].input[0] = "updated_lookup"
    source_path.write_bytes(model.SerializeToString())
    before = _hashes(ckpt)
    replace = text_encoder_split.os.replace
    failure_seen = False

    def reject_publication(source: Path, destination: Path) -> None:
        """模拟指定产物发布失败一次，恢复旧文件的移动继续正常执行。

        Args:
            source: 暂存文件或待恢复的历史主干。
            destination: 本次原子替换的目标路径。
        Returns:
            非失败轮次替换成功时无返回值。
        """
        nonlocal failure_seen
        if not failure_seen and destination == ckpt / failed_filename:
            failure_seen = True
            raise OSError("模拟产物发布失败")
        replace(source, destination)

    monkeypatch.setattr(text_encoder_split.os, "replace", reject_publication)
    with pytest.raises(OSError, match="产物发布失败"):
        split_text_encoder(ckpt)
    assert _hashes(ckpt) == before


@pytest.mark.parametrize("failed_filename", ["text_encoder_lookup.onnx", "text_encoder_trunk.onnx", "text_encoder_lookup.json"])
def test_backup_failure_restores_all_original_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_filename: str,
) -> None:
    """备份任一历史产物失败时，已经移走的旧文件也必须恢复。

    Args:
        tmp_path: 隔离的测试目录。
        monkeypatch: 原子替换入口的隔离工具。
        failed_filename: 本次模拟无法移到备份位置的产物。
    Returns:
        无返回值。
    """
    ckpt, _ = _write_checkpoint(tmp_path)
    split_text_encoder(ckpt)
    before = _hashes(ckpt)
    replace = text_encoder_split.os.replace

    def reject_backup(source: Path, destination: Path) -> None:
        """阻止指定旧文件移到备份位置，保留其他恢复操作。

        Args:
            source: 当前需要移动的旧文件或暂存产物。
            destination: 备份文件或正式目标路径。
        Returns:
            非失败移动完成时无返回值。
        """
        if source == ckpt / failed_filename and destination.suffix == ".backup":
            raise OSError("模拟产物备份失败")
        replace(source, destination)

    monkeypatch.setattr(text_encoder_split.os, "replace", reject_backup)
    with pytest.raises(OSError, match="产物备份失败"):
        split_text_encoder(ckpt)
    assert _hashes(ckpt) == before
