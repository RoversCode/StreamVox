"""保留原始公式，将文本嵌入查表与主干拆为独立 ONNX 图。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any


def _external_range(ckpt: Path, tensor: Any) -> tuple[Path, int, int | None]:
    """验证外部权重的相对路径和声明的文件区间。

    Args:
        ckpt: 已解析的检查点目录。
        tensor: 使用外部存储的张量元数据。
    Returns:
        权重文件的绝对路径、字节偏移和可选的声明长度。
    """
    external = {entry.key: entry.value for entry in tensor.external_data}
    location = Path(external.get("location", ""))
    if not external.get("location") or location.is_absolute() or PureWindowsPath(external["location"]).anchor:
        raise ValueError("外部权重必须使用检查点内的相对路径。")
    path = (ckpt / location).resolve()
    if not path.is_relative_to(ckpt):
        raise ValueError("外部权重路径必须位于检查点内。")
    if not path.is_file():
        raise ValueError(f"外部权重文件不存在：{location}")
    offset_text, length_text = external.get("offset", "0"), external.get("length")
    # 解析规则须与原图的十进制字节区间一致，拒绝下划线和非 ASCII 数字。
    if any(value is not None and (not value.isascii() or not value.isdecimal()) for value in (offset_text, length_text)):
        raise ValueError("外部权重的偏移和长度必须是非负整数。")
    offset = int(offset_text)
    length = int(length_text) if length_text is not None else None
    file_size = path.stat().st_size
    if offset > file_size or (length is not None and offset + length > file_size):
        raise ValueError("外部权重声明的区间超出文件边界。")
    return path, offset, length


def _publish_assets(staged_assets: dict[Path, Path]) -> None:
    """共同发布已校验的产物，失败时恢复全部原始文件状态。

    Args:
        staged_assets: 正式路径到同目录暂存文件的映射。
    Returns:
        全部产物发布成功后无返回值。
    """
    backups = {}
    temporary_backups = []
    published = []
    try:
        try:
            for path in staged_assets:
                if not path.exists() and not path.is_symlink():
                    continue
                with NamedTemporaryFile(dir=path.parent, prefix=".text-encoder-", suffix=".backup", delete=False) as temporary:
                    backup = Path(temporary.name)
                temporary_backups.append(backup)
                os.replace(path, backup)
                backups[path] = backup
            for path, temporary in staged_assets.items():
                os.replace(temporary, path)
                published.append(path)
        except BaseException:
            # 回滚本身失败时保留备份文件，避免清理流程丢失原始产物。
            for backup in backups.values():
                temporary_backups.remove(backup)
            for path in reversed(published):
                path.unlink(missing_ok=True)
            for path, backup in backups.items():
                os.replace(backup, path)
            raise
    finally:
        for backup in temporary_backups:
            backup.unlink(missing_ok=True)


def split_text_encoder(ckpt: Path) -> set[Path]:
    """拆出原始单精度查表图，生成主干图和可移植的查表记录。

    Args:
        ckpt: 含原始文本编码图及其外部权重的检查点目录。
    Returns:
        新增的查表图、主干图和查表记录三个路径。
    """
    import onnx

    ckpt = Path(ckpt).resolve(strict=True)
    source_path = ckpt / "text_encoder.onnx"
    model = onnx.load_model(source_path, load_external_data=False)
    if not model.graph.node:
        raise ValueError("原图必须以文本嵌入 Gather 节点开头。")
    gather = model.graph.node[0]
    if (
        gather.op_type != "Gather" or gather.domain not in ("", "ai.onnx")
        or len(gather.input) != 2 or gather.input[1] != "input_ids" or len(gather.output) != 1
    ):
        raise ValueError("原图首节点必须是读取 input_ids 的文本嵌入 Gather。")
    axis = next((onnx.helper.get_attribute_value(item) for item in gather.attribute if item.name == "axis"), 0)
    if axis != 0:
        raise ValueError("文本嵌入 Gather 必须沿词表维度使用 axis=0。")
    embedding = next((item for item in model.graph.initializer if item.name == gather.input[0]), None)
    if (
        embedding is None or embedding.data_type != onnx.TensorProto.FLOAT
        or len(embedding.dims) != 2 or any(size <= 0 for size in embedding.dims)
        or embedding.data_location != onnx.TensorProto.EXTERNAL
    ):
        raise ValueError("文本嵌入必须是二维、非空、外部存储的 float32 initializer。")
    if (
        sum(embedding.name in node.input for node in model.graph.node) != 1
        or any(item.name == embedding.name for item in (*model.graph.input, *model.graph.output))
    ):
        raise ValueError("文本嵌入只能由首个 Gather 使用，不能同时作为图输入或输出。")
    inputs = {item.name: item for item in model.graph.input}
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise ValueError("原图必须保留 input_ids 和 attention_mask 输入。")
    if len(inputs["input_ids"].type.tensor_type.shape.dim) != 2:
        raise ValueError("input_ids 必须具有批次和序列两个维度。")

    external_paths = set()
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        path, _, _ = _external_range(ckpt, tensor)
        external_paths.add(path)
    embedding_path, offset, declared_length = _external_range(ckpt, embedding)
    length = math.prod(embedding.dims) * 4
    if declared_length is not None and declared_length != length:
        raise ValueError("文本嵌入的外部长度必须等于二维形状对应的单精度字节数。")
    if offset + length > embedding_path.stat().st_size:
        raise ValueError("文本嵌入的完整数据超出外部文件边界。")

    # 先检查原图，避免追加的新输入掩盖源图已有的未定义引用。
    try:
        onnx.checker.check_model(str(source_path))
    except onnx.checker.ValidationError as error:
        raise ValueError("原始文本编码图的节点或引用无效。") from error
    expected_nodes = [node.SerializeToString() for node in model.graph.node[1:]]
    expected_initializers = [
        tensor.SerializeToString() for tensor in model.graph.initializer if tensor.name != embedding.name
    ]
    expected_lookup_nodes = [gather.SerializeToString()]
    expected_lookup_initializers = [embedding.SerializeToString()]
    metadata = {
        "version": 1,
        "lookup_input_name": gather.output[0],
        "embedding": {
            "location": embedding_path.relative_to(ckpt).as_posix(), "offset": offset,
            "length": length, "shape": list(embedding.dims), "dtype": "float32",
            "initializer_name": embedding.name,
        },
    }
    lookup_input = onnx.helper.make_tensor_value_info(
        gather.output[0], onnx.TensorProto.FLOAT, [None, None, embedding.dims[1]],
    )
    for target, source in zip(
        lookup_input.type.tensor_type.shape.dim, inputs["input_ids"].type.tensor_type.shape.dim,
    ):
        target.CopyFrom(source)
    lookup_model = onnx.ModelProto()
    lookup_model.CopyFrom(model)
    lookup_model.graph.CopyFrom(onnx.helper.make_graph(
        [gather], "text_encoder_lookup", [inputs["input_ids"]], [lookup_input], [embedding],
    ))
    # 后续缩放和结束符号分支仍读取原节点及原始编号，不能将负编号归一化。
    del model.graph.node[0]
    model.graph.initializer.remove(embedding)
    model.graph.input.append(lookup_input)

    lookup_graph_path = ckpt / "text_encoder_lookup.onnx"
    trunk_path = ckpt / "text_encoder_trunk.onnx"
    metadata_path = ckpt / "text_encoder_lookup.json"
    assets = {lookup_graph_path, trunk_path, metadata_path}
    if any(path.resolve() in external_paths | {source_path.resolve()} for path in assets):
        raise ValueError("拆分输出路径与原图或原始外部权重冲突。")
    temporary_paths = []
    staged_assets = {}
    try:
        for path, graph, nodes, initializers in (
            (lookup_graph_path, lookup_model, expected_lookup_nodes, expected_lookup_initializers),
            (trunk_path, model, expected_nodes, expected_initializers),
        ):
            assert [node.SerializeToString() for node in graph.graph.node] == nodes
            assert [tensor.SerializeToString() for tensor in graph.graph.initializer] == initializers
            with NamedTemporaryFile(dir=ckpt, prefix=".text-encoder-", suffix=".onnx", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary_paths.append(temporary_path)
                temporary.write(graph.SerializeToString())
            # 使用文件路径校验，避免将大型外部权重载入 Python 内存。
            onnx.checker.check_model(str(temporary_path))
            written = onnx.load_model(temporary_path, load_external_data=False)
            assert [node.SerializeToString() for node in written.graph.node] == nodes
            assert [tensor.SerializeToString() for tensor in written.graph.initializer] == initializers
            staged_assets[path] = temporary_path
        with NamedTemporaryFile(dir=ckpt, prefix=".text-encoder-", suffix=".json", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary_paths.append(temporary_path)
            temporary.write((json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        assert json.loads(temporary_path.read_text(encoding="utf-8")) == metadata
        staged_assets[metadata_path] = temporary_path
        _publish_assets(staged_assets)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return assets
