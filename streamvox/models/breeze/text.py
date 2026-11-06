"""在处理器查表，并用显卡执行保留原始单精度公式的文本主干。"""

from __future__ import annotations

import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np

from .ort import OrtGraph


def _checkpoint_file(ckpt: Path, relative: object) -> Path:
    """解析检查点内部的相对文件路径，拒绝越界和外部符号链接。

    Args:
        ckpt: 已解析的检查点目录。
        relative: 元数据声明的相对路径。
    Returns:
        检查点内部真实文件的绝对路径。
    """
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or PureWindowsPath(relative).anchor:
        raise ValueError("Breeze 文本资产必须使用检查点内部的相对路径。")
    path = (ckpt / relative).resolve(strict=True)
    if not path.is_relative_to(ckpt) or not path.is_file():
        raise ValueError("Breeze 文本资产必须是检查点内部的文件。")
    return path


class HybridTextEncoder:
    """只读映射原始嵌入表，并复用显卡文本主干的执行与资源管理。

    Args:
        ckpt_path: 包含文本主干和查表元数据的检查点目录。
        device_id: 显卡设备编号。
    Returns:
        与文本编码图一致的运行、会话检查和关闭接口。
    """

    def __init__(self, ckpt_path: str | Path, device_id: int = 0) -> None:
        self.backend, self.device_id = "cuda", device_id
        self._graph = None
        self._lookup = None
        self._table = None
        self._lookup_input_name = None
        ckpt = Path(ckpt_path).resolve(strict=True)
        if not ckpt.is_dir():
            raise ValueError("Breeze 文本检查点路径必须为目录。")
        graph_path = _checkpoint_file(ckpt, "text_encoder_trunk.onnx")
        lookup_path = _checkpoint_file(ckpt, "text_encoder_lookup.onnx")
        metadata_path = _checkpoint_file(ckpt, "text_encoder_lookup.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or type(metadata.get("version")) is not int or metadata["version"] != 1:
            raise ValueError("Breeze 文本查表元数据版本无效。")
        embedding = metadata.get("embedding")
        if not isinstance(embedding, dict) or embedding.get("dtype") != "float32":
            raise ValueError("Breeze 文本嵌入表必须使用单精度浮点权重。")
        initializer_name = embedding.get("initializer_name")
        if not isinstance(initializer_name, str) or not initializer_name:
            raise ValueError("Breeze 文本嵌入表初始化张量名称无效。")
        shape = embedding.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Breeze 文本嵌入表形状必须包含两个正整数。")
        offset, length = embedding.get("offset"), embedding.get("length")
        if type(offset) is not int or offset < 0:
            raise ValueError("Breeze 文本嵌入表偏移必须为非负整数。")
        if type(length) is not int or length != math.prod(shape) * 4:
            raise ValueError("Breeze 文本嵌入表字节长度与单精度形状不匹配。")
        table_path = _checkpoint_file(ckpt, embedding.get("location"))
        if offset + length > table_path.stat().st_size:
            raise ValueError("Breeze 文本嵌入表超出外部权重文件边界。")
        lookup_input_name = metadata.get("lookup_input_name")
        if not isinstance(lookup_input_name, str) or not lookup_input_name or lookup_input_name in {"input_ids", "attention_mask"}:
            raise ValueError("Breeze 文本查表输入名称无效。")
        self._lookup_input_name = lookup_input_name
        try:
            self._table = np.memmap(table_path, dtype="<f4", mode="r", offset=offset, shape=tuple(shape), order="C")
            # 只读映射作为外部张量注入处理器图，由原始 Gather 执行查表。
            self._lookup = OrtGraph(lookup_path, "cpu", cpu_initializers={initializer_name: self._table})
            # 共用显卡执行器保留单精度计算，并负责释放接口元数据和底层会话。
            self._graph = OrtGraph(graph_path, "cuda", device_id)
        except BaseException:
            self.close()
            raise

    @property
    def session(self):
        """提供底层会话供后端检查和性能诊断。

        Returns:
            已加载的会话，关闭后为空值。
        """
        return None if self._graph is None else self._graph.session

    def run(self, feeds: dict[str, object]) -> dict[str, object]:
        """按原编号查表，继续执行原缩放、特殊符号和文本主干算子。

        Args:
            feeds: 原图的文本编号和有效词掩码。
        Returns:
            文本主干输出，显卡数组继续保留在设备端。
        """
        if self._graph is None or self._lookup is None or self._table is None:
            raise RuntimeError("Breeze 混合文本编码器已关闭。")
        lookup = self._lookup.run({"input_ids": feeds["input_ids"]})
        trunk_feeds = dict(feeds)
        trunk_feeds[self._lookup_input_name] = lookup[self._lookup_input_name]
        return self._graph.run(trunk_feeds)

    def close(self) -> None:
        """释放主干会话、接口记录和只读嵌入表映射。

        Returns:
            编码器资源释放后返回。
        """
        if self._graph is not None:
            self._graph.close()
            self._graph = None
        if self._lookup is not None:
            self._lookup.close()
            self._lookup = None
        if self._table is not None:
            self._table._mmap.close()
            self._table = None
        self._lookup_input_name = None
