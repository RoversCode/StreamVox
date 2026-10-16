"""只读加载词嵌入资源，并按模型的原始规则查表。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingResourceMetadata:
    """描述一份独立 embedding 资源。

    参数说明：
    - ``name``：资源名称，当前固定为 ``base_lm.embed_tokens``。
    - ``weight_path``：``.npy`` 权重文件路径。
    - ``dtype``：落盘权重的数据类型字符串。
    - ``vocab_size`` / ``hidden_size``：embedding 的二维矩阵尺寸。
    - ``use_mup`` / ``scale_emb``：查表后是否需要附加官方的 muP 缩放。
    """

    name: str
    weight_path: str
    dtype: str
    vocab_size: int
    hidden_size: int
    use_mup: bool
    scale_emb: float

    def to_dict(self) -> dict:
        """转为 JSON 友好的字典。"""

        return asdict(self)


class MMapEmbeddingTable:
    """基于 ``numpy.load(..., mmap_mode='r')`` 的 embedding 只读查表器。

    业务意图：
    - M0 先验证独立 embedding 资源方案的正确性，不把大词表查表逻辑塞进主 runtime。
    - 后续 M1 直接复用这一接口，把 ``text_token -> text_embed`` 稳定固定下来。
    """

    def __init__(self, metadata: EmbeddingResourceMetadata) -> None:
        """初始化 mmap 只读 embedding 表。"""

        self.metadata = metadata
        self.weight = np.load(metadata.weight_path, mmap_mode="r")
        if self.weight.shape != (metadata.vocab_size, metadata.hidden_size):
            raise ValueError(
                f"embedding shape mismatch: expected {(metadata.vocab_size, metadata.hidden_size)}, got {self.weight.shape}"
            )

    def lookup(self, token_ids: np.ndarray) -> np.ndarray:
        """执行 token -> embedding 查表，并附加官方缩放规则。

        参数说明：
        - ``token_ids``：一维或二维 token id 序列。

        返回结果：
        - ``float32`` 的 embedding 数组，布局保持与 ``token_ids`` 对齐，最后一维为 hidden size。
        """

        token_array = np.asarray(token_ids)

        # 这里显式转成 int64，避免 numpy 在不同平台下因为默认整型宽度差异造成索引行为不一致。
        token_array = token_array.astype(np.int64, copy=False)
        embed = np.asarray(self.weight[token_array], dtype=np.float32)
        if self.metadata.use_mup:
            embed = embed * np.float32(self.metadata.scale_emb)
        return embed


def load_embedding_metadata(metadata_path: str | Path) -> EmbeddingResourceMetadata:
    """从 ``.json`` 读取 embedding 元数据。"""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return EmbeddingResourceMetadata(**metadata)
