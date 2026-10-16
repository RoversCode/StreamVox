from __future__ import annotations

from pathlib import Path

import numpy as np

from ...utils.log import logger


class AssetsManager:
    """负责加载和持有 S2-Pro 的静态 embedding 资产。"""

    def __init__(self, ckpt_root: str | Path) -> None:
        self.ckpt_root = Path(ckpt_root)
        embeddings_dir = self.ckpt_root / "embeddings"
        self.paths = {
            "text_embeddings": embeddings_dir / "text_embeddings.npy",
            "codebook_embeddings": embeddings_dir / "codebook_embeddings.npy",
            "fast_embeddings": embeddings_dir / "fast_embeddings.npy",
        }
        self.load_all()

    def _load_required(self, name: str) -> np.ndarray:
        path = self.paths[name]
        if not path.exists():
            raise FileNotFoundError(f"缺失 S2-Pro embedding 资产: {path}")
        # embedding 是只读静态权重，使用 mmap 避免初始化时额外复制大数组。
        return np.load(path, mmap_mode="r")

    def load_all(self) -> None:
        logger.debug("[S2-Pro Assets] 初始化 %s embedding 权重...", self.ckpt_root)
        self.text_embeddings = self._load_required("text_embeddings")
        self.codebook_embeddings = self._load_required("codebook_embeddings")
        self.fast_embeddings = self._load_required("fast_embeddings")
        self.codebook_size = int(self.fast_embeddings.shape[0])
        self.text_embedding_dim = int(self.text_embeddings.shape[1])
        self.fast_embedding_dim = int(self.fast_embeddings.shape[1])
        logger.debug(
            "[S2-Pro Assets] embedding 权重加载完成: text=%s, codebook=%s, fast=%s",
            self.text_embeddings.shape,
            self.codebook_embeddings.shape,
            self.fast_embeddings.shape,
        )

    def get_text_embeddings(self, token_ids: np.ndarray | int) -> np.ndarray:
        """按 slow 主 token id 取 text embedding。"""
        return self.text_embeddings[token_ids]

    def get_codebook_embeddings(self, codebook_index: int, code_ids: np.ndarray | int) -> np.ndarray:
        """按 codebook 编号和 code id 取 slow acoustic codebook embedding。"""
        # S2-Pro 的 codebook embedding 在文件中按 codebook 顺序拼成一张大表，所以这里集中处理 offset。
        offset_ids = np.asarray(code_ids) + int(codebook_index) * self.codebook_size
        return self.codebook_embeddings[offset_ids]

    def get_fast_embedding(self, code_id: int) -> np.ndarray:
        """按上一层 code id 取 fast 模型的输入 embedding。"""
        return self.fast_embeddings[int(code_id)]
