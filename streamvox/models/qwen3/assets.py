"""
assets.py - 资产管理器
负责加载词表、投影矩阵和 Codec 嵌入表。
提供针对工匠模型加速的预投影嵌入表。
"""
import os
import numpy as np
from ...utils.log import logger

class AssetsManager:
    """
    负责加载和持有所有静态权重资产。
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        # 定义关键资产路径
        self.paths = {
            "text_table": os.path.join(model_dir, 'embeddings', "text_embedding_projected.npy"),
            "proj_w": os.path.join(model_dir, 'embeddings', "proj_weight.npy"),
            "proj_b": os.path.join(model_dir, 'embeddings', "proj_bias.npy")
        }

        self.load_all()

    def load_all(self):
        """加载所有权重资产到内存"""
        logger.debug(f"[Assets] 初始化 {self.model_dir} 权重...")

        # 1. 加载文本投影表
        if not os.path.exists(self.paths["text_table"]):
            raise FileNotFoundError(f"缺失核心资产: {self.paths['text_table']}")
            
        self.text_table = np.load(self.paths["text_table"], 'r')
        self.tts_pad = self.text_table[151671] # 预存 PAD 向量
        
        # 2. 加载投影矩阵 (2048 -> 1024) - 可选，针对 1.7B
        has_proj = os.path.exists(self.paths["proj_w"]) and os.path.exists(self.paths["proj_b"])
        if has_proj:
            self.proj = {
                "weight": np.load(self.paths["proj_w"], 'r'),
                "bias": np.load(self.paths["proj_b"], 'r')
            }
            pw = self.proj["weight"]
            pb = self.proj["bias"]
            logger.debug("[Assets] 已加载 2048->1024 投影矩阵 (针对 1.7B)")
        else:
            self.proj = None
            logger.debug(" [Assets] 未检测到投影矩阵，将跳过预投影 (针对 0.6B)")
        
        # 3. 加载 16 组 Codec Embedding Tables
        self.emb_tables = []
        self.emb_tables_1024 = []
        
        for i in range(16):
            path = os.path.join(self.model_dir, 'embeddings', f"codec_embedding_{i}.npy")
            if not os.path.exists(path):
                # 如果没有 16 组，尝试兼容某些旧版本或精简版
                logger.error(f"[Assets] 缺失第 {i} 组 Codec 嵌入表，尝试跳过...")
                continue
                
            table = np.load(path, 'r')
            self.emb_tables.append(table)
            
            # 预投影：针对 Predictor 模型（工匠模型）加速
            if self.proj is not None:
                # 1.7B 模式：从 2048 投影到 1024
                table_1024 = table @ pw.T + pb
                self.emb_tables_1024.append(table_1024)
            else:
                # 0.6B 模式：直接使用（已经是 1024 维度）
                self.emb_tables_1024.append(table)
            
        logger.debug(f"[Assets] 权重加载完成 (Codec 表数量: {len(self.emb_tables)})")

    def get_text_embedding(self, token_id: int) -> np.ndarray:
        return self.text_table[token_id]

    def get_codec_embedding(self, q_idx: int, code: int) -> np.ndarray:
        """获取原始 2048 维嵌入"""
        return self.emb_tables[q_idx][code]

    def get_codec_embedding_1024(self, q_idx: int, code: int) -> np.ndarray:
        """获取预投影后的 1024 维嵌入"""
        return self.emb_tables_1024[q_idx][code]
