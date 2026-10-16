"""离线写出及核对词嵌入资源。"""
from pathlib import Path
import json
import numpy as np
import torch
from streamvox.models.voxcpm2.embedding_runtime import EmbeddingResourceMetadata, MMapEmbeddingTable

def save_embedding_metadata(metadata: EmbeddingResourceMetadata, metadata_path: str | Path) -> None:
    """把 embedding 元数据写到 JSON 文件。"""

    # [参考迁移保留]
    # 当前 StreamVox 运行期只读取 manifest 中已有的 metadata；这里保留写盘能力，
    # 仅供离线导出脚本生成或修订独立 embedding 资产时使用。
    Path(metadata_path).write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def verify_text_embed_against_model(
    embedding_table: MMapEmbeddingTable,
    text_token: torch.Tensor,
    model_text_embed: torch.Tensor,
) -> dict[str, float]:
    """校验独立 embedding 查表结果是否与官方模型一致。

    返回结果：
    - ``max_abs_diff`` / ``mean_abs_diff``，供 M0 导出报告记录。
    """

    # [参考迁移保留]
    # 这个对拍入口只用于离线验收“独立 embedding 资源是否与官方模型一致”，
    # runtime 正常初始化不会调用它，避免在客户侧额外引入校验成本。
    exported_embed = embedding_table.lookup(text_token.detach().cpu().numpy())
    official_embed = model_text_embed.detach().to(torch.float32).cpu().numpy()
    abs_diff = np.abs(exported_embed - official_embed)
    return {
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
    }
