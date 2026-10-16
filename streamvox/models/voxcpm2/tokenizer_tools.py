"""无官方依赖的 VoxCPM2 tokenizer 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import List

from tokenizers import Tokenizer


class CharTokenizerWrapper:
    """把纯中文多字词 token 拆成单字 token 的轻量包装器。

    设计意图：
    - VoxCPM2 官方推理会对 tokenizer 做一次包裹，避免中文多字词 token
      直接进入后续文本-音频混合编排。
    - 这里保留同样的语义，但实现完全独立，不依赖 `voxcpm2_office`。
    """

    def __init__(self, base_tokenizer: Tokenizer) -> None:
        """构造包装器并预计算纯中文多字词词表。"""

        self.tokenizer = base_tokenizer
        # 关键变量：本地 tokenizer 词表，后续用于中文多字词拆分与 token-id 查询。
        self.vocab = self.tokenizer.get_vocab()
        self.multichar_tokens = {
            token
            for token in self.vocab.keys()
            if len(token) >= 2 and all("\u4e00" <= char <= "\u9fff" for char in token)
        }

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """执行分词，并把纯中文多字词拆回单字。"""

        if not isinstance(text, str):
            raise TypeError(f"expected string input, got {type(text)}")

        # tokenizers.Tokenizer 只暴露 encode 接口；这里读取 tokens 字段以保持参考实现的包装语义。
        tokens = self.tokenizer.encode(text, add_special_tokens=bool(kwargs.get("add_special_tokens", False))).tokens
        processed_tokens: list[str] = []
        for token in tokens:
            clean_token = token.replace("▁", "")
            if clean_token in self.multichar_tokens:
                processed_tokens.extend(list(clean_token))
            else:
                processed_tokens.append(token)
        return processed_tokens

    def __call__(self, text: str, **kwargs) -> List[int]:
        """执行 tokenization 并直接返回 token id 列表。"""

        tokens = self.tokenize(text, **kwargs)
        # token_to_id 可能在极端拆分场景返回 None；这种情况说明 tokenizer 词表缺失，继续推理只会产生错误输入。
        token_ids: list[int] = []
        for token in tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"voxcpm2 tokenizer cannot resolve token: {token!r}")
            token_ids.append(int(token_id))
        return token_ids

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        兼容 StreamVox 通用文本分段器的 tokenizer 接口。

        参数:
            text: 需要度量长度或切分的文本。
            **kwargs: 透传给 `__call__` 的 tokenizer 参数。

        返回:
            返回 token id 列表；通用分段器只依赖长度，不要求 offset 信息。

        边界异常:
            底层 token 无法映射为 id 时，沿用 `__call__` 的 ValueError。
        """

        # 通用 segment_text 使用 tokenizer.encode 计算 token 长度；这里复用 VoxCPM2 自己的中文拆字规则。
        return self(text, **kwargs)


def load_voxcpm2_tokenizer(model_path: str) -> CharTokenizerWrapper:
    """从本地模型目录加载 VoxCPM2 tokenizer 包装器。"""

    return CharTokenizerWrapper(Tokenizer.from_file(str(Path(model_path) / "tokenizer.json")))
