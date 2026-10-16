from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer

from ...utils.log import logger


SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"
SPECIAL_TOKEN_CONFIG_KEYS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "additional_special_tokens",
    "extra_special_tokens",
)


def _resolve_tokenizer_json_path(candidate: Path) -> Path | None:
    """
    将外部传入的 tokenizer 路径解析成 tokenizers 可直接读取的 tokenizer.json 文件路径。

    参数:
        candidate: 可能是 tokenizer.json 文件，也可能是包含 tokenizer.json 的目录。

    返回:
        当路径存在且可定位到 tokenizer.json 时返回文件路径；路径不存在或目录内缺少 tokenizer.json 时返回 None。

    边界异常:
        本函数只做路径判断，不读取文件内容；读取失败会交给上层加载流程统一处理。
    """
    # 优先兼容历史调用方式：S2-Pro runtime 传入的是 ckpt/tokenizer 目录。
    if candidate.is_dir():
        tokenizer_json_path = candidate / "tokenizer.json"
        if tokenizer_json_path.exists():
            return tokenizer_json_path
        return None

    # 新入口也允许直接传 tokenizer.json，便于测试和未来资产清单精确指向文件。
    if candidate.is_file():
        return candidate

    return None


def _append_config_token(tokens: list[str], token_payload: object) -> None:
    """
    从 tokenizer_config.json 的单个字段中提取特殊 token 字符串。

    参数:
        tokens: 当前已收集的特殊 token 列表，函数会原地追加新 token。
        token_payload: 配置字段的原始值，可能是字符串、列表或带 content 字段的字典。

    返回:
        无返回值；有效 token 会被追加到 tokens 中。

    边界异常:
        未知结构会被忽略，避免配置文件新增非 token 元数据时误报。
    """
    # 字符串就是最直接的特殊 token 声明，保留原值用于后续 token_to_id 校验。
    if isinstance(token_payload, str):
        tokens.append(token_payload)
        return

    # added_tokens_decoder 中的条目通常是字典，真正的 token 文本在 content 字段。
    if isinstance(token_payload, dict):
        content = token_payload.get("content")
        if isinstance(content, str):
            tokens.append(content)
        return

    # additional_special_tokens 这类字段是列表，需要逐项递归处理，兼容字符串和 AddedToken 字典两种形态。
    if isinstance(token_payload, list):
        for item in token_payload:
            _append_config_token(tokens, item)


def _load_config_special_tokens(tokenizer_json_path: Path) -> list[str]:
    """
    读取 tokenizer_config.json 中声明的特殊 token。

    参数:
        tokenizer_json_path: 已成功加载的 tokenizer.json 文件路径。

    返回:
        去重且保持配置顺序的特殊 token 列表；配置文件不存在时返回空列表。

    边界异常:
        tokenizer_config.json 内容不是合法 JSON 时会透出解析异常，阻止使用不完整或损坏的 tokenizer 资产。
    """
    # tokenizer_config.json 与 tokenizer.json 同目录，这是 HuggingFace tokenizer 资产的常规布局。
    config_path = tokenizer_json_path.with_name("tokenizer_config.json")
    if not config_path.exists():
        return []

    config = json.loads(config_path.read_text(encoding="utf-8"))
    collected_tokens: list[str] = []

    # 先读取明确的 special token 字段，业务意图是覆盖 eos/pad/im_start/semantic 等推理会直接拼进 prompt 的 token。
    for key in SPECIAL_TOKEN_CONFIG_KEYS:
        _append_config_token(collected_tokens, config.get(key))

    # added_tokens_decoder 可能包含额外 added token；即便其中部分不是 special，也应确保 tokenizer.json 能识别它们。
    added_tokens_decoder = config.get("added_tokens_decoder")
    if isinstance(added_tokens_decoder, dict):
        for token_payload in added_tokens_decoder.values():
            _append_config_token(collected_tokens, token_payload)

    # 去重保序，避免同一个 token 同时出现在 eos_token 和 additional_special_tokens 时重复报错。
    unique_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in collected_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        unique_tokens.append(token)
    return unique_tokens


def _validate_config_special_tokens(tokenizer: Tokenizer, tokenizer_json_path: Path) -> None:
    """
    校验 tokenizer_config.json 声明的特殊 token 都已写入 tokenizer.json。

    参数:
        tokenizer: 已通过 tokenizers.Tokenizer.from_file 加载的底层 tokenizer。
        tokenizer_json_path: tokenizer.json 文件路径，用于定位同目录 tokenizer_config.json。

    返回:
        无返回值；全部 token 均可查询到 id 时校验通过。

    边界异常:
        任一配置 token 无法通过 token_to_id 查询到时抛出 RuntimeError，避免 encode 时把特殊 token 拆碎。
    """
    config_special_tokens = _load_config_special_tokens(tokenizer_json_path)
    missing_tokens = [token for token in config_special_tokens if tokenizer.token_to_id(token) is None]

    # 如果配置声明了 tokenizer.json 不认识的 token，继续推理会导致同一文本和 AutoTokenizer 编码不一致，必须尽早失败。
    if missing_tokens:
        preview = ", ".join(missing_tokens[:20])
        raise RuntimeError(
            f"tokenizer_config.json special tokens are missing from tokenizer.json: {preview}"
        )


@dataclass
class InferenceTokenizer:
    """
    S2-Pro 推理侧 tokenizer 适配器，直接基于 tokenizers.Tokenizer 执行编码和解码。

    参数:
        model_path: 首选 tokenizer 路径，可为 tokenizer.json 文件或包含 tokenizer.json 的目录。
        fallback_model_path: 首选路径不可用或加载失败时使用的备用 tokenizer 路径。

    输出:
        实例化成功后会暴露 encode、decode、token_id、vocab_size 以及 semantic token 连续区间。

    边界异常:
        所有候选路径都不存在、缺少 tokenizer.json、tokenizer.json 无法被 tokenizers 加载、
        或词表中找不到任何 semantic token 时会抛出 RuntimeError。
    """

    model_path: str
    fallback_model_path: str | None = None

    # 用了dataclass ，InferenceTokenizer()实例化的时候，就会调用__post_init__方法
    def __post_init__(self) -> None:
        """
        初始化 tokenizers.Tokenizer，并预计算 semantic token 的 id 闭区间。

        参数:
            无显式参数；使用 dataclass 注入的 model_path 与 fallback_model_path。

        返回:
            无返回值；成功后会填充 _tokenizer、tokenizer_path、semantic_begin_id、semantic_end_id。

        边界异常:
            tokenizer 加载失败或 semantic token 缺失时抛出 RuntimeError，阻止后续推理使用错误词表。
        """
        # slow 文本 tokenizer 优先走 ckpts/slow_hf；加载失败时再回退到原始模型目录。
        tokenizer_candidates = [Path(self.model_path)]
        if self.fallback_model_path is not None:
            tokenizer_candidates.append(Path(self.fallback_model_path))

        # 这里保留成员初始化，确保加载失败分支也能给出明确状态，便于调试资产路径问题。
        self._tokenizer: Tokenizer | None = None
        self.tokenizer_path: Path | None = None
        last_load_error: Exception | None = None

        # 逐个尝试候选路径，业务意图是保持旧的 fallback 语义：首选 tokenizer 损坏时继续尝试备用目录。
        for candidate in tokenizer_candidates:
            tokenizer_json_path = _resolve_tokenizer_json_path(candidate)
            if tokenizer_json_path is None:
                continue

            try:
                # [旧逻辑已废弃]: 旧 transformers tokenizer 入口保持可用，但改为懒加载，避免 S2-Pro Vulkan 主路径导入本模块时加载 tokenizers native 库。
                # 新逻辑直接加载 tokenizer.json，彻底绕开 transformers.AutoTokenizer，避免运行时依赖 transformers。
                self._tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
                _validate_config_special_tokens(self._tokenizer, tokenizer_json_path)
                self.tokenizer_path = tokenizer_json_path
                break
            except Exception as exc:
                # 加载失败时继续尝试 fallback，避免单个损坏资产阻断可用备用路径。
                last_load_error = exc
                self._tokenizer = None
                self.tokenizer_path = None
                print(f"Unable to load tokenizer.json from: {candidate}")
                continue

        # 所有候选都失败时，合并候选列表给调用方，方便直接定位缺失或损坏的资产目录。
        if self._tokenizer is None:
            tried = ", ".join(str(p) for p in tokenizer_candidates)
            if last_load_error is not None:
                raise RuntimeError(f"Unable to load tokenizer.json from: {tried}. Last error: {last_load_error}")
            raise RuntimeError(f"Unable to load tokenizer.json from: {tried}")

        # semantic token 在词表里是连续编号，后续 slow/fast 会频繁用到这个闭区间。
        semantic_ids: list[int] = []
        for i in range(4096):
            token = SEMANTIC_TOKEN_TEMPLATE.format(i=i)
            token_id = self.token_id(token)
            if token_id is not None and token_id >= 0:
                semantic_ids.append(token_id)
        if not semantic_ids:
            raise RuntimeError("No semantic tokens found in tokenizer vocab")
        self.semantic_begin_id = min(semantic_ids)
        self.semantic_end_id = max(semantic_ids)

        logger.info(f"[s2-pro]：✅ Tokenizer 加载完毕")

    @property
    def vocab_size(self) -> int:
        """
        返回包含 added tokens 的完整词表大小。

        参数:
            无。

        返回:
            tokenizers 后端登记的完整 token 数量，包含 semantic 等新增特殊 token。

        边界异常:
            正常初始化后不会返回空值；若外部破坏 _tokenizer 状态，会触发断言失败。
        """
        tokenizer = self._require_tokenizer()
        return tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token id 列表，保持与旧 AutoTokenizer.encode(add_special_tokens=False) 一致的外部语义。

        参数:
            text: 需要编码的原始文本，可包含 S2-Pro 的特殊标记。

        返回:
            不自动追加模板特殊 token 的 token id 列表。

        边界异常:
            tokenizer.json 的 normalizer、pre_tokenizer 或 model 配置异常时，底层 tokenizers 异常会直接透出。
        """
        tokenizer = self._require_tokenizer()

        # add_special_tokens=False 只禁止模板自动追加特殊 token；文本中显式出现的特殊 token 仍由 tokenizer.json 识别。
        return tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: Iterable[int]) -> str:
        """
        将 token id 序列解码为文本。

        参数:
            tokens: 任意可迭代的整数 token id 序列。

        返回:
            保留特殊 token 的文本结果，对齐旧 AutoTokenizer.decode 默认不跳过特殊 token 的行为。

        边界异常:
            token id 不在词表内时由底层 tokenizers 处理，当前方法不吞掉异常或替换结果。
        """
        tokenizer = self._require_tokenizer()
        return tokenizer.decode(list(tokens), skip_special_tokens=False)

    def token_id(self, token: str) -> int:
        """
        查询单个 token 在词表中的 id。

        参数:
            token: 需要查询的 token 字符串，通常是 semantic 或对话控制特殊 token。

        返回:
            找到时返回非负整数 id；未找到时返回 -1，兼容旧 convert_tokens_to_ids 包装逻辑。

        边界异常:
            正常初始化后不会返回 None；底层 tokenizer 状态异常时会触发断言失败。
        """
        tokenizer = self._require_tokenizer()
        token_id = tokenizer.token_to_id(token)
        return -1 if token_id is None else token_id

    def _require_tokenizer(self) -> Tokenizer:
        """
        获取已完成初始化的 tokenizers.Tokenizer 实例。

        参数:
            无。

        返回:
            非空的 tokenizers.Tokenizer 实例。

        边界异常:
            如果初始化流程被外部绕过或成员被篡改，会抛出 RuntimeError 明确指出状态非法。
        """
        # 这里使用显式卫语句，避免后续方法在 None 上触发难以定位的 AttributeError。
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized")

        return self._tokenizer
