"""验证请求级文本批次、独立参考前缀及有界转录缓存。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from streamvox.models.breeze import templates
from streamvox.models.breeze.prompt import BreezePromptAsset


class FakeTokenizer:
    """用固定编号观察官方分词往返及各文本段的处理顺序。"""

    def __init__(self):
        self.calls = []
        self.tokens = {
            "[S0]参考。": [2, 11, 12, 1],
            "[S0]另一参考。": [2, 41, 42, 1],
            "[S0]<ins_bos>自然。<ins_eos>目标。": [2, 21, 22, 23, 24, 1],
            "[S0]目标。": [2, 31, 1],
        }
        self.source_ids = {text: [100 + index] for index, text in enumerate(self.tokens)}

    def encode(self, text, *, add_special_tokens):
        """返回预先声明的编号，拒绝未经官方往返的目标输入。

        Args:
            text: 当前文本段或带起止符号的重编码文本。
            add_special_tokens: 是否添加首轮特殊符号。
        Returns:
            含固定编号的编码结果。
        """
        self.calls.append(("encode", text, add_special_tokens))
        if add_special_tokens:
            return SimpleNamespace(ids=self.source_ids[text])
        rendered = {f"<bos>{segment}<eos>": ids for segment, ids in self.tokens.items()}
        return SimpleNamespace(ids=rendered[text])

    def decode(self, ids, *, skip_special_tokens):
        """保留特殊符号，让测试能够区分首轮编码和重新编码。

        Args:
            ids: 首轮文本编号。
            skip_special_tokens: 是否跳过特殊符号。
        Returns:
            带起止符号的完整文本。
        """
        self.calls.append(("decode", tuple(ids), skip_special_tokens))
        assert skip_special_tokens is False
        source = next(text for text, source_ids in self.source_ids.items() if source_ids == ids)
        return f"<bos>{source}<eos>"


class FakeGraph:
    """记录图输入，以词编号、位置和调用轮次区分输出来源。

    Args:
        kind: 文本编码或音频查表职责。
    """

    def __init__(self, kind):
        self.kind, self.calls = kind, []

    def run(self, feeds):
        """生成可检查分支、填充和跨请求缓存的确定性输出。

        Args:
            feeds: 图的文本编号、有效词掩码或音频编码。
        Returns:
            四维隐藏空间中的嵌入。
        """
        self.calls.append({name: value.copy() for name, value in feeds.items()})
        if self.kind == "audio":
            return {"embeddings": np.full((*feeds["codes"].shape[:2], 4), 7, np.float32)}
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        result = np.empty((*ids.shape, 4), dtype=np.float32)
        result[:, :, 0] = ids
        result[:, :, 1] = np.arange(ids.shape[1])
        result[:, :, 2] = mask.sum(axis=1, keepdims=True)
        result[:, :, 3] = len(self.calls)
        return {"embeddings": result}


@pytest.fixture
def components():
    """提供完全运行于处理器的可观察组件。

    Returns:
        分词器、文本图和音频查表图。
    """
    return FakeTokenizer(), FakeGraph("text"), FakeGraph("audio")


@pytest.fixture
def prompt():
    """提供两帧参考音频及完整转录。

    Returns:
        可按需移除音频编码的参考资产。
    """
    return BreezePromptAsset(
        role_name="参考", model="breeze-tts-2-gguf", sample_rate=24000,
        prompt_text="参考。", audio_codes=np.full((2, 16), 9, dtype=np.int64),
    )


def test_reference_prefix_can_be_prepared_without_target_text(components, prompt):
    """准备阶段可以独立得到完整参考前缀，转录和结束帧各处理一次。"""
    tokenizer, encoder, audio = components
    cache = {}
    prefix = templates.build_reference_prefix(
        tokenizer, encoder, audio, prompt=prompt, hidden_size=4,
        eos_code=0, num_codebooks=16, reference_cache=cache,
    )
    assert prefix.shape == (1, 7, 4) and prefix.dtype == np.float32
    assert len(encoder.calls) == len(audio.calls) == 1
    np.testing.assert_array_equal(encoder.calls[0]["input_ids"], [[2, 11, 12, 1]])
    np.testing.assert_array_equal(encoder.calls[0]["attention_mask"], [[1, 1, 1, 1]])
    np.testing.assert_array_equal(audio.calls[0]["codes"][0, :2], prompt.audio_codes)
    np.testing.assert_array_equal(audio.calls[0]["codes"][0, -1], np.zeros(16, dtype=np.int64))
    np.testing.assert_array_equal(prefix[0, :4, 0], [2, 11, 12, 1])
    np.testing.assert_array_equal(prefix[0, 4:], np.full((3, 4), 7))
    assert list(cache) == [(2, 11, 12, 1)]
    assert [call[1] for call in tokenizer.calls if call[0] == "encode" and call[2]] == ["[S0]参考。"]


def test_empty_reference_prefix_avoids_encoding_and_preserves_cache(components, prompt):
    """无参考音频时返回空值，已有参考缓存保持可供后续角色复用。"""
    tokenizer, encoder, audio = components
    previous = np.zeros((1, 1, 4), dtype=np.float32)
    cache = {(99,): previous}
    prefix = templates.build_reference_prefix(
        tokenizer, encoder, audio,
        prompt=replace(prompt, audio_codes=np.empty((0, 16), dtype=np.int64)),
        hidden_size=4, eos_code=0, num_codebooks=16, reference_cache=cache,
    )
    assert prefix is None
    assert not tokenizer.calls and not encoder.calls and not audio.calls
    assert list(cache) == [(99,)] and cache[(99,)] is previous


def test_cfg_targets_use_independent_padded_rows_and_trim_outputs(components, prompt):
    """正负文本只执行一次编码，填充输出不会进入各自预填充。"""
    tokenizer, encoder, audio = components
    empty_prompt = replace(prompt, audio_codes=np.empty((0, 16), dtype=np.int64))
    branches, reference_length = templates.build_prompt_embeddings(
        tokenizer, encoder, audio, text="目标。", prompt=empty_prompt,
        instructions=["自然。", ""], hidden_size=4, eos_code=0, num_codebooks=16,
    )

    assert reference_length == 0
    assert len(encoder.calls) == 1
    assert not audio.calls
    np.testing.assert_array_equal(encoder.calls[0]["input_ids"], [
        [2, 21, 22, 23, 24, 1], [2, 31, 1, 0, 0, 0],
    ])
    np.testing.assert_array_equal(encoder.calls[0]["attention_mask"], [
        [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0],
    ])
    assert [value.shape for value in branches] == [(1, 6, 4), (1, 3, 4)]
    assert all(value.dtype == np.float32 for value in branches)
    np.testing.assert_array_equal(branches[0][0, :, 0], [2, 21, 22, 23, 24, 1])
    np.testing.assert_array_equal(branches[1][0, :, 0], [2, 31, 1])
    np.testing.assert_array_equal(branches[1][0, :, 1], [0, 1, 2])
    np.testing.assert_array_equal(branches[1][0, :, 2], [3, 3, 3])
    assert tokenizer.calls == [
        ("encode", "[S0]<ins_bos>自然。<ins_eos>目标。", True),
        ("decode", (102,), False),
        ("encode", "<bos>[S0]<ins_bos>自然。<ins_eos>目标。<eos>", False),
        ("encode", "[S0]目标。", True),
        ("decode", (103,), False),
        ("encode", "<bos>[S0]目标。<eos>", False),
    ]


def test_cfg_reference_text_and_audio_are_prepared_once(components, prompt):
    """两个分支共享独立编码的参考前缀，结束帧查表只执行一次。"""
    tokenizer, encoder, audio = components
    cache = {}
    branches, reference_length = templates.build_prompt_embeddings(
        tokenizer, encoder, audio, text="目标。", prompt=prompt,
        instructions=["自然。", ""], hidden_size=4, eos_code=0, num_codebooks=16,
        reference_cache=cache,
    )

    assert reference_length == 7
    assert [value.shape for value in branches] == [(1, 13, 4), (1, 10, 4)]
    assert len(encoder.calls) == 2
    assert encoder.calls[0]["input_ids"].shape == (1, 4)
    assert encoder.calls[1]["input_ids"].shape == (2, 6)
    assert len(audio.calls) == 1
    np.testing.assert_array_equal(audio.calls[0]["codes"][0, :2], prompt.audio_codes)
    np.testing.assert_array_equal(audio.calls[0]["codes"][0, -1], np.zeros(16, dtype=np.int64))
    np.testing.assert_array_equal(branches[0][:, :7], branches[1][:, :7])
    np.testing.assert_array_equal(branches[0][0, :4, 0], [2, 11, 12, 1])
    np.testing.assert_array_equal(branches[0][0, 4:7], np.full((3, 4), 7))
    assert list(cache) == [(2, 11, 12, 1)]
    assert [call[1] for call in tokenizer.calls if call[0] == "encode" and call[2]] == [
        "[S0]参考。", "[S0]<ins_bos>自然。<ins_eos>目标。", "[S0]目标。",
    ]


def test_target_is_reencoded_when_matching_the_cached_reference(components, prompt):
    """相同目标每次请求仍编码，只有参考转录能够跨请求复用。"""
    tokenizer, encoder, audio = components
    cache = {}
    requests = [templates.build_prompt_embeddings(
        tokenizer, encoder, audio, text="参考。", prompt=prompt,
        instructions=[""], hidden_size=4, eos_code=0, num_codebooks=16,
        reference_cache=cache,
    ) for _ in range(2)]

    assert len(encoder.calls) == 3
    assert len(audio.calls) == 2
    assert len(cache) == 1
    for call in encoder.calls:
        np.testing.assert_array_equal(call["input_ids"], [[2, 11, 12, 1]])
        np.testing.assert_array_equal(call["attention_mask"], [[1, 1, 1, 1]])
    first, second = (request[0][0] for request in requests)
    np.testing.assert_array_equal(first[:, :7], second[:, :7])
    np.testing.assert_array_equal(first[0, 7:, 3], [2, 2, 2, 2])
    np.testing.assert_array_equal(second[0, 7:, 3], [3, 3, 3, 3])


def test_reference_switch_replaces_the_only_cached_transcript(components, prompt):
    """切回已离开的参考时重新编码，缓存始终只保留最近一条。"""
    tokenizer, encoder, audio = components
    cache = {}
    for selected_prompt, expected_ids in [
        (prompt, (2, 11, 12, 1)),
        (replace(prompt, prompt_text="另一参考。"), (2, 41, 42, 1)),
        (prompt, (2, 11, 12, 1)),
    ]:
        templates.build_prompt_embeddings(
            tokenizer, encoder, audio, text="目标。", prompt=selected_prompt,
            instructions=[""], hidden_size=4, eos_code=0, num_codebooks=16,
            reference_cache=cache,
        )
        assert list(cache) == [expected_ids]
    assert len(encoder.calls) == 6
    np.testing.assert_array_equal(encoder.calls[4]["input_ids"], [[2, 11, 12, 1]])


@pytest.mark.parametrize("limit, expected_calls, expected_cache_size", [(63, 4, 0), (64, 3, 1)])
def test_reference_cache_honors_size_limit_and_discards_old_entry(
    components, prompt, monkeypatch, limit, expected_calls, expected_cache_size,
):
    """超过上限的参考不驻留，等于上限时可以缓存且旧参考被替换。"""
    assert templates.REFERENCE_CACHE_LIMIT_BYTES == 16 * 1024 * 1024
    monkeypatch.setattr(templates, "REFERENCE_CACHE_LIMIT_BYTES", limit)
    tokenizer, encoder, audio = components
    cache = {(99,): np.zeros((1, 1, 4), dtype=np.float32)}
    for _ in range(2):
        templates.build_prompt_embeddings(
            tokenizer, encoder, audio, text="目标。", prompt=prompt,
            instructions=[""], hidden_size=4, eos_code=0, num_codebooks=16,
            reference_cache=cache,
        )
    assert (99,) not in cache
    assert len(cache) == expected_cache_size
    assert len(encoder.calls) == expected_calls


@pytest.mark.parametrize("instruction, expected_ids", [("", [2, 31, 1]), ("自然。", [2, 21, 22, 23, 24, 1])])
def test_single_branch_keeps_unpadded_graph_inputs(components, prompt, instruction, expected_ids):
    """单分支继续使用原始长度和全有效掩码，不引入额外填充。"""
    tokenizer, encoder, audio = components
    branches, reference_length = templates.build_prompt_embeddings(
        tokenizer, encoder, audio, text="目标。",
        prompt=replace(prompt, audio_codes=np.empty((0, 16), dtype=np.int64)),
        instructions=[instruction], hidden_size=4, eos_code=0, num_codebooks=16,
    )
    assert reference_length == 0
    assert len(branches) == len(encoder.calls) == 1
    np.testing.assert_array_equal(encoder.calls[0]["input_ids"], [expected_ids])
    np.testing.assert_array_equal(encoder.calls[0]["attention_mask"], np.ones((1, len(expected_ids))))


def test_hidden_size_mismatch_is_rejected(components, prompt):
    """图输出维度必须符合主干输入约定。"""
    tokenizer, encoder, audio = components
    with pytest.raises(ValueError, match="维度与主干不匹配"):
        templates.build_prompt_embeddings(
            tokenizer, encoder, audio, text="目标。", prompt=prompt,
            instructions=[""], hidden_size=5, eos_code=0, num_codebooks=16,
        )


def test_empty_instruction_list_is_rejected_before_encoding(components, prompt):
    """请求必须包含至少一个分支，空请求不应消耗编码资源。"""
    tokenizer, encoder, audio = components
    with pytest.raises(ValueError, match="至少一个"):
        templates.build_prompt_embeddings(
            tokenizer, encoder, audio, text="目标。", prompt=prompt,
            instructions=[], hidden_size=4, eos_code=0, num_codebooks=16,
        )
    assert not tokenizer.calls and not encoder.calls and not audio.calls
