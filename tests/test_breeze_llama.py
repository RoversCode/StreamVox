"""用真实微型浮点模型验证原生解码、分支隔离与缓存生命周期。"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from threading import RLock
from unittest.mock import Mock

import gguf
import numpy as np
import pytest

from streamvox.models.breeze.llama import LlamaDecoder
from streamvox.models.llama_native import NativeLibrary


def _tiny_model(path: Path) -> dict[str, np.ndarray]:
    """生成没有量化且具有真实注意力和位置编码的微型 Llama。

    Args:
        path: 临时 GGUF 路径。
    Returns:
        供独立 NumPy 参考实现使用的原始权重。
    """
    rng = np.random.default_rng(83)
    weights = {
        "token_embd.weight": rng.normal(0, 0.3, (9, 8)).astype(np.float32),
        "output.weight": rng.normal(0, 0.3, (9, 8)).astype(np.float32),
        "output_norm.weight": np.ones(8, dtype=np.float32),
    }
    for layer in range(2):
        for name, shape in (
            ("attn_q", (8, 8)), ("attn_k", (4, 8)), ("attn_v", (4, 8)),
            ("attn_output", (8, 8)), ("ffn_gate", (16, 8)),
            ("ffn_up", (16, 8)), ("ffn_down", (8, 16)),
        ):
            weights[f"blk.{layer}.{name}.weight"] = rng.normal(0, 0.2, shape).astype(np.float32)
        for name in ("attn_norm", "ffn_norm"):
            weights[f"blk.{layer}.{name}.weight"] = np.ones(8, dtype=np.float32)
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_name("Breeze 原生接口数值测试")
    writer.add_tokenizer_model("none")
    writer.add_vocab_size(9)
    writer.add_context_length(1024)
    writer.add_embedding_length(8)
    writer.add_block_count(2)
    writer.add_feed_forward_length(16)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(4)
    writer.add_rope_freq_base(500000.0)
    writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    for name, value in weights.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return weights


def _reference(weights: dict[str, np.ndarray], embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """独立计算相邻旋转布局、分组查询注意力和门控前馈。

    Args:
        weights: 微型模型的 FP32 权重。
        embeddings: 批次、长度、隐藏维度的完整前缀。
    Returns:
        各分支最后位置的最终隐藏状态与完整词表分数。
    """
    batch, length, _hidden = embeddings.shape
    state = embeddings.copy()
    mask = np.triu(np.full((length, length), -np.inf, dtype=np.float32), 1)
    theta = np.arange(length, dtype=np.float32)[:, None] / (500000.0 ** (np.arange(0, 4, 2, dtype=np.float32) / 4))
    cosine, sine = np.cos(theta)[None, :, None], np.sin(theta)[None, :, None]
    for layer in range(2):
        prefix = f"blk.{layer}."
        norm = state / np.sqrt(np.mean(state * state, axis=-1, keepdims=True) + 1e-5)
        query = (norm @ weights[prefix + "attn_q.weight"].T).reshape(batch, length, 2, 4)
        key = (norm @ weights[prefix + "attn_k.weight"].T).reshape(batch, length, 1, 4)
        value = (norm @ weights[prefix + "attn_v.weight"].T).reshape(batch, length, 1, 4)
        for projected in (query, key):
            even, odd = projected[..., ::2].copy(), projected[..., 1::2].copy()
            projected[..., ::2] = even * cosine - odd * sine
            projected[..., 1::2] = even * sine + odd * cosine
        query, key, value = (item.transpose(0, 2, 1, 3) for item in (query, key, value))
        scores = (query @ key.transpose(0, 1, 3, 2)) * 0.5 + mask
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attention = (probabilities @ value).transpose(0, 2, 1, 3).reshape(batch, length, 8)
        state += attention @ weights[prefix + "attn_output.weight"].T
        norm = state / np.sqrt(np.mean(state * state, axis=-1, keepdims=True) + 1e-5)
        gate = norm @ weights[prefix + "ffn_gate.weight"].T
        activated = gate / (1.0 + np.exp(-gate))
        state += (activated * (norm @ weights[prefix + "ffn_up.weight"].T)) @ weights[prefix + "ffn_down.weight"].T
    state = state / np.sqrt(np.mean(state * state, axis=-1, keepdims=True) + 1e-5)
    final = state[:, -1]
    return final, final @ weights["output.weight"].T


@pytest.fixture
def tiny(tmp_path: Path):
    """仅在当前平台具备发行原生库时构建真实微型模型。

    Args:
        tmp_path: 测试临时目录。
    Returns:
        GGUF 路径和独立参考权重。
    """
    directory = Path(__file__).resolve().parents[1] / "streamvox" / "bin"
    if not any((directory / name).is_file() for name in ("llama.dll", "libllama.so", "libllama.dylib")):
        pytest.skip("当前环境没有发行原生库，无法执行原生 CPU 数值验证")
    path = tmp_path / "tiny.gguf"
    return path, _tiny_model(path)


def test_batched_sequences_and_cached_tokens_match_reference(tiny) -> None:
    """独立条件分支、嵌入前缀和续写 token 必须匹配完整因果参考。"""
    path, weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=16)
    rng = np.random.default_rng(18)
    prefix = rng.normal(size=(2, 3, 8)).astype(np.float32)
    tokens = np.asarray([[2], [5]], dtype=np.int64)
    try:
        first = decoder.decode(embeddings=prefix, positions=[0, 0], sequence_ids=[1, 0])
        hidden, logits = _reference(weights, prefix)
        np.testing.assert_allclose(first["hidden"], hidden, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(first["logits"], logits, atol=3e-5, rtol=3e-5)
        saved = first["hidden"].copy()
        second = decoder.decode(tokens=tokens, positions=[3, 3], sequence_ids=[1, 0], logit_start=2, logit_count=3)
        complete = np.concatenate((prefix, weights["token_embd.weight"][tokens]), axis=1)
        hidden, logits = _reference(weights, complete)
        np.testing.assert_allclose(second["hidden"], hidden, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(second["logits"], logits[:, 2:5], atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(first["hidden"], saved)
    finally:
        decoder.close()
        decoder.close()


def test_reset_and_reserve_preserve_weights_and_clear_cache(tiny) -> None:
    """清缓存和扩容后使用同一模型权重重新计算，结果必须可复现。"""
    path, _weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=4, embeddings=False)
    try:
        tokens = np.asarray([[1, 2, 3]], dtype=np.int32)
        original = decoder.decode(tokens=tokens, positions=[0], sequence_ids=[0])
        model = decoder._model
        decoder.reset()
        repeated = decoder.decode(tokens=tokens, positions=[0], sequence_ids=[0])
        np.testing.assert_array_equal(original["logits"], repeated["logits"])
        decoder.reserve(16)
        assert decoder._model == model
        assert decoder.n_ctx == 16
        expanded = decoder.decode(tokens=tokens, positions=[0], sequence_ids=[0])
        np.testing.assert_array_equal(original["logits"], expanded["logits"])
        assert expanded["hidden"] is None
    finally:
        decoder.close()


@pytest.mark.parametrize("extract_hidden", [False, True])
def test_different_prefix_lengths_can_continue_in_one_batch(tiny, extract_hidden: bool) -> None:
    """不同指令长度的条件分支在各自绝对位置续写时保持独立上下文。"""
    path, weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=16, embeddings=extract_hidden)
    rng = np.random.default_rng(42)
    prefixes = [rng.normal(size=(1, length, 8)).astype(np.float32) for length in (2, 5)]
    tokens = np.asarray([[4], [7]], dtype=np.int32)
    try:
        for branch, prefix in enumerate(prefixes):
            decoder.decode(embeddings=prefix, positions=[0], sequence_ids=[branch])
        result = decoder.decode(tokens=tokens, positions=[2, 5], sequence_ids=[0, 1], logit_start=1, logit_count=4)
        expected = [_reference(weights, np.concatenate((prefix, weights["token_embd.weight"][tokens[branch:branch + 1]]), axis=1))
                    for branch, prefix in enumerate(prefixes)]
        np.testing.assert_allclose(result["logits"], np.concatenate([value[1][:, 1:5] for value in expected]), atol=3e-5, rtol=3e-5)
        if extract_hidden:
            np.testing.assert_allclose(result["hidden"], np.concatenate([value[0] for value in expected]), atol=3e-5, rtol=3e-5)
        else:
            assert result["hidden"] is None
    finally:
        decoder.close()


def test_long_prefill_chunks_and_batch_reuse_match_reference(tiny) -> None:
    """长提示分块提交与整段因果解码一致，短续写复用既有输入存储。"""
    path, weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=600)
    tokens = np.arange(520, dtype=np.int32)[None] % 9
    try:
        output = decoder.decode(tokens=tokens, positions=[0], sequence_ids=[0])
        hidden, logits = _reference(weights, weights["token_embd.weight"][tokens])
        np.testing.assert_allclose(output["hidden"], hidden, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(output["logits"], logits, atol=5e-5, rtol=5e-5)
        address = ct.cast(decoder._batches[False][0].token, ct.c_void_p).value
        skipped = decoder.decode(tokens=np.asarray([[2]], dtype=np.int32), positions=[520], sequence_ids=[0], output=False)
        assert skipped == {"hidden": None, "logits": None}
        assert ct.cast(decoder._batches[False][0].token, ct.c_void_p).value == address
        final = decoder.decode(tokens=np.asarray([[3]], dtype=np.int32), positions=[521], sequence_ids=[0])
        complete = np.concatenate((tokens, [[2, 3]]), axis=1)
        hidden, logits = _reference(weights, weights["token_embd.weight"][complete])
        np.testing.assert_allclose(final["hidden"], hidden, atol=5e-5, rtol=5e-5)
        np.testing.assert_allclose(final["logits"], logits, atol=5e-5, rtol=5e-5)
    finally:
        decoder.close()


def test_truncated_prefix_matches_fresh_target_prefill(tiny) -> None:
    """保留参考前缀并替换目标后，与清空缓存后的完整预填充一致。"""
    path, weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=16)
    prefix = np.random.default_rng(57).normal(size=(1, 4, 8)).astype(np.float32)
    first_target = np.asarray([[1, 2, 3]], dtype=np.int32)
    second_target = np.asarray([[8, 6]], dtype=np.int32)
    try:
        decoder.decode(embeddings=prefix, positions=[0], sequence_ids=[0], output=False)
        decoder.decode(tokens=first_target, positions=[4], sequence_ids=[0])
        decoder.remove_sequence(np.int64(0), np.int64(4))
        reused = decoder.decode(tokens=second_target, positions=[4], sequence_ids=[0])
        complete = np.concatenate((prefix, weights["token_embd.weight"][second_target]), axis=1)
        decoder.reset()
        fresh = decoder.decode(embeddings=complete, positions=[0], sequence_ids=[0])
        hidden, logits = _reference(weights, complete)
        np.testing.assert_allclose(reused["hidden"], fresh["hidden"], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(reused["logits"], fresh["logits"], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(reused["hidden"], hidden, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(reused["logits"], logits, atol=3e-5, rtol=3e-5)
    finally:
        decoder.close()


@pytest.mark.parametrize("arguments", [
    (-1, 0), (2, 0), (True, 0), (0, -1), (0, 17),
    (0, 0.0), (0, 2, 1), (0, 0, -2), (0, 0, 17), (0, 0, False),
])
def test_remove_sequence_rejects_invalid_boundaries(tiny, monkeypatch, arguments: tuple) -> None:
    """序列编号、位置类型与区间越界必须在调用原生接口前被拒绝。"""
    path, _weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=16)
    remove = Mock()
    monkeypatch.setattr(decoder._api, "llama_memory_seq_rm", remove)
    try:
        with pytest.raises(ValueError):
            decoder.remove_sequence(*arguments)
        remove.assert_not_called()
    finally:
        decoder.close()


def test_remove_sequence_reports_native_failure_and_closed_context(tiny, monkeypatch) -> None:
    """原生截断失败必须上报，已关闭的上下文拒绝后续序列操作。"""
    path, _weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=16)
    remove = Mock(return_value=False)
    monkeypatch.setattr(decoder._api, "llama_memory_seq_rm", remove)
    try:
        with pytest.raises(RuntimeError, match="无法删除序列 0.*\\[4, -1\\)"):
            decoder.remove_sequence(0, 4)
        remove.assert_called_once_with(decoder._api.llama_get_memory(decoder._context), 0, 4, -1)
    finally:
        decoder.close()
    with pytest.raises(RuntimeError, match="已经关闭"):
        decoder.remove_sequence(0, 0)


def test_explicit_backend_does_not_fall_back() -> None:
    """指定后端没有设备时必须拒绝，不能转用已注册的 CPU。"""
    library = NativeLibrary.__new__(NativeLibrary)
    library._lock = RLock()
    library._load_backend = Mock(return_value=23)
    library.ggml_backend_reg_dev_count = Mock(return_value=0)
    library.ggml_backend_reg_dev_get = Mock()
    with pytest.raises(RuntimeError, match="vulkan.*可用设备数为 0"):
        library.device("vulkan", 0)
    library._load_backend.assert_called_once_with("vulkan")
    library.ggml_backend_reg_dev_get.assert_not_called()


def test_decode_rejects_shared_branch_identity_and_overflow(tiny) -> None:
    """重复分支身份与上下文越界必须在进入原生代码前被拒绝。"""
    path, _weights = tiny
    decoder = LlamaDecoder(path, backend="cpu", n_ctx=4)
    try:
        with pytest.raises(ValueError, match="不同序列编号"):
            decoder.decode(tokens=np.zeros((2, 1), np.int32), positions=[0, 0], sequence_ids=[0, 0])
        with pytest.raises(ValueError, match="缓存容量"):
            decoder.decode(tokens=np.zeros((1, 2), np.int32), positions=[3], sequence_ids=[0])
        with pytest.raises(ValueError, match="词表范围"):
            decoder.decode(tokens=np.asarray([[9]], np.int32), positions=[0], sequence_ids=[0])
    finally:
        decoder.close()
    with pytest.raises(RuntimeError, match="已经关闭"):
        decoder.reset()
