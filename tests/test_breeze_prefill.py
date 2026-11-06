"""验证单分支前缀缓存、多分支完整预填充和请求失效边界。"""

from __future__ import annotations

import numpy as np
import pytest

from streamvox.models.breeze.prefill import BackbonePrefill


class FakeNative:
    """记录原生操作并保存独立序列，供预填充生命周期测试观察。

    Args:
        backend: 用于选择物理微批次对齐规则的后端名称。
    """

    def __init__(self, backend: str = "cpu") -> None:
        self.backend = backend
        self.n_ctx = 16
        self.sequences: dict[int, np.ndarray] = {}
        self.decode_calls: list[dict] = []
        self.remove_calls: list[tuple[int, int, int]] = []
        self.events: list[str] = []
        self.reset_count = 0
        self.fail_decode = self.fail_remove = False

    def reserve(self, n_ctx: int) -> None:
        """扩容时模拟原生上下文重建，已有容量则保留缓存。

        Args:
            n_ctx: 请求的每序列容量。
        Returns:
            容量准备完成后返回。
        """
        self.events.append("reserve")
        if n_ctx > self.n_ctx:
            self.n_ctx = n_ctx
            self.sequences.clear()

    def reset(self) -> None:
        """清空所有序列，同时保留可观察的历史操作。

        Returns:
            序列状态清空后返回。
        """
        self.events.append("reset")
        self.reset_count += 1
        self.sequences.clear()

    def decode(self, *, embeddings: np.ndarray, positions: list[int], sequence_ids: list[int], output: bool = True) -> dict:
        """按绝对位置追加嵌入，拒绝未清理的旧后缀。

        Args:
            embeddings: 当前分支提交的参考或目标嵌入。
            positions: 单个分支的起始绝对位置。
            sequence_ids: 单个独立序列编号。
            output: 是否提取分数和隐藏状态。
        Returns:
            根据完整序列生成的隐藏状态和分数。
        """
        self.events.append("decode")
        self.decode_calls.append({
            "embeddings": embeddings.copy(), "positions": positions.copy(),
            "sequence_ids": sequence_ids.copy(), "output": output,
        })
        if self.fail_decode:
            self.fail_decode = False
            raise RuntimeError("模拟原生解码失败")
        sequence, position = sequence_ids[0], positions[0]
        previous = self.sequences.get(sequence, embeddings[:, :0])
        assert position == previous.shape[1]
        complete = np.concatenate((previous, embeddings), axis=1)
        self.sequences[sequence] = complete
        if not output:
            return {"hidden": None, "logits": None}
        hidden = complete.sum(axis=1)
        return {"hidden": hidden, "logits": hidden[:, :1].copy()}

    def remove_sequence(self, sequence: int, start: int, end: int = -1) -> None:
        """记录分支清理并截断后缀，允许模拟原生清理失败。

        Args:
            sequence: 待清理的序列编号。
            start: 删除区间的起点。
            end: 删除区间的终点，负一表示到末尾。
        Returns:
            后缀或整个序列删除后返回。
        """
        self.events.append("remove")
        self.remove_calls.append((sequence, start, end))
        if self.fail_remove:
            self.fail_remove = False
            raise RuntimeError("模拟原生清理失败")
        assert end == -1
        if start == 0:
            self.sequences.pop(sequence, None)
            return
        self.sequences[sequence] = self.sequences[sequence][:, :start].copy()


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_same_reference_reuses_suffix_for_different_targets(backend: str) -> None:
    """冷请求计算完整提示，热请求仅追加新目标并保留一条参考前缀。"""
    native = FakeNative(backend)
    prefill = BackbonePrefill(native)
    reference = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    first = np.concatenate((reference, np.full((1, 2, 4), 20, np.float32)), axis=1)
    second = np.concatenate((reference, np.full((1, 3, 4), 40, np.float32)), axis=1)

    cold = prefill.decode([first], reference_length=4)
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], first)
    assert native.decode_calls[-1]["positions"] == [0]
    prefill.finish(True)
    assert native.remove_calls == [(0, 4, -1)]
    np.testing.assert_array_equal(native.sequences[0], reference)

    reset_count = native.reset_count
    hot = prefill.decode([second.copy(order="F")], reference_length=4)
    assert native.reset_count == reset_count
    assert native.decode_calls[-1]["positions"] == [4]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], second[:, 4:])
    np.testing.assert_array_equal(hot[0]["hidden"], second.sum(axis=1))
    assert not np.array_equal(cold[0]["hidden"], hot[0]["hidden"])
    prefill.finish(True)
    assert set(native.sequences) == {0}
    np.testing.assert_array_equal(native.sequences[0], reference)


def test_mutated_reference_content_invalidates_same_prompt_object() -> None:
    """同一可变提示数组的参考内容改变后，必须完整重算前缀。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    prompt = np.arange(24, dtype=np.float32).reshape(1, 6, 4)
    prefill.decode([prompt], reference_length=4)
    prefill.finish(True)
    reset_count = native.reset_count

    prompt[0, 1, 2] += 10
    prefill.decode([prompt], reference_length=4)
    assert native.reset_count == reset_count + 1
    assert native.decode_calls[-1]["positions"] == [0]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], prompt)
    prefill.finish(True)
    np.testing.assert_array_equal(native.sequences[0], prompt[:, :4])


def test_reserve_growth_invalidates_reference_but_existing_capacity_preserves_it() -> None:
    """容量足够时继续命中，原生上下文扩容后必须重新建立前缀。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    prompt = np.arange(24, dtype=np.float32).reshape(1, 6, 4)
    prefill.decode([prompt], reference_length=4)
    prefill.finish(True)

    prefill.reserve(8)
    prefill.decode([prompt], reference_length=4)
    assert native.decode_calls[-1]["positions"] == [4]
    prefill.finish(True)
    reset_count = native.reset_count

    prefill.reserve(32)
    prefill.decode([prompt], reference_length=4)
    assert native.n_ctx == 32
    assert native.reset_count == reset_count + 1
    assert native.decode_calls[-1]["positions"] == [0]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], prompt)
    prefill.finish(True)


def test_cfg_requests_recompute_full_prompts_and_invalidate_single_branch_cache() -> None:
    """单分支缓存切换到多分支后完整重算，后续单分支也重新建立参考。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    reference = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    prompts = [np.concatenate((reference, np.full((1, length, 4), value, np.float32)), axis=1)
               for length, value in ((2, 20), (3, 30))]
    prefill.decode([prompts[0]], reference_length=4)
    prefill.finish(True)
    np.testing.assert_array_equal(native.sequences[0], reference)

    for _ in range(2):
        event_start = len(native.events)
        branches = prefill.decode(prompts, reference_length=4)
        assert native.events[event_start:] == ["reset", "decode", "decode"]
        for sequence, prompt in enumerate(prompts):
            call = native.decode_calls[-2 + sequence]
            assert call["positions"] == [0]
            assert call["sequence_ids"] == [sequence]
            np.testing.assert_array_equal(call["embeddings"], prompt)
            np.testing.assert_array_equal(native.sequences[sequence], prompt)
            np.testing.assert_array_equal(branches[sequence]["hidden"], prompt.sum(axis=1))
        prefill.finish(True)
        assert native.sequences == {}
        assert native.events[-1] == "reset"

    prefill.decode([prompts[0]], reference_length=4)
    assert native.decode_calls[-1]["positions"] == [0]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], prompts[0])
    prefill.finish(True)
    assert set(native.sequences) == {0}
    np.testing.assert_array_equal(native.sequences[0], reference)


@pytest.mark.parametrize(("backend", "reference_length", "retained_length"), [
    ("cpu", 0, 0), ("cuda", 0, 0), ("cuda", 1, 0),
    ("cuda", 3, 2), ("cuda", 5, 4),
])
def test_odd_cuda_reference_recomputes_boundary_and_empty_reference_does_not_cache(
    backend: str, reference_length: int, retained_length: int,
) -> None:
    """奇数参考的边界词随目标重算，没有可保留前缀时每次完整预填充。"""
    native = FakeNative(backend)
    prefill = BackbonePrefill(native)
    prompt = np.arange((reference_length + 2) * 4, dtype=np.float32).reshape(1, reference_length + 2, 4)
    prefill.decode([prompt], reference_length)
    prefill.finish(True)
    if retained_length:
        np.testing.assert_array_equal(native.sequences[0], prompt[:, :retained_length])
    else:
        assert native.sequences == {}

    reset_count = native.reset_count
    prompt[:, reference_length:] += 100
    prefill.decode([prompt], reference_length)
    assert native.decode_calls[-1]["positions"] == [retained_length]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], prompt[:, retained_length:])
    assert native.reset_count == reset_count + int(retained_length == 0)
    prefill.finish(True)


@pytest.mark.parametrize("failure", ["cancelled", "decode_failed", "cleanup_failed"])
def test_failed_or_cancelled_request_invalidates_cached_reference(failure: str) -> None:
    """取消、解码失败和收尾失败均清空缓存，下一请求从完整提示开始。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    prompt = np.arange(24, dtype=np.float32).reshape(1, 6, 4)
    prefill.decode([prompt], reference_length=4)
    prefill.finish(True)

    if failure == "decode_failed":
        native.fail_decode = True
        with pytest.raises(RuntimeError, match="模拟原生解码失败"):
            prefill.decode([prompt], reference_length=4)
        prefill.finish(False)
    else:
        prefill.decode([prompt], reference_length=4)
        native.fail_remove = failure == "cleanup_failed"
        if failure == "cleanup_failed":
            with pytest.raises(RuntimeError, match="模拟原生清理失败"):
                prefill.finish(True)
        else:
            prefill.finish(False)
    assert native.sequences == {}

    reset_count = native.reset_count
    prefill.decode([prompt], reference_length=4)
    assert native.reset_count == reset_count + 1
    assert native.decode_calls[-1]["positions"] == [0]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], prompt)
    prefill.finish(True)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_warm_prepares_once_and_first_targets_decode_only_suffixes(backend: str) -> None:
    """提前准备参考且重复准备不重算，首次及后续目标仅提交各自后缀。"""
    native = FakeNative(backend)
    prefill = BackbonePrefill(native)
    reference = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    prefill.warm(reference)
    assert native.events == ["reset", "decode"]
    assert native.decode_calls[-1]["positions"] == [0]
    assert native.decode_calls[-1]["sequence_ids"] == [0]
    assert native.decode_calls[-1]["output"] is False
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], reference)

    prefill.warm(reference.astype(np.float64, order="F"))
    assert native.events == ["reset", "decode"]
    for length, value in ((2, 20), (3, 30)):
        prompt = np.concatenate((reference, np.full((1, length, 4), value, np.float32)), axis=1)
        prefill.decode([prompt], reference_length=4)
        call = native.decode_calls[-1]
        assert call["positions"] == [4]
        assert call["output"] is True
        np.testing.assert_array_equal(call["embeddings"], prompt[:, 4:])
        prefill.finish(True)
    assert native.n_ctx == 16
    assert "reserve" not in native.events


def test_warm_recomputes_mutated_reference_content() -> None:
    """可变参考内容改变后，预准备必须替换旧缓存。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    reference = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    prefill.warm(reference)
    reference[0, 1, 2] += 10
    prefill.warm(reference)
    assert native.events == ["reset", "decode", "reset", "decode"]
    np.testing.assert_array_equal(native.sequences[0], reference)
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], reference)


def test_warm_cuda_recomputes_odd_reference_tail_with_target() -> None:
    """奇数参考末词不进入驻留前缀，改变末词后仍在目标预填充中重新计算。"""
    native = FakeNative("cuda")
    prefill = BackbonePrefill(native)
    reference = np.arange(20, dtype=np.float32).reshape(1, 5, 4)
    prefill.warm(reference)
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], reference[:, :4])
    reference[:, -1] += 100
    prefill.warm(reference)
    assert len(native.decode_calls) == 1

    prompt = np.concatenate((reference, np.full((1, 2, 4), 30, np.float32)), axis=1)
    output = prefill.decode([prompt], reference_length=5)
    assert native.decode_calls[-1]["positions"] == [4]
    np.testing.assert_array_equal(native.decode_calls[-1]["embeddings"], prompt[:, 4:])
    np.testing.assert_array_equal(output[0]["hidden"], prompt.sum(axis=1))
    prefill.finish(True)


@pytest.mark.parametrize(("backend", "length"), [("cpu", 0), ("cuda", 0), ("cuda", 1)])
def test_warm_without_aligned_reference_clears_existing_cache(backend: str, length: int) -> None:
    """参考对齐后为空时清空旧状态，后续目标完整预填充。"""
    native = FakeNative(backend)
    prefill = BackbonePrefill(native)
    prefill.warm(np.ones((1, 4, 4), dtype=np.float32))
    decode_count = len(native.decode_calls)
    reference = np.zeros((1, length, 4), dtype=np.float32)
    prefill.warm(reference)
    assert native.sequences == {}
    assert len(native.decode_calls) == decode_count
    prompt = np.concatenate((reference, np.ones((1, 2, 4), dtype=np.float32)), axis=1)
    prefill.decode([prompt], reference_length=length)
    assert native.decode_calls[-1]["positions"] == [0]
    prefill.finish(True)


def test_warm_failure_clears_partial_state_and_allows_retry() -> None:
    """预准备失败后重置状态和内容记录，相同参考重试会重新计算。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    reference = np.ones((1, 4, 4), dtype=np.float32)
    native.fail_decode = True
    with pytest.raises(RuntimeError, match="模拟原生解码失败"):
        prefill.warm(reference)
    assert native.events == ["reset", "decode", "reset"]
    assert native.sequences == {}

    prefill.warm(reference)
    assert len(native.decode_calls) == 2
    np.testing.assert_array_equal(native.sequences[0], reference)


def test_warm_rejects_active_target_request() -> None:
    """目标请求尚未收尾时拒绝预准备，保留活动请求的原生状态。"""
    native = FakeNative()
    prefill = BackbonePrefill(native)
    reference = np.ones((1, 4, 4), dtype=np.float32)
    prompt = np.concatenate((reference, np.zeros((1, 2, 4), dtype=np.float32)), axis=1)
    prefill.decode([prompt], reference_length=4)
    events = native.events.copy()
    with pytest.raises(RuntimeError, match="上一请求尚未完成"):
        prefill.warm(reference)
    assert native.events == events
    np.testing.assert_array_equal(native.sequences[0], prompt)
    prefill.finish(False)
    prefill.warm(reference)
    assert native.decode_calls[-1]["output"] is False
