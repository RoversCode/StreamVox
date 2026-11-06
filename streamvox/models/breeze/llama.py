"""使用原生 Llama 和 Qwen3 图执行 Breeze 的独立分支解码。"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from threading import RLock
from typing import Sequence

import numpy as np

from ..llama_native import GGML_TYPE_F32, NativeBatch, native_library


class LlamaDecoder:
    """保留独立序列缓存、原始浮点权重和 FP32 缓存的原生解码器。

    Args:
        model_path: 已转换且经过校验的 GGUF 模型。
        backend: 明确选择的 cpu、cuda 或 vulkan 后端。
        device_id: 指定后端中的设备索引。
        n_ctx: 每条序列允许的缓存长度。
        n_seq_max: 同一上下文的独立序列数上限。
        embeddings: 是否同时返回最终归一化后的隐藏状态。
    """

    def __init__(
        self, model_path: Path, *, backend: str, device_id: int = 0,
        n_ctx: int, n_seq_max: int = 2, embeddings: bool = True,
    ) -> None:
        if backend not in {"cpu", "cuda", "vulkan"}:
            raise ValueError(f"不支持的 llama.cpp 后端：{backend}")
        if not isinstance(n_seq_max, int) or isinstance(n_seq_max, bool) or n_seq_max <= 0:
            raise ValueError("n_seq_max 必须为正整数。")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ValueError("n_ctx 必须为正整数。")
        model_path = Path(model_path).resolve(strict=True)
        self._lock = RLock()
        self._api = native_library()
        self._model = self._context = None
        self._batches: dict[bool, tuple[NativeBatch, int]] = {}
        self.backend, self.device_id = backend, device_id
        self.n_seq_max, self.embeddings, self.n_ctx = n_seq_max, embeddings, 0
        self._device_list = (ct.c_void_p * 2)(self._api.device(backend, device_id), None)
        parameters = self._api.llama_model_default_params()
        # 显式设备列表避免同进程其他模型已注册的后端影响本模型的选择。
        parameters.devices = self._device_list
        parameters.n_gpu_layers = 0 if backend == "cpu" else -1
        parameters.split_mode, parameters.main_gpu = 0, 0
        self._model = self._api.llama_model_load_from_file(str(model_path).encode("utf-8"), parameters)
        if not self._model:
            raise RuntimeError(f"llama.cpp 无法加载模型：{model_path}")
        try:
            self.hidden_size = self._api.llama_model_n_embd_out(self._model)
            self.input_size = self._api.llama_model_n_embd_inp(self._model)
            vocab = self._api.llama_model_get_vocab(self._model)
            self.vocab_size = self._api.llama_vocab_n_tokens(vocab)
            self.reserve(n_ctx)
        except BaseException:
            self.close()
            raise

    def reserve(self, n_ctx: int) -> None:
        """扩容每序列缓存并丢弃旧状态，同时保留模型权重。

        Args:
            n_ctx: 请求的每序列容量；小于当前容量时不重建。
        """
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ValueError("n_ctx 必须为正整数。")
        if n_ctx * self.n_seq_max > np.iinfo(np.int32).max:
            raise ValueError("总缓存容量超过原生接口上限。")
        with self._lock:
            if not self._model:
                raise RuntimeError("llama.cpp 解码器已经关闭。")
            if self._context and n_ctx <= self.n_ctx:
                return
            if self._context:
                self._api.llama_free(self._context)
                self._context = None
            parameters = self._api.llama_context_default_params()
            parameters.n_ctx, parameters.n_seq_max = n_ctx * self.n_seq_max, self.n_seq_max
            self._batch_limit = min(512, parameters.n_ctx)
            parameters.n_batch = self._batch_limit
            # b8683 的 CUDA 四到十六列矩阵核直接执行 TF32，绕过 cuBLAS 精度标志。
            # 两词物理微批次保留完整逻辑批次和分支缓存，避开该 TF32 路径。
            parameters.n_ubatch = min(self._batch_limit, 2) if self.backend == "cuda" else self._batch_limit
            parameters.type_k = parameters.type_v = GGML_TYPE_F32
            # 固定版本的融合注意力会把 FP32 键和值转成 FP16，不能用于无降精度基准。
            parameters.flash_attn_type = 0
            parameters.pooling_type, parameters.attention_type = 0, 0
            parameters.embeddings = self.embeddings
            parameters.offload_kqv = parameters.op_offload = self.backend != "cpu"
            parameters.kv_unified = True
            # 频率表与模型元数据共同定义位置编码；上下文扩容不启用额外外推。
            parameters.yarn_ext_factor = 0.0
            self._context = self._api.llama_init_from_model(self._model, parameters)
            if not self._context:
                raise RuntimeError(f"llama.cpp 无法创建每序列容量为 {n_ctx} 的上下文。")
            self.n_ctx, self._extract_hidden = n_ctx, self.embeddings

    def decode(
        self, *, embeddings: np.ndarray | None = None, tokens: np.ndarray | None = None,
        positions: Sequence[int], sequence_ids: Sequence[int], output: bool = True,
        logit_start: int = 0, logit_count: int | None = None,
    ) -> dict[str, np.ndarray | None]:
        """推进独立分支，并复制各分支最后位置的隐藏状态与所需分数。

        Args:
            embeddings: 批次、长度、隐藏维度的浮点输入。
            tokens: 批次、长度的整数输入，与嵌入输入二选一。
            positions: 各分支本次输入的起始绝对位置。
            sequence_ids: 各分支在缓存中的独立序列编号。
            output: 是否提取最后位置的结果。
            logit_start: 所需输出词表区间的起点。
            logit_count: 区间长度；为空时取到词表末尾。
        Returns:
            拥有独立存储的 hidden 和 logits；未请求的结果为 None。
        """
        if (embeddings is None) == (tokens is None):
            raise ValueError("embeddings 和 tokens 必须且只能提供一个。")
        embedded = embeddings is not None
        values = np.asarray(embeddings if embedded else tokens)
        expected_dimensions = 3 if embedded else 2
        if values.ndim != expected_dimensions or 0 in values.shape[:2]:
            raise ValueError("解码输入必须具有非空的批次和序列维度。")
        batch_size, length = values.shape[:2]
        if len(positions) != batch_size or len(sequence_ids) != batch_size:
            raise ValueError("positions、sequence_ids 必须分别对应每个分支。")
        if batch_size > self.n_seq_max or len(set(sequence_ids)) != batch_size:
            raise ValueError("同批分支必须使用不同序列编号，且不能超过 n_seq_max。")
        for position, sequence in zip(positions, sequence_ids):
            if not isinstance(position, (int, np.integer)) or not 0 <= position <= self.n_ctx - length:
                raise ValueError("本次输入位置超出每序列缓存容量。")
            if not isinstance(sequence, (int, np.integer)) or not 0 <= sequence < self.n_seq_max:
                raise ValueError("序列编号超出 n_seq_max。")
        if embedded:
            if values.shape[2] != self.input_size:
                raise ValueError(f"嵌入维度必须为 {self.input_size}。")
            values = np.ascontiguousarray(values, dtype=np.float32)
        else:
            if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= self.vocab_size):
                raise ValueError("token 编号必须为词表范围内的整数。")
            values = np.ascontiguousarray(values, dtype=np.int32)
        count = self.vocab_size - logit_start if logit_count is None else logit_count
        if not isinstance(logit_start, int) or not isinstance(count, int) or min(logit_start, count) < 0 or logit_start + count > self.vocab_size:
            raise ValueError("logits 切片超出词表范围。")
        with self._lock:
            if not self._context:
                raise RuntimeError("llama.cpp 解码器已经关闭或上下文创建失败。")
            # 长提示分块提交，不按模型上下文上限预留巨大的输入和计算图。
            chunk_length = max(1, self._batch_limit // batch_size)
            for start in range(0, length, chunk_length):
                stop = min(length, start + chunk_length)
                chunk = np.ascontiguousarray(values[:, start:stop])
                extract = output and stop == length
                hidden_output = self.embeddings and extract
                if hidden_output != self._extract_hidden:
                    self._api.llama_set_embeddings(self._context, hidden_output)
                    self._extract_hidden = hidden_output
                native = self._batch(embedded, batch_size * (stop - start))
                pointer = native.embd if embedded else native.token
                ct.memmove(pointer, chunk.ctypes.data, chunk.nbytes)
                native.n_tokens = batch_size * (stop - start)
                for branch, (position, sequence) in enumerate(zip(positions, sequence_ids)):
                    for offset in range(stop - start):
                        index = branch * (stop - start) + offset
                        native.pos[index] = position + start + offset
                        native.n_seq_id[index], native.seq_id[index][0] = 1, sequence
                        native.logits[index] = int(hidden_output or (extract and offset == stop - start - 1))
                status = self._api.llama_decode(self._context, native)
                if status != 0:
                    raise RuntimeError(f"llama.cpp 解码失败，状态码为 {status}。")
            if not output:
                return {"hidden": None, "logits": None}
            hidden_rows, logit_rows = [], []
            final_length = stop - start
            for branch in range(batch_size):
                index = (branch + 1) * final_length - 1
                pointer = self._api.llama_get_logits_ith(self._context, index)
                if not pointer:
                    raise RuntimeError("llama.cpp 未返回请求的 logits。")
                # 原生库已同步结果；只复制请求区间，不复制整个联合码本词表。
                logit_rows.append(np.ctypeslib.as_array(pointer, shape=(self.vocab_size,))[logit_start:logit_start + count].copy())
                if self.embeddings:
                    pointer = self._api.llama_get_embeddings_ith(self._context, index)
                    if not pointer:
                        raise RuntimeError("llama.cpp 未返回请求的隐藏状态。")
                    hidden_rows.append(np.ctypeslib.as_array(pointer, shape=(self.hidden_size,)).copy())
            return {"hidden": np.stack(hidden_rows) if hidden_rows else None, "logits": np.stack(logit_rows)}

    def _batch(self, embedded: bool, count: int) -> NativeBatch:
        """复用两种输入模式的原生批次，并仅在容量不足时增长。

        Args:
            embedded: 是否分配浮点嵌入输入。
            count: 本次需要的词数。
        Returns:
            满足容量要求的原生批次。
        """
        previous = self._batches.get(embedded)
        if previous and previous[1] >= count:
            return previous[0]
        if previous:
            self._api.llama_batch_free(previous[0])
            del self._batches[embedded]
        capacity = min(self._batch_limit, max(count, 32, previous[1] * 2 if previous else 0))
        native = self._api.llama_batch_init(capacity, self.input_size if embedded else 0, 1)
        if not native.pos or not (native.embd if embedded else native.token):
            self._api.llama_batch_free(native)
            raise MemoryError("无法分配 llama.cpp 输入批次。")
        self._batches[embedded] = native, capacity
        return native

    def remove_sequence(self, sequence: int, start: int, end: int = -1) -> None:
        """删除指定分支在半开位置区间内的缓存，保留区间外的状态。

        Args:
            sequence: 需要截断的独立序列编号。
            start: 删除区间的起始绝对位置。
            end: 删除区间的结束位置；负一表示删除到序列末尾。
        Returns:
            原生缓存删除成功后返回。
        """
        with self._lock:
            if not self._context:
                raise RuntimeError("llama.cpp 解码器已经关闭或上下文创建失败。")
            if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)) or not 0 <= sequence < self.n_seq_max:
                raise ValueError("序列编号超出 n_seq_max。")
            if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= start <= self.n_ctx:
                raise ValueError("缓存区间起点必须位于 0..n_ctx。")
            if isinstance(end, bool) or not isinstance(end, (int, np.integer)) or (end != -1 and not start <= end <= self.n_ctx):
                raise ValueError("缓存区间终点必须为 -1，或位于起点与 n_ctx 之间。")
            memory = self._api.llama_get_memory(self._context)
            if not self._api.llama_memory_seq_rm(memory, int(sequence), int(start), int(end)):
                raise RuntimeError(f"llama.cpp 无法删除序列 {sequence} 的缓存区间 [{start}, {end})。")

    def reset(self) -> None:
        """清除全部分支的缓存，让下一帧或请求从独立状态开始。"""
        with self._lock:
            if not self._context:
                raise RuntimeError("llama.cpp 解码器已经关闭。")
            self._api.llama_memory_clear(self._api.llama_get_memory(self._context), True)

    def close(self) -> None:
        """按上下文、批次、模型的顺序释放资源，允许重复调用。"""
        with self._lock:
            if self._context:
                self._api.llama_free(self._context)
                self._context = None
            for batch, _capacity in self._batches.values():
                self._api.llama_batch_free(batch)
            self._batches.clear()
            if self._model:
                self._api.llama_model_free(self._model)
                self._model = None
