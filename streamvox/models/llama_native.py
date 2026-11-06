"""加载发行包内的 llama.cpp b8683，并声明解码所需的最小原生接口。"""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
import sys
from threading import RLock

from streamvox_runtime import initialize_nvidia_runtime

from ..utils.log import logger


LLAMA_TAG = "b8683"
LLAMA_COMMIT = "d0a6dfeb28a09831d904fc4d910ddb740da82834"
GGML_TYPE_F32 = 0
FloatPointer = ct.POINTER(ct.c_float)
IntPointer = ct.POINTER(ct.c_int32)
LogCallback = ct.CFUNCTYPE(None, ct.c_int, ct.c_char_p, ct.c_void_p)


class ModelParams(ct.Structure):
    """与固定版本头文件逐字段一致的模型加载参数。"""

    _fields_ = [
        ("devices", ct.POINTER(ct.c_void_p)),
        ("tensor_buft_overrides", ct.c_void_p),
        ("n_gpu_layers", ct.c_int32),
        ("split_mode", ct.c_int32),
        ("main_gpu", ct.c_int32),
        ("tensor_split", FloatPointer),
        ("progress_callback", ct.c_void_p),
        ("progress_callback_user_data", ct.c_void_p),
        ("kv_overrides", ct.c_void_p),
        ("vocab_only", ct.c_bool),
        ("use_mmap", ct.c_bool),
        ("use_direct_io", ct.c_bool),
        ("use_mlock", ct.c_bool),
        ("check_tensors", ct.c_bool),
        ("use_extra_bufts", ct.c_bool),
        ("no_host", ct.c_bool),
        ("no_alloc", ct.c_bool),
    ]


class ContextParams(ct.Structure):
    """与固定版本头文件逐字段一致的上下文参数。"""

    _fields_ = [
        ("n_ctx", ct.c_uint32), ("n_batch", ct.c_uint32),
        ("n_ubatch", ct.c_uint32), ("n_seq_max", ct.c_uint32),
        ("n_threads", ct.c_int32), ("n_threads_batch", ct.c_int32),
        ("rope_scaling_type", ct.c_int), ("pooling_type", ct.c_int),
        ("attention_type", ct.c_int), ("flash_attn_type", ct.c_int),
        ("rope_freq_base", ct.c_float), ("rope_freq_scale", ct.c_float),
        ("yarn_ext_factor", ct.c_float), ("yarn_attn_factor", ct.c_float),
        ("yarn_beta_fast", ct.c_float), ("yarn_beta_slow", ct.c_float),
        ("yarn_orig_ctx", ct.c_uint32), ("defrag_thold", ct.c_float),
        ("cb_eval", ct.c_void_p), ("cb_eval_user_data", ct.c_void_p),
        ("type_k", ct.c_int), ("type_v", ct.c_int),
        ("abort_callback", ct.c_void_p), ("abort_callback_data", ct.c_void_p),
        ("embeddings", ct.c_bool), ("offload_kqv", ct.c_bool),
        ("no_perf", ct.c_bool), ("op_offload", ct.c_bool),
        ("swa_full", ct.c_bool), ("kv_unified", ct.c_bool),
        ("samplers", ct.c_void_p), ("n_samplers", ct.c_size_t),
    ]


class NativeBatch(ct.Structure):
    """由原生库分配并释放的批次内存布局。"""

    _fields_ = [
        ("n_tokens", ct.c_int32), ("token", IntPointer),
        ("embd", FloatPointer), ("pos", IntPointer),
        ("n_seq_id", IntPointer), ("seq_id", ct.POINTER(IntPointer)),
        ("logits", ct.POINTER(ct.c_int8)),
    ]


class NativeLibrary:
    """共享原生库句柄，并只注册当前请求需要的计算后端。

    Args:
        directory: 同一发行构建的动态库目录。
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve(strict=True)
        self._lock = RLock()
        self._handles: list[object] = []
        manifest_path = self.directory / "native-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("llama_tag"), manifest.get("llama_commit")) != (LLAMA_TAG, LLAMA_COMMIT):
                raise RuntimeError("llama.cpp 原生资产与 b8683 接口版本不一致。")
        if os.name == "nt":
            self._handles.append(os.add_dll_directory(str(self.directory)))
        prefix, suffix = ("", ".dll") if os.name == "nt" else ("lib", ".dylib" if sys.platform == "darwin" else ".so")
        self._prefix, self._suffix = prefix, suffix
        mode = getattr(ct, "RTLD_GLOBAL", 0)
        self.base = ct.CDLL(str(self.directory / f"{prefix}ggml-base{suffix}"), mode=mode)
        self.ggml = ct.CDLL(str(self.directory / f"{prefix}ggml{suffix}"), mode=mode)
        self.llama = ct.CDLL(str(self.directory / f"{prefix}llama{suffix}"), mode=mode)
        self._bind()
        self._log_callback = LogCallback(self._log)
        self.llama_log_set(self._log_callback, None)
        self._load_backend("cpu")
        self.llama_backend_init()
        logger.info("[llama.cpp] 原生接口=%s 源码=%s", LLAMA_TAG, LLAMA_COMMIT)

    def _bind(self) -> None:
        """声明固定版本的函数签名，避免指针和按值结构体被截断。"""
        declarations = {
            "llama_log_set": (None, [LogCallback, ct.c_void_p]),
            "llama_backend_init": (None, []),
            "llama_model_default_params": (ModelParams, []),
            "llama_model_load_from_file": (ct.c_void_p, [ct.c_char_p, ModelParams]),
            "llama_model_free": (None, [ct.c_void_p]),
            "llama_model_n_embd": (ct.c_int32, [ct.c_void_p]),
            "llama_model_n_embd_inp": (ct.c_int32, [ct.c_void_p]),
            "llama_model_n_embd_out": (ct.c_int32, [ct.c_void_p]),
            "llama_model_get_vocab": (ct.c_void_p, [ct.c_void_p]),
            "llama_vocab_n_tokens": (ct.c_int32, [ct.c_void_p]),
            "llama_context_default_params": (ContextParams, []),
            "llama_init_from_model": (ct.c_void_p, [ct.c_void_p, ContextParams]),
            "llama_free": (None, [ct.c_void_p]),
            "llama_batch_init": (NativeBatch, [ct.c_int32, ct.c_int32, ct.c_int32]),
            "llama_batch_free": (None, [NativeBatch]),
            "llama_decode": (ct.c_int32, [ct.c_void_p, NativeBatch]),
            "llama_set_embeddings": (None, [ct.c_void_p, ct.c_bool]),
            "llama_get_logits_ith": (FloatPointer, [ct.c_void_p, ct.c_int32]),
            "llama_get_embeddings_ith": (FloatPointer, [ct.c_void_p, ct.c_int32]),
            "llama_get_memory": (ct.c_void_p, [ct.c_void_p]),
            "llama_memory_clear": (None, [ct.c_void_p, ct.c_bool]),
            "llama_memory_seq_rm": (ct.c_bool, [ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_int32]),
        }
        for name, (result, arguments) in declarations.items():
            function = getattr(self.llama, name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name, function)
        for name, result, arguments in (
            ("ggml_backend_load", ct.c_void_p, [ct.c_char_p]),
            ("ggml_backend_reg_by_name", ct.c_void_p, [ct.c_char_p]),
            ("ggml_backend_reg_dev_count", ct.c_size_t, [ct.c_void_p]),
            ("ggml_backend_reg_dev_get", ct.c_void_p, [ct.c_void_p, ct.c_size_t]),
            ("ggml_backend_dev_name", ct.c_char_p, [ct.c_void_p]),
        ):
            owner = self.ggml if name in {"ggml_backend_load", "ggml_backend_reg_by_name"} else self.base
            function = getattr(owner, name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name, function)

    @staticmethod
    def _log(level: int, message: bytes | None, _user_data: int) -> None:
        """将原生日志接入现有日志系统，常规加载详情仅在调试级别输出。

        Args:
            level: 原生日志级别。
            message: 原生传来的文本。
            _user_data: 未使用的原生回调上下文。
        """
        if not message:
            return
        text = message.decode("utf-8", errors="replace").rstrip()
        if not text:
            return
        if level == 4:
            logger.error("[llama.cpp] %s", text)
        elif level == 3:
            logger.warning("[llama.cpp] %s", text)
        else:
            logger.debug("[llama.cpp] %s", text)

    def _load_backend(self, backend: str) -> int:
        """注册指定后端；缺少库或初始化失败时直接报错。

        Args:
            backend: 已校验的后端名称。
        Returns:
            后端注册句柄。
        """
        existing = self.ggml_backend_reg_by_name(backend.encode())
        if existing:
            return existing
        stem = f"{self._prefix}ggml-{backend}"
        path = self.directory / f"{stem}{self._suffix}"
        if backend == "cpu":
            # 按官方评分选择本机支持的指令集；不能按文件名字母序猜测兼容性。
            best_score = 0
            for candidate in sorted(self.directory.glob(f"{stem}-*{self._suffix}")):
                try:
                    handle = ct.CDLL(str(candidate))
                    score = handle.ggml_backend_score
                    score.argtypes, score.restype = [], ct.c_int
                    candidate_score = score()
                except (AttributeError, OSError):
                    continue
                self._handles.append(handle)
                if candidate_score > best_score:
                    best_score, path = candidate_score, candidate
        if not path.is_file():
            raise RuntimeError(f"发行包缺少 llama.cpp {backend} 后端：{path}")
        registration = self.ggml_backend_load(str(path).encode("utf-8"))
        if not registration:
            raise RuntimeError(f"llama.cpp {backend} 后端加载失败：{path}")
        logger.info("[llama.cpp] 已加载后端=%s 库=%s", backend, path.name)
        return registration

    def device(self, backend: str, device_id: int) -> int:
        """只从指定后端枚举设备，禁止借用其他已注册后端继续运行。

        Args:
            backend: cpu、cuda 或 vulkan。
            device_id: 指定后端内的设备索引。
        Returns:
            经原生库验证存在的设备句柄。
        """
        if backend not in {"cpu", "cuda", "vulkan"}:
            raise ValueError(f"不支持的 llama.cpp 后端：{backend}")
        if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
            raise ValueError("llama.cpp device_id 必须为非负整数。")
        with self._lock:
            if backend == "cuda":
                initialize_nvidia_runtime(consumer="llama", required=True)
            registration = self._load_backend(backend)
            count = self.ggml_backend_reg_dev_count(registration)
            if device_id >= count:
                raise RuntimeError(f"llama.cpp {backend} 没有设备 {device_id}，可用设备数为 {count}。")
            device = self.ggml_backend_reg_dev_get(registration, device_id)
            if not device:
                raise RuntimeError(f"llama.cpp {backend} 设备 {device_id} 初始化失败。")
            name = self.ggml_backend_dev_name(device).decode("utf-8", errors="replace")
            logger.info("[llama.cpp] 实际后端=%s 设备=%s 索引=%s", backend, name, device_id)
            return device


_library: NativeLibrary | None = None
_library_lock = RLock()


def native_library() -> NativeLibrary:
    """按进程复用发行包中的原生库和后端注册状态。

    Returns:
        已初始化且始终保留原生回调引用的共享库对象。
    """
    global _library
    with _library_lock:
        if _library is None:
            _library = NativeLibrary(Path(__file__).resolve().parents[1] / "bin")
        return _library
