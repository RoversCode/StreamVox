import sys
import os
import ctypes
import codecs
from streamvox_runtime import initialize_nvidia_runtime
import struct
import time
import atexit
from collections import deque, Counter
from pathlib import Path

import numpy as np
try:
    import gguf
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
except ImportError:
    export_dir = Path(__file__).resolve().parents[1] / "export"
    if str(export_dir) not in sys.path:
        sys.path.insert(0, str(export_dir))
    import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from typing import List, Union, Set, Optional
from os.path import relpath
from . import logger

# =========================================================================
# Configuration
# =========================================================================
# QUIET_LOGS = True 时，不打印任何日志。但现在我们路由到 logger。
QUIET_LOGS = True
_log_callback_ref = None
# 动态后端加载状态只属于当前进程；value 保存 ggml_backend_load 返回的注册句柄，shutdown 时需要显式卸载。
_loaded_dynamic_backend_regs: dict[str, int] = {}
_dll_directory_handles = []
# Vulkan 动态后端一旦被当前进程使用，退出阶段需要走专门保护，避免 native 全局析构器输出段错误。
_vulkan_backend_used = False
# 退出保护只能注册一次，避免多个 S2-Pro runtime 反复创建时堆叠 atexit 回调。
_vulkan_exit_guard_armed = False


def _model_path_for_native_load(model_path: Path, lib_dir: Path) -> Path:
    try:
        return Path(relpath(model_path, lib_dir))
    except ValueError:
        return model_path


def _main_gpu_for_native_load(n_gpu_layers: int, main_gpu: int) -> int:
    return -1 if n_gpu_layers == 0 else main_gpu



# =========================================================================
# Type Definitions
# =========================================================================

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class llama_model_kv_override_value(ctypes.Union):
    _fields_ = [
        ("val_i64", ctypes.c_int64),
        ("val_f64", ctypes.c_double),
        ("val_bool", ctypes.c_bool),
        ("val_str", ctypes.c_char * 128),
    ]


class llama_model_kv_override(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("tag", ctypes.c_int32),
        ("key", ctypes.c_char * 128),
        ("value", llama_model_kv_override_value),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

class llama_sampler_chain_params(ctypes.Structure):
    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]

class llama_logit_bias(ctypes.Structure):
    _fields_ = [
        ("token", llama_token),
        ("bias", ctypes.c_float),
    ]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# =========================================================================
# Llama Library Bindings
# =========================================================================

# Global library references
llama = None
ggml = None
ggml_base = None

# Global function pointers
llama_log_set = None
llama_backend_init = None
llama_backend_free = None
llama_model_default_params = None
llama_model_load_from_file = None
llama_model_free = None
llama_model_get_vocab = None
llama_model_rope_type = None
llama_context_default_params = None
llama_init_from_model = None
llama_free = None
llama_batch_init = None
llama_batch_free = None
llama_decode = None
llama_get_logits = None
llama_get_logits_ith = None
llama_get_embeddings = None
llama_get_embeddings_ith = None
llama_tokenize = None
llama_vocab_n_tokens = None
llama_vocab_eos = None
llama_token_to_piece = None
llama_get_memory = None
llama_memory_clear = None
llama_state_get_size = None
llama_state_get_data = None
llama_state_set_data = None
llama_model_n_embd = None

# Sampler
llama_sampler_chain_default_params = None
llama_sampler_chain_init = None
llama_sampler_chain_add = None
llama_sampler_init_greedy = None
llama_sampler_init_dist = None
llama_sampler_init_temp = None
llama_sampler_init_top_k = None
llama_sampler_init_top_p = None
llama_sampler_sample = None
llama_sampler_free = None
llama_sampler_init_min_p = None
llama_sampler_init_penalties = None
llama_sampler_accept = None

def _requires_dynamic_backend_scan(requested_backend: str | None) -> bool:
    """
    判断本次模型加载是否需要扫描 llama.cpp 动态后端。

    参数:
        requested_backend: runtime 已经解析出的 GGUF 后端名称；常见值为 cpu、cuda、vulkan、metal 或 None。

    返回:
        True 表示必须调用 ggml_backend_load_all 注册动态后端；False 表示仅加载基础库即可。

    边界异常:
        未知 backend 会按不扫描处理，由上层 runtime 的设备解析负责保证传入值合法。
    """
    # 环境变量保留为人工诊断开关，便于在特殊二进制包上强制扫描全部 backend。
    if os.environ.get("S2_PRO_LOAD_ALL_BACKENDS", "0") == "1":
        return True

    # CPU 不需要动态 GPU backend；GPU backend 通常以 libggml-vulkan/libggml-cuda 等插件形式存在，必须扫描注册。
    return requested_backend in {"cuda", "vulkan", "metal"}


def _first_existing_backend(lib_dir: Path, filenames: list[str]) -> Path | None:
    """
    从候选文件名中选择第一个存在的动态后端库。

    参数:
        lib_dir: StreamVox 打包 llama.cpp 动态库的目录。
        filenames: 按优先级排列的候选文件名。

    返回:
        返回第一个存在的 backend 动态库路径；找不到时返回 None。

    边界异常:
        本函数只做文件存在性判断，不校验 ABI；ABI 问题会在实际加载时暴露。
    """
    # 按优先级选择明确存在的文件，避免用通配误加载不属于当前平台的 backend。
    for filename in filenames:
        candidate = lib_dir / filename
        if candidate.exists():
            return candidate
    return None


def _select_cpu_backend_path(lib_dir: Path) -> Path | None:
    """
    选择当前包内可用的 llama.cpp CPU 动态后端。

    参数:
        lib_dir: StreamVox 打包 llama.cpp 动态库的目录。

    返回:
        返回一个可加载的 CPU backend 路径；如果当前 libggml 已静态包含 CPU backend，则可能返回 None。

    边界异常:
        多个 CPU 架构变体同时存在时优先选择通用 x64 变体，避免误选当前机器不支持的指令集。
    """
    if sys.platform == "win32":
        preferred = _first_existing_backend(
            lib_dir,
            [
                "ggml-cpu.dll",
                "ggml-cpu-x64.dll",
                "ggml-cpu-sse42.dll",
            ],
        )
        if preferred is not None:
            return preferred
        cpu_variants = sorted(lib_dir.glob("ggml-cpu-*.dll"))
        return cpu_variants[0] if cpu_variants else None
    if sys.platform == "darwin":
        return _first_existing_backend(lib_dir, ["libggml-cpu.dylib"])

    # 互联网发布包常把 CPU 后端拆成多个 x86 变体；x64 是通用兜底，优先级高于具体微架构优化包。
    preferred = _first_existing_backend(
        lib_dir,
        [
            "libggml-cpu.so",
            "libggml-cpu-x64.so",
            "libggml-cpu-sse42.so",
        ],
    )
    if preferred is not None:
        return preferred

    # 如果发布包没有通用文件名，就取排序后的第一个 CPU backend，保证 CPU 模式至少有机会注册设备。
    cpu_variants = sorted(lib_dir.glob("libggml-cpu-*.so"))
    return cpu_variants[0] if cpu_variants else None


def _select_gpu_backend_path(lib_dir: Path, requested_backend: str | None) -> Path | None:
    """
    选择 runtime 已解析出的目标 GPU 动态后端。

    参数:
        lib_dir: StreamVox 打包 llama.cpp 动态库的目录。
        requested_backend: runtime 已确认的 GGUF 后端名称。

    返回:
        返回目标 GPU backend 动态库路径；CPU 或未找到时返回 None。

    边界异常:
        未知 backend 返回 None，让后续模型加载自然失败并交给 runtime fallback。
    """
    if requested_backend == "cuda":
        return _first_existing_backend(lib_dir, ["libggml-cuda.so", "ggml-cuda.dll", "libggml-cuda.dylib"])
    if requested_backend == "vulkan":
        return _first_existing_backend(lib_dir, ["libggml-vulkan.so", "ggml-vulkan.dll", "libggml-vulkan.dylib"])
    if requested_backend == "metal":
        return _first_existing_backend(lib_dir, ["libggml-metal.dylib"])
    return None


def _backend_name_from_path(path: Path) -> str | None:
    """
    从 backend 动态库文件名推断 llama.cpp registry 中的后端名称。

    参数:
        path: backend 动态库路径。

    返回:
        返回 cpu、cuda、vulkan、metal 等 registry 名称；无法识别时返回 None。

    边界异常:
        本函数只做文件名模式判断，不访问文件系统。
    """
    name = path.name.lower()
    if "cpu" in name:
        return "cpu"
    if "cuda" in name:
        return "cuda"
    if "vulkan" in name:
        return "vulkan"
    if "metal" in name:
        return "metal"
    return None


def _backend_registered(backend_name: str | None) -> bool:
    """
    判断 llama.cpp 后端是否已经注册。

    参数:
        backend_name: registry 后端名称，例如 cpu、cuda、vulkan。

    返回:
        True 表示当前进程已经能通过 ggml registry 找到该后端。

    边界异常:
        当前 libggml 不导出查询 API 或 backend_name 为空时返回 False。
    """
    if not backend_name:
        return False
    try:
        ggml_backend_reg_by_name = ggml.ggml_backend_reg_by_name
    except AttributeError:
        return False
    ggml_backend_reg_by_name.argtypes = [ctypes.c_char_p]
    ggml_backend_reg_by_name.restype = ctypes.c_void_p
    return bool(ggml_backend_reg_by_name(backend_name.encode("utf-8")))


def _load_backend_file(path: Path) -> bool:
    """
    通过 ggml_backend_load 精准注册一个动态 backend 文件。

    参数:
        path: 需要加载的 backend 动态库绝对路径。

    返回:
        True 表示本次或之前已经成功注册；False 表示加载失败。

    边界异常:
        ctypes 调用失败或 llama.cpp 返回空注册指针时记录警告，不直接中断，交给模型加载阶段决定是否 fallback。
    """
    normalized = str(path.resolve())
    backend_name = _backend_name_from_path(path)
    # CUDA 这类编译产物可能已由 libllama/libggml 直接注册，文件存在但无需再通过 ggml_backend_load 加载。
    if _backend_registered(backend_name):
        logger.debug("[s2-pro] llama.cpp backend 已注册，无需动态加载: %s", backend_name)
        return True
    # 同一进程内重复加载同一个 backend 没有收益，且可能让底层注册表产生重复项。
    if normalized in _loaded_dynamic_backend_regs:
        return True

    try:
        ggml_backend_load = ggml.ggml_backend_load
    except AttributeError:
        logger.warning("[s2-pro] 当前 libggml 不导出 ggml_backend_load，无法精准加载动态 backend: %s", normalized)
        return False

    ggml_backend_load.argtypes = [ctypes.c_char_p]
    ggml_backend_load.restype = ctypes.c_void_p
    reg = ggml_backend_load(normalized.encode("utf-8"))
    if not reg:
        # 部分 llama.cpp 构建会在 load 调用过程中完成注册但返回空指针；注册成功就不应误报 warning。
        if _backend_registered(backend_name):
            logger.debug("[s2-pro] llama.cpp backend 已由当前二进制注册: %s", backend_name)
            return True
        logger.warning("[s2-pro] llama.cpp 动态 backend 加载失败: %s", normalized)
        return False

    _loaded_dynamic_backend_regs[normalized] = int(reg)
    logger.info("[s2-pro] llama.cpp 动态 backend 已加载: %s", path.name)
    return True


def unload_dynamic_backends() -> None:
    """
    卸载当前进程由 S2-Pro 精准加载过的 llama.cpp 动态 backend。

    参数:
        本函数不接收参数，使用模块内记录的 backend 注册句柄。

    返回:
        无返回值；成功后会清空已加载 backend 记录。

    边界异常:
        卸载阶段异常会被记录并吞掉，避免 shutdown 期间掩盖用户主流程结果。
    """
    # 没有通过精准路径加载过 backend 时无需处理；全量扫描模式下没有逐项句柄，也不在这里卸载。
    if not _loaded_dynamic_backend_regs:
        return

    # Vulkan backend 在 S2-Pro slow/fast 双 GGUF 场景下，即使 context/model 已经断开 native 指针，
    # 退出阶段主动 ggml_backend_unload 仍可能触发驱动或 llama.cpp registry 析构段错误。
    # 默认跳过手动卸载，交给操作系统在进程退出时回收动态库；诊断时可显式打开强制卸载开关复现底层问题。
    has_vulkan_backend = any("vulkan" in Path(path).name for path in _loaded_dynamic_backend_regs)
    force_unload = os.environ.get("S2_PRO_FORCE_UNLOAD_DYNAMIC_BACKENDS", "0") == "1"
    if has_vulkan_backend and not force_unload:
        loaded_names = ", ".join(Path(path).name for path in _loaded_dynamic_backend_regs)
        logger.info("[s2-pro] Vulkan 动态 backend 跳过手动卸载，交由进程退出回收: %s", loaded_names)
        _loaded_dynamic_backend_regs.clear()
        return

    try:
        ggml_backend_unload = ggml.ggml_backend_unload
    except AttributeError:
        logger.warning("[s2-pro] 当前 libggml 不导出 ggml_backend_unload，无法卸载动态 backend")
        return

    ggml_backend_unload.argtypes = [ctypes.c_void_p]
    ggml_backend_unload.restype = None
    # 按加载的逆序卸载，确保 GPU backend 先于 CPU backend 清理，贴近手工验证过的安全释放顺序。
    for path, reg in reversed(list(_loaded_dynamic_backend_regs.items())):
        try:
            ggml_backend_unload(ctypes.c_void_p(reg))
            logger.info("[s2-pro] llama.cpp 动态 backend 已卸载: %s", Path(path).name)
        except Exception as exc:
            logger.warning("[s2-pro] llama.cpp 动态 backend 卸载失败 %s: %s", path, exc)
    _loaded_dynamic_backend_regs.clear()


def _load_dynamic_backends_if_needed(requested_backend: str | None, lib_dir: Path) -> None:
    """
    按 runtime 的后端决策精准加载 llama.cpp 动态后端插件。

    参数:
        requested_backend: runtime 已确认要使用的 GGUF 后端；CPU 会加载 CPU backend，GPU 会加载 CPU + 目标 GPU backend。
        lib_dir: StreamVox 打包 llama.cpp 动态库的目录。

    返回:
        无返回值；成功加载的 backend 会被记录到进程级集合。

    边界异常:
        单个 backend 加载失败时仅记录警告，后续模型加载会自然失败并交给 runtime 回退 CPU。
    """
    global _vulkan_backend_used
    # 记录 Vulkan 使用事实，后续 shutdown 会据此安装进程退出保护；这不改变加载流程本身。
    if requested_backend == "vulkan":
        _vulkan_backend_used = True

    # 人工诊断开关保留原始“扫描全部”能力，但默认路径使用精准加载，规避部分 Vulkan 发布包退出阶段崩溃。
    if os.environ.get("S2_PRO_LOAD_ALL_BACKENDS", "0") == "1":
        try:
            ggml_backend_load_all = ggml.ggml_backend_load_all
        except AttributeError:
            logger.warning("[s2-pro] 当前 libggml 不导出 ggml_backend_load_all，无法扫描全部动态 backend")
            return
        ggml_backend_load_all.argtypes = []
        ggml_backend_load_all.restype = None
        ggml_backend_load_all()
        logger.info("[s2-pro] llama.cpp 动态后端全量扫描完毕，目标 GGUF 后端: %s", requested_backend or "auto")
        return

    # llama.cpp 即使走 GPU，也需要 CPU backend 兜底 unsupported ops 和基础调度；动态 backend 包下 CPU 也必须显式注册。
    cpu_backend = _select_cpu_backend_path(lib_dir)
    if cpu_backend is not None:
        _load_backend_file(cpu_backend)

    # 只有 runtime 判定 GGUF 需要 GPU 时，才额外加载目标 GPU backend，避免把不相关 backend 一起拉进进程。
    if _requires_dynamic_backend_scan(requested_backend):
        gpu_backend = _select_gpu_backend_path(lib_dir, requested_backend)
        if gpu_backend is not None:
            _load_backend_file(gpu_backend)


def _exit_without_native_vulkan_destructors() -> None:
    """
    在解释器退出最后阶段绕过不稳定的 Vulkan native 全局析构器。

    参数:
        无。

    返回:
        无返回值；函数会直接以 0 状态结束当前进程。

    边界异常:
        os._exit 不运行后续 Python atexit 回调，因此只在 S2-Pro Vulkan shutdown 完成后注册。
    """
    # 尽量刷新 Python 层输出，避免用户在退出前打印的信息丢失；随后直接结束进程绕过 C++ 析构崩溃点。
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


def arm_vulkan_exit_guard() -> None:
    """
    为 S2-Pro Vulkan 运行注册进程退出保护。

    参数:
        无。

    返回:
        无返回值；满足条件时注册一次 atexit 回调。

    边界异常:
        仅在显式 shutdown 之后启用，避免影响 SDK 正常推理过程；可通过环境变量关闭用于底层崩溃诊断。
    """
    global _vulkan_exit_guard_armed
    # 非 Vulkan 路径没有这个 native 析构问题，不注册任何进程级退出保护。
    if not _vulkan_backend_used:
        return

    # 诊断 native 崩溃时允许关闭保护，便于保留原始 core dump 和底层退出码。
    if os.environ.get("S2_PRO_DISABLE_VULKAN_EXIT_GUARD", "0") == "1":
        return

    # 多次 shutdown 只需要一个退出保护，避免重复注册导致日志和行为难以推断。
    if _vulkan_exit_guard_armed:
        return

    atexit.register(_exit_without_native_vulkan_destructors)
    _vulkan_exit_guard_armed = True
    logger.info("[s2-pro] Vulkan 进程退出保护已启用，避免 native backend 析构阶段段错误")


def init_llama_lib(requested_backend: str | None = None):
    """
    初始化 llama.cpp 库，并按 runtime 决策注册必要的动态后端。

    参数:
        requested_backend: S2-Pro runtime 已解析出的 GGUF 后端；CPU 不扫描动态 GPU 插件，GPU 后端会扫描。

    返回:
        无返回值；初始化后会绑定当前模块需要的 llama.cpp C API。

    边界异常:
        动态库缺失或 ABI 不匹配时会由 ctypes 抛出异常；动态 backend 扫描失败时记录日志并交给模型加载阶段兜底。
    """
    global llama, ggml, ggml_base
    global llama_log_set, llama_backend_init, llama_backend_free
    global llama_model_default_params, llama_model_load_from_file, llama_model_free, llama_model_get_vocab
    global llama_model_rope_type
    global llama_context_default_params, llama_init_from_model, llama_free
    global llama_batch_init, llama_batch_free, llama_batch_get_one
    global llama_decode, llama_get_logits, llama_get_logits_ith, llama_get_embeddings, llama_tokenize
    global llama_get_embeddings_ith
    global llama_get_memory, llama_memory_clear, llama_state_get_size, llama_state_get_data
    global llama_state_set_data, llama_model_n_embd
    global llama_vocab_n_tokens, llama_vocab_eos, llama_token_to_piece
    global llama_sampler_chain_default_params, llama_sampler_chain_init, llama_sampler_chain_add
    global llama_sampler_init_greedy, llama_sampler_init_dist, llama_sampler_init_temp
    global llama_sampler_init_top_k, llama_sampler_init_top_p, llama_sampler_sample, llama_sampler_free
    global llama_sampler_init_min_p, llama_sampler_init_penalties, llama_sampler_accept
    global llama_sampler_init_logit_bias
    global _log_callback_ref

    if llama is not None:
        lib_dir = Path(__file__).resolve().parents[2] / "bin"
        initialize_nvidia_runtime(consumer="llama", required=requested_backend == "cuda")
        _load_dynamic_backends_if_needed(requested_backend, lib_dir)
        return

    # S2-Pro 使用自己的 llama 绑定，但复用 StreamVox 统一打包的 llama.cpp 动态库目录。
    lib_dir = Path(__file__).resolve().parents[2] / "bin"
    initialize_nvidia_runtime(consumer="llama", required=requested_backend == "cuda")

    # uv 虚拟环境底层仍可能复用 conda 的 python，可先显式预加载系统 libstdc++。
    for candidate in (
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/lib/x86_64-linux-gnu/libstdc++.so.6",
    ):
        if not os.path.exists(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue

    # DLL 命名处理
    if sys.platform == "win32":
        GGML_DLL = "ggml.dll"
        GGML_BASE_DLL = "ggml-base.dll"
        LLAMA_DLL = "llama.dll"
    elif sys.platform == "darwin":
        GGML_DLL = "libggml.dylib"
        GGML_BASE_DLL = "libggml-base.dylib"
        LLAMA_DLL = "libllama.dylib"
    else:
        GGML_DLL = "libggml.so"
        GGML_BASE_DLL = "libggml-base.so"
        LLAMA_DLL = "libllama.so"

    ggml = ctypes.CDLL(os.path.join(str(lib_dir), GGML_DLL))
    ggml_base = ctypes.CDLL(os.path.join(str(lib_dir), GGML_BASE_DLL))
    llama = ctypes.CDLL(os.path.join(str(lib_dir), LLAMA_DLL))

    # 设置日志回调
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    llama_log_set = llama.llama_log_set
    llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
    llama_log_set.restype = None
    
    # 默认开启日志路由
    configure_logging(quiet=QUIET_LOGS)

    # 后端加载由 StreamVox 的 device 解析结果驱动：动态 backend 包下精准注册 CPU 和目标 GPU 插件。
    _load_dynamic_backends_if_needed(requested_backend, lib_dir)

    llama_backend_init = llama.llama_backend_init
    llama_backend_init.argtypes = []
    llama_backend_init.restype = None
    llama_backend_init()

    # 绑定其他函数
    llama_backend_free = llama.llama_backend_free
    llama_backend_free.argtypes = []
    llama_backend_free.restype = None

    llama_model_default_params = llama.llama_model_default_params
    llama_model_default_params.argtypes = []
    llama_model_default_params.restype = llama_model_params

    llama_model_load_from_file = llama.llama_model_load_from_file
    llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama_model_load_from_file.restype = ctypes.c_void_p

    llama_model_free = llama.llama_model_free
    llama_model_free.argtypes = [ctypes.c_void_p]
    llama_model_free.restype = None

    llama_model_get_vocab = llama.llama_model_get_vocab
    llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    llama_model_get_vocab.restype = ctypes.c_void_p

    llama_model_rope_type = llama.llama_model_rope_type
    llama_model_rope_type.argtypes = [ctypes.c_void_p]
    llama_model_rope_type.restype = ctypes.c_int32

    llama_model_n_embd = llama.llama_model_n_embd
    llama_model_n_embd.argtypes = [ctypes.c_void_p]
    llama_model_n_embd.restype = ctypes.c_int32

    # Context
    llama_context_default_params = llama.llama_context_default_params
    llama_context_default_params.argtypes = []
    llama_context_default_params.restype = llama_context_params

    llama_init_from_model = llama.llama_init_from_model
    llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama_init_from_model.restype = ctypes.c_void_p

    llama_free = llama.llama_free
    llama_free.argtypes = [ctypes.c_void_p]
    llama_free.restype = None

    # Batch
    llama_batch_init = llama.llama_batch_init
    llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama_batch_init.restype = llama_batch

    llama_batch_free = llama.llama_batch_free
    llama_batch_free.argtypes = [llama_batch]
    llama_batch_free.restype = None
    
    llama_batch_get_one = llama.llama_batch_get_one
    llama_batch_get_one.argtypes = [ctypes.POINTER(llama_token), ctypes.c_int32]
    llama_batch_get_one.restype = llama_batch

    # Decode
    llama_decode = llama.llama_decode
    llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama_decode.restype = ctypes.c_int32

    # Logits
    llama_get_logits = llama.llama_get_logits
    llama_get_logits.argtypes = [ctypes.c_void_p]
    llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_logits_ith = llama.llama_get_logits_ith
    llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_embeddings = llama.llama_get_embeddings
    llama_get_embeddings.argtypes = [ctypes.c_void_p]
    llama_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

    llama_get_embeddings_ith = llama.llama_get_embeddings_ith
    llama_get_embeddings_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)

    # Tokenize
    llama_tokenize = llama.llama_tokenize
    llama_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(llama_token), ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    llama_tokenize.restype = ctypes.c_int32

    # Vocab
    llama_vocab_n_tokens = llama.llama_vocab_n_tokens
    llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    llama_vocab_n_tokens.restype = ctypes.c_int32

    llama_vocab_eos = llama.llama_vocab_eos
    llama_vocab_eos.argtypes = [ctypes.c_void_p]
    llama_vocab_eos.restype = llama_token

    llama_token_to_piece = llama.llama_token_to_piece
    llama_token_to_piece.argtypes = [ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
    llama_token_to_piece.restype = ctypes.c_int

    # Memory (KV Cache)
    llama_get_memory = llama.llama_get_memory
    llama_get_memory.argtypes = [ctypes.c_void_p]
    llama_get_memory.restype = ctypes.c_void_p

    llama_memory_clear = llama.llama_memory_clear
    llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    llama_memory_clear.restype = None

    llama_state_get_size = llama.llama_state_get_size
    llama_state_get_size.argtypes = [ctypes.c_void_p]
    llama_state_get_size.restype = ctypes.c_size_t

    llama_state_get_data = llama.llama_state_get_data
    llama_state_get_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    llama_state_get_data.restype = ctypes.c_size_t

    llama_state_set_data = llama.llama_state_set_data
    llama_state_set_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    llama_state_set_data.restype = ctypes.c_size_t

    # Sampler
    llama_sampler_chain_default_params = llama.llama_sampler_chain_default_params
    llama_sampler_chain_default_params.argtypes = []
    llama_sampler_chain_default_params.restype = llama_sampler_chain_params

    llama_sampler_chain_init = llama.llama_sampler_chain_init
    llama_sampler_chain_init.argtypes = [llama_sampler_chain_params]
    llama_sampler_chain_init.restype = ctypes.c_void_p

    llama_sampler_chain_add = llama.llama_sampler_chain_add
    llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    llama_sampler_chain_add.restype = None

    llama_sampler_init_greedy = llama.llama_sampler_init_greedy
    llama_sampler_init_greedy.argtypes = []
    llama_sampler_init_greedy.restype = ctypes.c_void_p

    llama_sampler_init_dist = llama.llama_sampler_init_dist
    llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    llama_sampler_init_dist.restype = ctypes.c_void_p

    llama_sampler_init_temp = llama.llama_sampler_init_temp
    llama_sampler_init_temp.argtypes = [ctypes.c_float]
    llama_sampler_init_temp.restype = ctypes.c_void_p

    llama_sampler_init_top_k = llama.llama_sampler_init_top_k
    llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    llama_sampler_init_top_k.restype = ctypes.c_void_p

    llama_sampler_init_top_p = llama.llama_sampler_init_top_p
    llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    llama_sampler_init_top_p.restype = ctypes.c_void_p

    llama_sampler_sample = llama.llama_sampler_sample
    llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    llama_sampler_sample.restype = llama_token

    llama_sampler_free = llama.llama_sampler_free
    llama_sampler_free.argtypes = [ctypes.c_void_p]
    llama_sampler_free.restype = None

    llama_sampler_init_logit_bias = llama.llama_sampler_init_logit_bias
    llama_sampler_init_logit_bias.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(llama_logit_bias)]
    llama_sampler_init_logit_bias.restype = ctypes.c_void_p

    llama_sampler_init_min_p = llama.llama_sampler_init_min_p
    llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    llama_sampler_init_min_p.restype = ctypes.c_void_p

    llama_sampler_init_penalties = llama.llama_sampler_init_penalties
    llama_sampler_init_penalties.argtypes = [ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    llama_sampler_init_penalties.restype = ctypes.c_void_p

    llama_sampler_accept = llama.llama_sampler_accept
    llama_sampler_accept.argtypes = [ctypes.c_void_p, llama_token]
    llama_sampler_accept.restype = None


def load_model(model_path: str, n_gpu_layers=-1, main_gpu=0, split_mode=0, backend: str | None = None):
    """
    加载 GGUF 模型（自动处理初始化和路径编码）
    
    Args:
        model_path: GGUF 模型文件路径
        n_gpu_layers: 需要 offload 到 GPU 的层数；0 表示强制 CPU。
        main_gpu: llama.cpp 主 GPU 索引。
        split_mode: llama.cpp 多 GPU 切分模式；S2-Pro 当前固定使用单卡。
        backend: runtime 已解析出的 GGUF 后端，用于决定是否扫描 Vulkan/CUDA/Metal 动态插件。
        
    Returns:
        model: llama_model 指针
    """
    # 模型文件按相对路径交给 llama.cpp，工作目录切到统一 bin 目录以确保依赖库可解析。
    lib_dir = Path(__file__).resolve().parents[2] / "bin"
    model_path = Path(model_path).resolve()
    native_model_path = _model_path_for_native_load(model_path, lib_dir)

    # 跳转到 dll 所在目录，并将其加到 Path
    original_cwd = Path.cwd()
    try:
        os.chdir(lib_dir)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(os.getcwd())
        os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']
        logger.debug("Changed directory to: %s", Path.cwd())

        # 初始化 backend，载入模型；动态后端是否扫描由 runtime 的 GGUF 后端决策驱动。
        init_llama_lib(requested_backend=backend)
        model_params = llama_model_default_params()
        # StreamVox 的公开 device 入口需要能控制 GGUF 是否上 GPU；参考代码固定本地调试，
        # 这里把 GPU 层数和主卡继续透传给 llama.cpp，但仍保持 S2-Pro 自己的 llama 绑定独立存在。
        model_params.n_gpu_layers = n_gpu_layers
        model_params.main_gpu = _main_gpu_for_native_load(n_gpu_layers, main_gpu)
        model_params.split_mode = split_mode
        kv_overrides = _build_model_kv_overrides(model_path)
        if kv_overrides is not None:
            model_params.kv_overrides = ctypes.cast(kv_overrides, ctypes.c_void_p)
        model = llama_model_load_from_file(
            native_model_path.as_posix().encode('utf-8'),
            model_params
        )
    finally:
        # 后端初始化或原生加载失败后仍需恢复目录，避免后续回退和相对路径输出受到影响。
        os.chdir(original_cwd)

    if model:
        logger.debug("Restored directory to: %s", Path.cwd())
        return model
    else:
        logger.error(f'当前路径：{Path.cwd()}')
        logger.error(f'模型绝对路径：{model_path.as_posix()}')
        logger.error(f'模型可访问性：{model_path.exists()}')
        logger.error(f"模型加载失败: {model_path}")
        return None


def _kv_key_bytes(key: str) -> bytes:
    encoded = key.encode("utf-8")
    if len(encoded) >= 128:
        raise ValueError(f"metadata override key too long: {key!r}")
    return encoded


def _kv_str_bytes(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= 128:
        raise ValueError(f"metadata override value too long: {value!r}")
    return encoded


def _append_kv_override(overrides, key: str, value):
    item = llama_model_kv_override()
    item.key = _kv_key_bytes(key)
    if isinstance(value, bool):
        item.tag = 2
        item.val_bool = value
    elif isinstance(value, int):
        item.tag = 0
        item.val_i64 = value
    elif isinstance(value, float):
        item.tag = 1
        item.val_f64 = value
    else:
        item.tag = 3
        item.val_str = _kv_str_bytes(str(value))
    overrides.append(item)


def _build_model_kv_overrides(model_path: Path):
    model_name = Path(model_path).name
    env_candidates = []
    if "slow" in model_name:
        env_candidates.append("S2_PRO_SLOW_ARCH_OVERRIDE")
    if "fast" in model_name:
        env_candidates.append("S2_PRO_FAST_ARCH_OVERRIDE")
    env_candidates.append("S2_PRO_ARCH_OVERRIDE")

    arch_override = None
    for env_name in env_candidates:
        value = os.environ.get(env_name)
        if value:
            arch_override = value.strip()
            logger.info("Applying architecture override from %s=%s for %s", env_name, arch_override, model_name)
            break

    if not arch_override:
        return None

    overrides = []
    source_arch = _read_model_architecture(model_path)
    _append_kv_override(overrides, "general.architecture", arch_override)
    _mirror_arch_metadata(overrides, model_path, source_arch, arch_override)

    terminator = llama_model_kv_override()
    terminator.key = b""
    overrides.append(terminator)

    array_type = llama_model_kv_override * len(overrides)
    return array_type(*overrides)


def _read_model_architecture(model_path: Path) -> str | None:
    try:
        reader = gguf.GGUFReader(str(model_path))
    except Exception as exc:
        logger.warning("Failed to read GGUF metadata from %s: %s", model_path, exc)
        return None
    field = reader.get_field("general.architecture")
    if field is None:
        return None
    value = field.contents()
    return str(value) if value is not None else None


def _mirror_arch_metadata(overrides, model_path: Path, source_arch: str | None, target_arch: str) -> None:
    if not source_arch or source_arch == target_arch:
        return

    try:
        reader = gguf.GGUFReader(str(model_path))
    except Exception as exc:
        logger.warning("Failed to mirror GGUF metadata from %s: %s", model_path, exc)
        return

    source_prefix = f"{source_arch}."
    target_prefix = f"{target_arch}."
    mirrored = 0
    for key, field in reader.fields.items():
        if not key.startswith(source_prefix):
            continue
        if field.types and field.types[0].name == "ARRAY":
            continue
        value = field.contents()
        if value is None:
            continue
        alias_key = target_prefix + key[len(source_prefix):]
        _append_kv_override(overrides, alias_key, value)
        mirrored += 1
    logger.info(
        "Mirrored %d GGUF metadata keys from %s.* to %s.* for %s",
        mirrored,
        source_arch,
        target_arch,
        model_path.name,
    )


def create_context(model, n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq_max=1, 
                   embeddings=False, pooling_type=0, flash_attn=True, 
                   offload_kqv=True, no_perf=True, n_threads=None):
    """创建 ASR 专用的上下文"""
    params = llama_context_default_params()
    params.n_ctx = n_ctx
    params.n_batch = n_batch
    params.n_ubatch = n_ubatch
    params.n_seq_max = n_seq_max
    params.embeddings = embeddings
    params.pooling_type = pooling_type
    params.flash_attn_type = 1 if flash_attn else 0  # 1 = ON, 0 = OFF (auto typically uses what's available)
    params.offload_kqv = offload_kqv
    params.no_perf = no_perf
    
    if n_threads:
        params.n_threads = n_threads
        params.n_threads_batch = n_threads
    else:
        params.n_threads = os.cpu_count() // 2
        params.n_threads_batch = os.cpu_count()

    return llama_init_from_model(model, params)


class LlamaModel:
    """模型的面向对象封装"""
    def __init__(self, path, n_gpu_layers=-1, main_gpu=0, split_mode=0, backend: str | None = None):
        """
        创建 S2-Pro GGUF 模型句柄。

        参数:
            path: GGUF 模型文件路径。
            n_gpu_layers: 需要 offload 到 GPU 的层数；CPU 回退时应传 0。
            main_gpu: llama.cpp 主 GPU 索引。
            split_mode: llama.cpp 多 GPU 切分模式；当前业务固定为单卡。
            backend: runtime 已解析出的 GGUF 后端，用于触发必要的动态 backend 注册。

        返回:
            构造函数无显式返回；成功后 self.ptr 持有 llama_model 指针。

        边界异常:
            llama.cpp 返回空模型指针时抛出 RuntimeError，调用方可据此执行 CPU 回退。
        """
        # backend 记录当前模型的 native 释放策略；部分 Vulkan 动态后端在双 GGUF 场景下主动 free 会触发退出崩溃。
        self.backend = backend
        self.ptr = load_model(path, n_gpu_layers=n_gpu_layers, main_gpu=main_gpu, split_mode=split_mode, backend=backend)
        if not self.ptr:
            raise RuntimeError(f"failed to load GGUF model: {path}")
        self.rope_type = llama_model_rope_type(self.ptr) if self.ptr else None
            
        self.vocab = llama_model_get_vocab(self.ptr)
        self.vocab_size = llama_vocab_n_tokens(self.vocab)
        self.n_embd = llama_model_n_embd(self.ptr)
        self.eos_token = llama_vocab_eos(self.vocab)
        module_name = "Slow LM" if "slow" in Path(path).stem.lower() else "Fast LM"
        normalized_backend = str(self.backend or "").strip().lower()
        if normalized_backend == "vulkan":
            device_label = "Vulkan:0"
        elif normalized_backend == "cuda":
            device_label = "CUDA:0"
        elif normalized_backend == "metal":
            device_label = "Metal:0"
        elif normalized_backend == "cpu" or int(n_gpu_layers) == 0:
            device_label = "CPU"
        else:
            device_label = "Unknown"
        precision_label = "GGUF"
        for token in ("q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_1", "q5_0", "q4_k_m", "q4_k_s", "q4_1", "q4_0", "f16", "f32"):
            if token in Path(path).stem.lower():
                precision_label = token.upper()
                break
        logger.info("✅ [s2-pro] [%s] [%s] [%s]", module_name, device_label, precision_label)

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = True) -> List[int]:
        """(Native) 文本转 Token ID 列表"""
        return text_to_tokens(self.vocab, text, add_special, parse_special)

    def detokenize(self, tokens: List[int]) -> str:
        """(Native) Token ID 列表转文本"""
        if tokens is None or len(tokens) == 0: return ""
        all_bytes = b"".join([self.token_to_bytes(tid) for tid in tokens])
        return all_bytes.decode('utf-8', errors='replace')

    def token_to_bytes(self, token_id: int) -> bytes:
        """(Native) 单个 Token 转字节"""
        return token_to_bytes(self.vocab, token_id)
        
    def token_to_piece(self, token_id: int) -> str:
        """(Native) 单个 Token 转字符串 Piece"""
        return self.token_to_bytes(token_id).decode('utf-8', errors='replace')

    def token_bos(self) -> int:
        return llama_vocab_bos(self.vocab)

    def token_eos(self) -> int:
        return llama_vocab_eos(self.vocab)
        
    def token_to_id(self, text: str) -> int:
        """(Native) 单个 Token 字符串转 ID (仅限 Exact Match)"""
        # 利用 tokenize 来查找 ID
        res = self.tokenize(text, add_special=False, parse_special=True)
        return res[0] if res else -1

    def detach_native(self) -> None:
        """
        断开 Python 对象与 native llama_model 指针的关联。

        参数:
            本函数不接收参数。

        返回:
            无返回值；调用后 __del__ 不会再释放 native 模型指针。

        边界异常:
            仅用于 Vulkan 动态后端的退出崩溃规避；调用后当前对象不能继续参与推理。
        """
        # Vulkan 动态后端在 S2-Pro slow/fast 双模型场景主动 free 不稳定，断开指针让 OS 在进程退出时回收。
        self.ptr = None

    def free_native(self, force: bool = False) -> None:
        """
        释放 native llama_model 指针。

        参数:
            force: 为 True 时即使当前 backend 是 Vulkan 也执行 native free，用于验证安全释放路径。

        返回:
            无返回值；释放后 self.ptr 会被置空，避免重复释放。

        边界异常:
            llama_model_free 的 native 异常无法被 Python 捕获；调用方必须保证 context 已先释放。
        """
        if not self.ptr:
            return
        # 默认析构仍保护 Vulkan，实验或显式 shutdown 可传 force=True 走真实释放。
        if getattr(self, "backend", None) == "vulkan" and not force:
            self.ptr = None
            return
        llama_model_free(self.ptr)
        self.ptr = None

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            # 当前互联网 Vulkan backend 在 S2-Pro slow/fast 双模型场景下主动 llama_model_free 会导致进程退出段错误。
            # 这里仅对 Vulkan 路径跳过 native free，交给进程退出时的 OS 资源回收；CUDA/CPU 仍走正常释放。
            self.free_native(force=False)


class LlamaContext:
    """上下文的面向对象封装"""
    def __init__(self, model, n_ctx=2048, n_batch=2048, n_ubatch=512, n_seq_max=1, 
                 embeddings=False, pooling_type=0, flash_attn=True, 
                 offload_kqv=True, no_perf=True, n_threads=None, n_threads_batch=None):
        self.model = model # 保持模型引用防止被释放
        params = llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_ubatch = n_ubatch
        params.n_seq_max = n_seq_max
        params.embeddings = embeddings
        params.pooling_type = pooling_type
        params.flash_attn_type = 1 if flash_attn else 0
        params.offload_kqv = offload_kqv
        params.no_perf = no_perf
        
        # 线程配置
        cpu_count = os.cpu_count() or 4
        if n_threads:
            params.n_threads = n_threads
        else:
            params.n_threads = cpu_count // 2

        if n_threads_batch:
            params.n_threads_batch = n_threads_batch
        else:
            params.n_threads_batch = n_threads if n_threads else cpu_count

        self.ptr = llama_init_from_model(model.ptr, params)
        if not self.ptr:
            raise RuntimeError("上下文初始化失败")

    def decode(self, batch):
        struct = batch.struct if hasattr(batch, 'struct') else batch
        return llama_decode(self.ptr, struct)

    def decode_token(self, token_id):
        """
        原子操作：设置单 Token Batch 并执行解码
        """
        return self.decode(get_one_batch(token_id))

    def get_logits(self):
        """获取 Batch 中最后一个启用 Logits 的 Token 的输出"""
        return llama_get_logits(self.ptr)

    def get_logits_ith(self, i: int):
        """获取 Batch 中第 i 个 Token 的 Logits 输出 (前提是该 Token 启用了 Logits 标志)"""
        return llama_get_logits_ith(self.ptr, i)

    def get_embeddings(self):
        return llama_get_embeddings(self.ptr)

    def get_embeddings_ith(self, i: int):
        return llama_get_embeddings_ith(self.ptr, i)

    def clear_kv_cache(self):
        mem = llama_get_memory(self.ptr)
        llama_memory_clear(mem, True)

    def detach_native(self) -> None:
        """
        断开 Python 对象与 native llama_context 指针的关联。

        参数:
            本函数不接收参数。

        返回:
            无返回值；调用后 __del__ 不会再释放 native context 指针。

        边界异常:
            仅用于 Vulkan 动态后端的退出崩溃规避；调用后当前对象不能继续 decode。
        """
        # context 必须先于 model 断开；否则 __del__ 仍可能调用 llama_free 触发 Vulkan backend 崩溃。
        self.ptr = None

    def free_native(self, force: bool = False) -> None:
        """
        释放 native llama_context 指针。

        参数:
            force: 为 True 时即使当前模型 backend 是 Vulkan 也执行 native free，用于验证安全释放路径。

        返回:
            无返回值；释放后 self.ptr 会被置空，避免重复释放。

        边界异常:
            llama_free 的 native 异常无法被 Python 捕获；调用方必须保证不再访问当前 context。
        """
        if not self.ptr:
            return
        # 默认析构仍保护 Vulkan，显式 shutdown 可在确认安全时传 force=True 走真实释放。
        if getattr(getattr(self, "model", None), "backend", None) == "vulkan" and not force:
            self.ptr = None
            return
        llama_free(self.ptr)
        self.ptr = None

    def get_state(self) -> bytes: # 取 llama context state
        size = int(llama_state_get_size(self.ptr))
        if size <= 0:
            return b""
        buffer = (ctypes.c_uint8 * size)()
        written = int(
            llama_state_get_data(
                self.ptr,
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.c_size_t(size),
            )
        )
        if written <= 0:
            raise RuntimeError("llama_state_get_data returned no data")
        return bytes(buffer[:written])

    def set_state(self, state: bytes) -> None: # 根据state 恢复 llama context
        payload = bytes(state)
        size = len(payload)
        if size == 0:
            self.clear_kv_cache()
            return
        buffer = ctypes.create_string_buffer(payload, size)
        restored = int(
            llama_state_set_data(
                self.ptr,
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.c_size_t(size),
            )
        )
        if restored != size:
            raise RuntimeError(
                f"llama_state_set_data restored {restored} bytes, expected {size}"
            )

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            # 与 LlamaModel 的 Vulkan 释放策略保持一致：Vulkan 双 context 主动 llama_free 会触发 native 退出崩溃。
            self.free_native(force=False)


class LlamaBatch:
    """Batch 的面向对象封装，支持直接属性访问"""
    def __init__(self, n_tokens, embd_dim=0, n_seq_max=1):
        self.struct = llama_batch_init(n_tokens, embd_dim, n_seq_max)
        self.n_tokens_max = n_tokens

    @property
    def n_tokens(self): return self.struct.n_tokens
    @n_tokens.setter
    def n_tokens(self, val): self.struct.n_tokens = val

    @property
    def token(self): return self.struct.token
    @property
    def embd(self): return self.struct.embd
    @property
    def pos(self): return self.struct.pos
    @property
    def n_seq_id(self): return self.struct.n_seq_id
    @property
    def seq_id(self): return self.struct.seq_id
    @property
    def logits(self): return self.struct.logits

    def set_embd(self, data: np.ndarray, pos: Union[np.ndarray, int] = 0, seq_id: int = 0):
        """
        高阶接口：直接注入 Embedding 数据并初始化位置信息
        告诉 llama.cpp：用这些 embedding 做一次 decode，而不是用 token id 查 embedding
        
        Args:
            data: Embedding 数据 [n_tokens, dim]
            pos: 位置信息。
                 - 若为 int，则视为起始偏移量，自动生成 [offset, offset+1, ...]
                 - 若为 np.ndarray，则直接拷贝到 pos buffer 
                    np.ndarray pos 只是允许外部显式传 position buffer
                    是否需要 1D 还是 4D，取决于底层模型的 n_pos_per_embd
                    当前项目 slow/fast 都只用 1D position
            seq_id: 序列 ID
        """
        n_tokens = data.shape[0]
        if n_tokens > self.n_tokens_max:
            raise ValueError(f"Batch 空间不足: {n_tokens} > {self.n_tokens_max}")
        
        # 1. 内存移动 (Embedding)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        ctypes.memmove(self.embd, data.ctypes.data, data.nbytes)
        
        # 2. 位置信息处理 (Position)
        if isinstance(pos, int):
            # 自动生成线性位置
            pos_offset = pos
            for i in range(n_tokens):
                self.pos[i] = pos_offset + i
        elif isinstance(pos, np.ndarray):
            # 注意：不检查 pos 长度是否等于 n_tokens，因为复杂位置编码可能存在 stride。
            # 但必须确保不超过 batch capacity
            if not pos.flags['C_CONTIGUOUS']:
                pos = np.ascontiguousarray(pos)
            
            # 使用 memmove 直接拷贝
            # self.pos 是 ctypes 指针，可以直接操作
            ctypes.memmove(self.pos, pos.ctypes.data, pos.nbytes)
        else:
            raise TypeError(f"Unsupported pos type: {type(pos)}")

        # 3. 设置其他元数据
        self.n_tokens = n_tokens
        for i in range(n_tokens):
            self.n_seq_id[i] = 1
            self.seq_id[i][0] = seq_id
            self.logits[i] = 1 if i == n_tokens - 1 else 0
        
        return self

    def __del__(self):
        if hasattr(self, 'struct'):
            llama_batch_free(self.struct)

def get_one_batch(token_id: int):
    """
    底层极限优化：用于单 Token 生成的无分配 Batch 构造。
    相当于 C++ 的 llama_batch_get_one(&token, 1)。
    它不依赖 llama_batch_init 的内存分配，并允许底层自动推断 pos。
    """
    token_arr = (llama_token * 1)(token_id)
    return llama_batch_get_one(token_arr, 1)

class LlamaSampler:
    """采样器的面向对象封装"""
    def __init__(
        self, 
        temperature: float = 0.8, 
        top_k: int = 50, 
        top_p: float = 1.0, 
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_last_n: int = 64,
        seed: Optional[int] = None, 
        logit_bias: Optional[dict] = None, 
        n_vocab: int = 0
    ):
        import time
        if seed is None:
            seed = int(time.time())
            
        sparams = llama_sampler_chain_default_params()
        self.ptr = llama_sampler_chain_init(sparams)
        
        # 1. Logit Bias (最高优先级)
        if logit_bias and n_vocab > 0 and isinstance(logit_bias, dict):
            n_bias = len(logit_bias)
            BiasArray = llama_logit_bias * n_bias
            bias_data = BiasArray()
            for i, (token, bias) in enumerate(logit_bias.items()):
                bias_data[i].token = token
                bias_data[i].bias = bias
            llama_sampler_chain_add(self.ptr, llama_sampler_init_logit_bias(n_vocab, n_bias, bias_data))

        # 2. Penalties (重复/频率/存在惩罚)
        has_penalty = (repeat_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0)
        if has_penalty:
            # llama.cpp 会自动管理历史 rings
            llama_sampler_chain_add(self.ptr, llama_sampler_init_penalties(
                penalty_last_n, repeat_penalty, frequency_penalty, presence_penalty
            ))

        # 3. 采样过滤器 (顺序很重要)
        if temperature > 0:
            if top_k > 0:
                llama_sampler_chain_add(self.ptr, llama_sampler_init_top_k(top_k))
            if top_p < 1.0:
                llama_sampler_chain_add(self.ptr, llama_sampler_init_top_p(top_p, 1))
            if 0.0 < min_p < 1.0:
                llama_sampler_chain_add(self.ptr, llama_sampler_init_min_p(min_p, 1))
            
            llama_sampler_chain_add(self.ptr, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(self.ptr, llama_sampler_init_dist(seed))
        else:
            llama_sampler_chain_add(self.ptr, llama_sampler_init_greedy())

        self._neg_inf = -1e10

    def accept(self, token_id: int):
        """同步 Token 到采样链历史 (native)"""
        if self.ptr:
            llama_sampler_accept(self.ptr, token_id)

    def sample(self, ctx, idx=-1, limit_start=None, limit_end=None, allow_tokens=None):
        """采样一个 Token，支持范围限制和白名单豁免"""
        ctx_ptr = ctx.ptr if hasattr(ctx, 'ptr') else ctx
            
        # 处理 Logits 约束 (Range Limit + Allow-list)
        if (limit_start is not None or limit_end is not None) and hasattr(ctx, 'get_logits'):
            n_vocab = llama_vocab_n_tokens(ctx.model.vocab)
            logits_ptr = ctx.get_logits_ith(idx) # 获取指定索引的 logits
            logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
            
            s = max(0, limit_start) if limit_start is not None else 0
            e = min(n_vocab, limit_end) if limit_end is not None else n_vocab
            
            # 创建掩码：默认全灭，然后开启指定范围和白名单
            mask = np.ones(n_vocab, dtype=bool)
            mask[s:e] = False # 范围内的不抹除
            if allow_tokens:
                for t in allow_tokens:
                    if 0 <= t < n_vocab:
                        mask[t] = False # 白名单内地不抹除
            
            logits[mask] = self._neg_inf
        
        # 使用原生 Chain 进行采样。
        # 注意：llama_sampler_sample 会自动对 Chain 调用 accept()。
        return llama_sampler_sample(self.ptr, ctx_ptr, idx)

    def free(self):
        """释放采样器资源"""
        if hasattr(self, 'ptr') and self.ptr:
            llama_sampler_free(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

    def __del__(self):
        self.free()


class ASRStreamDecoder:
    """ASR 专属流式解码器，集成字节解码与 ASRReporter 交互"""
    def __init__(self, vocab, reporter=None):
        self.vocab = vocab
        self.reporter = reporter
        self.byte_decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
        self.generated_text = ""
        self.tokens_generated = 0
        self.tokens = []

    def push(self, token_id: int):
        """推入 Token，返回新解码的文字片段"""
        raw_bytes = token_to_bytes(self.vocab, token_id)
        text_piece = self.byte_decoder.decode(raw_bytes, final=False)
        self.tokens.append(text_piece)
        self.tokens_generated += 1
        
        self.generated_text += text_piece
        
        if self.reporter:
            self.reporter.stream(text_piece)
            
        return text_piece

    def flush(self):
        """清空残余字节并返回"""
        remaining = self.byte_decoder.decode(b"", final=True)
        self.tokens.append(remaining)
        self.generated_text += remaining
        return remaining


def python_log_callback(level, message, user_data):
    """
    llama.cpp 日志回调函数
    level: 
        2 = ERROR
        3 = WARN
        4 = INFO
        5 = DEBUG
    """
    if not message: return
    try:
        msg_str = message.decode('utf-8', errors='replace').strip()
        if not msg_str or msg_str in ['.', '\n']: return
        
        if level == 2:
            logger.error(f"[llama.cpp] {msg_str}")
        elif level == 3:
            logger.warning(f"[llama.cpp] {msg_str}")
        elif level == 4:
            logger.info(f"[llama.cpp] {msg_str}")
        elif level >= 5:
            logger.debug(f"[llama.cpp] {msg_str}")
        else:
            logger.info(f"[llama.cpp] {msg_str}")
    except Exception as e:
        # 防止回调错误导致程序崩溃
        print(f"日志回调出错: {e}")

def configure_logging(quiet=False):
    """配置 llama.cpp 日志回调"""
    global _log_callback_ref
    if not llama_log_set: return
    
    LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    if not quiet:
        _log_callback_ref = LOG_CALLBACK(python_log_callback)
        llama_log_set(_log_callback_ref, None)
    else:
        # 如果需要静默，可以传递一个空函数，或者将 logger 级别调高
        _log_callback_ref = LOG_CALLBACK(lambda l, m, u: None)
        llama_log_set(_log_callback_ref, None)



# =========================================================================
# Embedding Table
# =========================================================================



class LlamaEmbeddingTable:
    """动态反量化 Embedding 表，支持 table[ids] 语法"""
    def __init__(self, raw_data, qtype):
        self.raw_data = raw_data
        self.qtype = qtype
        
    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, tokens):
        from gguf.quants import dequantize
        
        # 如果是原生 float 类型，直接返回
        if self.raw_data.dtype in (np.float32, np.float16):
            return self.raw_data[tokens].astype(np.float32)
            
        # 调用官方库进行高性能反量化
        return dequantize(self.raw_data[tokens], self.qtype.value)




def _skip_gguf_value(mm, offs, v_type):
    # UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
    fixed = [1, 1, 2, 2, 4, 4, 4, 1, -1, -2, 8, 8, 8]
    val_len = fixed[v_type]
    if val_len > 0:
        return offs + val_len
    elif val_len == -1: # string
        slen = struct.unpack_from("<Q", mm, offs)[0]
        return offs + 8 + slen
    elif val_len == -2: # array
        itype, alen = struct.unpack_from("<IQ", mm, offs)
        offs += 12
        if itype == 8: # string array
            for _ in range(alen):
                slen = struct.unpack_from("<Q", mm, offs)[0]
                offs += 8 + slen
        else:
            item_len = fixed[itype]
            if item_len > 0:
                offs += item_len * alen
            else:
                raise ValueError("Nested arrays or unknown type not supported in fast skip")
        return offs

def get_token_embeddings_gguf(model_path, target_tensor="token_embd.weight"):
    """
    超极速 GGUF Embedding 提取 (直接二进制寻址)
    避免加载整个模型、避免解析包含 15 万词条的 tokenizer 对象。耗时降至 < 50ms。
    """
    t_start = time.time()
    mm = np.memmap(model_path, mode='r')
    
    # 获取文件头信息
    tensor_count, kv_count = struct.unpack_from("<QQ", mm, 8)
    offs = 24
    alignment = 32
    
    # 光速跃过/扫描所有 KV 字段
    for _ in range(kv_count):
        key_len = struct.unpack_from("<Q", mm, offs)[0]
        offs += 8
        if key_len == 17 and mm[offs:offs+17].tobytes() == b'general.alignment':
            offs += 17
            v_type = struct.unpack_from("<I", mm, offs)[0]
            offs += 4
            if v_type == 4: # UINT32
                alignment = struct.unpack_from("<I", mm, offs)[0]
                offs += 4
                continue
        else:
            offs += key_len
            
        v_type = struct.unpack_from("<I", mm, offs)[0]
        offs += 4
        offs = _skip_gguf_value(mm, offs, v_type)
        
    # 扫描 Tensor Infos 搜寻我们想要的张量
    target_rel_offset = None
    target_type = None
    target_shape = None # GGUF shape 是倒序的 [n_embd, vocab_size]
    
    target_bytes = target_tensor.encode('utf-8')
    for _ in range(tensor_count):
        name_len = struct.unpack_from("<Q", mm, offs)[0]
        offs += 8
        is_target = False
        if name_len == len(target_bytes) and mm[offs:offs+name_len].tobytes() == target_bytes:
            is_target = True
        offs += name_len
        
        n_dims = struct.unpack_from("<I", mm, offs)[0]
        offs += 4
        
        shape = struct.unpack_from(f"<{n_dims}Q", mm, offs) # 返回元组
        offs += 8 * n_dims
        
        t_type = struct.unpack_from("<I", mm, offs)[0]
        offs += 4
        
        rel_offset = struct.unpack_from("<Q", mm, offs)[0]
        offs += 8
        
        if is_target:
            target_shape = shape
            target_type = t_type
            target_rel_offset = rel_offset
            
    # 计算数据区起始点并加载张量
    padding = offs % alignment
    if padding != 0:
        offs += (alignment - padding)
    data_offset = offs
    
    if target_shape is None:
        logger.error(f"无法在 {model_path} 中找到 {target_tensor}")
        return None
        
    abs_offset = data_offset + target_rel_offset
    n_embd = target_shape[0]     # 特征维度
    vocab_size = target_shape[1] # 词表大小
    
    qtype = GGMLQuantizationType(target_type)
    if qtype in GGML_QUANT_SIZES:
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        bytes_per_row = (n_embd // block_size) * type_size
    else:
        # F32 或 F16
        if qtype == GGMLQuantizationType.F32:
            bytes_per_row = n_embd * 4
        elif qtype == GGMLQuantizationType.F16:
            bytes_per_row = n_embd * 2
        else:
            raise ValueError(f"未知的数据格式支持: {qtype.name}")

    total_bytes = vocab_size * bytes_per_row
    raw_data = mm[abs_offset : abs_offset + total_bytes]
    
    if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        if qtype == GGMLQuantizationType.F32:
            raw_data = raw_data.view(np.float32).reshape(vocab_size, n_embd)
        else:
            raw_data = raw_data.view(np.float16).reshape(vocab_size, n_embd)
    else:
        raw_data = raw_data.reshape(vocab_size, bytes_per_row)
        
    total_time = time.time() - t_start
    logger.info(f"--- [QwenASR] 已极速载入 Embedding 视图 ({total_time*1000:.1f}ms) ---")
    logger.info(f"    - 量化格式: {qtype.name} ({n_embd} dims, {vocab_size} tokens)")
    
    return LlamaEmbeddingTable(raw_data, qtype)



# =========================================================================
# Utilities
# =========================================================================


def text_to_tokens(vocab, text, add_special=False, parse_special=True):
    text_bytes = text.encode("utf-8")
    n_tokens_max = len(text_bytes) + 32
    tokens = (llama_token * n_tokens_max)()
    n = llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, add_special, parse_special)
    return [tokens[i] for i in range(n)] if n >= 0 else []

def token_to_bytes(vocab, token_id):
    buf = ctypes.create_string_buffer(256)
    n = llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
    return buf.raw[:n] if n > 0 else b""
