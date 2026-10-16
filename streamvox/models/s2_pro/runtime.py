from __future__ import annotations

from ..onnx_backend import resolve_explicit_onnx_backend

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import yaml

from ...model_bundle import validate_local_bundle
from ...utils.log import logger
from .. import llama_backend as llama_backend_utils
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, RuntimeInfo, SerializedPromptAsset
from ..text_processing import TextPreprocessConfig
from . import llama
from .assets import AssetsManager
from .codec import TrueStreamingCodecEncoderOrtRuntime
from .codec_common import CodecOnnxConfig
from .decoder import S2ProDecoderAPI
from .schema.decoder_protocol import DecoderResponse
from .text import S2_PRO_TEXT_PREPROCESS_CONFIG, preprocess_text
from .prompt_builder import build_role_prefix_prompt, build_text_values, build_turn_prompt, InferenceTokenizer, IM_END_TOKEN
from .schema.record import S2ProTiming, log_s2_pro_performance
from .spec import S2_PRO_4B_GGUF_SPEC, S2ProModelSpec


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "s2_pro.yaml"
S2_PRO_CONTEXT_RESET_MARGIN = 60 * 22  # 留空1分钟的上下文


@dataclass(slots=True)
class S2ProSampleConfig:
    temperature: float
    top_p: float
    top_k: int


@dataclass(slots=True)
class S2ProConfig:
    sample_rate: int = 44100
    n_ctx: int = 12288
    min_n_ctx: int = 4096
    max_n_ctx: int = 32768
    chunk_size: int = 21  # 1秒
    max_length: int = 450 # 300字符
    min_length: int = 40
    slow_n_gpu_layers: int = -1
    fast_n_gpu_layers: int = -1
    llama_cpu_threads: int = 0
    llama_cpu_threads_batch: int = 0
    llama_cpu_numa_strategy: str = "disabled"
    decoder_start_method: str = "spawn"
    slow_filename: str = "slow_hf_q8_0.gguf"
    fast_filename: str = "fast_hf_q8_0.gguf"
    codec_encode_filename: str = "codec_encode_fp16.onnx"
    codec_decode_filename: str = "codec_decode_stream_fp16.onnx"
    codec_meta_filename: str = "codec_true_streaming_meta_fp16.json"
    tokenizer_dirname: str = "tokenizer"
    codec_frame_length: int = 2048
    codec_hop_length: int = 512
    codec_num_codebooks: int = 10
    codec_stream_code_frames: int = 21
    generation_sample: S2ProSampleConfig = field(default_factory=lambda: S2ProSampleConfig(0.9, 0.8, 30))
    ras_sample: S2ProSampleConfig = field(default_factory=lambda: S2ProSampleConfig(1.0, 0.9, 30))


@dataclass(slots=True)
class S2ProPromptAsset(BasePromptAsset):
    prompt_text: str | list[str] | None = None
    prompt_codes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    decoder_state: dict[str, np.ndarray] = field(default_factory=dict)
    # Prompt Decoder预热时，不满21frame的code
    decoder_tail_codes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    # make_prompt 阶段完成的 slow prefill 快照；合成时可直接恢复，避免每次使用角色都重复计算 system/reference prompt。
    slow_prefix_state: bytes | None = None
    # slow_prefix_state 对应的上下文写入位置，也就是 system/reference prompt 的 token 长度。
    slow_prefix_position: int = 0


@dataclass(slots=True)
class S2ProResolvedDevice:
    request: str
    gguf_backend: str
    decoder_backend: str
    gpu_index: int | None = None

    @property
    def normalized(self) -> str:
        if self.gpu_index is None:
            return self.decoder_backend
        return f"{self.decoder_backend}:{self.gpu_index}"

    @property
    def gguf_gpu_enabled(self) -> bool:
        return self.gguf_backend in {"cuda", "vulkan", "metal"}


@dataclass(slots=True)
class CpuTopology:
    logical_cpus: int
    physical_cores: int
    sockets: int
    physical_cores_per_socket: int


@dataclass(slots=True)
class LlamaCpuThreadPlan:
    n_threads: int
    n_threads_batch: int
    source: str
    topology: CpuTopology


@dataclass(slots=True)
class CodeStepResult:
    semantic_token: int
    codes: np.ndarray
    step_idx: int
    is_last: bool = False


@dataclass(slots=True)
class _S2ProStreamError:
    """
    S2-Pro 流式合成后台线程通过输出队列回传给前台生成器的异常包装。

    参数:
        error: 后台线程捕获到的原始异常对象。

    返回:
        返回一个仅用于线程间传递错误状态的轻量对象。

    异常:
        本数据结构本身不抛出异常。
    """

    error: Exception


def load_s2_pro_config() -> S2ProConfig:
    if not CONFIG_PATH.exists():
        return S2ProConfig()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime", {})
    paths = payload.get("paths", {})
    codec = payload.get("codec", {})
    sample_config = payload.get("sample_config", {})
    generation_sample = sample_config.get("generation", sample_config.get("slow", {}))
    ras_sample = sample_config.get("ras", {})
    # default_n_ctx 是当前模型未收到用户显式 n_ctx 时使用的默认上下文长度。
    # 兼容旧配置里的 n_ctx 字段，避免旧包升级后缺少 default_n_ctx 时无法按原值启动。
    default_n_ctx = int(runtime.get("default_n_ctx", runtime.get("n_ctx", 12288)))
    # min_n_ctx / max_n_ctx 属于模型私有能力边界，后续不同模型可在 yaml 中声明不同取值。
    min_n_ctx = int(runtime.get("min_n_ctx", 4096))
    max_n_ctx = int(runtime.get("max_n_ctx", 32768))
    return S2ProConfig(
        sample_rate=int(runtime.get("sample_rate", 44100)),
        n_ctx=default_n_ctx,
        min_n_ctx=min_n_ctx,
        max_n_ctx=max_n_ctx,
        chunk_size=int(runtime.get("chunk_size", 21)),
        max_length=int(runtime.get("max_length", 400)),
        min_length=int(runtime.get("min_length", 40)),
        slow_n_gpu_layers=int(runtime.get("slow_n_gpu_layers", -1)),
        fast_n_gpu_layers=int(runtime.get("fast_n_gpu_layers", -1)),
        llama_cpu_threads=int(runtime.get("llama_cpu_threads", 0)),
        llama_cpu_threads_batch=int(runtime.get("llama_cpu_threads_batch", 0)),
        llama_cpu_numa_strategy=str(runtime.get("llama_cpu_numa_strategy", "disabled")),
        decoder_start_method=str(runtime.get("decoder_start_method", "spawn")),
        slow_filename=str(paths.get("slow_filename", "slow_hf_q8_0.gguf")),
        fast_filename=str(paths.get("fast_filename", "fast_hf_q8_0.gguf")),
        codec_encode_filename=str(paths.get("codec_encode_filename", "codec_encode_fp16.onnx")),
        codec_decode_filename=str(paths.get("codec_decode_filename", "codec_decode_stream_fp16.onnx")),
        codec_meta_filename=str(paths.get("codec_meta_filename", "codec_true_streaming_meta_fp16.json")),
        tokenizer_dirname=str(paths.get("tokenizer_dirname", "tokenizer")),
        codec_frame_length=int(codec.get("frame_length", 2048)),
        codec_hop_length=int(codec.get("hop_length", 512)),
        codec_num_codebooks=int(codec.get("num_codebooks", 10)),
        codec_stream_code_frames=int(codec.get("stream_code_frames", runtime.get("chunk_size", 21))),
        generation_sample=S2ProSampleConfig(
            temperature=float(generation_sample.get("temperature", 0.9)),
            top_p=float(generation_sample.get("top_p", 0.8)),
            top_k=int(generation_sample.get("top_k", 30)),
        ),
        ras_sample=S2ProSampleConfig(
            temperature=float(ras_sample.get("temperature", 1.0)),
            top_p=float(ras_sample.get("top_p", 0.9)),
            top_k=int(ras_sample.get("top_k", 30)),
        ),
    )


def _clamp_n_ctx(config: S2ProConfig, n_ctx: int) -> int:
    """
    将 S2-Pro 上下文窗口限制在当前模型配置声明的范围内。

    参数:
        config: 已从 yaml 读取的 S2-Pro 配置，包含上下文窗口边界。
        n_ctx: 配置默认值或引擎入口给出的上下文长度。

    返回:
        返回位于 [config.min_n_ctx, config.max_n_ctx] 区间内的上下文长度。

    异常:
        本函数不主动抛出异常；调用方负责保证输入可转为整数。
    """
    # 先把调用方传入值转成整数，后续比较、日志和返回值都使用同一个规范化输入，避免重复转换产生分歧。
    requested_n_ctx = int(n_ctx)
    # 上下文边界由模型配置提供，避免把当前 S2-Pro 版本的能力假设写死在通用引擎或 runtime 常量里。
    clamped_n_ctx = max(config.min_n_ctx, min(config.max_n_ctx, requested_n_ctx))
    # 当用户输入或 yaml 默认值超出模型声明范围时，仍然按边界保护启动，但必须显式告警，方便定位配置没有生效的原因。
    if clamped_n_ctx != requested_n_ctx:
        logger.warning(
            "S2-Pro n_ctx=%s is outside supported range [%s, %s], using %s",
            requested_n_ctx,
            config.min_n_ctx,
            config.max_n_ctx,
            clamped_n_ctx,
        )
    return clamped_n_ctx


def _ort_provider_available(provider: str) -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return provider in ort.get_available_providers()


def _resolve_llama_backend(target: ExecutionTarget) -> tuple[str, int | None]:
    """
    解析 S2-Pro GGUF 的 llama.cpp backend。

    参数:
        target: adapter 生成的 llama runtime family 目标。

    返回:
        返回 `(backend, gpu_index)`；CPU 路径的 gpu_index 为 None。

    异常:
        target.mode 非 auto/cpu/gpu 时抛出 ValueError。
    """
    resolution = llama_backend_utils.select_llama_backend(
        mode=target.mode,
        device_index=target.device_index,
        model_label="s2-pro",
        explicit_backend=target.backend_preference,
    )
    return resolution.backend, resolution.gpu_index


def _resolve_onnx_backend(target: ExecutionTarget) -> tuple[str, int | None]:
    explicit = resolve_explicit_onnx_backend(target)
    if explicit is not None:
        return explicit
    if target.mode == "cpu":
        return "cpu", None
    if os.name == "nt":
        if _ort_provider_available("CUDAExecutionProvider"):
            return "cuda", target.device_index if target.device_index is not None else 0
        if _ort_provider_available("DmlExecutionProvider"):
            return "dml", target.device_index if target.device_index is not None else 0
        if target.mode == "gpu":
            logger.warning("[Device Fallback][IMPORTANT] 用户请求 S2-Pro decoder GPU，但 DML 不可用，将回退到 CPU")
        return "cpu", None
    if _ort_provider_available("CUDAExecutionProvider"):
        return "cuda", target.device_index if target.device_index is not None else 0
    if target.mode == "gpu":
        logger.warning("[Device Fallback][IMPORTANT] 用户请求 S2-Pro decoder GPU，但 CUDAExecutionProvider 不可用，将回退到 CPU")
    return "cpu", None


def resolve_s2_pro_device(device: str, execution_plan: ExecutionPlan | None) -> S2ProResolvedDevice:
    if execution_plan is None:
        raise ValueError("s2-pro runtime requires an execution plan")
    llama_target = execution_plan.get("llama")
    onnx_target = execution_plan.get("onnx")
    if llama_target is None or onnx_target is None:
        raise ValueError("s2-pro execution plan must include llama and onnx targets")
    gguf_backend, gguf_index = _resolve_llama_backend(llama_target)
    decoder_backend, decoder_index = _resolve_onnx_backend(onnx_target)
    return S2ProResolvedDevice(
        request=device,
        gguf_backend=gguf_backend,
        decoder_backend=decoder_backend,
        gpu_index=decoder_index if decoder_index is not None else gguf_index,
    )


def _available_logical_cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except Exception:
        pass
    return os.cpu_count() or 1


def _detect_cpu_topology() -> CpuTopology:
    # 1. 获取系统当前可用的“逻辑 CPU”总数（如果开启了超线程，这个数字通常是物理核心数的两倍）。
    logical_cpus = _available_logical_cpu_count()
    
    try:
        # 2. os.sched_getaffinity(0) 获取当前进程被允许运行在哪些 CPU 核心上。
        # 很多时候系统或容器会限制进程只能用特定的核心。sorted() 将这些核心的 ID 排个序。
        cpu_ids = sorted(os.sched_getaffinity(0))
    except Exception:
        # 3. 如果获取失败（比如在不支持该 API 的操作系统上，如 Windows），
        # 就直接假设所有逻辑 CPU 都可以使用，生成一个 [0, 1, ..., logical_cpus-1] 的列表。
        cpu_ids = list(range(logical_cpus))
        
    # 4. core_keys 是一个集合（set），用来存放 (物理封装ID, 物理核心ID) 的组合。
    # 集合会自动去重，这能帮我们算出到底有多少个真正的“物理核心”。
    core_keys: set[tuple[int, int]] = set()
    
    # 5. package_to_cores 是一个字典，用来记录每个“物理 CPU 插槽（Socket/Package）”里有哪些物理核心。
    package_to_cores: dict[int, set[int]] = {}
    
    # 6. 开始遍历每一个我们可以使用的 CPU ID。
    for cpu_id in cpu_ids:
        # 7. 构建 Linux 系统下记录 CPU 拓扑信息的目录路径。
        # Linux 会将硬件信息映射为文件系统，也就是 /sys/devices/system/cpu/...
        topo_dir = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        try:
            # 8. 读取 physical_package_id 文件，这代表当前核心属于主板上的哪个物理 CPU 芯片（插槽）。
            package_id = int((topo_dir / "physical_package_id").read_text(encoding="utf-8").strip())
            # 9. 读取 core_id 文件，这代表当前核心在所属 CPU 芯片上的物理编号。
            core_id = int((topo_dir / "core_id").read_text(encoding="utf-8").strip())
        except Exception:
            # 10. 如果读取文件失败（比如系统不是 Linux，或者没有权限），就跳过当前循环，看下一个 ID。
            continue
            
        # 11. 将读取到的 (芯片ID, 物理核心ID) 存入集合。
        # 为什么需要这个？因为超线程技术会让一个“物理核心”虚拟出两个“逻辑 CPU”。
        # 这两个逻辑 CPU 的 (芯片ID, 物理核心ID) 是一模一样的，通过 set 可以完美去重，只算一个物理核心。
        core_keys.add((package_id, core_id))
        
        # 12. 记录这个物理 CPU 芯片（package_id）下拥有这个物理核心（core_id）。
        package_to_cores.setdefault(package_id, set()).add(core_id)
        
    # 13. 如果 core_keys 里面有数据，说明我们成功从 Linux 文件系统中读取到了底层拓扑信息。
    if core_keys:
        # 14. 计算“单个 CPU 插槽中最多的物理核心数”。
        # 比如你是双路 CPU，CPU_1 有 8 核，CPU_2 也有 8 核，这里就会算出 8。
        physical_per_socket = max((len(cores) for cores in package_to_cores.values()), default=len(core_keys))
        
        # 15. 返回精准的硬件拓扑信息对象：(逻辑CPU数, 总物理核心数, CPU插槽数, 单插槽物理核心数)
        return CpuTopology(logical_cpus, len(core_keys), len(package_to_cores), physical_per_socket)
        
    # 16. 如果前面读取 Linux 文件系统失败了（比如在 Windows/Mac 上），就走“估算”逻辑。
    # 估算方法很简单：假设所有的机器都开启了超线程，所以把“逻辑CPU数”除以 2（向下取整），作为物理核心数的估算值。
    estimated = max(1, logical_cpus // 2)
    
    # 17. 返回估算的拓扑信息：单插槽，物理核心数是估算出来的。
    return CpuTopology(logical_cpus, estimated, 1, estimated)


def _resolve_llama_cpu_thread_plan(config: S2ProConfig, resolved_device: S2ProResolvedDevice) -> LlamaCpuThreadPlan:
    # 1. 调用第一个函数，获取 CPU 拓扑信息。
    topology = _detect_cpu_topology()
    
    # 2. 如果检测到当前模型是跑在 GPU 上的（gguf_gpu_enabled）。
    if resolved_device.gguf_gpu_enabled:
        # 3. 那就不需要 CPU 参与主力计算了，直接返回 CPU 线程数为 0 的方案。
        return LlamaCpuThreadPlan(0, 0, "gpu", topology)
        
    # 4. 尝试从系统的环境变量中读取用户强制指定的线程数（普通推理线程 和 批处理线程）。
    env_threads = int(os.getenv("STREAMVOX_S2_PRO_LLAMA_CPU_THREADS", "0") or "0")
    env_threads_batch = int(os.getenv("STREAMVOX_S2_PRO_LLAMA_CPU_THREADS_BATCH", "0") or "0")
    
    # --- 下面开始确定“普通推理线程数” (n_threads) ---
    if env_threads > 0:
        # 5. 优先级最高：如果环境变量配置了，就用环境变量的值。
        # 但为了安全，不能超过系统本身的逻辑 CPU 总数（用 min 函数限制）。
        n_threads = min(env_threads, topology.logical_cpus)
        source = "env" # 标记来源为环境变量
    elif config.llama_cpu_threads > 0:
        # 6. 优先级次之：如果代码配置对象 (config) 里指定了，就用 config 的值。
        n_threads = min(config.llama_cpu_threads, topology.logical_cpus)
        source = "config" # 标记来源为配置文件
    else:
        # 7. 优先级最低（默认行为）：自动推断。
        # 重点来了：这里使用了拓扑信息里的 `physical_cores_per_socket`（单插槽物理核心数）。
        # 这正是为了避免超线程和跨插槽导致的性能损耗，是 llama.cpp 推荐的性能最优解。
        n_threads = min(topology.physical_cores_per_socket, topology.logical_cpus)
        source = "auto-topology" # 标记来源为自动拓扑计算
        
    # --- 下面开始确定“批处理线程数” (n_threads_batch) ---
    # （批处理通常发生在刚输入一大段提示词需要消化时，有时能利用比普通生成更多的线程）
    if env_threads_batch > 0:
        # 8. 如果环境变量指定了批处理线程数，使用它（同样不能超过系统上限）。
        n_threads_batch = min(env_threads_batch, topology.logical_cpus)
    elif config.llama_cpu_threads_batch > 0:
        # 9. 如果配置文件指定了，使用它。
        n_threads_batch = min(config.llama_cpu_threads_batch, topology.logical_cpus)
    else:
        # 10. 如果都没指定，那么批处理的线程数就和普通推理的线程数保持一致。
        n_threads_batch = n_threads
        
    # 11. 最后返回决定好的线程计划。
    # max(1, n_threads): 确保至少有 1 个线程在干活。
    # max(n_threads, n_threads_batch): 确保批处理的线程数绝对不会少于普通推理的线程数。
    return LlamaCpuThreadPlan(max(1, n_threads), max(n_threads, n_threads_batch), source, topology)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def sample_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
    allow_range: tuple[int, int] | None = None,
    allow_tokens: list[int] | None = None,
) -> int:
    """
    从 logits 中按温度、top-p、top-k 规则采样一个 token。

    参数:
        logits: 当前模型输出的原始 logits，长度等于候选词表或 codebook 大小。
        rng: 当前会话共用的随机数生成器，用于保证同一次合成内采样序列可复现。
        temperature: 温度参数；小于等于 0 时退化为贪心选择。
        top_p: nucleus sampling 阈值，只保留累计概率达到该阈值所需的高概率候选。
        top_k: 只保留概率最高的 k 个候选；小于等于 0 时不启用 top-k 截断。
        allow_range: 可选的连续 token 白名单区间，用于限制 slow 只能采样 semantic token。
        allow_tokens: 额外允许的离散 token，例如 `<|im_end|>` 停止符不在 semantic 连续区间内也要保留。

    返回:
        返回最终采样到的 token id 或 code id。

    异常:
        当 numpy 概率归一化或随机采样失败时会透出对应异常。
    """
    # 词表大小来自 logits 第一维；后续所有候选索引都必须落在这个范围内。
    vocab_size = int(logits.shape[0])
    # 如果调用方提供了连续白名单区间，只允许该区间内的 token 参与采样。
    if allow_range is not None:
        start, end = allow_range
        base_indices = np.arange(max(0, start), min(vocab_size - 1, end) + 1, dtype=np.int64)
    # 未限制 token 范围时，默认整张 logits 都是合法候选。
    else:
        base_indices = np.arange(vocab_size, dtype=np.int64)
        start, end = 0, vocab_size - 1
    # allow_tokens 用来补充离散合法 token，典型场景是 slow semantic 区间之外的结束符。
    extras = []
    for token in allow_tokens or []:
        token = int(token)
        if 0 <= token < vocab_size and (allow_range is None or not (start <= token <= end)):
            extras.append(token)
    # 将连续白名单和额外离散 token 合并成最终候选集合，再从 logits 中取出对应分数。
    candidate_indices = np.concatenate([base_indices, np.asarray(extras, dtype=np.int64)]) if extras else base_indices
    candidate_logits = logits[candidate_indices].astype(np.float32, copy=False)
    # top-k 先做粗裁剪，减少低概率候选进入后续排序和 top-p 计算。
    if 0 < top_k < candidate_logits.shape[0]:
        topk_order = np.argpartition(candidate_logits, -top_k)[-top_k:]
        candidate_logits = candidate_logits[topk_order]
        candidate_indices = candidate_indices[topk_order]
    # top-p 需要按概率从高到低累计，因此先把候选 logits 降序排列。
    order = np.argsort(candidate_logits)[::-1]
    sorted_logits = candidate_logits[order]
    sorted_indices = candidate_indices[order]
    sorted_probs = _softmax(sorted_logits)
    # nucleus sampling 保留累计概率覆盖 top_p 的最小前缀，并至少保留一个候选避免空集合。
    if top_p < 1.0:
        keep_count = int(np.searchsorted(np.cumsum(sorted_probs), top_p, side="right")) + 1
        sorted_logits = sorted_logits[: max(1, min(keep_count, sorted_logits.shape[0]))]
        sorted_indices = sorted_indices[: sorted_logits.shape[0]]
    # temperature <= 0 表示不采样，直接选择当前候选里 logits 最大的 token。
    if temperature <= 0:
        return int(sorted_indices[0])
    # 温度缩放后重新归一化概率，最后用同一个 rng 抽样，保证 slow 和 fast 的随机过程受同一 seed 控制。
    probs = _softmax(sorted_logits / max(temperature, 1e-5)).astype(np.float64, copy=False)
    probs /= probs.sum()
    return int(sorted_indices[int(rng.choice(sorted_indices.shape[0], p=probs))])


class SlowRuntime:
    def __init__(
        self,
        gguf_path: Path,
        tokenizer: InferenceTokenizer,
        assets: AssetsManager,
        num_codebooks: int,
        codebook_size: int,
        n_ctx: int,
        n_gpu_layers: int,
        main_gpu: int,
        gguf_backend: str,
        n_threads: int | None,
        n_threads_batch: int | None,
    ) -> None:
        """
        初始化 S2-Pro slow GGUF 推理器。

        参数:
            gguf_path: slow 模型 GGUF 文件路径。
            tokenizer: slow 文本 tokenizer，用于 system/reference/target 文本编码。
            assets: S2-Pro 共享资产管理器，提供 embedding、codebook 等辅助张量。
            num_codebooks: codec codebook 数量，用于构造 slow 输入 embedding。
            codebook_size: 每个 codebook 的离散 token 空间大小。
            n_ctx: slow llama.cpp 上下文窗口大小。
            n_gpu_layers: 需要 offload 到 GPU 的层数；CPU 回退时传 0。
            main_gpu: llama.cpp 主 GPU 索引。
            gguf_backend: runtime 已解析出的 GGUF 后端，用于驱动动态 backend 注册。
            n_threads: CPU 推理线程数；为 None 时交给 llama.cpp 默认策略。
            n_threads_batch: CPU batch 线程数；为 None 时交给 llama.cpp 默认策略。

        返回:
            构造函数无显式返回；成功后持有 slow 模型和上下文。

        边界异常:
            GGUF 模型加载或上下文初始化失败时直接抛出异常，由外层 runtime 决定是否 CPU 回退。
        """
        self.model = llama.LlamaModel(
            str(gguf_path),
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
            split_mode=0,
            backend=gguf_backend,
        )
        self.context = llama.LlamaContext(
            self.model,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            embeddings=True,
            flash_attn=True,
            offload_kqv=True,
            no_perf=True,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
        )
        self.tokenizer = tokenizer
        self.assets = assets
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.scale = float(np.sqrt(num_codebooks + 1))
        self.n_embd = self.assets.text_embedding_dim
        self.im_end_id = tokenizer.token_id(IM_END_TOKEN) # 停止符
        self.vocab_size = self.model.vocab_size

    def shutdown(self) -> None:
        """
        关闭 slow runtime 持有的 native llama.cpp 资源。

        参数:
            本函数不接收参数。

        返回:
            无返回值；Vulkan 路径会主动断开 native 指针，非 Vulkan 路径交给对象析构释放。

        边界异常:
            shutdown 之后当前 runtime 不能继续推理。
        """
        # 当前 Vulkan 发布包默认仍保留 detach 兜底；诊断开关用于验证真实 native free 是否能解决进程退出阶段析构崩溃。
        if getattr(self.model, "backend", None) == "vulkan":
            if os.environ.get("S2_PRO_FREE_VULKAN_NATIVE", "0") == "1":
                self.context.free_native(force=True)
                self.model.free_native(force=True)
            else:
                self.context.detach_native()
                self.model.detach_native()
        # 释放顺序必须是 context 先、model 后；context 内部还持有 model 引用，反向清理会触发 Vulkan backend 退出崩溃。
        self.context = None
        self.model = None

    def clear(self) -> None:
        self.context.clear_kv_cache()

    def build_embeddings(self, values: np.ndarray) -> np.ndarray:
        """
        将 S2-Pro 的多行 token/codebook 矩阵转换为 slow 模型输入 embedding。

        参数:
            values: 形状为 `(num_codebooks + 1, T)` 的输入矩阵；第 0 行是文本或 semantic token，其余行是声学 codebook。

        返回:
            返回形状为 `(T, hidden_dim)` 的 slow 输入 embedding。

        异常:
            当 token id 或 code id 越界时会透出底层 embedding 数组索引异常。
        """
        # S2-Pro 原始模型输入不是 token id，而是 text embedding 与各 codebook embedding 的组合。
        main = values[0]
        # 第 0 行所有 token 先统一查文本 embedding；普通文本位置后续不会叠加声学 codebook。
        x = self.assets.get_text_embeddings(main).copy()
        # 只有 semantic token 位置才代表上一帧 speech token，需要把各层 acoustic codebook embedding 融入 slow 输入。
        semantic_mask = (main >= self.tokenizer.semantic_begin_id) & (main <= self.tokenizer.semantic_end_id)
        # 没有 semantic token 时，说明本段全是纯文本控制 token，直接返回文本 embedding。
        if values.shape[1] and semantic_mask.any():
            # codebook embedding 逐层累加，得到当前 semantic token 对应的完整声学条件。
            vq_sum = np.zeros_like(x)
            for i in range(self.num_codebooks):
                vq_sum += self.assets.get_codebook_embeddings(i, values[i + 1])
            # 仅 semantic 位置融合 acoustic embedding，避免污染 system/user/assistant 等普通文本 token。
            x[semantic_mask] += vq_sum[semantic_mask]
            # 按官方结构对 text+codebook embedding 做缩放，保持输入幅度和训练时一致。
            x[semantic_mask] /= self.scale
        return x

    def _decode_embeddings(self, embeddings: np.ndarray, positions: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """
        将一段 embedding 写入 slow llama context，并取回最后一个位置的 logits 和 hidden。

        参数:
            embeddings: 已经由 build_embeddings 构造好的 slow 输入 embedding。
            positions: 这些 embedding 在当前 llama context 中的绝对位置。

        返回:
            返回最后一个 token 位置的 logits 和 hidden state；logits 用于采样下一个 semantic token，hidden 交给 fast 模型生成 codebook。

        异常:
            当 llama_decode 返回非零状态码时抛出 RuntimeError。
        """
        # llama.cpp 的 embedding 模式需要显式构造 batch，并把每个 embedding 写到指定 position。
        batch = llama.LlamaBatch(len(embeddings), embd_dim=self.n_embd, n_seq_max=1)
        batch.set_embd(embeddings.astype(np.float32), pos=positions, seq_id=0)
        # decode 会推进当前 slow context 的 KV cache；失败时不能继续采样，否则上下文状态不可信。
        rc = self.context.decode(batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")
        # slow logits 只取最后一个位置，代表“在当前上下文后应该生成的下一个 semantic token”分布。
        logits = np.ctypeslib.as_array(self.context.get_logits_ith(-1), shape=(self.vocab_size,)).copy()
        # hidden 同样取最后一个位置，它是 fast 模型生成当前帧多层 acoustic codebook 的条件。
        hidden = np.ctypeslib.as_array(self.context.get_embeddings_ith(-1), shape=(self.n_embd,)).copy()
        return logits, hidden

    def prefill(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        从空 KV cache 开始预填充一段 slow prompt。

        参数:
            values: 需要完整写入 slow context 的 prompt 输入矩阵。

        返回:
            返回 prompt 最后位置的 logits 和 hidden。

        异常:
            当 embedding 构造或 llama decode 失败时透出对应异常。
        """
        # prefill 必须从干净上下文开始，否则新 prompt 会被旧会话历史污染。
        self.clear()
        # prompt 位置从 0 开始连续写入，为后续 snapshot_state 保存完整 reference prompt KV cache。
        return self._decode_embeddings(self.build_embeddings(values), np.arange(values.shape[1], dtype=np.int32))

    def decode_values(self, values: np.ndarray, start_position: int) -> tuple[np.ndarray, np.ndarray]:
        """
        在已有 slow context 后追加一段输入矩阵。

        参数:
            values: 要追加写入的输入矩阵，通常是当前 batch 的 user/assistant 前缀。
            start_position: 追加段在 llama context 中的起始绝对位置。

        返回:
            返回追加段最后位置的 logits 和 hidden。

        异常:
            当 embedding 构造或 llama decode 失败时透出对应异常。
        """
        # 追加段必须接在 prefix_state 对应位置之后，保持 llama KV cache 的 position 连续。
        positions = np.arange(start_position, start_position + values.shape[1], dtype=np.int32)
        return self._decode_embeddings(self.build_embeddings(values), positions)

    def snapshot_state(self) -> bytes:
        return self.context.get_state()

    def restore_state(self, state: bytes) -> None:
        # 恢复快照前先清空当前 KV cache，避免不同 role 或不同 batch 的历史状态混在一起。
        self.context.clear_kv_cache()
        # state 是 make_prompt 或 batch 结束时保存的 llama context 二进制快照。
        self.context.set_state(state)

    def step(self, semantic_token: int, codes: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        """
        将刚生成的一帧 semantic token 和 acoustic codes 写回 slow context。

        参数:
            semantic_token: slow 刚采样出的 semantic token。
            codes: fast 根据该 semantic token 生成的多层 acoustic codebook。
            position: 当前帧在 llama context 中的绝对位置。

        返回:
            返回写入该帧后最后位置的 logits 和 hidden，用于采样下一帧 semantic token。

        异常:
            当 embedding 构造或 llama decode 失败时透出对应异常。
        """
        # slow 的单帧输入仍保持 `(num_codebooks + 1, 1)` 结构，第 0 行放 semantic token。
        values = np.zeros((self.num_codebooks + 1, 1), dtype=np.int64)
        values[0, 0] = semantic_token
        # 其余行放 fast 生成的 acoustic codebook，使 slow 下一步能看到刚生成的音频历史。
        values[1:, 0] = codes
        return self._decode_embeddings(self.build_embeddings(values), np.asarray([position], dtype=np.int32))


class FastRuntime:
    def __init__(
        self,
        gguf_path: Path,
        assets: AssetsManager,
        codebook_size: int,
        n_gpu_layers: int,
        main_gpu: int,
        gguf_backend: str,
        n_threads: int | None,
        n_threads_batch: int | None,
    ) -> None:
        """
        初始化 S2-Pro fast GGUF 推理器。

        参数:
            gguf_path: fast 模型 GGUF 文件路径。
            assets: S2-Pro 共享资产管理器，提供 fast 输入所需的 embedding 资产。
            codebook_size: 每个 codebook 的离散 token 空间大小。
            n_gpu_layers: 需要 offload 到 GPU 的层数；CPU 回退时传 0。
            main_gpu: llama.cpp 主 GPU 索引。
            gguf_backend: runtime 已解析出的 GGUF 后端，用于驱动动态 backend 注册。
            n_threads: CPU 推理线程数；为 None 时交给 llama.cpp 默认策略。
            n_threads_batch: CPU batch 线程数；为 None 时交给 llama.cpp 默认策略。

        返回:
            构造函数无显式返回；成功后持有 fast 模型和上下文。

        边界异常:
            GGUF 模型加载或上下文初始化失败时直接抛出异常，由外层 runtime 决定是否 CPU 回退。
        """
        self.model = llama.LlamaModel(
            str(gguf_path),
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
            split_mode=0,
            backend=gguf_backend,
        )
        self.context = llama.LlamaContext(
            self.model,
            n_ctx=16,
            n_batch=1,
            embeddings=True,
            flash_attn=False,
            offload_kqv=True,
            no_perf=True,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
        )
        self.assets = assets
        self.codebook_size = codebook_size
        self.n_embd = self.assets.fast_embedding_dim

    def shutdown(self) -> None:
        """
        关闭 fast runtime 持有的 native llama.cpp 资源。

        参数:
            本函数不接收参数。

        返回:
            无返回值；Vulkan 路径会主动断开 native 指针，非 Vulkan 路径交给对象析构释放。

        边界异常:
            shutdown 之后当前 runtime 不能继续推理。
        """
        # fast 与 slow 同时存在时会触发当前 Vulkan backend 的退出析构问题，释放策略必须保持一致。
        if getattr(self.model, "backend", None) == "vulkan":
            if os.environ.get("S2_PRO_FREE_VULKAN_NATIVE", "0") == "1":
                self.context.free_native(force=True)
                self.model.free_native(force=True)
            else:
                self.context.detach_native()
                self.model.detach_native()
        # 释放顺序必须是 context 先、model 后；fast/slow 都遵循同一 native 生命周期约束。
        self.context = None
        self.model = None

    def clear(self) -> None:
        self.context.clear_kv_cache()

    def _decode_one(self, embedding: np.ndarray, position: int) -> np.ndarray:
        batch = llama.LlamaBatch(1, embd_dim=self.n_embd, n_seq_max=1)
        batch.set_embd(embedding.reshape(1, -1).astype(np.float32), pos=np.asarray([position], dtype=np.int32), seq_id=0)
        rc = self.context.decode(batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")
        return np.ctypeslib.as_array(self.context.get_logits_ith(-1), shape=(self.codebook_size,)).copy()

    def generate_codes(
        self,
        semantic_token: int,
        hidden_state: np.ndarray,
        semantic_begin_id: int,
        num_codebooks: int,
        temperature: float,
        top_p: float,
        top_k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        根据一个 slow semantic token 生成对应的一帧多层 acoustic codebook。

        参数:
            semantic_token: slow 模型采样出的当前帧 semantic token。
            hidden_state: slow 模型当前上下文最后位置的 hidden state，作为 fast 第 0 步条件。
            semantic_begin_id: semantic token 连续区间起点，用于把 semantic token 映射到第一层 code id。
            num_codebooks: 需要生成的 acoustic codebook 层数。
            temperature: fast 采样温度。
            top_p: fast nucleus sampling 阈值。
            top_k: fast top-k 阈值。
            rng: 当前会话共用随机数生成器。

        返回:
            返回形状为 `(num_codebooks,)` 的 acoustic code id 数组。

        异常:
            当 fast llama decode 或采样失败时透出对应异常。
        """
        # fast 模型只负责当前一帧 codebook 的自回归展开，每帧开始前必须清空它自己的短上下文。
        self.clear()
        # 第一层 code 与 semantic token 直接对齐；超出范围时裁剪到 codebook 合法区间作为保护。
        first_code = int(np.clip(semantic_token - semantic_begin_id, 0, self.codebook_size - 1))
        # 第 0 个 fast 输入是 slow hidden，用来把文本/历史语义条件注入 fast codebook 生成过程。
        self._decode_one(hidden_state, 0)
        codes = [first_code]
        prev_code = first_code
        # 后续 codebook 层逐层自回归：上一层 code 的 embedding 作为下一层输入。
        for position in range(1, num_codebooks):
            logits = self._decode_one(self.assets.get_fast_embedding(prev_code), position)
            # 每一层 codebook 都在自己的 codebook_size 词表内采样，采样参数沿用生成配置。
            prev_code = sample_logits(logits, rng, temperature, top_p, top_k)
            codes.append(prev_code)
        return np.asarray(codes, dtype=np.int64)


class S2ProRuntime:
    model_name = S2_PRO_4B_GGUF_SPEC.model_name

    def __init__(
        self,
        model_dir: Path,
        model_spec: S2ProModelSpec | None = None,
        device: str = "auto",
        n_ctx: int | None = None,
        verify_model_sha256: bool = True,
        execution_plan: ExecutionPlan | None = None,
        text_preprocess_config: TextPreprocessConfig | None = None,
    ) -> None:
        start = time.time()
        self.model_spec = model_spec or S2_PRO_4B_GGUF_SPEC
        self.model_name = self.model_spec.model_name
        self.bundle_root = Path(model_dir).resolve()
        self.device = device
        self.execution_plan = execution_plan
        self.text_preprocess_config = text_preprocess_config or S2_PRO_TEXT_PREPROCESS_CONFIG
        self.config = load_s2_pro_config()
        # 引擎显式传入 n_ctx 时优先使用用户意图；未传时使用 yaml 的 default_n_ctx，并按当前模型自己的边界裁剪。
        self.config.n_ctx = _clamp_n_ctx(self.config, n_ctx if n_ctx is not None else self.config.n_ctx)
        self.resolved_device = resolve_s2_pro_device(device, execution_plan)
        self.llama_cpu_thread_plan = _resolve_llama_cpu_thread_plan(self.config, self.resolved_device)
        self.ckpt_root = self.bundle_root / "ckpt"
        validate_local_bundle(
            self.bundle_root,
            self.model_name,
            verify_model_sha256=verify_model_sha256,
        )
        # 初始化预热只允许真正执行一次；重复调用直接返回，避免重复污染 decoder 和 GGUF 的启动统计。
        self._init_warmed = False
        self.paths = {
            "slow_gguf": self.ckpt_root / self.config.slow_filename,
            "fast_gguf": self.ckpt_root / self.config.fast_filename,
            "tokenizer": self.ckpt_root / self.config.tokenizer_dirname,
            "codec_encode": self.ckpt_root / self.config.codec_encode_filename,
            "codec_decode": self.ckpt_root / self.config.codec_decode_filename,
            "codec_meta": self.ckpt_root / self.config.codec_meta_filename,
        }
        self._validate_assets()
        self._init_runtime()
        asset_count = sum(1 for path in self.bundle_root.rglob("*") if path.is_file())
        self.runtime_info = RuntimeInfo(
            model=self.model_name,
            adapter=self.model_spec.adapter_name,
            bundle_loaded=True,
            asset_count=asset_count,
            metadata={
                "bundle_root": str(self.bundle_root),
                "device_request": str(self.device),
                "resolved_device": self.resolved_device.normalized,
                "gguf_backend": self.resolved_device.gguf_backend,
                "decoder_backend": self.resolved_device.decoder_backend,
                "init_seconds": f"{time.time() - start:.3f}",
            },
        )

    @property
    def sample_rate(self) -> int:
        return self.codec_config.sample_rate

    def _validate_assets(self) -> None:
        # dir，文件夹存在，path.exists()会返回True
        missing = [name for name, path in self.paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing s2-pro runtime assets under {self.bundle_root}: {missing}")

    def _providers_for_decoder(self) -> list[str]:
        if self.resolved_device.decoder_backend == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.resolved_device.decoder_backend == "dml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _init_runtime(self) -> None:
        self.codec_config = CodecOnnxConfig(
            sample_rate=self.config.sample_rate,
            frame_length=self.config.codec_frame_length,
            hop_length=self.config.codec_hop_length,
            num_codebooks=self.config.codec_num_codebooks,
            stream_code_frames=self.config.codec_stream_code_frames,
        )
        # 主进程初始化 encoder，确保 TTSEngine 初始化完成前 codec encoder 已经就绪。
        self.codec_encoder = TrueStreamingCodecEncoderOrtRuntime(
            self.ckpt_root,
            providers=self._providers_for_decoder(),
            encode_name=self.config.codec_encode_filename,
            meta_name=self.config.codec_meta_filename,
        )
        self.tokenizer = InferenceTokenizer(str(self.paths["tokenizer"]))
        self.assets = AssetsManager(self.ckpt_root)

        self.num_codebooks = int(self.codec_config.num_codebooks)
        self.codebook_size = self.assets.codebook_size
        self._load_gguf_runtimes()
        self.decoder = S2ProDecoderAPI(
            self.ckpt_root,
            providers=self._providers_for_decoder(),
            decode_name=self.config.codec_decode_filename,
            meta_name=self.config.codec_meta_filename,
            start_method=self.config.decoder_start_method,
        )
        if not self.decoder.wait_until_ready(timeout=60):
            raise RuntimeError("s2-pro decoder process failed to become ready")

    def _load_gguf_runtimes(self) -> None:
        """
        按已解析的 GGUF 后端加载 S2-Pro slow/fast 两个 llama.cpp 模型。

        参数:
            无显式参数；使用 self.resolved_device、self.config、self.paths 等 runtime 初始化状态。

        返回:
            无返回值；成功后会填充 self.slow 和 self.fast。

        边界异常:
            GPU 后端加载失败时会记录异常并回退 CPU 重试；CPU 重试仍失败时异常继续抛出。
        """
        # S2-Pro 的 device 自动分配结果已经落在 resolved_device，这里只负责把结果传给 llama 绑定层执行。
        try:
            self._create_gguf_runtimes_from_resolved_device()
            return
        except Exception:
            if not self.resolved_device.gguf_gpu_enabled:
                raise
            logger.exception(
                "[Device Fallback][IMPORTANT] S2-Pro llama.cpp %s 后端加载模型失败，将强制回退到 CPU 重新加载",
                self.resolved_device.gguf_backend,
            )

        # GPU 后端失败只影响 GGUF slow/fast；ONNX codec/decoder 的后端继续使用原先解析结果，避免误伤独立可用的 CUDA/DML 解码。
        # 如果 slow 已经加载成功但 fast 失败，先断开旧 runtime 引用，让 native 模型句柄尽早释放后再走 CPU 重试。
        self.slow = None
        self.fast = None
        self.resolved_device.gguf_backend = "cpu"
        self.llama_cpu_thread_plan = _resolve_llama_cpu_thread_plan(self.config, self.resolved_device)
        self._create_gguf_runtimes_from_resolved_device()

    def _create_gguf_runtimes_from_resolved_device(self) -> None:
        """
        基于当前 resolved_device 创建 slow/fast GGUF runtime。

        参数:
            无显式参数；当前方法只消费对象内已经完成校验的配置和资产路径。

        返回:
            无返回值；成功后 self.slow/self.fast 指向可用 runtime。

        边界异常:
            任一 GGUF 模型或上下文初始化失败时直接抛出，交由外层决定是否 CPU 回退。
        """
        # GPU 索引为空时按 llama.cpp 单卡默认值 0 处理；CPU 模式下该值不会参与实际 offload。
        gpu_index = 0 if self.resolved_device.gpu_index is None else self.resolved_device.gpu_index
        # GPU 层数只在 GGUF 后端确认为 GPU 时启用；CPU 回退会强制置 0，避免 llama.cpp 继续尝试设备 offload。
        slow_layers = self.config.slow_n_gpu_layers if self.resolved_device.gguf_gpu_enabled else 0
        fast_layers = self.config.fast_n_gpu_layers if self.resolved_device.gguf_gpu_enabled else 0
        # CPU 线程计划只在 CPU 路径有业务意义；GPU 路径保留 None，让 llama.cpp 使用自己的默认调度。
        n_threads = self.llama_cpu_thread_plan.n_threads or None
        n_threads_batch = self.llama_cpu_thread_plan.n_threads_batch or None
        self.slow = SlowRuntime(
            self.paths["slow_gguf"],
            self.tokenizer,
            self.assets,
            self.num_codebooks,
            self.codebook_size,
            self.config.n_ctx,
            slow_layers,
            gpu_index,
            self.resolved_device.gguf_backend,
            n_threads,
            n_threads_batch,
        )
        self.fast = FastRuntime(
            self.paths["fast_gguf"],
            self.assets,
            self.codebook_size,
            fast_layers,
            gpu_index,
            self.resolved_device.gguf_backend,
            n_threads,
            n_threads_batch,
        )

    def create_session(self) -> "S2ProSession":
        return S2ProSession(self)

    def warmup_for_init(self) -> None:
        """
        在 S2-Pro runtime 初始化完成后，主动走一遍 ONNX 与 GGUF 的最小合法预热路径。

        参数:
            本函数不接收显式参数；依赖当前 runtime 已完成 codec encoder、decoder、slow/fast GGUF 的加载。

        返回:
            无返回值；预热成功后会把 `_init_warmed` 标记为 True。

        边界异常:
            任一 ONNX/GGUF 预热步骤失败都会直接抛出异常，阻止 TTSEngine 进入“半初始化成功”状态。
        """
        # 幂等保护必须放在入口处；TTSEngine 和单测都可能重复调用该钩子，但真实预热只应执行一次。
        if self._init_warmed:
            return
        logger.info("[s2-pro] 开始初始化预热：ONNX + GGUF")
        # 先走 ONNX encoder 最短合法路径：codec encoder 只要求固定的 [1, 1, frame_length] 音频结构。
        warmup_audio = np.zeros((1, 1, self.config.codec_frame_length), dtype=np.float32)
        self.codec_encoder.encode_audio(warmup_audio)

        # decoder 预热不依赖真实语义，只要求一块满足 true-streaming 形状约束的全零 codes。
        warmup_codes = np.zeros(
            (self.num_codebooks, self.config.codec_stream_code_frames),
            dtype=np.int64,
        )
        self.decoder.prewarm_decoder_state(
            warmup_codes,
            task_id="engine-init-warmup",
        )

        # 选一个稳定存在且单 token 的文本控制 token 作为 slow 最小输入；
        # 若词表布局变动导致该 token 不可用，再回退到普通文本编码结果的首个 token。
        min_text_token = int(self.tokenizer.token_id(IM_END_TOKEN))
        if min_text_token < 0:
            encoded_tokens = self.tokenizer.encode("a")
            if not encoded_tokens:
                raise RuntimeError("s2-pro warmup could not resolve a minimal text token")
            min_text_token = int(encoded_tokens[0])

        # GGUF 预热的目标只是把 slow/fast 的关键路径拉热，不应把预热状态残留给真实请求。
        try:
            slow_values = np.zeros((self.num_codebooks + 1, 1), dtype=np.int64)
            slow_values[0, 0] = min_text_token
            self.slow.prefill(slow_values)

            # fast 路径不要求真实 slow hidden；这里只需要满足 embedding runner 的最小输入约束。
            fast_hidden_state = np.zeros((self.fast.n_embd,), dtype=np.float32)
            self.fast.generate_codes(
                semantic_token=int(self.tokenizer.semantic_begin_id),
                hidden_state=fast_hidden_state,
                semantic_begin_id=int(self.tokenizer.semantic_begin_id),
                num_codebooks=self.num_codebooks,
                temperature=float(self.config.generation_sample.temperature),
                top_p=float(self.config.generation_sample.top_p),
                top_k=int(self.config.generation_sample.top_k),
                rng=np.random.default_rng(0),
            )
        finally:
            # 无论 GGUF 预热成功还是失败，都要把 slow/fast 的 KV cache 立即清干净。
            self.slow.clear()
            self.fast.clear()
        logger.info("[s2-pro] 初始化预热完成")
        self._init_warmed = True

    def shutdown(self) -> None:
        if hasattr(self, "decoder") and self.decoder is not None:
            self.decoder.shutdown()
            self.decoder = None
        self.codec_encoder = None
        if hasattr(self, "slow") and self.slow is not None:
            self.slow.shutdown()
        if hasattr(self, "fast") and self.fast is not None:
            self.fast.shutdown()
        self.slow = None
        self.fast = None
        # S2-Pro Vulkan 的 native 全局析构器在解释器退出阶段不稳定；显式 shutdown 完成后安装退出保护。
        llama.arm_vulkan_exit_guard()
        # llama.cpp 动态 backend 在 Vulkan 双 GGUF 场景下需要显式卸载，否则进程退出阶段可能触发 native 析构崩溃。
        llama.unload_dynamic_backends()


class S2ProSession:
    def __init__(self, runtime: S2ProRuntime) -> None:
        self.runtime = runtime
        self.decoder = runtime.decoder

    def _sample_semantic_token(
        self,
        logits: np.ndarray,
        sample_config: S2ProSampleConfig,
        ras_config: S2ProSampleConfig,
        rng: np.random.Generator,
        semantic_history: list[int],
    ) -> int:
        """
        从 slow logits 中采样下一帧 semantic token，并在重复时触发 RAS 退避采样。

        参数:
            logits: slow 模型在当前上下文最后位置输出的 token 分布。
            sample_config: 常规 semantic token 采样参数。
            ras_config: 重复惩罚采样参数；只在短窗口重复时使用。
            rng: 当前会话共用随机数生成器。
            semantic_history: 当前 batch 已经生成的 semantic token 历史，用于检测短期重复。

        返回:
            返回采样出的 semantic token；可能是普通 semantic token，也可能是 `<|im_end|>` 结束符。

        异常:
            当 logits 采样失败时透出 sample_logits 的异常。
        """
        # slow 只允许生成 semantic token 连续区间，另外显式允许 `<|im_end|>` 作为当前 assistant 轮次的结束信号。
        semantic = sample_logits(
            logits,
            rng,
            sample_config.temperature,
            sample_config.top_p,
            sample_config.top_k,
            allow_range=(self.runtime.tokenizer.semantic_begin_id, self.runtime.tokenizer.semantic_end_id),
            allow_tokens=[self.runtime.slow.im_end_id],
        )
        # 如果非结束符 semantic token 在最近 10 步内重复，说明语义流可能开始打转，触发 RAS 参数重新采样一次。
        if semantic != self.runtime.slow.im_end_id and semantic in semantic_history[-10:]:
            # RAS 只在 slow 语义 token 重复时启用，使用独立配置避免影响常规 slow/fast 采样。
            semantic = sample_logits(
                logits,
                rng,
                ras_config.temperature,
                ras_config.top_p,
                ras_config.top_k,
                allow_range=(self.runtime.tokenizer.semantic_begin_id, self.runtime.tokenizer.semantic_end_id),
                allow_tokens=[self.runtime.slow.im_end_id],
            )
        # 无论是否触发 RAS，都把最终 token 写入历史，供后续短窗口重复检测使用。
        semantic_history.append(semantic)
        return semantic

    def _generate_codes_stream_from_context(
        self,
        prefix_position: int,
        suffix_values: np.ndarray,
        generation_sample: S2ProSampleConfig,
        ras_sample: S2ProSampleConfig,
        rng: np.random.Generator,
        timing: S2ProTiming | None = None,
    ) -> Iterator[CodeStepResult]:
        """
        从当前 slow llama context 继续追加当前 batch，并流式生成 acoustic code 帧。

        参数:
            prefix_position: 当前 llama context 中下一写入位置。
            suffix_values: 当前 batch 的 user 文本与 assistant voice 前缀矩阵。
            generation_sample: 常规 semantic token 与 fast codebook 采样参数。
            ras_sample: semantic token 短期重复时使用的 RAS 采样参数。
            rng: 当前会话共用随机数生成器。
            timing: 可选性能统计容器；传入时记录 slow/fast/采样等关键阶段耗时。

        返回:
            逐帧 yield CodeStepResult；每个结果包含一帧 acoustic codebook 和是否到达结束符的信息。

        异常:
            当 slow decode、fast code 生成或采样失败时透出对应异常。
        """
        # 调用方保证 slow context 已经位于 prefix_position 对应的历史末尾；这里直接追加当前 user/assistant 前缀，避免每批次恢复快照。
        slow_prefill_start = time.time()
        logits, hidden = self.runtime.slow.decode_values(suffix_values, start_position=prefix_position)
        # slow 前缀追加是每个 batch 的首个重计算点，单独记录有助于判断长文本分段是否过多。
        if timing is not None:
            timing.slow_prefill_times.append(time.time() - slow_prefill_start)
        # semantic_history 只记录当前 batch 内的 semantic token，用于 RAS 检测短期重复，不跨 batch 延续。
        semantic_history: list[int] = []
        # prompt_len 是写入 suffix 后的上下文长度；后续生成的每一帧都会从这个位置继续递增。
        prompt_len = prefix_position + suffix_values.shape[1]
        # 生成上限只由 context 剩余长度决定；文本分段已在进入 runtime 前间接管理。
        limit = self.runtime.config.n_ctx - prompt_len
        # 当前上下文已经没有生成空间时直接结束生成器，调用方会根据上层 reset 逻辑处理下一批。
        if limit <= 0:
            return
        # 第一帧 semantic token 来自 suffix 解码后的 logits，此时还没有 acoustic codebook 可写回 slow。
        semantic_sample_start = time.time()
        semantic = self._sample_semantic_token(logits, generation_sample, ras_sample, rng, semantic_history)
        # semantic 采样耗时通常很小，但在 CPU 上可帮助判断采样逻辑是否成为瓶颈。
        if timing is not None:
            timing.semantic_sample_times.append(time.time() - semantic_sample_start)
        # step 记录当前 batch 已经写回 slow context 的 acoustic 帧数，同时参与 context 位置计算。
        step = 0
        # semantic token 为结束符或达到 context 剩余上限时停止；这两个条件分别对应模型主动结束和上下文保护结束。
        while step < limit and semantic != self.runtime.slow.im_end_id:
            # fast 模型把一个 semantic token 和 slow hidden 展开成当前帧的多层 acoustic codebook。
            fast_generate_start = time.time()
            codes = self.runtime.fast.generate_codes(
                semantic_token=semantic,
                hidden_state=hidden,
                semantic_begin_id=self.runtime.tokenizer.semantic_begin_id,
                num_codebooks=self.runtime.num_codebooks,
                temperature=generation_sample.temperature,
                top_p=generation_sample.top_p,
                top_k=generation_sample.top_k,
                rng=rng,
            )
            # fast 展开是 S2-Pro 每帧 acoustic code 的核心开销，按帧记录后可聚合平均值和总耗时。
            if timing is not None:
                timing.fast_generate_times.append(time.time() - fast_generate_start)
            # 将刚生成的 semantic token 与 acoustic codes 写回 slow context，让 slow 下一步看到完整音频历史。
            slow_step_start = time.time()
            logits, hidden = self.runtime.slow.step(semantic, codes, prompt_len + step)
            # slow step 反映语义模型在逐帧生成阶段的增量解码成本。
            if timing is not None:
                timing.slow_step_times.append(time.time() - slow_step_start)
            # 写回一帧后递增步数；下一帧在 llama context 中会写到后一个绝对位置。
            step += 1
            # 每次完整产出一帧 acoustic code 后累计步数，便于日志中直接观察生成帧数。
            if timing is not None:
                timing.total_steps += 1
            # 如果已经用完剩余 context，就强制把下一 token 视作结束符，避免继续采样并写越界。
            if step >= limit:
                next_semantic = self.runtime.slow.im_end_id
            else:
                semantic_sample_start = time.time()
                next_semantic = self._sample_semantic_token(logits, generation_sample, ras_sample, rng, semantic_history)
                # 后续帧 semantic 采样同样计入总采样成本，便于和 slow/fast 模型耗时分开比较。
                if timing is not None:
                    timing.semantic_sample_times.append(time.time() - semantic_sample_start)
            # codes 以 `(num_codebooks, 1)` 形式交给上层流式 decoder 缓冲；is_last 告诉调用方本 batch 是否结束。
            yield CodeStepResult(semantic_token=semantic, codes=codes[:, None].copy(), step_idx=step - 1, is_last=(next_semantic == self.runtime.slow.im_end_id))
            # 当前帧产出后，把预采样的 next_semantic 作为下一轮循环的 semantic token。
            semantic = next_semantic

    def _prefill_role_prompt(self, prompt: S2ProPromptAsset) -> tuple[bytes, int]:
        """
        读取 S2-Pro prompt 中已经预填充好的 slow cache 状态。

        参数:
            prompt: 已由 make_prompt 构建的 S2-Pro prompt 资产；该资产必须已经携带 llama context state。

        返回:
            返回二元组，包含 slow cache 快照和当前位置；调用方会在此状态后追加 user/assistant turn。

        异常:
            当 prompt 中缺少 slow cache 快照或位置非法时抛出 ValueError。
        """
        # make_prompt 阶段已经完成 system/reference prompt 的 slow prefill；合成阶段只允许复用快照，避免重复推进 llama context。
        if prompt.slow_prefix_state is None:
            raise ValueError("s2-pro prompt is missing slow_prefix_state; rebuild the prompt with make_prompt")

        # slow_prefix_position 必须指向 reference prompt 结束后的下一个写入位置，否则后续 turn 无法正确接到缓存末尾。
        if prompt.slow_prefix_position <= 0:
            raise ValueError("s2-pro prompt has invalid slow_prefix_position; rebuild the prompt with make_prompt")

        # 返回 prompt 内保存的 llama context state；上下文重置时也复用同一份快照，不再做任何 prefill 计算。
        return prompt.slow_prefix_state, int(prompt.slow_prefix_position)

    def _append_assistant_end_to_cache(self, prefix_position: int) -> int:
        """
        将 assistant 轮次结束符写入当前 slow cache。

        参数:
            prefix_position: 当前 cache 中下一 token 的写入位置。

        返回:
            返回写入 <|im_end|> 后的新位置。

        异常:
            当 llama decode 失败时会透出 RuntimeError。
        """
        # 给batch加上尾巴： <|im_end|>\n，下一轮 user 才是合法聊天结构。
        end_values = build_text_values(
            self.runtime.tokenizer,
            self.runtime.num_codebooks,
            IM_END_TOKEN + "\n",
        )

        # 如果上下文已经没有空间写结束符，调用方会在下一批前重置 cache，这里直接返回原位置。
        if prefix_position + int(end_values.shape[1]) >= self.runtime.config.n_ctx:
            return prefix_position

        # 写入结束符只推进 slow cache，不产生音频 codebook 输出。
        self.runtime.slow.decode_values(end_values, start_position=prefix_position)
        return prefix_position + int(end_values.shape[1])

    def _should_reset_context(self, prefix_position: int, turn_values_len: int) -> bool:
        """
        判断下一批文本是否需要先重置上下文。

        参数:
            prefix_position: 当前 slow cache 已占用位置。
            turn_values_len: 下一批 user/assistant 前缀 token 长度。

        返回:
            如果接近 n_ctx 阈值则返回 True，否则返回 False。

        异常:
            本函数不主动抛出异常。
        """
        # 保留少量余量给 assistant 结束符和至少若干个语义 token，避免刚进入生成就触顶。
        return prefix_position + turn_values_len + S2_PRO_CONTEXT_RESET_MARGIN >= self.runtime.config.n_ctx

    def _submit_decoder_chunk(
        self,
        *,
        codes: np.ndarray,
        task_id: str,
        prompt: S2ProPromptAsset,
        decoder_state_loaded: bool,
        timing: S2ProTiming | None,
        last_chunk_time: float,
    ) -> tuple[bool, float]:
        """
        向 true-streaming decoder 提交一个完整 chunk，并更新与该 chunk 相关的状态。

        参数:
            codes: 已经凑满一个流式窗口的 acoustic code，形状应为 `(num_codebooks, stream_frames)`。
            task_id: 当前 synthesize 请求对应的 decoder task_id。
            prompt: 角色 prompt 资产；首包解码时需要从中注入 decoder 预热状态。
            decoder_state_loaded: 当前请求是否已经向 decoder worker 注入过 prompt state。
            timing: 可选性能统计对象；存在时会记录本次攒满 chunk 的墙钟耗时。
            last_chunk_time: 上一个 chunk 提交时刻，用于计算当前 chunk 的攒码耗时。

        返回:
            返回二元组：新的 `decoder_state_loaded` 标志位，以及新的 `last_chunk_time`。

        异常:
            当跨进程投递 decode 请求失败时透出异常。
        """
        # chunk 送入 decoder 前先记录“从上次提交到这次凑满窗口”的真实墙钟时间；它反映生成侧吞吐，不包含 decoder 计算。
        if timing is not None:
            timing.chunk_gen_times.append(time.time() - last_chunk_time)
            last_chunk_time = time.time()
        self.decoder.decode(
            codes,
            task_id=task_id,
            is_final=False,
            stream=True,
            decoder_state=(prompt.decoder_state or None) if not decoder_state_loaded else None,
            decoder_tail_codes=prompt.decoder_tail_codes if (not decoder_state_loaded and prompt.decoder_tail_codes.size) else None,
        )
        # prompt 预热状态只允许在首个 chunk 注入一次；后续由 decoder worker 维护会话状态。
        return True, last_chunk_time

    def _finalize_decoder_task(
        self,
        *,
        chunk_buffer: list[np.ndarray],
        task_id: str,
        prompt: S2ProPromptAsset,
        decoder_state_loaded: bool,
        timing: S2ProTiming | None,
        last_chunk_time: float,
    ) -> None:
        """
        以 final 请求收尾当前 decoder task，确保 worker flush 尾帧并释放状态。

        参数:
            chunk_buffer: 尚未送入 decoder 的尾部 code 片段列表。
            task_id: 当前 synthesize 请求对应的 decoder task_id。
            prompt: 角色 prompt 资产；若请求期间从未解码过任何 chunk，则 final 请求仍需注入首包 state。
            decoder_state_loaded: 当前请求是否已经注入过 decoder 初始状态。
            timing: 可选性能统计对象；存在且 final_codes 非空时会记录最后一次攒码耗时。
            last_chunk_time: 上一个 chunk 提交时刻，用于补记尾帧对应的生成耗时。

        返回:
            本函数无返回值；会阻塞到 decoder worker 返回 FINISH。

        异常:
            当 final decode 失败或 worker 无法正常收尾时透出异常。
        """
        # flush 阶段需要把不足一个 stream window 的尾帧也送进 decoder；没有尾帧时传空矩阵，仅用于驱动 worker 清理状态。
        final_codes = (
            np.concatenate(chunk_buffer, axis=1)
            if chunk_buffer
            else np.zeros((self.runtime.num_codebooks, 0), dtype=np.int64)
        )
        if timing is not None and final_codes.shape[-1] > 0:
            timing.chunk_gen_times.append(time.time() - last_chunk_time)
        self.decoder.decode(
            final_codes,
            task_id=task_id,
            is_final=True,
            stream=True,
            decoder_state=(prompt.decoder_state or None) if not decoder_state_loaded else None,
            decoder_tail_codes=prompt.decoder_tail_codes if (not decoder_state_loaded and prompt.decoder_tail_codes.size) else None,
        )

    def _run_synthesis_stream_worker(
        self,
        *,
        task_id: str,
        batches: list[str],
        prompt: S2ProPromptAsset,
        timing: S2ProTiming | None,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        generation_options: dict[str, object],
    ) -> None:
        """
        在后台线程中执行 S2-Pro 的 code 生成与 decoder chunk 提交逻辑。

        参数:
            batches: 预处理后的文本批次列表。
            prompt: 当前角色的 prompt 资产。
            timing: 可选性能统计对象；由后台线程记录生成侧耗时，前台生成器记录 decoder 输出耗时。
            output_queue: 对外输出队列；仅在后台线程发生异常时写入错误包装，正常音频由 decoder listener 直接 fan-out 进来。
            stop_event: 当前流式请求的停止信号；前台生成器提前关闭时后台线程会尽快停止继续生成。
            generation_options: 调用方传入的采样参数覆盖项。

        返回:
            本函数无返回值；其生命周期由前台生成器通过线程对象管理。

        异常:
            本函数内部会捕获异常并通过 output_queue 回传 `_S2ProStreamError`，不直接向线程外抛出。
        """
        try:
            prompt_restore_start = time.time()
            prompt_prefix_state, prompt_prefix_position = self._prefill_role_prompt(prompt)
            self.runtime.slow.restore_state(prompt_prefix_state)
            if timing is not None:
                timing.prompt_restore_time = time.time() - prompt_restore_start

            prefix_position = prompt_prefix_position
            rng = np.random.default_rng(42)
            stream_frames = int(self.runtime.codec_config.stream_code_frames)
            chunk_buffer: list[np.ndarray] = []
            decoder_state_loaded = False
            last_chunk_time = time.time()

            shared_sample = S2ProSampleConfig(
                temperature=float(generation_options.get("temperature", self.runtime.config.generation_sample.temperature)),
                top_p=float(generation_options.get("top_p", self.runtime.config.generation_sample.top_p)),
                top_k=int(generation_options.get("top_k", self.runtime.config.generation_sample.top_k)),
            )
            ras_sample = self.runtime.config.ras_sample

            for idx, batch_text in enumerate(batches):
                if stop_event.is_set():
                    break
                logger.info(f"[s2-pro]-正在推理第{idx}段文本：{batch_text}")
                turn_prompt = build_turn_prompt(self.runtime.tokenizer, self.runtime.num_codebooks, batch_text)

                if self._should_reset_context(prefix_position, int(turn_prompt.values.shape[1])):
                    logger.info("S2-Pro 上下文接近 n_ctx=%s，已清空 slow cache 并从参考 prompt 继续后续文本", self.runtime.config.n_ctx)
                    self.runtime.slow.restore_state(prompt_prefix_state)
                    prefix_position = prompt_prefix_position
                    if self._should_reset_context(prefix_position, int(turn_prompt.values.shape[1])):
                        raise ValueError("s2-pro batch cannot fit into n_ctx after context reset")

                batch_frames = 0
                generated_codes: list[np.ndarray] = []

                for event in self._generate_codes_stream_from_context(
                    prefix_position=prefix_position,
                    suffix_values=turn_prompt.values,
                    generation_sample=shared_sample,
                    ras_sample=ras_sample,
                    rng=rng,
                    timing=timing,
                ):
                    if stop_event.is_set():
                        break
                    chunk_buffer.append(event.codes)
                    generated_codes.append(event.codes)
                    batch_frames += int(event.codes.shape[-1])
                    buffered_frames = sum(int(item.shape[-1]) for item in chunk_buffer)
                    if buffered_frames >= stream_frames:
                        codes = np.concatenate(chunk_buffer, axis=1)
                        send, remain = codes[:, :stream_frames], codes[:, stream_frames:]
                        decoder_state_loaded, last_chunk_time = self._submit_decoder_chunk(
                            codes=send,
                            task_id=task_id,
                            prompt=prompt,
                            decoder_state_loaded=decoder_state_loaded,
                            timing=timing,
                            last_chunk_time=last_chunk_time,
                        )
                        chunk_buffer = [remain] if remain.shape[-1] else []
                    if event.is_last:
                        break

                prefix_position += int(turn_prompt.values.shape[1]) + batch_frames
                reached_context_limit = prefix_position >= self.runtime.config.n_ctx - S2_PRO_CONTEXT_RESET_MARGIN
                if reached_context_limit:
                    self.runtime.slow.restore_state(prompt_prefix_state)
                    prefix_position = prompt_prefix_position
                    continue
                if generated_codes:
                    prefix_position = self._append_assistant_end_to_cache(prefix_position)

            # 即使前台提前停止消费，也尽量发送一次 final decode，把 decoder worker 中该 task 的会话状态收干净。
            self._finalize_decoder_task(
                chunk_buffer=chunk_buffer,
                task_id=task_id,
                prompt=prompt,
                decoder_state_loaded=decoder_state_loaded,
                timing=timing,
                last_chunk_time=last_chunk_time,
            )
        except Exception as exc:
            output_queue.put(_S2ProStreamError(exc))

    def synthesize(
        self,
        text: str,
        prompt: S2ProPromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        max_length: int | None = None,
        min_length: int | None = None,
        remove_meaningless_chars: bool = False,
        generation_options: dict[str, object] | None = None,
    ) -> Iterable[np.ndarray]:
        """
        使用 S2-Pro runtime 按流式方式合成目标文本。

        参数:
            text: 用户输入的待合成文本；进入模型前会先做 speaker 解析、分段和清洗。
            prompt: make_prompt 生成的 S2ProPromptAsset，内部包含参考音频 token、decoder 预热状态和 slow prefix cache。
            stream: 是否按流式路径解码；当前实现固定走 true-streaming decoder 路径。
            track_performance: 是否记录性能数据；开启后会打印文本预处理、slow/fast 生成和 decoder 解码等关键耗时。
            max_length: 文本预处理阶段的单段上限；为空时使用模型配置。
            min_length: 文本预处理阶段合并短句的下限；为空时使用模型配置。
            remove_meaningless_chars: 是否移除模型文本规则里定义的无意义字符。
            generation_options: 模型私有采样参数，例如 temperature、top_p、top_k。

        返回:
            以生成器形式逐段产出 float32 PCM 音频数组。

        异常:
            当 prompt 类型错误、文本预处理为空、上下文无法容纳当前 batch、slow/fast/decoder 推理失败时抛出异常。
        """
        # S2-Pro session 只能消费 S2ProPromptAsset；提前校验类型，避免后续访问模型专属字段时报晦涩异常。
        if not isinstance(prompt, S2ProPromptAsset):
            raise TypeError("s2-pro runtime requires S2ProPromptAsset")
        # 只有用户显式开启性能跟踪时才创建统计容器，默认路径不增加列表写入和时间采样成本。
        timing = S2ProTiming() if track_performance else None
        # generation_options 允许调用方只覆盖部分采样参数；None 时用空字典统一后续读取逻辑。
        options = generation_options or {}
        # 进入模型前先做文本预处理，把用户原始文本转换成 S2-Pro 官方推理约束下的 batch 列表。
        preprocess_start = time.time()
        processed = preprocess_text(
            text,
            language="auto", # 这个参数目前在streamvox无用
            max_length=max_length if max_length is not None else self.runtime.config.max_length, 
            min_length=min_length if min_length is not None else self.runtime.config.min_length,
            tokenizer=self.runtime.tokenizer,
            config=self.runtime.text_preprocess_config,
            remove_meaningless_chars=remove_meaningless_chars,
        )
        # 文本预处理包含 speaker 解析、清洗和分段，长文本场景下需要单独观察。
        if timing is not None:
            timing.preprocess_time = time.time() - preprocess_start
        # S2-Pro 的 preprocess_text 已经完成 speaker 解析、单说话人补标签、300 bytes 折算分段和 5 speaker 合批。
        batches = processed.segments
        # 预处理后没有任何可合成片段时，继续推理只会构造空 prompt，因此直接报错提示调用方输入无效。
        if not batches:
            raise ValueError("text preprocessing produced no segments")
        
        # 当前 synthesize 的 decoder task_id 必须在主线程先创建：一方面供后台 worker 提交 chunk，另一方面供前台即时订阅音频流。
        current_task_id = str(uuid.uuid4())
        # output_queue 同时承载两类消息：decoder listener fan-out 进来的 DecoderResponse，以及后台线程回传的异常对象。
        output_queue = self.decoder.subscribe_stream(current_task_id)

        # stop_event 用于在调用方提前结束迭代时通知后台线程停止继续生成长文本。
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._run_synthesis_stream_worker,
            kwargs={
                "task_id": current_task_id,
                "batches": batches,
                "prompt": prompt,
                "timing": timing,
                "output_queue": output_queue,
                "stop_event": stop_event,
                "generation_options": options,
            },
            daemon=True,
        )
        worker.start()

        try:
            while True:
                item = output_queue.get()
                # 后台线程一旦遇到异常，立即通过队列透传给前台生成器，保持现有对外接口的同步异常语义。
                if isinstance(item, _S2ProStreamError):
                    raise item.error
                if not isinstance(item, DecoderResponse):
                    continue
                # AUDIO 是真正需要吐给 SDK 的有效载荷；一旦 listener 拿到音频，这里会立刻对外 yield。
                if item.msg_type == "AUDIO" and item.audio is not None:
                    if timing is not None:
                        timing.decoder_compute_times.append(item.compute_time)
                    yield item.audio
                    continue
                # FINISH 说明 decoder worker 已经把当前 task 的尾帧 flush 并清理完成，此时整个流式请求可以结束。
                if item.msg_type == "FINISH":
                    break
            # 只有前台把整个流完整消费到 FINISH，性能日志才算完整；这与旧实现“完整消费后打印”保持一致。
            if timing is not None:
                while len(timing.chunk_gen_times) < len(timing.decoder_compute_times):
                    timing.chunk_gen_times.append(0.0)
                log_s2_pro_performance(timing)
        finally:
            # 无论正常结束、调用方提前 break，还是后台异常，都要先解除订阅，避免队列继续累计历史消息。
            stop_event.set()
            self.decoder.unsubscribe_stream(current_task_id, output_queue)
            worker.join(timeout=1.0)


def prompt_asset_from_payload(
    *,
    role_name: str,
    model_name: str,
    sample_rate: int,
    prompt_text: str | list[str] | None,
    prompt_codes: np.ndarray,
    decoder_state: dict[str, np.ndarray] | None = None,
    decoder_tail_codes: np.ndarray | None = None,
    slow_prefix_state: bytes | None = None,
    slow_prefix_position: int = 0,
    metadata: dict[str, str] | None = None,
) -> S2ProPromptAsset:
    """
    从 adapter 产出的中间数据构建 S2-Pro prompt 资产。

    参数:
        role_name: prompt 角色名。
        model_name: 当前模型名。
        sample_rate: prompt 音频采样率。
        prompt_text: 单条或多条参考文本。
        prompt_codes: 已拼接的参考音频 codec token。
        decoder_state: 单参考音频场景下的 decoder 预热状态。
        decoder_tail_codes: decoder 预热后尚未凑满流式窗口的尾部 token。
        slow_prefix_state: make_prompt 阶段完成的 slow prefill 快照。
        slow_prefix_position: slow prefill 快照对应的上下文位置。
        metadata: prompt 附加元数据。

    返回:
        返回可被缓存和合成流程复用的 S2ProPromptAsset。

    异常:
        当数组转换失败时会透出 numpy 的异常。
    """
    return S2ProPromptAsset(
        role_name=role_name,
        model=model_name,
        sample_rate=sample_rate,
        metadata=metadata or {},
        prompt_text=prompt_text,
        prompt_codes=np.asarray(prompt_codes, dtype=np.int64),
        decoder_state={name: np.asarray(value).copy() for name, value in (decoder_state or {}).items()},
        decoder_tail_codes=np.asarray(decoder_tail_codes if decoder_tail_codes is not None else np.zeros((0, 0), dtype=np.int64), dtype=np.int64),
        slow_prefix_state=slow_prefix_state,
        slow_prefix_position=int(slow_prefix_position),
    )


def _normalize_s2_pro_prompt_text(prompt_text: str | list[str] | None) -> list[str]:
    """
    将 S2-Pro prompt 文本统一成列表。

    参数:
        prompt_text: 单条、多条或缺省参考文本。

    返回:
        返回供 build_role_prefix_prompt 使用的文本列表。

    异常:
        当文本缺失或存在空白文本时抛出 ValueError。
    """
    # 多说话人参考通常传入多条文本；每条文本后续会由 prompt_builder 补 speaker tag。
    if isinstance(prompt_text, list):
        normalized = prompt_text
    # 单说话人参考保持为单元素列表，使单条和多条走同一套 prefix 构造逻辑。
    elif isinstance(prompt_text, str):
        normalized = [prompt_text]
    # S2-Pro 参考音频 token 必须和参考文本对齐，缺文本会让 prompt 语义不完整。
    else:
        raise ValueError("s2-pro prompt_text is required")
    # 每条参考文本都必须包含有效内容，避免构造只有 speaker 标签但没有参考语义的 system prompt。
    if not normalized or any(not item.strip() for item in normalized):
        raise ValueError("s2-pro prompt_text entries must be non-empty")
    return normalized


def build_s2_pro_role_prefix_state(
    runtime: S2ProRuntime,
    *,
    prompt_text: str | list[str] | None,
    prompt_codes: np.ndarray,
) -> tuple[bytes, int]:
    """
    预计算 S2-Pro system/reference prompt 的 slow cache。

    参数:
        runtime: 已加载的 S2-Pro runtime。
        prompt_text: 单条或多条参考文本。
        prompt_codes: 已拼接的参考音频 codec token。

    返回:
        返回 slow cache 快照和该快照对应的上下文位置。

    异常:
        当参考文本为空、参考 token 为空、prompt 过长或 llama decode 失败时抛出异常。
    """
    # 文本统一成列表后交给 prompt_builder；它会负责缺省 speaker tag 的补齐和最终序列拼接。
    normalized_text = _normalize_s2_pro_prompt_text(prompt_text)
    # prompt_codes 必须是完整的 codebook 矩阵，多参考音频在 adapter 阶段已经按时间维拼接。
    normalized_codes = np.asarray(prompt_codes, dtype=np.int64)
    if normalized_codes.ndim != 2 or normalized_codes.shape[0] != runtime.num_codebooks or normalized_codes.shape[1] <= 0:
        raise ValueError(f"Expected prompt_codes shape ({runtime.num_codebooks}, T), got {normalized_codes.shape}")
    # s2-pro 的参考 prompt 序列由 system 文本、参考文本和参考 speech token 构成；多文本会被拼接到同一个 system reference 中。
    role_prompt = build_role_prefix_prompt(
        runtime.tokenizer,
        runtime.num_codebooks,
        normalized_text,
        [normalized_codes],
    )
    # 参考 prompt 本身不能占满上下文，否则后续 user batch 和 assistant 生成 token 没有可用空间。
    if int(role_prompt.values.shape[1]) + S2_PRO_CONTEXT_RESET_MARGIN >= runtime.config.n_ctx:
        raise ValueError("s2-pro reference prompt is too long for the configured n_ctx")
    # make_prompt 阶段提前执行 slow prefill；后续按角色合成时只恢复快照并追加 user/assistant turn。
    runtime.slow.prefill(role_prompt.values)
    return runtime.slow.snapshot_state(), int(role_prompt.values.shape[1])


def serialize_prompt_asset(asset: S2ProPromptAsset) -> SerializedPromptAsset:
    # decoder_state 是多个命名数组，序列化时拆成独立 payload，避免 pickle 并保持 npz 可直接读取。
    decoder_state_arrays = {
        f"decoder_state__{name}": np.asarray(value)
        for name, value in asset.decoder_state.items()
    }
    # slow_prefix_state 是 llama context state 的 bytes 快照，其中包含 KV cache；转成 uint8 数组后即可安全写入 allow_pickle=False 的 npz。
    slow_prefix_state_array = (
        np.frombuffer(asset.slow_prefix_state, dtype=np.uint8)
        if asset.slow_prefix_state is not None
        else np.zeros((0,), dtype=np.uint8)
    )
    return SerializedPromptAsset(
        role_name=asset.role_name,
        model=asset.model,
        sample_rate=asset.sample_rate,
        metadata=asset.metadata,
        arrays={
            "prompt_codes": np.asarray(asset.prompt_codes, dtype=np.int64),
            "decoder_tail_codes": np.asarray(asset.decoder_tail_codes, dtype=np.int64),
            "slow_prefix_state": slow_prefix_state_array,
            **decoder_state_arrays,
        },
        json_data={
            "prompt_text": asset.prompt_text,
            "decoder_state_names": sorted(asset.decoder_state),
            "slow_prefix_position": int(asset.slow_prefix_position),
            "has_slow_prefix_state": asset.slow_prefix_state is not None,
        },
    )


def deserialize_prompt_asset(payload: SerializedPromptAsset) -> S2ProPromptAsset:
    # decoder_state 采用名称索引恢复，兼容不同 decoder backend 暴露出的状态张量集合。
    state_names = [str(name) for name in payload.json_data.get("decoder_state_names", [])]
    decoder_state = {
        name: np.asarray(payload.arrays[f"decoder_state__{name}"])
        for name in state_names
        if f"decoder_state__{name}" in payload.arrays
    }
    # 新版本缓存会带 slow_prefix_state；旧版本缓存没有该字段时保持 None，并在首次合成时现场补算。
    slow_prefix_array = np.asarray(payload.arrays.get("slow_prefix_state", np.zeros((0,), dtype=np.uint8)), dtype=np.uint8)
    # has_slow_prefix_state 是“字段是否存在”的语义，不能用数组长度判断；这样即使极端情况下 state 为 b"" 也能字节级还原。
    slow_prefix_state = bytes(slow_prefix_array.tobytes()) if bool(payload.json_data.get("has_slow_prefix_state", False)) else None
    return S2ProPromptAsset(
        role_name=payload.role_name,
        model=payload.model,
        sample_rate=payload.sample_rate,
        metadata=payload.metadata,
        prompt_text=payload.json_data.get("prompt_text"),
        prompt_codes=np.asarray(payload.arrays["prompt_codes"], dtype=np.int64),
        decoder_state=decoder_state,
        decoder_tail_codes=np.asarray(payload.arrays.get("decoder_tail_codes", np.zeros((0, 0), dtype=np.int64)), dtype=np.int64),
        slow_prefix_state=slow_prefix_state,
        slow_prefix_position=int(payload.json_data.get("slow_prefix_position", 0)),
    )
