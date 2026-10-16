from __future__ import annotations

from ..onnx_backend import resolve_explicit_onnx_backend

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer
import yaml

from ...model_bundle import validate_local_bundle
from ...utils.log import logger
from ...utils.performance import format_perf_value
from .. import llama_backend as llama_backend_utils
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, RuntimeInfo, SerializedPromptAsset
from . import llama
from .assets import AssetsManager
from .decoder import DecoderAPI
from .encoder import CodecEncoder, SpeakerEncoder
from .predictor import Predictor
from .prompt_builder import PromptBuilder
from .schema.constant import LANGUAGE_MAP, SPECIAL_TOKENS, map_language
from .schema.decoder_protocol import DecoderResponse, DecoderState
from .schema.item import PromptData, VoiceClonePromptItem
from .schema.record import Timing
from .spec import QWEN3_CLONE_17B_SPEC, Qwen3CloneModelSpec
from .talker import TalkerPredictor
from .text import preprocess_text
from ..text_processing import TextPreprocessConfig


MODEL_NAME = QWEN3_CLONE_17B_SPEC.model_name
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "qwen3_tts.yaml"


@dataclass(slots=True)
class Qwen3CloneConfig:
    sample_rate: int = 24000
    chunk_size: int = 12
    n_ctx: int = 4096
    min_n_ctx: int = 4096
    max_n_ctx: int = 32768
    max_length: int = 300
    min_length: int = 30
    talker_n_gpu_layers: int = -1
    predictor_n_gpu_layers: int = -1
    llama_cpu_threads: int = 0
    llama_cpu_threads_batch: int = 0
    llama_cpu_numa_strategy: str = "disabled"
    decoder_start_method: str = "spawn"
    talker_filename: str = "qwen3_tts_talker.q5_k.gguf"
    predictor_filename: str = "qwen3_tts_predictor.q8_0.gguf"
    decoder_filename: str = "qwen3_tts_decoder.fp16.onnx"
    codec_encoder_filename: str = "qwen3_tts_codec_encoder.fp16.onnx"
    speaker_encoder_filename: str = "qwen3_tts_speaker_encoder.fp16.onnx"
    tokenizer_filename: str = "tokenizer.json"


@dataclass(slots=True)
class Qwen3ClonePromptAsset(BasePromptAsset):
    prompt_text: str | None = None
    speaker_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    prompt_codes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int64)
    )
    decoder_state: DecoderState | None = None


@dataclass(slots=True)
class Qwen3ResolvedDevice:
    '''
        qwen3 clone GGUF 系列的最终后端决策结果。
        这里保存的是“已经结合平台和当前能力解析完”的 backend。
    '''

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
class _Qwen3StreamError:
    """
    qwen3 流式后台线程通过输出队列回传给前台生成器的异常包装。

    参数:
        error: 后台线程捕获到的原始异常对象。

    返回:
        返回一个仅用于线程间传递错误状态的轻量对象。

    异常:
        本数据结构本身不抛出异常。
    """

    error: Exception


def load_qwen3_config() -> Qwen3CloneConfig:
    '''加载配置文件'''
    if not CONFIG_PATH.exists():
        return Qwen3CloneConfig()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime", {})
    paths = payload.get("paths", {})
    # default_n_ctx 是当前模型未收到用户显式 n_ctx 时使用的默认上下文长度。
    # 兼容旧配置里的 n_ctx 字段，避免旧包升级后缺少 default_n_ctx 时无法按原值启动。
    default_n_ctx = int(runtime.get("default_n_ctx", runtime.get("n_ctx", 4096)))

    return Qwen3CloneConfig(
        sample_rate=int(runtime.get("sample_rate", 24000)),
        chunk_size=int(runtime.get("chunk_size", 12)),
        n_ctx=default_n_ctx,
        min_n_ctx=int(runtime.get("min_n_ctx", 4096)),
        max_n_ctx=int(runtime.get("max_n_ctx", 32768)),
        max_length=int(runtime.get("max_length", 300)),
        min_length=int(runtime.get("min_length", 30)),
        talker_n_gpu_layers=int(runtime.get("talker_n_gpu_layers", -1)),
        predictor_n_gpu_layers=int(runtime.get("predictor_n_gpu_layers", -1)),
        llama_cpu_threads=int(runtime.get("llama_cpu_threads", 0)),
        llama_cpu_threads_batch=int(runtime.get("llama_cpu_threads_batch", 0)),
        llama_cpu_numa_strategy=str(runtime.get("llama_cpu_numa_strategy", "disabled")),
        decoder_start_method=str(runtime.get("decoder_start_method", "spawn")),
        talker_filename=str(paths.get("talker_filename", "qwen3_tts_talker.q5_k.gguf")),
        predictor_filename=str(paths.get("predictor_filename", "qwen3_tts_predictor.q8_0.gguf")),
        decoder_filename=str(paths.get("decoder_filename", "qwen3_tts_decoder.fp16.onnx")),
        codec_encoder_filename=str(paths.get("codec_encoder_filename", "qwen3_tts_codec_encoder.fp16.onnx")),
        speaker_encoder_filename=str(paths.get("speaker_encoder_filename", "qwen3_tts_speaker_encoder.fp16.onnx")),
        tokenizer_filename=str(paths.get("tokenizer_filename", "tokenizer.json")),
    )


def _clamp_n_ctx(config: Qwen3CloneConfig, n_ctx: int) -> int:
    """
    将 Qwen3 上下文窗口限制在当前模型配置声明的范围内。

    参数:
        config: 已从 yaml 读取的 Qwen3 配置，包含上下文窗口边界。
        n_ctx: 配置默认值或引擎入口给出的上下文长度。

    返回:
        返回位于 [config.min_n_ctx, config.max_n_ctx] 区间内的上下文长度。

    异常:
        本函数不主动抛出异常；调用方负责保证输入可转为整数。
    """
    # 先把调用方传入值转成整数，后续比较、日志和返回值都使用同一个规范化输入，避免重复转换产生分歧。
    requested_n_ctx = int(n_ctx)
    # 上下文边界由模型配置提供，避免把当前 Qwen3 版本的能力假设写死在通用引擎或 runtime 常量里。
    clamped_n_ctx = max(config.min_n_ctx, min(config.max_n_ctx, requested_n_ctx))
    # 当用户输入或 yaml 默认值超出模型声明范围时，仍然按边界保护启动，但必须显式告警，方便定位配置没有生效的原因。
    if clamped_n_ctx != requested_n_ctx:
        logger.warning(
            "Qwen3 n_ctx=%s is outside supported range [%s, %s], using %s",
            requested_n_ctx,
            config.min_n_ctx,
            config.max_n_ctx,
            clamped_n_ctx,
        )
    return clamped_n_ctx


def _ort_provider_available(provider: str) -> bool:
    # decoder 的最终执行能力以 onnxruntime 实际注册的 provider 为准，避免仅凭显卡存在就误判。
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return provider in ort.get_available_providers()


def _resolve_llama_backend(target: ExecutionTarget) -> tuple[str, int | None]:
    '''
        llama.cpp 的 GPU 后端由 bin 决定。
        上层只表达“要不要 GPU”，这里再把它落到当前实际打包的后端上。
    '''
    resolution = llama_backend_utils.select_llama_backend(
        mode=target.mode,
        device_index=target.device_index,
        model_label="qwen3",
        explicit_backend=target.backend_preference,
    )
    return resolution.backend, resolution.gpu_index


def _resolve_onnx_backend(target: ExecutionTarget) -> tuple[str, int | None]:
    '''
        onnxruntime 的设备映射规则：
        - Windows 有显卡：DML
        - Ubuntu/Linux 有 NVIDIA 显卡：CUDA
        - 没显卡：CPU
    '''
    explicit = resolve_explicit_onnx_backend(target)
    if explicit is not None:
        return explicit
    if target.mode == "cpu":
        return "cpu", None
    if target.mode == "auto":
        if os.name == "nt":
            if _ort_provider_available("CUDAExecutionProvider"):
                return "cuda", target.device_index if target.device_index is not None else 0
            if _ort_provider_available("DmlExecutionProvider"):
                return "dml", target.device_index if target.device_index is not None else 0
            return "cpu", None
        if _ort_provider_available("CUDAExecutionProvider"):
            return "cuda", target.device_index if target.device_index is not None else 0
        return "cpu", None
    if target.mode == "gpu":
        if os.name == "nt":
            if _ort_provider_available("CUDAExecutionProvider"):
                return "cuda", target.device_index if target.device_index is not None else 0
            if _ort_provider_available("DmlExecutionProvider"):
                return "dml", target.device_index if target.device_index is not None else 0
            logger.warning(
                "[Device Fallback][IMPORTANT] 用户请求 decoder GPU，但 DmlExecutionProvider 不可用，decoder 将回退到 CPU"
            )
            return "cpu", None
        if _ort_provider_available("CUDAExecutionProvider"):
            return "cuda", target.device_index if target.device_index is not None else 0
        logger.warning(
            "[Device Fallback][IMPORTANT] 用户请求 decoder GPU，但 CUDAExecutionProvider 不可用，decoder 将回退到 CPU"
        )
        return "cpu", None
    raise ValueError(f"unsupported onnx target mode: {target.mode}")


def resolve_qwen3_device(
    device: str,
    execution_plan: ExecutionPlan | None,
) -> Qwen3ResolvedDevice:
    '''
        qwen3 runtime 只消费 execution plan，不直接关心 TTSEngine 的 device 设计。
        这样未来别的模型可以沿用同一套 device 入口，但用不同 family 翻译。
    '''
    if execution_plan is None:
        raise ValueError("qwen3 runtime requires an execution plan")
    llama_target = execution_plan.get("llama")
    onnx_target = execution_plan.get("onnx")
    if llama_target is None or onnx_target is None:
        raise ValueError("qwen3 execution plan must include llama and onnx targets")

    gguf_backend, gguf_index = _resolve_llama_backend(llama_target)
    decoder_backend, decoder_index = _resolve_onnx_backend(onnx_target)

    preferred_index = decoder_index if decoder_index is not None else gguf_index
    return Qwen3ResolvedDevice(
        request=device,
        gguf_backend=gguf_backend,
        decoder_backend=decoder_backend,
        gpu_index=preferred_index,
    )


def _available_logical_cpu_count() -> int:
    """
    优先读取当前进程真正可用的 CPU 数量。
    这样在 taskset/cgroup/容器里不会错误地把整机核数都吃满。
    """
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except Exception:
        pass
    return os.cpu_count() or 1


def _detect_cpu_topology() -> CpuTopology:
    """
    尽量从 Linux sysfs 读取物理核心拓扑。

    为什么要做这一步：
    - 逻辑线程数很多时，直接把 llama 线程开到 `os.cpu_count() // 2`，
      在双路机器上很容易跨 NUMA 抢缓存，反而比少线程更慢；
    - 我们更关心“物理核心数”和“每路 socket 的核心数”，
      因为这更接近 llama CPU 推理的甜点区间。
    """
    logical_cpus = _available_logical_cpu_count()
    cpu_ids: list[int]
    try:
        cpu_ids = sorted(os.sched_getaffinity(0))
    except Exception:
        cpu_ids = list(range(logical_cpus))

    core_keys: set[tuple[int, int]] = set()
    package_to_cores: dict[int, set[int]] = {}
    for cpu_id in cpu_ids:
        topo_dir = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        package_file = topo_dir / "physical_package_id"
        core_file = topo_dir / "core_id"
        try:
            package_id = int(package_file.read_text(encoding="utf-8").strip())
            core_id = int(core_file.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        core_keys.add((package_id, core_id))
        package_to_cores.setdefault(package_id, set()).add(core_id)

    if core_keys:
        physical_cores = len(core_keys)
        sockets = len(package_to_cores)
        physical_per_socket = max((len(cores) for cores in package_to_cores.values()), default=physical_cores)
        return CpuTopology(
            logical_cpus=logical_cpus,
            physical_cores=physical_cores,
            sockets=sockets,
            physical_cores_per_socket=physical_per_socket,
        )

    # 回退策略：拿不到拓扑时，至少避免把 SMT 线程全部当成独立大核。
    estimated_physical = max(1, logical_cpus // 2)
    return CpuTopology(
        logical_cpus=logical_cpus,
        physical_cores=estimated_physical,
        sockets=1,
        physical_cores_per_socket=estimated_physical,
    )


def _env_positive_int(name: str) -> int:
    """
    读取调优环境变量。

    为什么保留环境变量入口：
    - 调参阶段要快速搜索线程组合，不应该每次都改 yaml 或代码；
    - 但正式默认值仍由配置和自动策略决定，环境变量只作为临时覆盖。
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"[CPU Tune] 忽略非法环境变量 {name}={raw!r}")
        return 0
    return max(0, value)


def _env_text(name: str) -> str:
    """读取字符串环境变量，主要用于 CPU 调优阶段快速切换 NUMA 策略。"""
    return os.getenv(name, "").strip()


def _resolve_llama_cpu_thread_plan(config: Qwen3CloneConfig, resolved_device: Qwen3ResolvedDevice) -> LlamaCpuThreadPlan:
    """
    统一生成 CPU 模式下的 llama 线程计划。

    策略原则：
    - GPU 模式不干预，避免误伤已调好的显卡推理；
    - CPU 模式默认优先“单路 socket 的物理核心数”，
      这是为了减少双路 NUMA 机器上的跨 socket 抖动；
    - batch 线程可以更激进一些，用到更多逻辑线程来提升 prefill 吞吐。
    """
    topology = _detect_cpu_topology()

    if resolved_device.gguf_gpu_enabled:
        return LlamaCpuThreadPlan(
            n_threads=0,
            n_threads_batch=0,
            source="gpu",
            topology=topology,
        )

    env_threads = _env_positive_int("STREAMVOX_LLAMA_CPU_THREADS")
    env_threads_batch = _env_positive_int("STREAMVOX_LLAMA_CPU_THREADS_BATCH")

    if env_threads > 0:
        n_threads = min(env_threads, topology.logical_cpus)
        source = "env"
    elif config.llama_cpu_threads > 0:
        n_threads = min(config.llama_cpu_threads, topology.logical_cpus)
        source = "config"
    else:
        # 默认把解码线程压在单路物理核范围内，避免双路机器过度扩线程。
        n_threads = min(topology.physical_cores_per_socket, topology.logical_cpus)
        source = "auto-topology"

    if env_threads_batch > 0:
        n_threads_batch = min(env_threads_batch, topology.logical_cpus)
        source = "env"
    elif config.llama_cpu_threads_batch > 0:
        n_threads_batch = min(config.llama_cpu_threads_batch, topology.logical_cpus)
        source = "config"
    else:
        # 当前经验上 batch 线程不宜再激进翻倍。
        # 对这台双路机器，过高的 batch 线程会显著拖慢 prefill 和总 RTf。
        n_threads_batch = n_threads

    n_threads = max(1, n_threads)
    n_threads_batch = max(n_threads, n_threads_batch)
    return LlamaCpuThreadPlan(
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        source=source,
        topology=topology,
    )


def _resolve_llama_cpu_numa_strategy(config: Qwen3CloneConfig, resolved_device: Qwen3ResolvedDevice) -> str:
    """
    解析 CPU 模式下的 llama NUMA 策略。

    只在 CPU 路径生效：
    - GPU 路径不需要这层优化；
    - 环境变量优先级高于配置，方便 benchmark 时快速切换；
    - 默认保持 disabled，只有显式启用时才改变行为。
    """
    if resolved_device.gguf_gpu_enabled:
        return "disabled"
    env_strategy = _env_text("STREAMVOX_LLAMA_NUMA_STRATEGY")
    if env_strategy:
        return env_strategy
    return config.llama_cpu_numa_strategy


class Qwen3Runtime:
    model_name = MODEL_NAME

    def __init__(
        self,
        model_dir: Path,
        model_spec: Qwen3CloneModelSpec | None = None,
        device: str = "auto",
        n_ctx: int | None = None,
        verify_model_sha256: bool = True,
        execution_plan: ExecutionPlan | None = None,
        text_preprocess_config: TextPreprocessConfig | None = None,
    ) -> None:
        # 兼容旧调用方：未传规格时仍按最早接入的 1.7B 模型加载。
        self.model_spec = model_spec or QWEN3_CLONE_17B_SPEC
        self.model_name = self.model_spec.model_name
        self.bundle_root = Path(model_dir).resolve()
        self.device = device
        self.execution_plan = execution_plan
        # 这里把模型声明的文本能力固化到 runtime，后续 session 只消费统一配置，
        # 不再依赖 qwen3 私有规则文件，便于别的模型直接复用同一套预处理实现。
        self.text_preprocess_config = text_preprocess_config
        self.resolved_device = resolve_qwen3_device(device, execution_plan)
        self.config = load_qwen3_config()
        # 引擎显式传入 n_ctx 时优先使用用户意图；未传时使用 yaml 的 default_n_ctx，并按当前模型自己的边界裁剪。
        self.config.n_ctx = _clamp_n_ctx(self.config, n_ctx if n_ctx is not None else self.config.n_ctx)
        self.llama_cpu_numa_strategy = _resolve_llama_cpu_numa_strategy(self.config, self.resolved_device)
        self.llama_cpu_thread_plan = _resolve_llama_cpu_thread_plan(self.config, self.resolved_device)
        self.ckpt_root = self.bundle_root / "ckpt"
        # 同系列不同大小的 GGUF 权重放在不同 core 子目录；公共 ONNX/tokenizer 仍放在 ckpt 根目录。
        self.core_model_dir = self.ckpt_root / self.model_spec.core_subdir
        validate_local_bundle(
            self.bundle_root,
            self.model_name,
            verify_model_sha256=verify_model_sha256,
        )

        self.paths = {
            "talker_gguf": self.core_model_dir / self.config.talker_filename,
            "predictor_gguf": self.core_model_dir / self.config.predictor_filename,
            "decoder_onnx": self.ckpt_root / self.config.decoder_filename,
            "codec_enc_onnx": self.ckpt_root / self.config.codec_encoder_filename,
            "spk_enc_onnx": self.core_model_dir / self.config.speaker_encoder_filename,
            "tokenizer": self.ckpt_root / self.config.tokenizer_filename,
        }
        # 共享执行器锁负责把同一个 runtime 上的 synthesize 串行化，避免多请求并发污染同一套 GGUF context。
        self._synthesize_lock = threading.Lock()
        # 下面这些成员在 runtime 初始化期只创建一次，后续每次 synthesize 仅做状态重置和复用。
        self.talker_ctx: llama.LlamaContext | None = None
        self.predictor_ctx: llama.LlamaContext | None = None
        self.talker_batch: llama.LlamaBatch | None = None
        self.predictor_batch: llama.LlamaBatch | None = None
        self.talker: TalkerPredictor | None = None
        self.predictor: Predictor | None = None
        self.prompt_builder: PromptBuilder | None = None
        # 初始化预热只允许真正执行一次；重复调用直接短路，避免重复污染 decoder/talker/predictor 的启动路径统计。
        self._init_warmed = False
        self._validate_assets()
        self._init_runtime()
        

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def _validate_assets(self) -> None:
        required = ["talker_gguf", "predictor_gguf", "decoder_onnx", "tokenizer"]
        missing = [name for name in required if not self.paths[name].exists()]
        if missing:
            raise FileNotFoundError(
                f"missing qwen3 clone runtime assets under {self.bundle_root}: {missing}"
            )

    def _load_llama_models(self) -> tuple[llama.LlamaModel, llama.LlamaModel]:
        # llama.cpp 的 GPU 层数只在模型加载时生效；集中封装便于 GPU 失败后整体用 CPU 重试。
        gpu_index = 0 if self.resolved_device.gpu_index is None else self.resolved_device.gpu_index
        n_talker_gpu_layers = (
            self.config.talker_n_gpu_layers if self.resolved_device.gguf_gpu_enabled else 0
        )
        n_predictor_gpu_layers = (
            self.config.predictor_n_gpu_layers if self.resolved_device.gguf_gpu_enabled else 0
        )
        try:
            talker_model = llama.LlamaModel(
                str(self.paths["talker_gguf"]),
                n_gpu_layers=n_talker_gpu_layers,
                main_gpu=gpu_index,
                split_mode=0,
                backend=self.resolved_device.gguf_backend,
            )
            predictor_model = llama.LlamaModel(
                str(self.paths["predictor_gguf"]),
                n_gpu_layers=n_predictor_gpu_layers,
                main_gpu=gpu_index,
                split_mode=0,
                backend=self.resolved_device.gguf_backend,
            )
            return talker_model, predictor_model
        except Exception:
            if not self.resolved_device.gguf_gpu_enabled:
                raise
            logger.exception(
                "[Device Fallback][IMPORTANT] llama.cpp %s 后端加载模型失败，将强制回退到 CPU 重新加载",
                self.resolved_device.gguf_backend,
            )
            # 回退只影响 llama.cpp GGUF 部分；decoder 的 ONNX 后端保持之前解析结果，避免误伤 CUDA/DML 解码。
            self.resolved_device.gguf_backend = "cpu"
            self.llama_cpu_numa_strategy = _resolve_llama_cpu_numa_strategy(self.config, self.resolved_device)
            self.llama_cpu_thread_plan = _resolve_llama_cpu_thread_plan(self.config, self.resolved_device)
            talker_model = llama.LlamaModel(
                str(self.paths["talker_gguf"]),
                n_gpu_layers=0,
                main_gpu=0,
                split_mode=0,
                backend="cpu",
            )
            predictor_model = llama.LlamaModel(
                str(self.paths["predictor_gguf"]),
                n_gpu_layers=0,
                main_gpu=0,
                split_mode=0,
                backend="cpu",
            )
            return talker_model, predictor_model

    def _init_runtime(self) -> None:
        start = time.time()
        # llama NUMA 是进程级初始化开关，因此要在首次加载模型前就确定。
        llama.set_numa_strategy(self.llama_cpu_numa_strategy)
        self.assets = AssetsManager(str(self.core_model_dir))
        self.tokenizer = Tokenizer.from_file(str(self.paths["tokenizer"]))

        self.codec_encoder = None
        self.speaker_encoder = None
        if self.paths["codec_enc_onnx"].exists() and self.paths["spk_enc_onnx"].exists():
            self.codec_encoder = CodecEncoder(str(self.paths["codec_enc_onnx"]))
            self.speaker_encoder = SpeakerEncoder(str(self.paths["spk_enc_onnx"]))

        self.talker_model, self.predictor_model = self._load_llama_models()

        self.decoder = DecoderAPI(
            str(self.paths["decoder_onnx"]),
            device_backend=self.resolved_device.decoder_backend,
            device_id=self.resolved_device.gpu_index,
            chunk_size=self.config.chunk_size,
            start_method=self.config.decoder_start_method,
        )
        if not self.decoder.wait_until_ready(timeout=60):
            raise RuntimeError("qwen3 decoder process failed to become ready")
        # talker/predictor 的 GGUF context、batch 和 prompt builder 都提升到 runtime 生命周期，
        # 避免每次 synthesize 都重新创建原生对象。
        self._init_shared_executor()

        if not self.resolved_device.gguf_gpu_enabled:
            # 明确打印最终线程策略，方便用户在 CPU benchmark 时直接看到实际生效值。
            logger.info(
                "[CPU Tune] llama 线程配置已生效: "
                f"n_threads={self.llama_cpu_thread_plan.n_threads}, "
                f"n_threads_batch={self.llama_cpu_thread_plan.n_threads_batch}, "
                f"source={self.llama_cpu_thread_plan.source}, "
                f"logical_cpus={self.llama_cpu_thread_plan.topology.logical_cpus}, "
                f"physical_cores={self.llama_cpu_thread_plan.topology.physical_cores}, "
                f"sockets={self.llama_cpu_thread_plan.topology.sockets}, "
                f"physical_cores_per_socket={self.llama_cpu_thread_plan.topology.physical_cores_per_socket}"
            )

        asset_count = sum(1 for path in self.bundle_root.rglob("*") if path.is_file())
        self.runtime_info = RuntimeInfo(
            model=self.model_name,
            adapter=self.model_spec.adapter_name,
            bundle_loaded=True,
            asset_count=asset_count,
            metadata={
                "bundle_root": str(self.bundle_root),
                "core_model_dir": str(self.core_model_dir),
                "qwen3_size": self.model_spec.size_label,
                "device_request": str(self.device),
                "execution_plan": str(self.execution_plan.raw_device) if self.execution_plan is not None else "",
                "resolved_device": self.resolved_device.normalized,
                "gguf_backend": self.resolved_device.gguf_backend,
                "decoder_backend": self.resolved_device.decoder_backend,
                "gpu_index": (
                    "" if self.resolved_device.gpu_index is None else str(self.resolved_device.gpu_index)
                ),
                "llama_cpu_threads": str(self.llama_cpu_thread_plan.n_threads),
                "llama_cpu_threads_batch": str(self.llama_cpu_thread_plan.n_threads_batch),
                "llama_cpu_thread_source": self.llama_cpu_thread_plan.source,
                "llama_cpu_numa_strategy": self.llama_cpu_numa_strategy,
                "init_seconds": f"{time.time() - start:.3f}",
            },
        )

    def _init_shared_executor(self) -> None:
        """
        初始化 qwen3 runtime 级共享执行器。

        参数:
            本函数不接收显式参数；依赖当前 runtime 已完成模型、资产、tokenizer 与线程计划加载。

        返回:
            无返回值；成功后共享的 talker/predictor context、batch 与 prompt builder 会常驻在 runtime 上。

        边界异常:
            任一 llama context、batch 或执行器初始化失败时直接抛出异常，阻止 runtime 进入半初始化状态。
        """
        # CPU 线程计划在 runtime 初始化时已经解析完成；共享执行器直接复用同一份策略。
        n_threads = self.llama_cpu_thread_plan.n_threads or None
        n_threads_batch = self.llama_cpu_thread_plan.n_threads_batch or None
        # 先用局部变量完成整套原生对象构造，只有全部成功后才挂到 runtime 上，避免中途失败留下半成品成员。
        talker_ctx = llama.LlamaContext(
            self.talker_model,
            n_ctx=self.config.n_ctx,
            embeddings=True,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
        )
        predictor_ctx = llama.LlamaContext(
            self.predictor_model,
            n_ctx=64,
            embeddings=False,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
        )
        talker_batch = llama.LlamaBatch(
            self.config.n_ctx, embd_dim=self.talker_model.n_embd
        )
        predictor_batch = llama.LlamaBatch(
            2, embd_dim=self.predictor_model.n_embd
        )
        talker = TalkerPredictor(
            self.talker_model, talker_ctx, talker_batch, self.assets
        )
        predictor = Predictor(
            self.predictor_model,
            predictor_ctx,
            predictor_batch,
            self.assets,
        )
        prompt_builder = PromptBuilder(self.tokenizer, self.assets)

        self.talker_ctx = talker_ctx
        self.predictor_ctx = predictor_ctx
        self.talker_batch = talker_batch
        self.predictor_batch = predictor_batch
        self.talker = talker
        self.predictor = predictor
        self.prompt_builder = prompt_builder

    def _reset_shared_executor_state(self) -> None:
        """
        在每次 qwen3 synthesize 开始前重置共享执行器状态。

        参数:
            本函数不接收显式参数；只操作 runtime 当前持有的共享 talker/predictor 执行器。

        返回:
            无返回值；共享 GGUF context 会恢复到干净状态，供下一次 synthesize 复用。

        边界异常:
            当共享执行器尚未初始化完成时抛出 RuntimeError，避免请求在不完整状态下继续执行。
        """
        # 共享执行器缺任一核心成员都说明 runtime 初始化异常或已经被 shutdown；此时继续推理会直接污染错误路径。
        if self.talker is None or self.predictor is None or self.predictor_ctx is None:
            raise RuntimeError("qwen3 shared executor is not initialized")
        # talker 持有额外的 cur_pos、trailing_text_pool 与 step_idx，必须通过 clear_memory 一并归零。
        self.talker.clear_memory()
        # predictor 只依赖 llama.cpp KV cache；每次新请求开始前清空即可保证 Q1-Q15 生成不串请求。
        self.predictor_ctx.clear_kv_cache()

    def create_session(self) -> "Qwen3Session":
        return Qwen3Session(self)

    def warm_decoder_state(self, codes: np.ndarray) -> DecoderState:
        info = self.decoder.decode(codes, task_id=f"warmup-{uuid.uuid4()}", is_final=True, stream=False)
        return info.final_state

    def warmup_for_init(self) -> None:
        """
        在 qwen3 runtime 初始化完成后，主动走一遍 ONNX 与 GGUF 的最小合法推理路径。

        参数:
            本函数不接收显式参数；依赖当前 runtime 已完成 encoder、decoder、共享 talker/predictor 的加载。

        返回:
            无返回值；预热成功后会将 `_init_warmed` 标记为 True。

        边界异常:
            任一 ONNX/GGUF 预热步骤失败都会直接抛出异常，阻止 TTSEngine 进入“半初始化成功”状态。
        """
        # 幂等保护必须放在入口处；TTSEngine 和测试都可能重复调用该方法，但真实预热只允许执行一次。
        if self._init_warmed:
            logger.info("[qwen3] 初始化预热已完成，跳过重复执行")
            return
        # 共享执行器缺失说明 runtime 初始化未完成；继续预热只会制造更隐蔽的空指针错误。
        if self.prompt_builder is None or self.talker is None or self.predictor is None:
            raise RuntimeError("qwen3 shared executor is not initialized")

        logger.info("[qwen3] 开始初始化预热：ONNX + GGUF")
        # 固定使用 1 秒静音作为初始化预热输入，目标是稳定产出最短合法结构，而不是模拟真实业务语义。
        silence = np.zeros((self.sample_rate,), dtype=np.float32)
        # codebook 数量由当前已加载的 embedding table 决定，避免把“16 路 codec”硬编码到 warmup 实现里。
        num_codebooks = len(self.assets.emb_tables)
        if num_codebooks <= 0:
            raise RuntimeError("qwen3 assets do not contain codec embedding tables")

        prompt_codes = np.zeros((1, num_codebooks), dtype=np.int64)
        # speaker embedding 的维度与 text embedding 维度对齐，确保走 clone prompt builder 时满足当前输入协议。
        speaker_embedding = np.zeros((self.assets.text_table.shape[1],), dtype=np.float32)

        # 先走 ONNX codec encoder；如果 encoder 真存在，就必须拿它产出的真实合法 codes 继续后续 warmup。
        if self.codec_encoder is not None:
            prompt_codes = np.asarray(self.codec_encoder.encode(silence), dtype=np.int64)
            if prompt_codes.size == 0 or prompt_codes.ndim != 2 or prompt_codes.shape[0] == 0:
                raise RuntimeError("qwen3 codec encoder warmup produced empty prompt codes")
        # speaker encoder 只在存在时执行；缺失时回退为零向量，保持无 prompt encoder 资产的 bundle 仍可初始化。
        if self.speaker_encoder is not None:
            speaker_embedding = np.asarray(self.speaker_encoder.encode(silence), dtype=np.float32)

        # decoder worker 的完整路径要在初始化期先走通一遍，并把 warm state 复用给最小 GGUF prompt。
        decoder_state = self.warm_decoder_state(prompt_codes)

        # GGUF 预热必须严格从干净状态开始，并且无论成功失败都清空 talker/predictor 的残留 KV/cache。
        self._reset_shared_executor_state()
        try:
            prompt_item = VoiceClonePromptItem(
                prompt_code=prompt_codes,
                prompt_spk_embedding=speaker_embedding,
                icl_mode=False,
                prompt_text=None,
                state=decoder_state,
            )
            # 这里复用 runtime 当前真实在用的 clone prompt builder，避免 warmup 逻辑和正式 synthesize 输入协议漂移。
            prompt_data = self.prompt_builder.build_clone_prompt(
                "a",
                lang_id=LANGUAGE_MAP["english"],
                stream=True,
                prompt_item=prompt_item,
            )
            # 正式请求路径会把 decoder state 回填到 PromptData；初始化预热必须沿用同一协议。
            prompt_data.state = prompt_item.state

            # 先让 talker 走完整个最小 prefill，再把最后一个 hidden 交给 predictor 跑一帧最小合法 forward。
            master_hidden = self.talker.prefill(prompt_data, seq_id=0)
            # 第一个 codec token 优先复用真实 warmup code；只有极端空输入回退才使用 0。
            code_0 = int(prompt_codes[0, 0]) if prompt_codes.size > 0 else 0
            self.predictor.predict_frame(master_hidden, code_0, do_sample=False)
        finally:
            logger.info("[qwen3] 初始化预热：清理共享 GGUF 执行器状态")
            self._reset_shared_executor_state()

        self._init_warmed = True
        logger.info("[qwen3] 初始化预热完成")

    def shutdown(self) -> None:
        # 共享执行器从“每次请求临时创建”改成“runtime 常驻持有”之后，必须在 shutdown 时显式断开引用，
        # 否则 context/batch/native 指针会一直挂到 runtime 生命周期结束。
        self.prompt_builder = None
        self.talker = None
        self.predictor = None
        self.talker_batch = None
        self.predictor_batch = None
        self.talker_ctx = None
        self.predictor_ctx = None
        if hasattr(self, "decoder") and self.decoder is not None:
            self.decoder.shutdown()
            self.decoder = None
        self.codec_encoder = None
        self.speaker_encoder = None
        self.talker_model = None
        self.predictor_model = None


class Qwen3Session:
    def __init__(self, runtime: Qwen3Runtime) -> None:
        self.runtime = runtime
        self.assets = runtime.assets
        self.tokenizer = runtime.tokenizer
        self.n_ctx = runtime.config.n_ctx
        self.chunk_size = runtime.config.chunk_size
        self.sample_rate = runtime.sample_rate
        # session 现在只是一层轻量包装器：真正的 GGUF context/batch/executor 统一由 runtime 持有并跨请求复用。
        if runtime.talker is None or runtime.predictor is None or runtime.prompt_builder is None:
            raise RuntimeError("qwen3 shared executor is not initialized")
        self.talker_ctx = runtime.talker_ctx
        self.predictor_ctx = runtime.predictor_ctx
        self.talker_batch = runtime.talker_batch
        self.predictor_batch = runtime.predictor_batch
        self.talker = runtime.talker
        self.predictor = runtime.predictor
        self.prompt_builder = runtime.prompt_builder
        self.decoder = runtime.decoder
        self._synthesize_lock = runtime._synthesize_lock

    def build_prompt_data(
        self,
        text: str,
        prompt: Qwen3ClonePromptAsset,
        language: str,
        stream: bool = True,
        icl: bool = False
    ) -> PromptData:
        if not isinstance(prompt, Qwen3ClonePromptAsset):
            raise TypeError("qwen3 runtime requires Qwen3ClonePromptAsset")
        resolved_language = language
        if map_language(resolved_language) is None:
            resolved_language = "chinese"
        prompt_item = VoiceClonePromptItem(
            prompt_code=prompt.prompt_codes,
            prompt_spk_embedding=prompt.speaker_embedding,
            icl_mode=icl,
            prompt_text=prompt.prompt_text,
            state=prompt.decoder_state,
        )
        prompt_data = self.prompt_builder.build_clone_prompt(
            text,
            lang_id=map_language(resolved_language),
            stream=stream,
            prompt_item=prompt_item,
        )
        prompt_data.state = prompt_item.state
        return prompt_data

    def synthesize(
        self,
        text: str,
        prompt: Qwen3ClonePromptAsset,
        language: str = "auto",
        icl: bool = False,
        stream: bool = True,
        track_performance: bool = False,
        max_length: int|None = None,
        min_length: int|None = None,
        remove_meaningless_chars: bool = False,
        generation_options: dict[str, object] | None = None,
    ) -> Iterable[np.ndarray]:
        # 允许调用方直接控制分段上限/下限，用更短的 segment 换取更稳的峰值显存占用。
        processed = preprocess_text(
            text,
            language=language,
            max_length=max_length if max_length is not None else self.runtime.config.max_length,
            min_length=min_length if min_length is not None else self.runtime.config.min_length,
            tokenizer=self.tokenizer,
            config=self.runtime.text_preprocess_config,
            remove_meaningless_chars=remove_meaningless_chars,
        )
        if not processed.segments:
            raise ValueError("text preprocessing produced no segments")
        resolved_language = language
        if map_language(resolved_language) is None:
            resolved_language = processed.language
        total_timing = Timing() if track_performance else None
        lock_acquired = False
        try:
            # 从这里开始进入共享 GGUF context 的真实执行阶段；同一个 runtime 必须串行化整个请求生命周期。
            self._synthesize_lock.acquire()
            lock_acquired = True
            # 每次 synthesize 都先清空共享 talker/predictor，避免上一请求残留 KV/cache 污染当前 prompt。
            self.runtime._reset_shared_executor_state()

            for idx, segment in enumerate(processed.segments):
                self.talker.clear_memory()
                logger.info(f"正在推理第{idx}段文本：{segment}")
                timing = Timing() if track_performance else None
                prompt_data = self.build_prompt_data(
                    text=segment,
                    prompt=prompt,
                    language=resolved_language,
                    stream=stream,
                    icl=icl
                )
                yield from self._run_engine_loop_stream(
                    prompt_data=prompt_data,
                    timing=timing,
                    generation_options=generation_options or {},
                )
                if total_timing is not None and timing is not None:
                    # 多段文本时统一聚合到一次 synthesize 调用的总统计里。
                    total_timing.merge_from(timing)

            if track_performance and total_timing is not None:
                # 输出所有chunk的生成耗时
                # chunk 生成和 decoder 响应理论上一一对应；这里取较短长度，避免异常中断时性能日志反向制造新错误。
                for idx in range(min(len(total_timing.chunk_gen_times), len(total_timing.decoder_compute_times))):
                    msg = f"Chunk {idx} generation time: {format_perf_value(total_timing.chunk_gen_times[idx] + total_timing.decoder_compute_times[idx])}"
                    msg += f", generate code time: {format_perf_value(total_timing.chunk_gen_times[idx])}"
                    msg += f", decode time: {format_perf_value(total_timing.decoder_compute_times[idx])}"
                    logger.info(msg)

                logger.info(f"Total inference time: {format_perf_value(total_timing.total_inference_time)}")
                logger.info(f"Prefill time: {format_perf_value(total_timing.prefill_time)}")
                logger.info(f"First chunk audio generation time: {format_perf_value(total_timing.first_audio_latency)}, generate code time: {format_perf_value(total_timing.first_chunk_latency)}, decode time: {format_perf_value(total_timing.first_decode_latency)}")
                logger.info(f"Avg chunk code generation time: {format_perf_value(total_timing.avg_chunk_gen_time)}")
                logger.info(f"Avg chunk decoder time: {format_perf_value(total_timing.avg_decoder_time)}")
                logger.info(f"Avg chunk RTF: {format_perf_value(total_timing.avg_chunk_rtf)}")
        finally:
            # 锁必须覆盖到流式生成器真正结束；只有在 decoder worker 收尾完成后才能释放给下一次请求。
            if lock_acquired:
                self._synthesize_lock.release()
            
            

    def _create_sampler(self, *, temperature: float, top_p: float, top_k: int, min_p: float = 0.0, repeat_penalty: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, penalty_last_n: int = 128, seed: int = 42) -> llama.LlamaSampler:
        return llama.LlamaSampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            penalty_last_n=penalty_last_n,
            seed=seed,
        )

    def _run_engine_loop_gen(
        self,
        prompt_data: PromptData,
        timing: Timing | None,
        generation_options: dict[str, object],
    ):
        prefill_start = time.time()
        master_hidden = self.talker.prefill(prompt_data, seq_id=0)
        if timing is not None:
            timing.prefill_time = time.time() - prefill_start

        talker_sampler = self._create_sampler(
            temperature=float(generation_options.get("temperature", 0.9)),
            top_p=float(generation_options.get("top_p", 0.95)),
            top_k=int(generation_options.get("top_k", 50)),
            min_p=float(generation_options.get("min_p", 0.0)),
            repeat_penalty=float(generation_options.get("repeat_penalty", 1.05)),
            frequency_penalty=float(generation_options.get("frequency_penalty", 0.0)),
            presence_penalty=float(generation_options.get("presence_penalty", 0.0)),
            penalty_last_n=int(generation_options.get("penalty_last_n", 128)),
            seed=int(generation_options.get("seed", 42)),
        )
        predictor_sampler = self._create_sampler(
            temperature=float(generation_options.get("sub_temperature", 0.9)),
            top_p=float(generation_options.get("sub_top_p", 1.0)),
            top_k=int(generation_options.get("sub_top_k", 50)),
            seed=int(generation_options.get("sub_seed", 42)),
        )

        allow_tokens = {
            SPECIAL_TOKENS["EOS"],
            SPECIAL_TOKENS["PAD"],
            SPECIAL_TOKENS["BOS"],
        }
        max_steps = int(generation_options.get("max_new_tokens", self.n_ctx))

        try:
            for step_idx in range(max_steps):
                code_0 = talker_sampler.sample(
                    self.talker.ctx,
                    idx=-1,
                    limit_start=0,
                    limit_end=2048,
                    allow_tokens=allow_tokens,
                )
                talker_sampler.accept(code_0)
                if code_0 == SPECIAL_TOKENS["EOS"]:
                    break
 
                step_codes, step_embeds = self.predictor.predict_frame(
                    master_hidden,
                    code_0,
                    sampler=predictor_sampler,
                )

                summed_embed = np.sum(step_embeds, axis=0)
                master_hidden = self.talker.decode_step(summed_embed, seq_id=0)
    
                yield step_codes, summed_embed
        finally:
            talker_sampler.free()
            predictor_sampler.free()

    def _run_engine_loop_stream(
        self,
        prompt_data: PromptData,
        timing: Timing | None,
        generation_options: dict[str, object],
    ) -> Iterable[np.ndarray]:
        current_task_id = str(uuid.uuid4())
        output_queue = self.decoder.subscribe_stream(current_task_id)
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._run_engine_loop_stream_worker,
            kwargs={
                "current_task_id": current_task_id,
                "prompt_data": prompt_data,
                "timing": timing,
                "generation_options": generation_options,
                "output_queue": output_queue,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker.start()

        try:
            while True:
                item = output_queue.get()
                if isinstance(item, _Qwen3StreamError):
                    raise item.error
                if not isinstance(item, DecoderResponse):
                    continue
                if item.msg_type == "AUDIO" and item.audio is not None:
                    if timing is not None:
                        timing.decoder_compute_times.append(item.compute_time)
                    yield item.audio
                    continue
                if item.msg_type == "FINISH":
                    break
        finally:
            stop_event.set()
            self.decoder.unsubscribe_stream(current_task_id, output_queue)
            worker.join(timeout=1.0)

    def _submit_decoder_chunk(
        self,
        *,
        current_task_id: str,
        chunk_buffer: list[list[int]],
        prompt_data: PromptData,
        timing: Timing | None,
        last_chunk_time: float,
    ) -> float:
        """
        向 qwen3 decoder 提交一个完整 chunk，并返回新的 chunk 计时锚点。

        参数:
            current_task_id: 当前 synthesize 请求对应的 decoder task_id。
            chunk_buffer: 已经攒满一个 decode window 的 code 帧列表。
            prompt_data: 当前请求的 prompt 数据，首包会把其中的 decoder state 注入 worker。
            timing: 可选性能统计对象；存在时记录本次攒满 chunk 的墙钟耗时。
            last_chunk_time: 上一个 chunk 提交时刻。

        返回:
            返回新的 `last_chunk_time`，用于下一次 chunk 统计。

        异常:
            当 decode 请求跨进程投递失败时透出异常。
        """
        if timing is not None:
            timing.chunk_gen_times.append(time.time() - last_chunk_time)
            last_chunk_time = time.time()
        self.decoder.decode(
            np.asarray(chunk_buffer, dtype=np.int64),
            task_id=current_task_id,
            is_final=False,
            stream=True,
            state=prompt_data.state,
        )
        return last_chunk_time

    def _finalize_decoder_stream(
        self,
        *,
        current_task_id: str,
        chunk_buffer: list[list[int]],
        prompt_data: PromptData,
        timing: Timing | None,
        last_chunk_time: float,
    ) -> None:
        """
        以 final 请求收尾当前 qwen3 decoder task，确保 worker flush 尾部状态并发出 FINISH。

        参数:
            current_task_id: 当前 synthesize 请求对应的 decoder task_id。
            chunk_buffer: 尚未送入 decoder 的尾部 code 帧列表。
            prompt_data: 当前请求的 prompt 数据。
            timing: 可选性能统计对象；尾部有残留 chunk 时会补记最后一次攒码耗时。
            last_chunk_time: 上一个 chunk 提交时刻。

        返回:
            本函数无返回值；会阻塞到 decoder worker 返回 FINISH。

        异常:
            当 final decode 失败或 worker 无法正常收尾时透出异常。
        """
        if chunk_buffer and timing is not None:
            timing.chunk_gen_times.append(time.time() - last_chunk_time)
        self.decoder.decode(
            np.asarray(chunk_buffer, dtype=np.int64)
            if chunk_buffer
            else np.zeros((0, 16), dtype=np.int64),
            task_id=current_task_id,
            is_final=True,
            stream=True,
            state=prompt_data.state,
        )

    def _run_engine_loop_stream_worker(
        self,
        *,
        current_task_id: str,
        prompt_data: PromptData,
        timing: Timing | None,
        generation_options: dict[str, object],
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        """
        在后台线程中持续生成 qwen3 code 并提交 decoder chunk。

        参数:
            current_task_id: 当前 synthesize 请求对应的 decoder task_id。
            prompt_data: 当前请求的 prompt 数据。
            timing: 可选性能统计对象；后台线程记录 code 生成耗时，前台生成器记录 decoder 输出耗时。
            generation_options: 调用方传入的采样参数覆盖项。
            output_queue: 输出队列；仅在后台线程异常时写入 `_Qwen3StreamError`，正常音频由 decoder listener 直接 fan-out。
            stop_event: 当前流式请求的停止信号；前台提前结束时后台线程会尽快停止继续生成。

        返回:
            本函数无返回值。

        异常:
            本函数内部会捕获异常并通过 output_queue 回传 `_Qwen3StreamError`。
        """
        try:
            chunk_buffer: list[list[int]] = []
            last_chunk_time = time.time()
            for step_codes, _summed_vec in self._run_engine_loop_gen(
                prompt_data,
                timing,
                generation_options,
            ):
                if stop_event.is_set():
                    break
                chunk_buffer.append(step_codes)
                if len(chunk_buffer) < self.chunk_size:
                    continue
                last_chunk_time = self._submit_decoder_chunk(
                    current_task_id=current_task_id,
                    chunk_buffer=chunk_buffer,
                    prompt_data=prompt_data,
                    timing=timing,
                    last_chunk_time=last_chunk_time,
                )
                chunk_buffer = []

            self._finalize_decoder_stream(
                current_task_id=current_task_id,
                chunk_buffer=chunk_buffer,
                prompt_data=prompt_data,
                timing=timing,
                last_chunk_time=last_chunk_time,
            )
        except Exception as exc:
            output_queue.put(_Qwen3StreamError(exc))


def prompt_asset_from_clone_payload(
    *,
    role_name: str,
    sample_rate: int,
    prompt_text: str | None,
    speaker_embedding: np.ndarray,
    prompt_codes: np.ndarray,
    decoder_state: DecoderState | None,
    metadata: dict[str, str] | None = None,
    model_name: str = MODEL_NAME,
) -> Qwen3ClonePromptAsset:
    return Qwen3ClonePromptAsset(
        role_name=role_name,
        model=model_name,
        sample_rate=sample_rate,
        metadata=metadata or {},
        prompt_text=prompt_text,
        speaker_embedding=np.asarray(speaker_embedding, dtype=np.float32),
        prompt_codes=np.asarray(prompt_codes, dtype=np.int64),
        decoder_state=decoder_state,
    )


def serialize_prompt_asset(asset: Qwen3ClonePromptAsset) -> SerializedPromptAsset:
    """序列化prompt资产"""
    arrays: dict[str, np.ndarray] = {
        "speaker_embedding": np.asarray(asset.speaker_embedding, dtype=np.float32),
        "prompt_codes": np.asarray(asset.prompt_codes, dtype=np.int64),
    }

    json_data: dict[str, object] = {"prompt_text": asset.prompt_text}
    if asset.decoder_state is not None:
        arrays["decoder_pre_conv_history"] = np.asarray(asset.decoder_state.pre_conv_history)
        arrays["decoder_latent_buffer"] = np.asarray(asset.decoder_state.latent_buffer)
        arrays["decoder_conv_history"] = np.asarray(asset.decoder_state.conv_history)
        for idx, item in enumerate(asset.decoder_state.kv_cache):
            arrays[f"decoder_kv_cache_{idx}"] = np.asarray(item)
        json_data["decoder_skip_samples"] = int(asset.decoder_state.skip_samples)

    return SerializedPromptAsset(
        role_name=asset.role_name,
        model=asset.model,
        sample_rate=asset.sample_rate,
        metadata=asset.metadata,
        arrays=arrays,
        json_data=json_data,
    )


def deserialize_prompt_asset(payload: SerializedPromptAsset) -> Qwen3ClonePromptAsset:
    kv_items: list[np.ndarray] = []
    # 这里必须按数值下标恢复 KV cache 顺序，不能走字符串排序。
    # 否则 10/11 会排到 2/3 前面，持久化后的 decoder 记忆会被打乱，只会在 role_name 复用路径触发。
    kv_keys = sorted(
        (key for key in payload.arrays if key.startswith("decoder_kv_cache_")),
        key=lambda item: int(item.rsplit("_", 1)[1]),
    )
    for key in kv_keys:
        kv_items.append(np.asarray(payload.arrays[key]))
    decoder_state = None
    if "decoder_pre_conv_history" in payload.arrays:
        decoder_state = DecoderState(
            pre_conv_history=np.asarray(payload.arrays["decoder_pre_conv_history"]),
            latent_buffer=np.asarray(payload.arrays["decoder_latent_buffer"]),
            conv_history=np.asarray(payload.arrays["decoder_conv_history"]),
            kv_cache=kv_items,
            skip_samples=int(payload.json_data.get("decoder_skip_samples", 0)),
        )
    return Qwen3ClonePromptAsset(
        role_name=payload.role_name,
        model=payload.model,
        sample_rate=payload.sample_rate,
        metadata=payload.metadata,
        prompt_text=payload.json_data.get("prompt_text"),
        speaker_embedding=np.asarray(payload.arrays["speaker_embedding"], dtype=np.float32),
        prompt_codes=np.asarray(payload.arrays["prompt_codes"], dtype=np.int64),
        decoder_state=decoder_state,
    )
