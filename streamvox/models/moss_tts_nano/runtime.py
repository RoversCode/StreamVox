from __future__ import annotations

from ..onnx_backend import resolve_explicit_onnx_backend

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from streamvox_runtime import ensure_single_onnxruntime_distribution, initialize_nvidia_runtime, nvidia_library_directories
ensure_single_onnxruntime_distribution()
import onnxruntime as ort
import yaml

from ...model_bundle import validate_local_bundle
from ...utils.log import logger
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, RuntimeInfo, SerializedPromptAsset
from ..text_processing import TextPreprocessConfig
from .ort_runtime import MossOnnxCoreRuntime
from .schema.record import MossTiming, log_moss_performance
from .spec import MOSS_TTS_NANO_ONNX_SPEC, MossTTSNanoOnnxModelSpec
from .text import MOSS_TTS_NANO_TEXT_PREPROCESS_CONFIG, preprocess_text


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "moss_tts_nano_onnx.yaml"


@dataclass(slots=True)
class MossTTSNanoSampleConfig:
    """
    MOSS-TTS-Nano 采样参数配置。

    参数:
        sample_mode: ONNX local frame 采样路径，支持 greedy/fixed/full。
        do_sample: 是否启用随机采样。
        audio_temperature: audio codebook 采样温度。
        audio_top_p: audio codebook top-p。
        audio_top_k: audio codebook top-k。
        audio_repetition_penalty: audio token 重复惩罚。

    返回:
        该对象只承载配置值，不直接返回推理结果。

    边界异常:
        字段合法性由 adapter.normalize_synthesize_kwargs 和 runtime 加载阶段共同保证。
    """

    sample_mode: str = "fixed"
    do_sample: bool = True
    audio_temperature: float = 0.8
    audio_top_p: float = 0.95
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.2


@dataclass(slots=True)
class MossTTSNanoConfig:
    """
    MOSS-TTS-Nano runtime 配置。

    参数:
        sample_rate: 输出采样率；会被 codec meta 的真实值覆盖。
        max_new_frames: 单段文本最多生成的 audio frame 数。
        max_length: 默认文本分段 token 预算。
        min_length: StreamVox 分段后的短段合并下限。
        tts_dirname: ckpt 下 TTS ONNX 子目录名。
        codec_dirname: ckpt 下 codec ONNX 子目录名。
        browser_manifest_filename: TTS 子目录下 browser_poc_manifest 文件名。
        sample: 默认采样参数。

    返回:
        配置对象只用于 runtime 初始化和 session 默认参数。

    边界异常:
        yaml 字段类型无法转换时会在 load_moss_tts_nano_config 中抛出异常。
    """

    sample_rate: int = 48000
    max_new_frames: int = 375
    max_length: int = 80
    min_length: int = 12
    tts_dirname: str = "MOSS-TTS-Nano-100M-ONNX"
    codec_dirname: str = "MOSS-Audio-Tokenizer-Nano-ONNX"
    browser_manifest_filename: str = "browser_poc_manifest.json"
    sample: MossTTSNanoSampleConfig = field(default_factory=MossTTSNanoSampleConfig)


@dataclass(slots=True)
class MossTTSNanoPromptAsset(BasePromptAsset):
    """
    MOSS-TTS-Nano prompt 资产。

    参数:
        role_name: StreamVox prompt 角色名。
        model: prompt 所属 StreamVox 模型名。
        sample_rate: prompt 参考音频采样率，等于 codec 输出采样率。
        prompt_text: continuation 模式可选参考文本；voice_clone 默认不会使用该字段。
        prompt_audio_codes: 参考音频经 MOSS audio tokenizer 编码后的 frame/n_vq token。
        default_mode: 当前 prompt 默认合成模式；有参考音频时通常是 voice_clone，无参考时是 continuation。
        metadata: 额外追踪信息。

    返回:
        该对象由 adapter.make_prompt 或 make_default_prompt 构造，并可序列化到 PromptCache。

    边界异常:
        prompt model 与 runtime model 不一致时会在合成入口拒绝。
    """

    prompt_text: str | None = None
    prompt_audio_codes: np.ndarray = field(default_factory=lambda: np.zeros((0, 16), dtype=np.int32))
    default_mode: str = "voice_clone"


@dataclass(slots=True)
class MossTTSNanoResolvedDevice:
    """
    MOSS-TTS-Nano ONNX 后端解析结果。

    参数:
        request: 用户原始 device 字符串。
        onnx_backend: 最终选择的 ONNX 后端名称。
        gpu_index: GPU 设备序号；CPU 后端为空。

    返回:
        对象提供 normalized 和 providers 辅助属性。

    边界异常:
        本数据结构不主动抛出异常。
    """

    request: str
    onnx_backend: str
    gpu_index: int | None = None

    @property
    def normalized(self) -> str:
        """
        返回面向日志展示的设备字符串。

        参数:
            本属性无参数。

        返回:
            返回 cpu/cuda:N/dml:N 等规范化字符串。

        边界异常:
            本属性不主动抛出异常。
        """

        if self.gpu_index is None:
            return self.onnx_backend
        return f"{self.onnx_backend}:{self.gpu_index}"

    @property
    def providers(self) -> list[str]:
        """
        返回 ONNX Runtime provider 列表。

        参数:
            本属性无参数。

        返回:
            返回按优先级排列的 provider 名称列表。

        边界异常:
            本属性不主动抛出异常；不可用 provider 会在 session 创建时暴露。
        """

        if self.onnx_backend == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.onnx_backend == "dml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


def load_moss_tts_nano_config() -> MossTTSNanoConfig:
    """
    加载 MOSS-TTS-Nano yaml 配置。

    参数:
        本函数无显式参数；配置路径固定为 streamvox/models/config/moss_tts_nano_onnx.yaml。

    返回:
        返回完成默认值合并后的 MossTTSNanoConfig。

    边界异常:
        yaml 结构非法或字段无法转换为目标类型时抛出异常。
    """

    # 配置文件缺失时返回内置默认值，便于测试构造最小 runtime。
    if not CONFIG_PATH.exists():
        return MossTTSNanoConfig()

    # yaml 分 runtime、sample_config、paths 三块，避免路径和采样参数互相污染。
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime", {})
    paths = payload.get("paths", {})
    sample_config = payload.get("sample_config", {})
    return MossTTSNanoConfig(
        sample_rate=int(runtime.get("sample_rate", 48000)),
        max_new_frames=int(runtime.get("max_new_frames", 375)),
        max_length=int(runtime.get("max_length", 80)),
        min_length=int(runtime.get("min_length", 12)),
        tts_dirname=str(paths.get("tts_dirname", "MOSS-TTS-Nano-100M-ONNX")),
        codec_dirname=str(paths.get("codec_dirname", "MOSS-Audio-Tokenizer-Nano-ONNX")),
        browser_manifest_filename=str(paths.get("browser_manifest_filename", "browser_poc_manifest.json")),
        sample=MossTTSNanoSampleConfig(
            sample_mode=str(sample_config.get("sample_mode", "fixed")),
            do_sample=bool(sample_config.get("do_sample", True)),
            audio_temperature=float(sample_config.get("audio_temperature", 0.8)),
            audio_top_p=float(sample_config.get("audio_top_p", 0.95)),
            audio_top_k=int(sample_config.get("audio_top_k", 25)),
            audio_repetition_penalty=float(sample_config.get("audio_repetition_penalty", 1.2)),
        ),
    )


def _ort_provider_available(provider: str) -> bool:
    """
    判断当前环境是否提供指定 ONNX Runtime provider。

    参数:
        provider: provider 名称，例如 CUDAExecutionProvider。

    返回:
        provider 可用返回 True，否则返回 False。

    边界异常:
        onnxruntime 导入失败时返回 False，不向外抛出。
    """

    # provider 探测只用于设备回退，不应该因为环境异常阻断 CPU fallback。
    try:
        return provider in ort.get_available_providers()
    except Exception:
        return False


def _missing_windows_cuda_runtime_dlls() -> list[str]:
    """
    检查发行包是否具备 ONNX Runtime CUDA provider 的核心运行库。
    参数:
        本函数无显式参数；只检查共享加载器管理的目录。
    返回:
        返回缺失的 CUDA/cuDNN DLL 文件名列表；列表为空表示具备基础加载条件。
    边界异常:
        本函数只做轻量路径探测，不主动加载 DLL；PATH 读取异常时按缺失处理并回退 CPU。
    """

    # 业务意图：`ort.get_available_providers()` 只能证明 onnxruntime-gpu 包注册了 CUDA EP，
    # 不能证明 `onnxruntime_providers_cuda.dll` 的二级依赖已经能被 Windows loader 找到。
    # 探测与加载使用同一组目录，避免系统 CUDA 恰好存在时掩盖发行包缺库。
    required_dlls = (
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
    )
    extra_search_dirs = nvidia_library_directories(consumer="onnx_cuda")

    missing: list[str] = []
    for dll_name in required_dlls:
        if any((search_dir / dll_name).exists() for search_dir in extra_search_dirs):
            continue
        missing.append(dll_name)
    return missing


def _windows_cuda_provider_loadable() -> bool:
    """
    判断 Windows 上 CUDAExecutionProvider 是否具备基础可加载条件。
    参数:
        本函数无显式参数；读取当前 onnxruntime provider 注册表与 PATH 中的 CUDA 运行库。
    返回:
        CUDA EP 已注册且核心 CUDA/cuDNN DLL 可发现时返回 True，否则返回 False。
    边界异常:
        探测过程中不抛出异常；无法确认时保守返回 False，让上层进入 DML/CPU fallback。
    """

    if not _ort_provider_available("CUDAExecutionProvider"):
        return False
    missing_dlls = _missing_windows_cuda_runtime_dlls()
    if not missing_dlls:
        try:
            initialize_nvidia_runtime(consumer="onnx_cuda", required=True)
        except Exception as exc:
            logger.warning(
                "[Device Fallback][IMPORTANT] MOSS-TTS-Nano CUDA 依赖预加载失败，将回退到 DML/CPU: %s",
                exc,
            )
            return False
        return True
    logger.warning(
        "[Device Fallback][IMPORTANT] MOSS-TTS-Nano 检测到 CUDAExecutionProvider 已注册，"
        "但发行环境缺少 ONNX Runtime CUDA 依赖 DLL: %s；将跳过 CUDA 并回退到 DML/CPU。"
        "如需启用 CUDA，请安装 StreamVox 的 onnx-cuda 依赖组。",
        ", ".join(missing_dlls),
    )
    return False


def _resolve_moss_tts_nano_device(device: str, execution_plan: ExecutionPlan | None) -> MossTTSNanoResolvedDevice:
    """
    将 StreamVox 统一执行计划解析为 MOSS ONNX 后端。

    参数:
        device: 用户原始 device 字符串。
        execution_plan: adapter.resolve_execution_plan 生成的统一执行计划。

    返回:
        返回 MossTTSNanoResolvedDevice。

    边界异常:
        execution_plan 缺失 onnx target 时抛出 ValueError。
    """

    # runtime 必须拿到 adapter 生成的执行计划，避免直接解析 device 时和其它模型语义漂移。
    if execution_plan is None:
        raise ValueError("moss-tts-nano runtime requires an execution plan")
    onnx_target = execution_plan.get("onnx")
    if onnx_target is None:
        raise ValueError("moss-tts-nano execution plan must include onnx target")

    explicit = resolve_explicit_onnx_backend(onnx_target)
    if explicit is not None:
        backend, index = explicit
        return MossTTSNanoResolvedDevice(request=device, onnx_backend=backend, gpu_index=index)

    # CPU 模式显式指定 CPUExecutionProvider。
    if onnx_target.mode == "cpu":
        return MossTTSNanoResolvedDevice(request=device, onnx_backend="cpu", gpu_index=None)

    # ONNX family 在 Windows 上统一采用 CUDA -> DML -> CPU 的优先级。
    # 这样 MOSS 和 qwen3 / s2_pro / voxcpm2 的后端决策保持一致，避免同机型出现模型间行为分裂。
    requested_index = onnx_target.device_index if onnx_target.device_index is not None else 0
    if os.name == "nt":
        if _windows_cuda_provider_loadable():
            return MossTTSNanoResolvedDevice(request=device, onnx_backend="cuda", gpu_index=requested_index)
        if _ort_provider_available("DmlExecutionProvider"):
            return MossTTSNanoResolvedDevice(request=device, onnx_backend="dml", gpu_index=requested_index)
        if onnx_target.mode == "gpu":
            logger.warning("[Device Fallback][IMPORTANT] 用户请求 MOSS-TTS-Nano GPU，但 CUDA/DML 都不可用，将回退到 CPU")
        return MossTTSNanoResolvedDevice(request=device, onnx_backend="cpu", gpu_index=None)

    # 非 Windows 当前只启用 CUDA；其它 provider 不在本模型首版适配中引入。
    if _ort_provider_available("CUDAExecutionProvider"):
        return MossTTSNanoResolvedDevice(request=device, onnx_backend="cuda", gpu_index=requested_index)
    if onnx_target.mode == "gpu":
        logger.warning("[Device Fallback][IMPORTANT] 用户请求 MOSS-TTS-Nano GPU，但 CUDAExecutionProvider 不可用，将回退到 CPU")
    return MossTTSNanoResolvedDevice(request=device, onnx_backend="cpu", gpu_index=None)


def _prompt_codes_to_list(prompt_audio_codes: np.ndarray) -> list[list[int]]:
    """
    将 prompt audio codes ndarray 转换为 Python list。

    参数:
        prompt_audio_codes: frame/n_vq 的 numpy 数组。

    返回:
        返回 frame/n_vq 的普通整数列表。

    边界异常:
        数组维度不为二维时抛出 ValueError。
    """

    # 统一 dtype 后再检查维度，保证 PromptCache 恢复出来的数组也能稳定使用。
    codes = np.asarray(prompt_audio_codes, dtype=np.int32)
    if codes.size == 0:
        return []
    if codes.ndim != 2:
        raise ValueError(f"prompt_audio_codes must be a 2D array, got {codes.shape}")
    return [[int(value) for value in row] for row in codes.tolist()]


class MossTTSNanoRuntime:
    """
    MOSS-TTS-Nano StreamVox runtime。

    参数:
        model_dir: 本地 StreamVox 模型 bundle 根目录。
        model_spec: 当前模型规格。
        device: 用户传给 TTSEngine 的原始 device 字符串。
        verify_model_sha256: 是否校验 manifest 中声明的运行期资产 sha256。
        execution_plan: adapter 解析出的统一执行计划。
        text_preprocess_config: StreamVox 注入的文本预处理配置；为空时使用模型默认配置。

    返回:
        初始化后得到可创建 session 的 runtime。

    边界异常:
        bundle 缺失、manifest 模型名不匹配、资产缺失、ONNX 加载失败或 tokenizer 缺失时抛出异常。
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        model_spec: MossTTSNanoOnnxModelSpec = MOSS_TTS_NANO_ONNX_SPEC,
        device: str = "auto",
        verify_model_sha256: bool = True,
        execution_plan: ExecutionPlan | None = None,
        text_preprocess_config: TextPreprocessConfig | None = None,
    ) -> None:
        self.model_name = model_spec.model_name
        self.model_spec = model_spec
        self.bundle_root = Path(model_dir)
        self.ckpt_root = self.bundle_root / "ckpt"
        self.device = device
        self.execution_plan = execution_plan
        self.text_preprocess_config = text_preprocess_config or MOSS_TTS_NANO_TEXT_PREPROCESS_CONFIG

        # 先读取轻量配置，再进行 bundle 校验，失败时错误边界更清晰。
        self.config = load_moss_tts_nano_config()
        bundle = validate_local_bundle(
            self.bundle_root,
            self.model_name,
            verify_model_sha256=verify_model_sha256,
        )

        # 设备解析只关注 ONNX family；GPU 不可用时明确日志回退 CPU。
        self.resolved_device = _resolve_moss_tts_nano_device(device, execution_plan)

        # 低层 ONNX 执行器只接收 ckpt 根目录，不依赖 ref_code 任何文件。
        self.core = MossOnnxCoreRuntime(
            asset_root=self.ckpt_root,
            providers=self.resolved_device.providers,
            max_new_frames=self.config.max_new_frames,
            do_sample=self.config.sample.do_sample,
            sample_mode=self.config.sample.sample_mode,
        )

        # runtime_info 汇总模型目录、provider 和资产数量，便于用户定位初始化问题。
        self.runtime_info = RuntimeInfo(
            model=self.model_name,
            adapter=model_spec.adapter_name,
            bundle_loaded=True,
            asset_count=len(bundle.asset_paths),
            metadata={
                "model_dir": str(self.bundle_root),
                "device": self.resolved_device.normalized,
                "providers": ",".join(self.resolved_device.providers),
                "sample_rate": str(self.sample_rate),
            },
        )
        self._init_warmed = False
        self._shutdown_started = False
        # 最近一次 track_performance 统计挂在 runtime 上，便于引擎外部在会话结束后直接读取。
        self.last_timing: MossTiming | None = None

    @property
    def sample_rate(self) -> int:
        """
        返回 MOSS codec 输出采样率。

        参数:
            本属性无参数。

        返回:
            返回整数 Hz 采样率。

        边界异常:
            core 未初始化成功时访问会暴露 AttributeError。
        """

        return int(self.core.sample_rate)

    def create_session(self) -> "MossTTSNanoSession":
        """
        创建一次 MOSS 合成会话。

        参数:
            本函数无显式参数；会话复用当前 runtime 的 ONNX session。

        返回:
            返回 MossTTSNanoSession。

        边界异常:
            runtime 已 shutdown 后继续创建 session 时抛出 RuntimeError。
        """

        # shutdown 后底层 session 已释放，继续合成只会得到难读的 ONNX 错误，因此提前拒绝。
        if self._shutdown_started:
            raise RuntimeError("moss-tts-nano runtime has been shut down")
        return MossTTSNanoSession(self)

    def warmup_for_init(self) -> None:
        """
        可选初始化预热钩子。

        参数:
            本函数无显式参数；TTSEngine 初始化时会调用该钩子。

        返回:
            无返回值；预热成功后标记为已预热。

        边界异常:
            ONNX 预热失败时直接抛出异常，阻止引擎半初始化。
        """

        # MOSS 首包冷启动成本较高，初始化阶段必须预热关键 ONNX 图，避免首个真实请求承担 allocator 和图执行抖动。
        if self._init_warmed:
            return
        self.core.warmup()
        self._init_warmed = True

    def shutdown(self) -> None:
        """
        关闭 MOSS runtime 持有的 ONNX 资源。

        参数:
            本函数无显式参数。

        返回:
            无返回值；重复调用安全返回。

        边界异常:
            底层资源释放异常会被记录日志，不影响上层清理流程。
        """

        # shutdown 需要幂等，避免异常清理和用户显式 shutdown 重复触发资源释放。
        if self._shutdown_started:
            return
        self._shutdown_started = True
        try:
            self.core.shutdown()
        except Exception as exc:
            logger.warning("MOSS-TTS-Nano shutdown ignored backend cleanup error: %s", exc)


class MossTTSNanoSession:
    """
    MOSS-TTS-Nano 单次合成会话。

    参数:
        runtime: 已初始化的 MossTTSNanoRuntime。

    返回:
        session 本身不直接返回音频；synthesize 生成器逐块产出 float32 PCM。

    边界异常:
        文本为空、prompt 类型不匹配、模式非法或 ONNX 推理失败时抛出异常。
    """

    def __init__(self, runtime: MossTTSNanoRuntime) -> None:
        self.runtime = runtime
        # 最近一次 synthesize 的性能统计只在显式开启 track_performance 时更新，便于调用方事后查看。
        self.last_timing: MossTiming | None = None

    def synthesize(
        self,
        *,
        text: str,
        prompt: MossTTSNanoPromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        remove_meaningless_chars: bool = False,
        mode: str | None = None,
        seed: int | None = None,
    ) -> Iterable[np.ndarray]:
        """
        执行一次 MOSS-TTS-Nano 流式合成。

        参数:
            text: 待合成文本。
            prompt: MOSS prompt 资产，包含参考音频 codes 或纯文本默认 prompt。
            stream: 是否流式输出；当前 ONNX 适配总是按流式块输出。
            track_performance: 是否记录预处理、prefill、逐 chunk 与首包组成耗时。
            remove_meaningless_chars: 是否执行轻量无意义字符过滤。
            mode: voice_clone 或 continuation；为空时使用 prompt.default_mode。
            seed: 随机种子。

        返回:
            逐块产出一维 float32 PCM ndarray。

        边界异常:
            文本为空、prompt 与 runtime 模型不匹配、voice_clone 缺参考音频或 continuation 参数不成对时抛出异常。
        """

        # 入口先做业务级校验，避免 ONNX 后端收到空文本后报出难读 shape 错误。
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if prompt.model != self.runtime.model_name:
            raise ValueError("prompt model does not match current runtime")

        # 每次请求只允许重置随机种子；分段、最大帧数和采样参数全部走 yaml 默认值，避免公开调用面过宽。
        self.runtime.core.apply_generation_options(seed=seed)
        resolved_max_length = int(self.runtime.config.max_length)
        resolved_min_length = int(self.runtime.config.min_length)
        timing = MossTiming() if track_performance else None
        self.last_timing = timing
        self.runtime.last_timing = timing

        # StreamVox 只做轻量清理和分段，不移植参考项目的 WeText 或 robust normalizer。
        preprocess_start_time = time.perf_counter() if timing is not None else 0.0
        preprocess_result = preprocess_text(
            text=text,
            tokenizer=self.runtime.core.sp_model,
            max_length=resolved_max_length,
            min_length=resolved_min_length,
            config=self.runtime.text_preprocess_config,
            remove_meaningless_chars=remove_meaningless_chars,
        )
        if timing is not None:
            timing.preprocess_time = time.perf_counter() - preprocess_start_time
        if not preprocess_result.segments:
            raise ValueError("text has no valid content after preprocessing")

        # prompt_audio_codes 是 voice_clone 和带参考 continuation 的共同条件。
        prompt_audio_codes = _prompt_codes_to_list(prompt.prompt_audio_codes)
        resolved_mode = str(mode or prompt.default_mode or "continuation").strip().lower()

        # voice_clone 默认需要参考音频；纯文本默认 prompt 会走 continuation。
        if resolved_mode == "voice_clone" and not prompt_audio_codes:
            raise ValueError("voice_clone mode requires prompt audio")

        # 按 StreamVox 分段逐段构造 request rows，并把每段内部的 codec state 独立重置。
        try:
            if timing is not None:
                timing.total_segments = len(preprocess_result.segments)
            for segment in preprocess_result.segments:
                # request rows 构造虽然不重，但它位于首包前，因此要纳入首包延迟拆分。
                request_build_start_time = time.perf_counter() if timing is not None else 0.0
                request_rows = self.runtime.core.build_inference_request_rows(
                    text=segment,
                    mode=resolved_mode,
                    prompt_text=prompt.prompt_text,
                    prompt_audio_codes=prompt_audio_codes,
                )
                if timing is not None:
                    timing.request_build_times.append(time.perf_counter() - request_build_start_time)
                for audio_chunk in self.runtime.core.iter_synthesize_mono(
                    request_rows=request_rows,
                    timing=timing,
                ):
                    yield np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        finally:
            if timing is not None:
                log_moss_performance(timing)


def prompt_asset_from_codes(
    *,
    role_name: str,
    model_name: str,
    sample_rate: int,
    prompt_audio_codes: list[list[int]] | np.ndarray,
    prompt_text: str | None,
    default_mode: str,
    metadata: dict[str, str] | None = None,
) -> MossTTSNanoPromptAsset:
    """
    根据 audio codes 构造 MOSS prompt 资产。

    参数:
        role_name: prompt 角色名。
        model_name: StreamVox 模型名。
        sample_rate: prompt 采样率。
        prompt_audio_codes: frame/n_vq 参考音频 token。
        prompt_text: continuation 可选参考文本。
        default_mode: prompt 默认合成模式。
        metadata: 额外元信息。

    返回:
        返回 MossTTSNanoPromptAsset。

    边界异常:
        prompt_audio_codes 不是二维结构时抛出 ValueError。
    """

    # 空 prompt 用 shape=(0, 16) 表达，保证后续序列化和维度校验稳定。
    codes = np.asarray(prompt_audio_codes, dtype=np.int32)
    if codes.size == 0:
        codes = np.zeros((0, 16), dtype=np.int32)
    if codes.ndim != 2:
        raise ValueError(f"prompt_audio_codes must be 2D, got {codes.shape}")
    return MossTTSNanoPromptAsset(
        role_name=role_name,
        model=model_name,
        sample_rate=int(sample_rate),
        prompt_text=prompt_text,
        prompt_audio_codes=codes,
        default_mode=default_mode,
        metadata=dict(metadata or {}),
    )


def serialize_prompt_asset(asset: MossTTSNanoPromptAsset) -> SerializedPromptAsset:
    """
    序列化 MOSS prompt 资产。

    参数:
        asset: MossTTSNanoPromptAsset。

    返回:
        返回 PromptCache 可保存的 SerializedPromptAsset。

    边界异常:
        prompt_audio_codes 无法转为 ndarray 时抛出异常。
    """

    # 大数组放 arrays，轻量字符串放 json_data，保持 PromptCache 的跨版本可读性。
    return SerializedPromptAsset(
        role_name=asset.role_name,
        model=asset.model,
        sample_rate=asset.sample_rate,
        metadata=dict(asset.metadata),
        arrays={"prompt_audio_codes": np.asarray(asset.prompt_audio_codes, dtype=np.int32)},
        json_data={
            "prompt_text": asset.prompt_text,
            "default_mode": asset.default_mode,
        },
    )


def deserialize_prompt_asset(payload: SerializedPromptAsset) -> BasePromptAsset:
    """
    反序列化 MOSS prompt 资产。

    参数:
        payload: PromptCache 读取出的 SerializedPromptAsset。

    返回:
        返回 MossTTSNanoPromptAsset。

    边界异常:
        prompt_audio_codes 缺失时按空 prompt 处理；维度非法时抛出 ValueError。
    """

    # 旧缓存如果缺少数组字段，就按纯文本 continuation prompt 恢复，避免缓存结构升级直接失效。
    prompt_audio_codes = payload.arrays.get("prompt_audio_codes", np.zeros((0, 16), dtype=np.int32))
    return prompt_asset_from_codes(
        role_name=payload.role_name,
        model_name=payload.model,
        sample_rate=payload.sample_rate,
        prompt_audio_codes=prompt_audio_codes,
        prompt_text=payload.json_data.get("prompt_text"),
        default_mode=str(payload.json_data.get("default_mode") or "voice_clone"),
        metadata=payload.metadata,
    )
