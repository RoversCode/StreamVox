from __future__ import annotations

from ..onnx_backend import resolve_explicit_onnx_backend

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from ...model_bundle import validate_local_bundle
from ...utils.log import logger
from ...utils.performance import format_perf_value
from .. import llama_backend as llama_backend_utils
from ..base import BasePromptAsset, ExecutionPlan, RuntimeInfo, SerializedPromptAsset
from ..text_processing import TextPreprocessConfig
from . import llama
from .assets import (
    VoxCPM2AssetManifest,
    load_fsq_runtime,
    load_linear_runtime,
    load_stop_predictor_runtime,
    load_voxcpm2_asset_manifest,
)
from .decoder import VoxCPM2DecoderAPI
from .embedding_runtime import MMapEmbeddingTable
from .input_builder import HybridInputBuilder, HybridPreparedInputs
from .locdit_ort import LocDiTEulerOrtRunner
from .locdit_runtime import LocDiTPatchRuntime
from .ort_runtime import create_ort_session, run_single_input
from .record import _VoxCPM2Timing
from .schema.decoder_protocol import DecoderResponse
from .spec import VOXCPM2_GGUF_SPEC, VoxCPM2GGUFModelSpec
from .text import VOXCPM2_TEXT_PREPROCESS_CONFIG, preprocess_text
from .tokenizer_tools import load_voxcpm2_tokenizer


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "voxcpm2_gguf.yaml"


def _select_voxcpm2_gguf_backend(
    mode: str,
    device_index: int | None,
    explicit_backend: str | None = None,
) -> tuple[str, int | None]:
    """
    按统一 GGUF 规则解析 VoxCPM2 的 llama.cpp backend。

    参数:
        mode: adapter 或旧 device 字符串解析出的 auto/cpu/gpu 模式。
        device_index: 用户指定的 GPU 索引；为空时 GPU backend 默认使用 0。

    返回:
        返回 `(backend, gpu_index)`；CPU 路径的 gpu_index 为 None。

    异常:
        mode 非 auto/cpu/gpu 时抛出 ValueError。
    """

    resolution = llama_backend_utils.select_llama_backend(
        mode=mode,
        device_index=device_index,
        model_label="voxcpm2-gguf",
        explicit_backend=explicit_backend,
    )
    return resolution.backend, resolution.gpu_index


def resolve_voxcpm2_gguf_backend(device: str, execution_plan: ExecutionPlan | None) -> tuple[str, int | None]:
    """
    解析 VoxCPM2-GGUF 的 GGUF backend。

    参数:
        device: 用户入口传入的原始设备字符串。
        execution_plan: adapter 生成的执行计划；存在时优先使用其中的 llama 目标。

    返回:
        返回 ``(gguf_backend, gpu_index)``，backend 取值为 ``cpu/cuda/vulkan/metal``。

    边界异常:
        当用户强制请求 GPU 但当前包或宿主机不支持时，会记录告警并回退到 CPU。
    """

    if execution_plan is not None:
        llama_target = execution_plan.get("llama")
        if llama_target is not None:
            if llama_target.mode == "cpu":
                return "cpu", None
            if llama_target.mode in {"auto", "gpu"}:
                return _select_voxcpm2_gguf_backend(
                    llama_target.mode,
                    llama_target.device_index,
                    llama_target.backend_preference,
                )
            raise ValueError(f"unsupported llama target mode: {llama_target.mode}")

    normalized_device = str(device).strip().lower()
    if normalized_device == "cpu":
        return "cpu", None
    if normalized_device in {"auto", "gpu"} or normalized_device.startswith("gpu:"):
        gpu_index = None
        mode = normalized_device
        if normalized_device.startswith("gpu:"):
            index_text = normalized_device.split(":", 1)[1]
            gpu_index = int(index_text) if index_text.isdigit() else 0
            mode = "gpu"
        elif normalized_device == "gpu":
            gpu_index = 0
        return _select_voxcpm2_gguf_backend(mode, gpu_index)
    return "cpu", None


@dataclass(slots=True)
class VoxCPM2GGUFConfig:
    """
    VoxCPM2-GGUF runtime 配置。

    参数:
        n_ctx: GGUF 默认上下文长度。
        default_mode: 公开 API 在“有 prompt 但未显式传 mode”时使用的默认模式编号。
        max_length: 文本分段阶段的单段上限。
        min_length: 文本分段阶段合并短句的下限。
        inference_timesteps: LocDiT 扩散步数。
        cfg_value: classifier-free guidance 系数。
        streaming_prefix_len: 流式上下文 patch 窗口。
        base_n_gpu_layers: base LM GPU offload 层数。
        residual_n_gpu_layers: residual LM GPU offload 层数。
        decoder_start_method: decoder worker 多进程启动方式。

    返回:
        该类只承载配置，不主动返回推理结果。

    边界异常:
        配置文件解析阶段会把字段强制转为目标类型；非法 YAML 值会在 load 阶段抛出异常。
    """

    n_ctx: int = 8192
    default_mode: int = 1
    max_length: int = 300
    min_length: int = 20
    inference_timesteps: int = 10
    cfg_value: float = 2.0
    streaming_prefix_len: int = 4
    base_n_gpu_layers: int = -1
    residual_n_gpu_layers: int = -1
    decoder_start_method: str = "spawn"


@dataclass(slots=True)
class VoxCPM2PromptAsset(BasePromptAsset):
    """
    VoxCPM2-GGUF 单人 prompt 资产。

    参数:
        prompt_audio_feat: continuation 模式预编码好的 AudioVAE patch 表示。
        prompt_locenc_hidden: continuation 模式预编码好的 locenc 隐状态。
        prompt_text: continuation 模式使用的提示文本。
        reference_audio_feat: reference 模式预编码好的 AudioVAE patch 表示。
        reference_locenc_hidden: reference 模式预编码好的 locenc 隐状态。

    返回:
        本类作为 PromptCache 可序列化数据载体，不主动执行推理。

    边界异常:
        路径模式和内存音频模式都会在 make_prompt 阶段尽量预编码，
        这样 stream 阶段可以直接复用缓存，避免重复跑 AudioVAE encoder 与 locenc。
    """

    prompt_audio_feat: np.ndarray | None = None
    prompt_locenc_hidden: np.ndarray | None = None
    prompt_text: str | None = None
    reference_audio_feat: np.ndarray | None = None
    reference_locenc_hidden: np.ndarray | None = None


@dataclass(slots=True)
class _DecodeState:
    """单次 latent 生成循环的隐状态快照。"""

    lm_hidden: np.ndarray
    residual_hidden: np.ndarray
    prefix_patch: np.ndarray


@dataclass(slots=True)
class _VoxCPM2StreamError:
    """后台生成线程向前台生成器传递异常的轻量包装。"""

    error: Exception


class _EmbeddingLlamaRunner:
    """
    llama.cpp embedding 输入运行器。

    参数:
        gguf_path: GGUF 权重路径。
        n_ctx: 上下文长度。
        n_gpu_layers: GPU offload 层数。

    返回:
        初始化后可执行 prefill/step 并读取 hidden embedding。

    边界异常:
        底层 llama.cpp decode 返回非零状态时抛出 RuntimeError。
    """

    def __init__(
        self,
        *,
        gguf_path: str | Path,
        n_ctx: int,
        n_gpu_layers: int,
        gguf_backend: str,
        main_gpu: int,
    ) -> None:
        self.model = llama.LlamaModel(
            str(gguf_path),
            n_gpu_layers=int(n_gpu_layers),
            main_gpu=int(main_gpu),
            backend=str(gguf_backend),
        )
        self.ctx = llama.LlamaContext(
            self.model,
            n_ctx=int(n_ctx),
            n_batch=int(n_ctx),
            embeddings=True,
            flash_attn=True,
            offload_kqv=True,
            no_perf=True,
        )
        self.position = 0
        self._prefill_batch = llama.LlamaBatch(int(n_ctx), embd_dim=self.model.n_embd, n_seq_max=1)
        self._step_batch = llama.LlamaBatch(1, embd_dim=self.model.n_embd, n_seq_max=1)

    def reset(self) -> None:
        """清空 KV cache 与位置游标，准备开始一次新的合成。"""

        self.ctx.clear_kv_cache()
        self.position = 0

    def prefill(self, embeds: np.ndarray, output_all_tokens: bool = False) -> np.ndarray:
        """执行 embedding prefill，并返回最后一步或完整时间轴 hidden。"""

        embeds = np.asarray(embeds, dtype=np.float32)
        self._prefill_batch.set_embd(embeds, pos=0, seq_id=0)
        if output_all_tokens:
            # residual prefill 需要完整 base hidden 时间轴，因此显式打开所有 token 的 embedding 输出。
            for token_index in range(self._prefill_batch.n_tokens):
                self._prefill_batch.logits[token_index] = 1
        status = self.ctx.decode(self._prefill_batch)
        if status != 0:
            raise RuntimeError(f"llama prefill failed with status={status}")
        self.position = int(embeds.shape[0])
        if not output_all_tokens:
            return _extract_embedding(self.ctx, self._prefill_batch.n_tokens - 1, self.model.n_embd)
        return np.stack(
            [_extract_embedding(self.ctx, token_index, self.model.n_embd) for token_index in range(int(embeds.shape[0]))],
            axis=0,
        )

    def step(self, embed: np.ndarray) -> np.ndarray:
        """执行单步 embedding decode，并返回当前 hidden。"""

        embed = np.asarray(embed, dtype=np.float32)
        # voxcpm2_gguf 自己维护 StreamVox-bin 适配后的 llama.cpp 绑定；该绑定使用通用
        # set_embd 写入 embedding，避免依赖参考脚本里的单步私有快路径。
        self._step_batch.set_embd(embed.reshape(1, -1), pos=int(self.position), seq_id=0)
        status = self.ctx.decode(self._step_batch)
        if status != 0:
            raise RuntimeError(f"llama decode-step failed with status={status}")
        self.position += 1
        return _extract_embedding(self.ctx, 0, self.model.n_embd)

    def shutdown(self) -> None:
        """
        按安全顺序释放单路 GGUF runtime 的 native 资源。

        参数:
            本函数不接收参数。

        返回:
            无返回值；调用后当前 runner 不可继续参与推理。

        边界异常:
            Vulkan 路径默认走 detach 策略，避免双 GGUF 场景在解释器退出阶段触发 native 析构崩溃。
        """

        # batch 只依赖独立的 llama_batch_free，先断开 Python 引用，避免后续误以为 context 仍可用。
        self._prefill_batch = None
        self._step_batch = None
        if getattr(getattr(self, "model", None), "backend", None) == "vulkan":
            if os.environ.get("STREAMVOX_VOXCPM2_GGUF_FREE_VULKAN_NATIVE", "0") == "1":
                self.ctx.free_native(force=True)
                self.model.free_native(force=True)
            else:
                self.ctx.detach_native()
                self.model.detach_native()
        else:
            # CPU/CUDA/Metal 路径显式 free native 指针，避免把释放时机完全交给 GC。
            self.ctx.free_native(force=True)
            self.model.free_native(force=True)
        # context 必须先于 model 失效；反向释放会让 context 仍持有已失效模型指针。
        self.ctx = None
        self.model = None


class VoxCPM2GGUFRuntime:
    """
    VoxCPM2-GGUF StreamVox runtime。

    参数:
        model_dir: 标准模型 bundle 根目录。
        model_spec: 模型规格。
        device: 用户请求的设备字符串。
        n_ctx: 可选上下文长度覆盖。
        verify_model_sha256: 是否校验 manifest sha256。
        execution_plan: adapter 解析出的执行计划。
        text_preprocess_config: 文本预处理配置声明；为空时使用 VoxCPM2 默认配置。

    返回:
        初始化后得到可创建 session 的 runtime 对象。

    边界异常:
        缺失资产、bundle 校验失败、GGUF/ONNX 加载失败都会在初始化阶段抛出。
    """

    model_name = VOXCPM2_GGUF_SPEC.model_name

    def __init__(
        self,
        *,
        model_dir: Path,
        model_spec: VoxCPM2GGUFModelSpec | None = None,
        device: str = "auto",
        n_ctx: int | None = None,
        verify_model_sha256: bool = True,
        execution_plan: ExecutionPlan | None = None,
        text_preprocess_config: TextPreprocessConfig | None = None,
    ) -> None:
        start = time.time()
        self.model_spec = model_spec or VOXCPM2_GGUF_SPEC
        self.model_name = self.model_spec.model_name
        self.bundle_root = Path(model_dir).resolve()
        self.device = device
        self.execution_plan = execution_plan
        # 文本配置由 adapter/engine 注入；缺省时回退到模型自己的默认规则。
        self.text_preprocess_config = text_preprocess_config or VOXCPM2_TEXT_PREPROCESS_CONFIG
        self.config = load_voxcpm2_config()
        self.gguf_backend, self.gguf_gpu_index = resolve_voxcpm2_gguf_backend(device, execution_plan)
        self.onnx_backend, self.onnx_gpu_index = resolve_voxcpm2_onnx_device(device, execution_plan)
        if n_ctx is not None:
            self.config.n_ctx = int(n_ctx)

        validate_local_bundle(self.bundle_root, self.model_name, verify_model_sha256=verify_model_sha256)
        self.ckpt_root = self.bundle_root / "ckpt"
        self.assets = load_voxcpm2_asset_manifest(self.ckpt_root)
        # 最近一次流式任务编号只用于测试和诊断，不参与推理决策；smoke 脚本会据此读取 decoder 时间线。
        self.last_stream_task_id: str | None = None
        # 最近一次输入摘要用于 P0 验收 mode/context_len/text_length，不保存大数组，避免长文本时额外占内存。
        self.last_prepared_summary: dict[str, int | str] = {}
        # prompt 相关 ORT/decoder 预热只做一次，避免每次 make_prompt 都重复触发。
        self._prompt_pipeline_warmed = False
        # 初始化期 ONNX + GGUF 统一预热只允许真正执行一次，避免重复污染启动路径统计。
        self._init_warmed = False
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
                "gguf_backend": str(self.gguf_backend),
                "onnx_backend": str(self.onnx_backend),
                "init_seconds": f"{time.time() - start:.3f}",
            },
        )

    @property
    def sample_rate(self) -> int:
        """返回当前模型输出采样率。"""

        return int(self.assets.audio_output_sample_rate)

    def _init_runtime(self) -> None:
        """加载 GGUF、ONNX、embedding 和轻量 numpy 权重。"""

        prefer_onnx_gpu = self.onnx_backend != "cpu"
        # 当前按用户要求恢复成“只要 ONNX backend 不是 CPU，就统一优先走显卡 provider”。
        # 这样 locenc / locdit / audio_encoder 与 decoder 保持同一套设备选择策略，便于直接对比
        # Windows DML 与 Ubuntu CUDA 的真实性能差异。
        prefer_feature_onnx_gpu = prefer_onnx_gpu
        gguf_gpu_enabled = self.gguf_backend in {"cuda", "vulkan", "metal"}
        self.embedding_table = MMapEmbeddingTable(self.assets.embedding_metadata)
        self.text_tokenizer = load_voxcpm2_tokenizer(str(self.assets.tokenizer_path))
        self.locenc_session, _ = create_ort_session(
            str(self.assets.locenc_onnx_path),
            prefer_gpu=prefer_feature_onnx_gpu,
            device_id=self.onnx_gpu_index,
        )
        self.locdit_session, _ = create_ort_session(
            str(self.assets.locdit_onnx_path),
            prefer_gpu=prefer_feature_onnx_gpu,
            device_id=self.onnx_gpu_index,
        )
        self.audio_encoder_session, _ = create_ort_session(
            str(self.assets.audio_encoder_onnx_path),
            prefer_gpu=prefer_feature_onnx_gpu,
            device_id=self.onnx_gpu_index,
        )
        self.base_lm = _EmbeddingLlamaRunner(
            gguf_path=self.assets.base_lm_gguf_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.base_n_gpu_layers if gguf_gpu_enabled else 0,
            gguf_backend=self.gguf_backend,
            main_gpu=self.gguf_gpu_index if self.gguf_gpu_index is not None else 0,
        )
        self.residual_lm = _EmbeddingLlamaRunner(
            gguf_path=self.assets.residual_lm_gguf_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.residual_n_gpu_layers if gguf_gpu_enabled else 0,
            gguf_backend=self.gguf_backend,
            main_gpu=self.gguf_gpu_index if self.gguf_gpu_index is not None else 0,
        )
        self.enc_to_lm_proj = load_linear_runtime(self.assets.enc_to_lm_proj)
        self.fsq_layer = load_fsq_runtime(self.assets.fsq_layer)
        self.fusion_concat_proj = load_linear_runtime(self.assets.fusion_concat_proj)
        self.stop_predictor = load_stop_predictor_runtime(self.assets.stop_predictor)
        self.locdit_patch_runtime = LocDiTPatchRuntime(
            lm_to_dit_proj=load_linear_runtime(self.assets.lm_to_dit_proj),
            res_to_dit_proj=load_linear_runtime(self.assets.res_to_dit_proj),
            locdit_runner=LocDiTEulerOrtRunner(self.locdit_session),
            latent_dim=self.assets.latent_dim,
            patch_size=self.assets.patch_size,
        )
        self.input_builder = HybridInputBuilder(
            audio_encoder_session=self.audio_encoder_session,
            locenc_session=self.locenc_session,
            tokenizer=self.text_tokenizer,
            audio_start_token=self.assets.audio_start_token,
            ref_audio_start_token=self.assets.ref_audio_start_token,
            ref_audio_end_token=self.assets.ref_audio_end_token,
            patch_size=self.assets.patch_size,
            latent_dim=self.assets.latent_dim,
            chunk_size=self.assets.chunk_size,
            sample_rate=self.assets.audio_encode_sample_rate,
        )
        self.decoder = VoxCPM2DecoderAPI(
            decoder_stateful_onnx_path=self.assets.audio_decoder_stateful_onnx_path,
            metadata_path=self.assets.audio_vae_export_metadata_path,
            patch_size=self.assets.patch_size,
            decode_chunk_size=self.assets.decode_chunk_size,
            latent_dim=self.assets.latent_dim,
            prefer_gpu=prefer_onnx_gpu,
            device_id=self.onnx_gpu_index,
            start_method=self.config.decoder_start_method,
        )
        if not self.decoder.wait_until_ready(timeout=60):
            raise RuntimeError("voxcpm2-gguf decoder process failed to become ready")

    def ensure_prompt_pipeline_warm(self) -> None:
        """
        预热 prompt 相关推理链路。

        参数:
            本函数不接收参数。

        返回:
            无返回值；首次调用后会把 locdit 与 audio decoder 热起来。

        边界异常:
            预热失败时直接抛错，避免真实首轮 stream 在用户请求里暴露初始化抖动。
        """

        if self._prompt_pipeline_warmed:
            return

        lm_hidden_dim = int(self.locdit_patch_runtime.lm_to_dit_proj.weight.shape[1])
        residual_hidden_dim = int(self.locdit_patch_runtime.res_to_dit_proj.weight.shape[1])
        warmup_patch = self.locdit_patch_runtime.decode_patch(
            lm_hidden=np.zeros((lm_hidden_dim,), dtype=np.float32),
            residual_hidden=np.zeros((residual_hidden_dim,), dtype=np.float32),
            prefix_patch=np.zeros((self.assets.patch_size, self.assets.latent_dim), dtype=np.float32),
            inference_timesteps=int(self.config.inference_timesteps),
            cfg_value=float(self.config.cfg_value),
            rng=np.random.default_rng(0),
        )
        self.decoder.warmup(patch=warmup_patch)
        self._prompt_pipeline_warmed = True

    def warmup_for_init(self) -> None:
        """
        在 runtime 初始化完成后预热 VoxCPM2 的 ONNX 与 GGUF 关键路径。

        参数:
            本函数不接收显式参数；依赖当前 runtime 已完成 input_builder、prompt pipeline 与两路 GGUF runner 的初始化。

        返回:
            无返回值；预热成功后会把 `_init_warmed` 标记为 True。

        边界异常:
            任一步骤失败都会直接抛出异常，阻止 TTSEngine 进入“runtime 已加载但关键链路未热起”的半初始化状态。
        """

        # 幂等保护必须放在入口处；初始化钩子可能被测试或上层重复调用，但真实预热只允许跑一次。
        if self._init_warmed:
            return
        # 关键执行器缺失说明 runtime 尚未完整初始化；继续预热只会制造更隐蔽的空指针错误。
        if self.input_builder is None or self.base_lm is None or self.residual_lm is None:
            raise RuntimeError("voxcpm2-gguf runtime is not initialized")
        
        logger.info("[voxcpm2] 开始初始化预热：ONNX + GGUF")
        # 最短合法音频长度按单 patch 的原始波形跨度构造，目标只是打通 ONNX encoder 路径，不追求真实业务语义。
        warmup_waveform = np.zeros(
            (int(self.assets.patch_size) * int(self.assets.chunk_size),),
            dtype=np.float32,
        )
        # continuation 场景的 prompt 音频实际使用 left padding；这里沿用同一方向，避免 warmup 和正式请求协议漂移。
        warmup_audio_feat = self.input_builder.encode_audio(warmup_waveform, padding_mode="left")
        self.input_builder.encode_audio_feat(warmup_audio_feat)
        self.ensure_prompt_pipeline_warm()

        # GGUF 预热只要求满足 embedding runner 的最小输入约束，不要求 embedding 具有真实语义。
        base_prefill_embed = np.zeros((1, int(self.base_lm.model.n_embd)), dtype=np.float32)
        residual_prefill_embed = np.zeros((1, int(self.residual_lm.model.n_embd)), dtype=np.float32)

        # 无论 GGUF 预热成功还是失败，都必须清空两路 KV cache，避免污染后续真实 synthesize 请求。
        try:
            self.base_lm.prefill(base_prefill_embed)
            self.residual_lm.prefill(residual_prefill_embed)
        finally:
            self.base_lm.reset()
            self.residual_lm.reset()

        self._init_warmed = True
        logger.info("[voxcpm2] 初始化预热完成")

    def shutdown(self) -> None:
        """
        释放 decoder、GGUF native 句柄和动态 backend 注册。

        参数:
            本函数不接收参数。

        返回:
            无返回值；调用后当前 runtime 不能继续创建 session。

        边界异常:
            shutdown 允许重复调用；第二次及以后直接返回，避免重复 close 导致噪音。
        """

        if getattr(self, "_shutdown_started", False):
            return
        self._shutdown_started = True
        if hasattr(self, "decoder") and self.decoder is not None:
            self.decoder.shutdown()
            self.decoder = None
        # 两路 GGUF runtime 都持有独立的 context/model；必须显式收口，不能只依赖 GC 时机。
        if hasattr(self, "base_lm") and self.base_lm is not None:
            self.base_lm.shutdown()
            self.base_lm = None
        if hasattr(self, "residual_lm") and self.residual_lm is not None:
            self.residual_lm.shutdown()
            self.residual_lm = None
        # 其余 ORT 与 numpy runtime 只需要断开 Python 引用，让各自后端在当前进程中安全回收。
        self.embedding_table = None
        self.text_tokenizer = None
        self.locenc_session = None
        self.locdit_session = None
        self.audio_encoder_session = None
        self.enc_to_lm_proj = None
        self.fsq_layer = None
        self.fusion_concat_proj = None
        self.stop_predictor = None
        self.locdit_patch_runtime = None
        self.input_builder = None
        # 显式 shutdown 完成后再处理 llama.cpp backend 收尾，避免 context/model 仍存活时提前卸载动态库。
        llama.arm_vulkan_exit_guard()
        llama.unload_dynamic_backends()


class VoxCPM2GGUFSession:
    """
    VoxCPM2-GGUF 单次合成会话。

    参数:
        runtime: 已初始化的 VoxCPM2-GGUF runtime。

    返回:
        会话通过 synthesize 生成 float32 PCM 音频块。

    边界异常:
        文本为空、prompt 类型错误或底层推理失败时抛出异常。
    """

    def __init__(self, runtime: VoxCPM2GGUFRuntime) -> None:
        self.runtime = runtime
        self.decoder = runtime.decoder

    def synthesize(
        self,
        *,
        text: str,
        prompt: VoxCPM2PromptAsset,
        track_performance: bool = False,
        mode: str,
        control_text: str | None,
        prompt_text: str | None,
        min_length: int | None,
        max_length: int | None,
        remove_meaningless_chars: bool,
        seed: int | None,
    ) -> Iterable[np.ndarray]:
        """
        按并行流式方式合成文本，后台生成 patch，decoder worker 同步解码音频。

        参数:
            text: 用户传入的完整待合成文本。
            prompt: 当前请求使用的单人 prompt 资产。
            track_performance: 开启后记录并输出每个 chunk 的生成/解码耗时。
            mode/control_text/prompt_text: VoxCPM2-GGUF 私有模式和文本控制参数。
            min_length/max_length: 文本分段参数；max_length 同时用于换算生成 patch 的异常安全上限。
            remove_meaningless_chars: 是否执行通用控制字符清理。
            seed: 随机种子；扩散步数、CFG 和流式上下文窗口由模型配置固定。

        返回:
            逐段串行 yield float32 PCM 音频块。

        边界异常:
            prompt 类型错误、文本为空，或文本清理/分段后为空时抛出异常。
        """

        if not isinstance(prompt, VoxCPM2PromptAsset):
            raise TypeError("voxcpm2-gguf runtime requires VoxCPM2PromptAsset")
        if not isinstance(text, str) or not text:
            raise ValueError("voxcpm2-gguf synthesize requires non-empty text")

        resolved_max_length = max_length if max_length is not None else self.runtime.config.max_length
        # 文本预处理统一走 voxcpm2 自己的 text 模块，保证和 qwen3/s2-pro 一样由模型侧声明
        # 清理与分段规则，而不是在 runtime 主路径里散落公共工具调用。
        processed = preprocess_text(
            text=text,
            language="auto",
            max_length=resolved_max_length,
            min_length=min_length if min_length is not None else self.runtime.config.min_length,
            tokenizer=self.runtime.text_tokenizer,
            config=self.runtime.text_preprocess_config,
            remove_meaningless_chars=remove_meaningless_chars,
        )
        segments = processed.segments
        if not segments:
            raise ValueError("voxcpm2-gguf text preprocessing produced no segments")

        # 新逻辑直接把文本分段上限扩大两倍，作为 stop predictor 失效时的异常保护线。
        # 这样 max_length 的公开语义更稳定，也避免再暴露额外的 patch 上限配置。
        max_generation_patches = max(1, int(resolved_max_length) * 2)
        # 严格以调用入参为准：传了 seed 就复现，不传就走真正随机。
        effective_seed = None if seed is None else int(seed)
        total_timing = _VoxCPM2Timing() if track_performance else None

        # 多段文本顺序合成；每段内部的正常结束条件只看 stop predictor，patch 上限只做异常检测。
        for segment_index, segment in enumerate(segments):
            logger.info(f"[voxcpm2-gguf]-正在推理第{segment_index} 段文本：{segment}")
            # 每个 segment 单独统计，最后再汇总到整次 synthesize，避免长文本时只看到最后一段的数据。
            timing = _VoxCPM2Timing() if track_performance else None
            prompt_cache = None
            if mode != "text":
                prompt_audio_feat = prompt.prompt_audio_feat if mode in {"continuation", "ref_continuation"} else None
                prompt_locenc_hidden = (
                    prompt.prompt_locenc_hidden if mode in {"continuation", "ref_continuation"} else None
                )
                reference_audio_feat = prompt.reference_audio_feat if mode in {"ref", "ref_continuation"} else None
                reference_locenc_hidden = (
                    prompt.reference_locenc_hidden if mode in {"ref", "ref_continuation"} else None
                )
                # 按最终模式裁剪 continuation 提示文本，避免 ref 模式只使用参考音频时误触 prompt 音频对齐校验。
                effective_prompt_text = None
                if mode in {"continuation", "ref_continuation"}:
                    # continuation 类模式必须把提示音频与对应文本成对交给 input_builder；
                    # 调用入参优先于缓存资产，便于单次合成覆盖持久化角色里的旧文本。
                    effective_prompt_text = prompt_text if prompt_text is not None else prompt.prompt_text
                # 按四种 mode 构造标准 text/audio mask 输入。
                prompt_cache = self.runtime.input_builder.build_prompt_cache(
                    prompt_text=effective_prompt_text,
                    prompt_audio_feat=prompt_audio_feat,
                    prompt_locenc_hidden=prompt_locenc_hidden,
                    reference_audio_feat=reference_audio_feat,
                    reference_locenc_hidden=reference_locenc_hidden,
                )
            prepared = self.runtime.input_builder.build_standard_inputs(
                target_text=segment,
                prompt_cache=prompt_cache,
                control_text=control_text,
                streaming_prefix_len=int(self.runtime.config.streaming_prefix_len),
            )
            yield from self._synthesize_prepared(
                prepared=prepared,
                max_generation_patches=max_generation_patches,
                inference_timesteps=self.runtime.config.inference_timesteps,
                cfg_value=self.runtime.config.cfg_value,
                seed=None if effective_seed is None else int(effective_seed) + segment_index,
                timing=timing,
            )
            if total_timing is not None and timing is not None:
                total_timing.merge_from(timing)

        if track_performance and total_timing is not None:
            self._log_track_performance(total_timing)

    def _synthesize_prepared(
        self,
        *,
        prepared: HybridPreparedInputs,
        max_generation_patches: int,
        inference_timesteps: int,
        cfg_value: float,
        seed: int | None,
        timing: _VoxCPM2Timing | None,
    ) -> Iterable[np.ndarray]:
        """启动后台 patch 生成线程，并在需要时同步累积 chunk 级性能统计。"""

        task_id = str(uuid.uuid4())
        self.runtime.last_stream_task_id = task_id
        self.runtime.last_prepared_summary = {
            "mode": str(prepared.mode),
            "text_length": int(prepared.text_length),
            "context_len": int(prepared.context_len),
            "input_length": int(prepared.text_token.shape[0]),
            "max_generation_patches": int(max_generation_patches),
        }
        output_queue = self.decoder.subscribe_stream(task_id)
        stop_event = threading.Event()
        # seed 未显式传入时必须让每次运行都走真正随机路径，避免默认值退化成固定可复现输出。
        rng = np.random.default_rng(None if seed is None else int(seed))
        worker = threading.Thread(
            target=self._run_patch_worker,
            kwargs={
                "task_id": task_id,
                "prepared": prepared,
                "max_generation_patches": int(max_generation_patches),
                "inference_timesteps": int(inference_timesteps),
                "cfg_value": float(cfg_value),
                "rng": rng,
                "timing": timing,
                "output_queue": output_queue,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        worker.start()

        try:
            while True:
                item = output_queue.get()
                if isinstance(item, _VoxCPM2StreamError):
                    raise item.error
                if not isinstance(item, DecoderResponse):
                    continue
                if item.msg_type == "ERROR":
                    raise RuntimeError(item.error_message or "voxcpm2-gguf decoder failed")
                if item.msg_type == "AUDIO" and item.audio is not None:
                    # decoder worker 已经把本次 chunk 的 compute_time 带回主线程；
                    # 这里顺手记录下来，后续可和 patch 生成耗时一一对齐输出。
                    if timing is not None:
                        timing.decoder_compute_times.append(float(item.compute_time))
                    yield item.audio
                    continue
                if item.msg_type == "FINISH":
                    break
        finally:
            stop_event.set()
            self.decoder.unsubscribe_stream(task_id, output_queue)
            worker.join(timeout=1.0)

    def _run_patch_worker(
        self,
        *,
        task_id: str,
        prepared: HybridPreparedInputs,
        max_generation_patches: int,
        inference_timesteps: int,
        cfg_value: float,
        rng: np.random.Generator,
        timing: _VoxCPM2Timing | None,
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        """后台执行 prefill 和 latent patch 生成，并把 patch 立即提交给 decoder worker。"""

        try:
            for patch in self._generate_patches(
                prepared=prepared,
                max_generation_patches=max_generation_patches,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                rng=rng,
                timing=timing,
            ):
                if stop_event.is_set():
                    break
                self.decoder.decode_patch(task_id=task_id, patch=patch, is_final=False)
            self.decoder.decode_patch(task_id=task_id, patch=None, is_final=True) # 收尾，结束符
        except Exception as exc:
            output_queue.put(_VoxCPM2StreamError(exc))

    def _generate_patches(
        self,
        *,
        prepared: HybridPreparedInputs,
        max_generation_patches: int,
        inference_timesteps: int,
        cfg_value: float,
        rng: np.random.Generator,
        timing: _VoxCPM2Timing | None,
    ) -> Iterable[np.ndarray]:
        """执行 prefill + decode loop，并逐个产出 latent patch。"""

        # prefill 是 voxcpm2 首包前最重的同步阶段；单独拆出来统计，便于和后续 chunk 生成耗时区分。
        prefill_started_at = time.time()
        state = self._prefill(prepared)
        if timing is not None:
            timing.prefill_time += time.time() - prefill_started_at
        stopped = False
        for step_index in range(int(max_generation_patches)):
            # 每个 chunk 单独计时；这里的口径不是只看 LocDiT 出 patch 的时间，
            # 还要覆盖 patch 生成后的自回灌更新，否则下一个 chunk 真正准备时间会被漏记。
            chunk_started_at = time.time()
            curr_patch, stop_logits = self._decode_single_patch(
                lm_hidden=state.lm_hidden,
                residual_hidden=state.residual_hidden,
                prefix_patch=state.prefix_patch,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                rng=rng,
            )
            curr_patch = np.asarray(curr_patch, dtype=np.float32)
            state.prefix_patch = curr_patch

            chunk_timing_index: int | None = None
            if timing is not None:
                # 先为当前 chunk 占位，允许外层尽快把 patch 送进 decoder；
                # 等生成器恢复执行后，再把“自回灌更新”的耗时补写回同一项。
                timing.chunk_gen_times.append(0.0)
                chunk_timing_index = len(timing.chunk_gen_times) - 1

            yield curr_patch

            stop_flag = int(np.argmax(stop_logits, axis=-1))
            if stop_flag == 1:
                # stop 命中后，本 chunk 不再需要为下一轮准备 LM 状态；
                # 因此这里把当前实际发生过的耗时写回即可。
                if timing is not None and chunk_timing_index is not None:
                    timing.chunk_gen_times[chunk_timing_index] = time.time() - chunk_started_at
                stopped = True
                break

            # 这段自回灌更新会直接决定“下一个 chunk 何时能够开始生成”，
            # 因此必须计入当前 chunk 的 generation time，而不能遗漏。
            curr_embed = self._encode_patch_for_next_step(curr_patch)
            lm_hidden_raw = self.runtime.base_lm.step(curr_embed.reshape(-1))
            lm_hidden = self.runtime.fsq_layer.forward(lm_hidden_raw.reshape(1, -1)).reshape(-1)
            residual_input = _forward_concat_proj(self.runtime.fusion_concat_proj, lm_hidden, curr_embed.reshape(-1))
            residual_hidden = self.runtime.residual_lm.step(residual_input.reshape(-1))

            state.lm_hidden = np.asarray(lm_hidden, dtype=np.float32)
            state.residual_hidden = np.asarray(residual_hidden, dtype=np.float32)
            if timing is not None and chunk_timing_index is not None:
                timing.chunk_gen_times[chunk_timing_index] = time.time() - chunk_started_at
        if not stopped:
            raise RuntimeError(
                "voxcpm2-gguf generation exceeded max_generation_patches="
                f"{int(max_generation_patches)} before stop predictor fired"
            )

    def _prefill(self, prepared: HybridPreparedInputs) -> _DecodeState:
        """执行 embedding lookup、LocEnc、base LM 和 residual LM prefill。"""

        self.runtime.base_lm.reset()
        self.runtime.residual_lm.reset()
        text_embed = self.runtime.embedding_table.lookup(prepared.text_token)

        # make_prompt 已经把 prompt/reference 音频对应的 locenc 预编码好了；
        # 这里只在旧 PromptCache 缺少缓存时才回退到整段现算。
        if prepared.locenc_hidden is not None:
            locenc_hidden = np.asarray(prepared.locenc_hidden, dtype=np.float32)
        else:
            locenc_hidden = run_single_input(self.runtime.locenc_session, "patches", prepared.audio_feat[None, ...]).squeeze(0)

        feat_embed = self.runtime.enc_to_lm_proj.forward(locenc_hidden)
        text_mask = prepared.text_mask.astype(np.float32, copy=False)[:, None]
        feat_mask = prepared.feat_mask.astype(np.float32, copy=False)[:, None]

        combined_embed = text_mask * text_embed + feat_mask * feat_embed

        base_timeline_raw = self.runtime.base_lm.prefill(combined_embed, output_all_tokens=True)
        base_timeline = self.runtime.fsq_layer.forward(np.asarray(base_timeline_raw, dtype=np.float32)) * feat_mask + base_timeline_raw * text_mask

        residual_inputs = self.runtime.fusion_concat_proj.forward(np.concatenate([base_timeline, feat_mask * feat_embed], axis=-1))
        residual_hidden = self.runtime.residual_lm.prefill(residual_inputs)
        return _DecodeState(
            lm_hidden=np.asarray(base_timeline[-1], dtype=np.float32),
            residual_hidden=np.asarray(residual_hidden, dtype=np.float32),
            prefix_patch=np.asarray(prepared.audio_feat[-1], dtype=np.float32),
        )

    def _decode_single_patch(
        self,
        *,
        lm_hidden: np.ndarray,
        residual_hidden: np.ndarray,
        prefix_patch: np.ndarray,
        inference_timesteps: int,
        cfg_value: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行单步 LocDiT 生成和 stop predictor 判断。"""

        patch = self.runtime.locdit_patch_runtime.decode_patch(
            lm_hidden=lm_hidden,
            residual_hidden=residual_hidden,
            prefix_patch=prefix_patch,
            inference_timesteps=int(inference_timesteps),
            cfg_value=float(cfg_value),
            rng=rng,
        )
        stop_logits = self.runtime.stop_predictor.forward(lm_hidden.reshape(1, -1)).reshape(-1)
        return patch, stop_logits

    def _encode_patch_for_next_step(self, patch: np.ndarray) -> np.ndarray:
        """把当前生成 patch 编回 LM embedding 空间，供下一轮自回归使用。"""

        locenc_hidden = run_single_input(self.runtime.locenc_session, "patches", patch[None, None, ...]).squeeze(0)
        return np.asarray(self.runtime.enc_to_lm_proj.forward(locenc_hidden).squeeze(0), dtype=np.float32)

    def _log_track_performance(self, timing: _VoxCPM2Timing) -> None:
        """
        按 qwen3/s2-pro 的日志口径输出 voxcpm2 的 chunk 级性能统计。

        参数:
            timing: 当前 synthesize 调用聚合后的耗时统计。

        返回:
            本函数无返回值，只写日志。

        边界异常:
            chunk 生成与 decoder 音频返回在异常场景下可能不等长；
            这里使用较短长度对齐，避免性能日志本身再抛异常覆盖原始问题。
        """

        for idx in range(min(len(timing.chunk_gen_times), len(timing.decoder_compute_times))):
            message = (
                f"Chunk {idx} generation time: "
                f"{format_perf_value(timing.chunk_gen_times[idx] + timing.decoder_compute_times[idx])}"
            )
            message += f", generate patch time: {format_perf_value(timing.chunk_gen_times[idx])}"
            message += f", decode time: {format_perf_value(timing.decoder_compute_times[idx])}"
            logger.info(message)

        logger.info(f"Total inference time: {format_perf_value(timing.total_inference_time)}")
        logger.info(f"Prefill time: {format_perf_value(timing.prefill_time)}")
        logger.info(
            "First chunk audio generation time: %s, generate patch time: %s, decode time: %s",
            format_perf_value(timing.first_audio_latency),
            format_perf_value(timing.first_chunk_latency),
            format_perf_value(timing.first_decode_latency),
        )
        logger.info(f"Avg chunk patch generation time: {format_perf_value(timing.avg_chunk_gen_time)}")
        logger.info(f"Avg chunk decoder time: {format_perf_value(timing.avg_decoder_time)}")
        logger.info(f"Avg chunk RTF: {format_perf_value(timing.avg_chunk_rtf)}")


def load_voxcpm2_config() -> VoxCPM2GGUFConfig:
    """从 YAML 加载 VoxCPM2-GGUF runtime 默认配置。"""

    if not CONFIG_PATH.exists():
        return VoxCPM2GGUFConfig()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime", {})
    return VoxCPM2GGUFConfig(
        n_ctx=int(runtime.get("default_n_ctx", runtime.get("n_ctx", 8192))),
        default_mode=int(runtime.get("default_mode", 1)),
        max_length=int(runtime.get("max_length", 300)),
        min_length=int(runtime.get("min_length", 20)),
        inference_timesteps=int(runtime.get("inference_timesteps", 10)),
        cfg_value=float(runtime.get("cfg_value", 2.0)),
        streaming_prefix_len=int(runtime.get("streaming_prefix_len", 4)),
        base_n_gpu_layers=int(runtime.get("base_n_gpu_layers", -1)),
        residual_n_gpu_layers=int(runtime.get("residual_n_gpu_layers", -1)),
        decoder_start_method=str(runtime.get("decoder_start_method", "spawn")),
    )


def prompt_asset_from_payload(
    *,
    role_name: str,
    model_name: str,
    sample_rate: int,
    prompt_audio_feat: np.ndarray | None,
    prompt_locenc_hidden: np.ndarray | None,
    prompt_text: str | None,
    reference_audio_feat: np.ndarray | None,
    reference_locenc_hidden: np.ndarray | None,
    metadata: dict[str, str] | None = None,
) -> VoxCPM2PromptAsset:
    """按统一字段构造可缓存的 VoxCPM2 prompt 资产。"""

    return VoxCPM2PromptAsset(
        role_name=role_name,
        model=model_name,
        sample_rate=int(sample_rate),
        metadata=metadata or {},
        prompt_audio_feat=None if prompt_audio_feat is None else np.asarray(prompt_audio_feat, dtype=np.float32),
        prompt_locenc_hidden=None if prompt_locenc_hidden is None else np.asarray(prompt_locenc_hidden, dtype=np.float32),
        prompt_text=prompt_text,
        reference_audio_feat=None
        if reference_audio_feat is None
        else np.asarray(reference_audio_feat, dtype=np.float32),
        reference_locenc_hidden=None
        if reference_locenc_hidden is None
        else np.asarray(reference_locenc_hidden, dtype=np.float32),
    )


def serialize_prompt_asset(asset: VoxCPM2PromptAsset) -> SerializedPromptAsset:
    """把 VoxCPM2 prompt 资产转成 PromptCache 可写入的结构。"""

    arrays: dict[str, np.ndarray] = {}
    if asset.prompt_audio_feat is not None:
        arrays["prompt_audio_feat"] = np.asarray(asset.prompt_audio_feat, dtype=np.float32)
    if asset.prompt_locenc_hidden is not None:
        arrays["prompt_locenc_hidden"] = np.asarray(asset.prompt_locenc_hidden, dtype=np.float32)
    if asset.reference_audio_feat is not None:
        arrays["reference_audio_feat"] = np.asarray(asset.reference_audio_feat, dtype=np.float32)
    if asset.reference_locenc_hidden is not None:
        arrays["reference_locenc_hidden"] = np.asarray(asset.reference_locenc_hidden, dtype=np.float32)
    return SerializedPromptAsset(
        role_name=asset.role_name,
        model=asset.model,
        sample_rate=asset.sample_rate,
        metadata=dict(asset.metadata),
        arrays=arrays,
        json_data={
            "prompt_text": asset.prompt_text,
        },
    )


def deserialize_prompt_asset(payload: SerializedPromptAsset) -> VoxCPM2PromptAsset:
    """从 PromptCache 载荷恢复 VoxCPM2 prompt 资产。"""

    data = payload.json_data or {}
    return VoxCPM2PromptAsset(
        role_name=payload.role_name,
        model=payload.model,
        sample_rate=payload.sample_rate,
        metadata=dict(payload.metadata),
        prompt_audio_feat=None
        if "prompt_audio_feat" not in payload.arrays
        else np.asarray(payload.arrays["prompt_audio_feat"], dtype=np.float32),
        prompt_locenc_hidden=None
        if "prompt_locenc_hidden" not in payload.arrays
        else np.asarray(payload.arrays["prompt_locenc_hidden"], dtype=np.float32),
        prompt_text=data.get("prompt_text"),
        reference_audio_feat=None
        if "reference_audio_feat" not in payload.arrays
        else np.asarray(payload.arrays["reference_audio_feat"], dtype=np.float32),
        reference_locenc_hidden=None
        if "reference_locenc_hidden" not in payload.arrays
        else np.asarray(payload.arrays["reference_locenc_hidden"], dtype=np.float32),
    )


def _extract_embedding(ctx: llama.LlamaContext, index: int, n_embd: int) -> np.ndarray:
    """从 llama.cpp context 中读取单个 embedding 输出并复制到 numpy。"""

    emb_ptr = ctx.get_embeddings_ith(index)
    if not emb_ptr:
        raise RuntimeError(f"llama.cpp returned null embedding pointer for index {index}")
    return np.ctypeslib.as_array(emb_ptr, shape=(n_embd,)).copy().astype(np.float32, copy=False)


def _forward_concat_proj(module, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """执行 fusion_concat_proj([lhs, rhs])，集中处理拼接 dtype。"""

    concat_value = np.concatenate([lhs, rhs], axis=-1).astype(np.float32, copy=False)
    return module.forward(concat_value.reshape(1, -1)).reshape(-1)


def resolve_voxcpm2_onnx_device(device: str, execution_plan: ExecutionPlan | None) -> tuple[str, int | None]:
    """
    解析 VoxCPM2-GGUF 的 ONNX backend。

    参数:
        device: 用户入口传入的原始设备字符串。
        execution_plan: adapter 生成的执行计划；存在时优先使用其中的 onnx 目标。

    返回:
        返回 ``(onnx_backend, gpu_index)``，backend 取值为 ``cpu/cuda/dml``。

    可能异常:
        当用户强制请求 GPU 但当前平台缺少对应 provider 时，会记录告警并回退到 CPU。
    """

    onnx_target = None if execution_plan is None else execution_plan.get("onnx")
    if onnx_target is not None:
        explicit = resolve_explicit_onnx_backend(onnx_target)
        if explicit is not None:
            return explicit

    try:
        import onnxruntime as ort
    except Exception:
        if execution_plan is not None:
            onnx_target = execution_plan.get("onnx")
            if onnx_target is not None and onnx_target.mode == "gpu":
                logger.warning(
                    "[Device Fallback][IMPORTANT] 用户请求 voxcpm2-gguf ONNX GPU，但 onnxruntime 不可用，将回退到 CPU"
                )
        return "cpu", None

    available = set(ort.get_available_providers())

    def _resolve_from_mode(mode: str, index: int | None) -> tuple[str, int | None]:
        if mode == "cpu":
            return "cpu", None
        if mode in {"auto", "gpu"}:
            if os.name == "nt":
                if "CUDAExecutionProvider" in available:
                    return "cuda", 0 if index is None else int(index)
                if "DmlExecutionProvider" in available:
                    return "dml", 0 if index is None else int(index)
                if mode == "gpu":
                    logger.warning(
                        "[Device Fallback][IMPORTANT] 用户请求 voxcpm2-gguf ONNX GPU，但 DmlExecutionProvider 不可用，将回退到 CPU"
                    )
                return "cpu", None
            if "CUDAExecutionProvider" in available:
                return "cuda", 0 if index is None else int(index)
            if mode == "gpu":
                logger.warning(
                    "[Device Fallback][IMPORTANT] 用户请求 voxcpm2-gguf ONNX GPU，但 CUDAExecutionProvider 不可用，将回退到 CPU"
                )
            return "cpu", None
        raise ValueError(f"unsupported onnx target mode: {mode}")

    if execution_plan is not None:
        onnx_target = execution_plan.get("onnx")
        if onnx_target is not None:
            return _resolve_from_mode(onnx_target.mode, onnx_target.device_index)

    normalized_device = str(device).strip().lower()
    if normalized_device == "cpu":
        return "cpu", None
    if normalized_device == "auto":
        return _resolve_from_mode("auto", None)
    if normalized_device == "gpu":
        return _resolve_from_mode("gpu", 0)
    if normalized_device.startswith("gpu:"):
        index_text = normalized_device.split(":", 1)[1]
        return _resolve_from_mode("gpu", int(index_text) if index_text.isdigit() else 0)
    return "cpu", None
