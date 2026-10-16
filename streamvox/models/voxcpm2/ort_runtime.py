"""ORT session 与 provider 选择工具。"""

from __future__ import annotations

from ..onnx_backend import verify_session_provider

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np







from streamvox_runtime import ensure_single_onnxruntime_distribution, initialize_nvidia_runtime

ensure_single_onnxruntime_distribution()
import onnxruntime as ort

from ...utils.log import logger


@dataclass(frozen=True)
class OrtSessionBundle:
    """描述一个已加载的 ORT session。"""

    model_path: str
    providers: list[str]
    input_names: list[str]
    output_names: list[str]
    input_types: list[str]
    output_types: list[str]


def _ort_module_name_from_path(model_path: str) -> str:
    """
    提取 ORT 模块的人类可读名称。
    参数:
        model_path: ONNX 模型文件路径。
    返回:
        返回适合初始化摘要日志展示的模块名。
    边界异常:
        当文件名不在已知映射中时，回退为去扩展名后的原始文件名，避免日志直接丢失模块身份。
    """

    stem = Path(model_path).stem.lower()
    if "audio_vae_decoder" in stem:
        return "AudioVAE Decoder"
    if "locdit" in stem:
        return "LocDiT"
    if "speaker_encoder" in stem:
        return "Speaker Encoder"
    return Path(model_path).stem


def _ort_device_label(providers: list[str]) -> str:
    """
    归一化 ORT provider，输出稳定的设备摘要。
    参数:
        providers: session 实际激活的 provider 列表。
    返回:
        返回形如 `DML:0`、`CUDA:0`、`CPU` 的设备标签。
    边界异常:
        provider 列表为空时回退为 `Unknown`，避免初始化日志因 provider 探测异常而报错。
    """

    primary_provider = providers[0] if providers else ""
    if primary_provider == "DmlExecutionProvider":
        return "DML:0"
    if primary_provider == "CUDAExecutionProvider":
        return "CUDA:0"
    if primary_provider == "CPUExecutionProvider":
        return "CPU"
    if primary_provider:
        return primary_provider.replace("ExecutionProvider", "")
    return "Unknown"


def _ort_precision_label(bundle: OrtSessionBundle) -> str:
    """
    从 ONNX 文件名与 I/O dtype 推断最适合给人看的精度标签。
    参数:
        bundle: ORT session 的只读元信息。
    返回:
        返回初始化摘要所需的精度标签，例如 `FP16`、`FP32`。
    边界异常:
        先信任文件名显式标注；若文件名未标注，则回退到输入/输出 dtype 交集推断；都失败时返回 `Unknown`。
    """

    normalized_path = Path(bundle.model_path).stem.lower()
    if "fp16" in normalized_path:
        return "FP16"
    if "fp32" in normalized_path:
        return "FP32"

    joined_types = " ".join([*bundle.input_types, *bundle.output_types]).lower()
    if "float16" in joined_types:
        return "FP16"
    if "float)" in joined_types or "float32" in joined_types:
        return "FP32"
    return "Unknown"


def resolve_ort_providers(
    prefer_gpu: bool = True,
    *,
    device_id: int | None = None,
) -> list[str | tuple[str, dict[str, int]]]:
    """
    解析当前环境可用于 voxcpm2-gguf 的 ORT provider 列表。

    参数:
        prefer_gpu: 为 True 时优先尝试平台对应的 GPU provider。
        device_id: GPU provider 使用的设备索引；缺省时使用 0 号卡。

    返回:
        返回可直接传给 ``onnxruntime.InferenceSession(..., providers=...)`` 的 provider 列表。

    可能异常:
        本函数不主动抛出平台分支异常；GPU provider 不可用时自动回退到 CPU。
    """

    available = ort.get_available_providers()
    providers: list[str | tuple[str, dict[str, int]]] = []

    # Windows 端按产品约定优先走 DirectML；Linux 等平台继续优先 CUDA。
    # 两条 GPU 路径都显式保留 CPU fallback，避免 provider 初始化差异直接阻塞主链路。
    if prefer_gpu:
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {"device_id": 0 if device_id is None else int(device_id)},
                )
            )
        if os.name == "nt" and "DmlExecutionProvider" in available:
            providers.append(
                (
                    "DmlExecutionProvider",
                    {"device_id": 0 if device_id is None else int(device_id)},
                )
            )
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        providers = available
    return providers

def create_ort_session(
    model_path: str,
    *,
    prefer_gpu: bool = True,
    device_id: int | None = None,
) -> tuple[ort.InferenceSession, OrtSessionBundle]:
    """
    创建 ORT session，并返回用于日志/调试的元信息。

    参数:
        model_path: ONNX 模型路径。
        prefer_gpu: 为 True 时优先尝试 CUDAExecutionProvider。
        device_id: CUDA device 索引；为空时默认使用 0 号卡。

    返回:
        返回 `(session, bundle)`。

    边界异常:
        provider 实际可用性由当前宿主机与 onnxruntime 安装决定；CUDA 不可用时会自然回退到 CPU。
    """

    # 统一配置 ORT SessionOptions，优先启用图优化，并关闭线程自旋带来的额外空转开销。
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model_file = Path(model_path).resolve()
    providers = resolve_ort_providers(prefer_gpu=prefer_gpu, device_id=device_id)
    if any((item[0] if isinstance(item, tuple) else item) == "CUDAExecutionProvider" for item in providers):
        initialize_nvidia_runtime(consumer="onnx_cuda", required=True)
    session = ort.InferenceSession(
        str(model_file),
        sess_options=sess_opts,
        providers=providers,
    )
    verify_session_provider(session, providers)
    bundle = OrtSessionBundle(
        model_path=str(model_file),
        providers=session.get_providers(),
        input_names=[item.name for item in session.get_inputs()],
        output_names=[item.name for item in session.get_outputs()],
        input_types=[str(item.type) for item in session.get_inputs()],
        output_types=[str(item.type) for item in session.get_outputs()],
    )
    _log_ort_session_bundle(bundle=bundle, prefer_gpu=prefer_gpu, device_id=device_id)
    return session, bundle


def run_single_input(session: ort.InferenceSession, input_name: str, value: np.ndarray) -> np.ndarray:
    """运行一个单输入单输出的 ORT session。"""

    outputs = session.run(None, {input_name: np.asarray(value, dtype=np.float32)})
    if len(outputs) != 1:
        raise RuntimeError(f"expected single ORT output, got {len(outputs)}")
    return np.asarray(outputs[0], dtype=np.float32)


def _log_ort_session_bundle(
    *,
    bundle: OrtSessionBundle,
    prefer_gpu: bool,
    device_id: int | None,
) -> None:
    """
    记录 ORT session 初始化后的 provider 与精度摘要。

    参数:
        bundle: 已创建完成的 session 元信息。
        prefer_gpu: 本次初始化是否显式偏好 GPU provider。
        device_id: 期望绑定的设备索引；为空时表示使用 provider 默认设备。

    返回:
        无返回值；仅输出用于排查设备与精度问题的日志。

    边界异常:
        日志构造阶段不应影响主链路初始化，因此这里只做只读摘要拼装，不访问 session 的重型接口。
    """

    logger.info(
        "✅ [voxcpm2-gguf] [%s] [%s] [%s]",
        _ort_module_name_from_path(bundle.model_path),
        _ort_device_label(bundle.providers),
        _ort_precision_label(bundle),
    )
