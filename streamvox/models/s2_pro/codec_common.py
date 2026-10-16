from __future__ import annotations

import platform
from dataclasses import dataclass

import onnxruntime as ort

from ...utils.log import logger


@dataclass(slots=True)
class CodecOnnxConfig:
    sample_rate: int
    frame_length: int
    hop_length: int
    num_codebooks: int
    stream_code_frames: int = 21  # 流式每个chunk数量


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def resolve_ort_providers(preferred: list[str] | None = None) -> list[str]:
    available = ort.get_available_providers()
    if preferred is None:
        system_name = platform.system()
        # S2-Pro codec 与参考代码保持一致：Windows 优先 DML，Linux 优先 CUDA，最后回 CPU。
        if system_name == "Windows":
            preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    resolved = [provider for provider in preferred if provider in available]
    if not resolved:
        resolved = ["CPUExecutionProvider"]
    logger.debug("S2-Pro ONNX providers: %s (available=%s)", resolved, available)
    return resolved


def create_session_options() -> ort.SessionOptions:
    # 单条语音低延迟场景下顺序执行更稳定，避免 ORT 对小图做过度并行调度。
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 规定模型中的计算节点必须顺序排队执行，而不是多线程并行执行。
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # 开启 CPU 内存池。
    options.enable_cpu_mem_arena = True
    # 开启内存使用模式的预测与重用
    options.enable_mem_pattern = True
    # ORT 会把转置移过归一化，CUDA 快速归约因此使用浮点原子累加。
    # 固定输入也可能改变参考音频的整数码本；确定性模式关闭该快速归约，保留精度与公式。
    options.use_deterministic_compute = True
    return options
