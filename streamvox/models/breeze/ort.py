"""执行 Breeze 的 ONNX 编解码图并在设备端保留声码器状态。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streamvox_runtime import ensure_single_onnxruntime_distribution, initialize_nvidia_runtime

from ..base import ExecutionTarget
from ..onnx_backend import resolve_explicit_onnx_backend, verify_session_provider


def resolve_backend(target: ExecutionTarget) -> tuple[str, int]:
    """解析 ONNX 执行意图，显式 GPU 请求失败时直接报错。

    Args:
        target: 适配器解析的设备计划。
    Returns:
        后端名称与设备编号。
    """
    ensure_single_onnxruntime_distribution()
    import onnxruntime as ort

    explicit = resolve_explicit_onnx_backend(target)
    if explicit is not None:
        return explicit[0], explicit[1] or 0
    if target.mode == "cpu":
        return "cpu", 0
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "cuda", target.device_index or 0
    if "DmlExecutionProvider" in available:
        return "dml", target.device_index or 0
    if target.mode == "gpu":
        raise RuntimeError("Breeze 请求 GPU，但当前 ONNX Runtime 没有可用 GPU 后端。")
    return "cpu", 0


def as_numpy(value: object) -> np.ndarray:
    """只在读取小型分数或最终音频时把设备结果取回主机。

    Args:
        value: NumPy 数组或 ONNX 设备数组。
    Returns:
        主机数组。
    """
    return value if isinstance(value, np.ndarray) else value.numpy()


class OrtGraph:
    """持有单个辅助图，CUDA 声码器状态通过绑定直接传递。

    Args:
        path: ONNX 图路径。
        backend: 已解析的后端。
        device_id: GPU 设备编号。
        cuda_graph_stream: 固定 CUDA 图绑定所使用的专用流地址。
        cpu_initializers: 由调用方持有的处理器权重，用于零复制映射外部张量。
    Returns:
        可执行图对象。
    """

    def __init__(
        self, path: Path, backend: str, device_id: int = 0, *,
        cuda_graph_stream: int | None = None, cpu_initializers: dict[str, np.ndarray] | None = None,
    ) -> None:
        ensure_single_onnxruntime_distribution()
        initialize_nvidia_runtime(consumer="onnx_cuda", required=backend == "cuda")
        import onnxruntime as ort

        self.backend = backend
        self.device_id = device_id
        self._cpu_initializers = dict(cpu_initializers or {})
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 长度持续变化时避免为每个历史长度保留一份内存模式。
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # 每帧小图使用单线程，避免调度开销超过查表或单次投影本身。
        if path.name in {"audio_embedding.onnx", "depth_conditioner.onnx", "text_encoder_lookup.onnx"} or backend == "cuda":
            options.intra_op_num_threads = 1
        if self._cpu_initializers:
            if backend != "cpu":
                raise ValueError("外部只读映射权重只用于处理器会话。")
            names = list(self._cpu_initializers)
            options.add_external_initializers(names, [ort.OrtValue.ortvalue_from_numpy(self._cpu_initializers[name]) for name in names])
        # 多个常驻会话交替工作，空闲线程及时休眠以释放原生解码的处理器时间。
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        providers: list = ["CPUExecutionProvider"]
        if backend == "cuda":
            cuda_options = {
                "device_id": device_id, "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0",
                "use_tf32": "0",
            }
            if cuda_graph_stream is not None:
                cuda_options.update(enable_cuda_graph="1", user_compute_stream=str(cuda_graph_stream))
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                providers = []
            providers.insert(0, ("CUDAExecutionProvider", cuda_options))
        elif backend == "dml":
            providers.insert(0, ("DmlExecutionProvider", {"device_id": device_id}))
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers,
        )
        verify_session_provider(self.session, providers)
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

    def run(self, feeds: dict[str, object]) -> dict[str, object]:
        """执行图，CUDA 的大数组输出保持在设备端。

        Args:
            feeds: 输入名称到主机或设备数组的映射。
        Returns:
            输出名称到结果数组的映射。
        """
        if self.backend != "cuda":
            values = self.session.run(None, {name: as_numpy(value) for name, value in feeds.items()})
            return dict(zip((output.name for output in self.outputs), values, strict=True))
        binding = self.session.io_binding()
        for name, value in feeds.items():
            if isinstance(value, np.ndarray):
                binding.bind_cpu_input(name, np.ascontiguousarray(value))
            else:
                binding.bind_ortvalue_input(name, value)
        for output in self.outputs:
            binding.bind_output(output.name, "cuda", self.device_id)
        self.session.run_with_iobinding(binding)
        # 结果还会交给另一个会话，显式完成当前流，保证跨会话的设备数组可直接读取。
        binding.synchronize_outputs()
        values = binding.get_outputs()
        return dict(zip((output.name for output in self.outputs), values, strict=True))

    def close(self) -> None:
        """先释放接口元数据，再释放会话持有的模型和后端资源。

        Returns:
            会话资源释放后返回。
        """
        # 接口元数据持有原生会话引用，必须在销毁会话使用的外部流前一并释放。
        self.inputs = []
        self.outputs = []
        self.session = None
        self._cpu_initializers = {}
