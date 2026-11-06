"""用固定设备地址及单个 CUDA Graph 连续解码完整音频帧。"""

from __future__ import annotations

import json
from pathlib import Path
from threading import get_ident

import numpy as np

from .cuda import CudaStream, DEVICE_TO_DEVICE, DEVICE_TO_HOST, HOST_TO_DEVICE
from .ort import OrtGraph


class CudaStreamingDecoder(OrtGraph):
    """在同一持久工作线程上预捕获并重放完整 FP32 声码器。

    Args:
        path: 单帧声码器图路径，旁边须有完整状态元数据。
        device_id: 声码器使用的显卡编号。
    Returns:
        固定输入输出地址且通过独立流推进状态的解码器。
    """

    def __init__(self, path: Path, device_id: int = 0) -> None:
        self.session = None
        self._stream = None
        self._binding = None
        self._codes = self._position = self._audio = None
        self._host_codes = self._host_position = self._host_audio = None
        self._input_states = {}
        self._output_states = {}
        self._host_states = {}
        self._outputs = {}
        self._state_copies = []
        self._initialized = False
        self._warmed_up = False
        self._thread_id = None
        self._closed = False
        self._failed = False
        try:
            import onnxruntime as ort

            path = Path(path)
            metadata = json.loads(path.with_name("codec_metadata.json").read_text(encoding="utf-8"))
            self._state_specs = metadata["state_specs"]
            if int(metadata["num_codebooks"]) != 16 or len(self._state_specs) != 37:
                raise ValueError("Breeze 声码器要求完整十六码本及三十七份状态。")
            if any(item["dtype"] != "float32" for item in self._state_specs):
                raise ValueError("Breeze 声码器状态必须保持 FP32。")
            self._stream = CudaStream(device_id)
            super().__init__(path, "cuda", device_id, cuda_graph_stream=self._stream.address)
            self._feed_names = {"codes", "position", *[item["name"] for item in self._state_specs]}
            output_names = {"audio", *["next_" + item["name"] for item in self._state_specs]}
            if {item.name for item in self.inputs} != self._feed_names or {item.name for item in self.outputs} != output_names:
                raise ValueError("Breeze 声码器图接口与完整状态元数据不一致。")

            self._codes = ort.OrtValue.ortvalue_from_shape_and_type([1, 1, 16], np.int64, "cuda", device_id)
            self._position = ort.OrtValue.ortvalue_from_shape_and_type([1], np.int64, "cuda", device_id)
            audio_shape = [1, int(metadata["samples_per_frame"])]
            self._audio = ort.OrtValue.ortvalue_from_shape_and_type(audio_shape, np.float32, "cuda", device_id)
            self._input_states = {
                item["name"]: ort.OrtValue.ortvalue_from_shape_and_type(item["shape"], np.float32, "cuda", device_id)
                for item in self._state_specs
            }
            self._output_states = {
                item["name"]: ort.OrtValue.ortvalue_from_shape_and_type(item["shape"], np.float32, "cuda", device_id)
                for item in self._state_specs
            }
            self._host_codes = self._stream.allocate_host([1, 1, 16], np.int64)
            self._host_position = self._stream.allocate_host([1], np.int64)
            self._host_audio = self._stream.allocate_host(audio_shape, np.float32)
            self._host_states = {item["name"]: self._stream.allocate_host(item["shape"], np.float32) for item in self._state_specs}
            self._binding = self.session.io_binding()
            self._binding.bind_ortvalue_input("codes", self._codes)
            self._binding.bind_ortvalue_input("position", self._position)
            self._binding.bind_ortvalue_output("audio", self._audio)
            for item in self._state_specs:
                name = item["name"]
                self._binding.bind_ortvalue_input(name, self._input_states[name])
                self._binding.bind_ortvalue_output("next_" + name, self._output_states[name])
                self._state_copies.append((
                    self._input_states[name].data_ptr(), self._output_states[name].data_ptr(), self._host_states[name].nbytes,
                ))
            self._outputs = {"next_" + name: value for name, value in self._output_states.items()}
            self._run_options = ort.RunOptions()
            self._run_options.add_run_config_entry("gpu_graph_id", "0")
            # 提供程序默认同步会等待整卡，本解码器只等待自己的非阻塞流。
            self._run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"声码器初始化失败后的清理也发生异常：{cleanup_error}")
            raise

    def warmup(self) -> None:
        """在后续运行所属的持久线程上提前完成零帧图捕获。

        Returns:
            捕获完成且状态恢复为等待新请求后正常返回。
        """
        if self._closed:
            raise RuntimeError("Breeze CUDA 声码器已经关闭。")
        if self._failed:
            raise RuntimeError("Breeze CUDA 声码器未能完成流同步，无法复用。")
        if self._thread_id is not None and self._thread_id != get_ident():
            raise RuntimeError("Breeze CUDA 声码器预热与运行必须使用同一个持久线程。")
        if self._warmed_up:
            return
        self._thread_id = get_ident()
        try:
            self.run({
                "codes": np.zeros_like(self._host_codes),
                "position": np.zeros_like(self._host_position),
                **{name: np.zeros_like(value) for name, value in self._host_states.items()},
            })
        except BaseException:
            self._thread_id = None
            self._initialized = False
            raise
        self._initialized = False
        self._warmed_up = True

    def run(self, feeds: dict[str, object]) -> dict[str, object]:
        """在固定流上复制输入、推进状态、重放图并返回独立音频副本。

        Args:
            feeds: 码本、绝对位置及全部三十七份连续状态，新请求使用完整主机状态。
        Returns:
            主机音频数组及同名设备后继状态，设备缓冲在下一次调用时复用。
        """
        if self._closed:
            raise RuntimeError("Breeze CUDA 声码器已经关闭。")
        if self._failed:
            raise RuntimeError("Breeze CUDA 声码器未能完成流同步，无法复用。")
        if self._thread_id != get_ident():
            raise RuntimeError("Breeze CUDA 声码器须在已完成预热的同一个持久线程上运行。")
        if feeds.keys() != self._feed_names:
            raise ValueError("Breeze 声码器每步必须提供码本、位置及全部状态。")
        codes, position = feeds["codes"], feeds["position"]
        if not isinstance(codes, np.ndarray) or codes.dtype != np.int64 or codes.shape != (1, 1, 16):
            raise ValueError("Breeze 声码器码本必须是形状为一乘一乘十六的整数数组。")
        if not isinstance(position, np.ndarray) or position.dtype != np.int64 or position.shape != (1,):
            raise ValueError("Breeze 声码器位置必须是单元素整数数组。")
        self._stream.activate()
        host_states = [isinstance(feeds[item["name"]], np.ndarray) for item in self._state_specs]
        try:
            if all(host_states):
                # 每个请求覆盖完整固定输入状态，预捕获及上一请求的输出不会泄漏进来。
                for item in self._state_specs:
                    name = item["name"]
                    value = feeds[name]
                    if value.dtype != np.float32 or list(value.shape) != item["shape"]:
                        raise ValueError(f"Breeze 声码器初始状态格式错误：{name}")
                    host = self._host_states[name]
                    np.copyto(host, value)
                    self._stream.copy(self._input_states[name].data_ptr(), host.ctypes.data, host.nbytes, HOST_TO_DEVICE)
                self._initialized = True
            elif any(host_states) or not self._initialized:
                raise ValueError("Breeze 声码器须先接收完整主机初始状态，随后传递连续设备状态。")
            else:
                for item in self._state_specs:
                    name = item["name"]
                    if feeds[name].data_ptr() != self._output_states[name].data_ptr():
                        raise ValueError(f"Breeze 声码器设备状态没有按上一帧连续传递：{name}")
                # 单图捕获要求输入地址恒定，状态只在同一条流上从输出复制回输入。
                for destination, source, size in self._state_copies:
                    self._stream.copy(destination, source, size, DEVICE_TO_DEVICE)
            for value, host, device in ((codes, self._host_codes, self._codes), (position, self._host_position, self._position)):
                np.copyto(host, value)
                self._stream.copy(device.data_ptr(), host.ctypes.data, host.nbytes, HOST_TO_DEVICE)
            self.session.run_with_iobinding(self._binding, self._run_options)
            self._stream.copy(self._host_audio.ctypes.data, self._audio.data_ptr(), self._host_audio.nbytes, DEVICE_TO_HOST)
        except BaseException as error:
            self._initialized = False
            # 任务结束前排空已入队传输，下一请求才能安全改写持久锁页缓冲。
            try:
                self._stream.synchronize()
            except BaseException as cleanup_error:
                self._failed = True
                error.add_note(f"声码器故障后的流排空也发生异常，解码器无法复用：{cleanup_error}")
            raise
        try:
            self._stream.synchronize()
        except BaseException:
            # 同步失败时无法确认传输是否结束，必须保留缓冲并阻止后续请求复用。
            self._initialized = False
            self._failed = True
            raise
        return {"audio": self._host_audio.copy(), **self._outputs}

    def close(self) -> None:
        """等待本流后依次释放绑定、会话、设备缓冲及外部流。

        Returns:
            资源释放后正常返回，重复关闭无效。
        """
        if self._closed:
            return
        if self._stream is not None:
            self._stream.synchronize()
        self._binding = None
        super().close()
        self._codes = self._position = self._audio = None
        self._input_states.clear()
        self._output_states.clear()
        self._outputs.clear()
        self._host_states.clear()
        self._host_codes = self._host_position = self._host_audio = None
        self._state_copies.clear()
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._initialized = False
        self._thread_id = None
        self._closed = True
