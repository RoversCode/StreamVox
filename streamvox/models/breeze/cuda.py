"""管理声码器独立的 CUDA 流、锁页缓冲及异步内存复制。"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from streamvox_runtime import initialize_nvidia_runtime
from streamvox_runtime.native_loader import nvidia_library_directories


HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2
DEVICE_TO_DEVICE = 3


class CudaStream:
    """持有指定设备的非阻塞流和锁页主机缓冲。

    Args:
        device_id: 声码器使用的显卡编号。
    Returns:
        负责本流内存传输、同步及释放的资源对象。
    """

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
        self._runtime = None
        self._handle = ctypes.c_void_p()
        self._host_allocations: list[ctypes.c_void_p] = []
        try:
            initialize_nvidia_runtime(consumer="onnx_cuda", required=True)
            filename = "cudart64_12.dll" if os.name == "nt" else "libcudart.so.12"
            path = next((directory / filename for directory in nvidia_library_directories(consumer="onnx_cuda")
                         if (directory / filename).is_file()), None)
            if path is None:
                raise RuntimeError(f"未找到声码器所需的 NVIDIA 运行库：{filename}")
            if os.name == "nt":
                self._runtime = ctypes.WinDLL(str(path), winmode=0x00000100 | 0x00000800)
            else:
                self._runtime = ctypes.CDLL(str(path))
            signatures = {
                "cudaSetDevice": [ctypes.c_int],
                "cudaStreamCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
                "cudaStreamSynchronize": [ctypes.c_void_p],
                "cudaStreamDestroy": [ctypes.c_void_p],
                "cudaMemcpyAsync": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
                "cudaHostAlloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
                "cudaFreeHost": [ctypes.c_void_p],
            }
            for name, parameters in signatures.items():
                function = getattr(self._runtime, name)
                function.argtypes = parameters
                function.restype = ctypes.c_int
            self._runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
            self._runtime.cudaGetErrorString.restype = ctypes.c_char_p
            self.activate()
            self._check(self._runtime.cudaStreamCreateWithFlags(ctypes.byref(self._handle), 1))
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"CUDA 流初始化失败后的清理也发生异常：{cleanup_error}")
            raise

    @property
    def address(self) -> int:
        """返回供 ONNX Runtime 使用的外部流地址。

        Returns:
            流句柄的整数地址，关闭后为零。
        """
        return self._handle.value or 0

    def _check(self, status: int) -> None:
        """保留 CUDA 调用的原始错误说明。

        Args:
            status: CUDA 调用返回的状态码。
        Returns:
            调用成功时正常返回。
        """
        if status:
            message = self._runtime.cudaGetErrorString(status).decode("utf-8", errors="replace")
            raise RuntimeError(f"声码器 CUDA 调用失败，错误码 {status}，{message}")

    def activate(self) -> None:
        """使当前调用线程使用本流所属的 CUDA 设备。

        Returns:
            当前线程设备切换成功后正常返回。
        """
        self._check(self._runtime.cudaSetDevice(self.device_id))

    def allocate_host(self, shape: tuple[int, ...] | list[int], dtype: np.dtype) -> np.ndarray:
        """分配持续保活到流关闭的锁页主机数组。

        Args:
            shape: 数组固定形状。
            dtype: 数组数据类型。
        Returns:
            底层内存由本对象管理的主机数组。
        """
        if not self.address:
            raise RuntimeError("声码器 CUDA 流已经关闭。")
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize
        pointer = ctypes.c_void_p()
        self._check(self._runtime.cudaHostAlloc(ctypes.byref(pointer), size, 1))
        self._host_allocations.append(pointer)
        storage = (ctypes.c_ubyte * size).from_address(pointer.value)
        return np.ctypeslib.as_array(storage).view(dtype).reshape(shape)

    def copy(self, destination: int, source: int, size: int, kind: int) -> None:
        """把内存传输排入本流，调用方应先激活所属设备。

        Args:
            destination: 目标内存地址。
            source: 来源内存地址。
            size: 复制的字节数。
            kind: CUDA 定义的主机或设备复制方向。
        Returns:
            传输成功入队时正常返回。
        """
        if not self.address:
            raise RuntimeError("声码器 CUDA 流已经关闭。")
        self._check(self._runtime.cudaMemcpyAsync(destination, source, size, kind, self._handle))

    def synchronize(self) -> None:
        """只等待本流中的复制和计算完成。

        Returns:
            本流全部任务完成后正常返回，已关闭的流无需等待。
        """
        if self.address:
            self.activate()
            self._check(self._runtime.cudaStreamSynchronize(self._handle))

    def close(self) -> None:
        """同步本流后释放锁页缓冲及流句柄。

        Returns:
            资源释放后正常返回，重复关闭无效。
        """
        if not self.address and not self._host_allocations:
            return
        self.activate()
        self.synchronize()
        # 每次成功释放后立即移除记录，失败后重试不会重复释放已归还的内存。
        while self._host_allocations:
            self._check(self._runtime.cudaFreeHost(self._host_allocations[-1]))
            self._host_allocations.pop()
        if self.address:
            self._check(self._runtime.cudaStreamDestroy(self._handle))
            self._handle = ctypes.c_void_p()
