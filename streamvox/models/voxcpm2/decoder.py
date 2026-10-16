from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import sys
import threading
import time
import types
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np

from ...utils.log import logger
from .schema.decoder_protocol import DecodeRequest, DecoderResponse


class VoxCPM2DecoderAPI:
    """
    VoxCPM2-GGUF AudioVAE decoder 主进程代理。

    参数:
        decoder_stateful_onnx_path: AudioVAE stateful decoder ONNX 路径。
        metadata_path: decoder state metadata 路径。
        patch_size: latent patch 时间维。
        decode_chunk_size: patch 对应输出采样窗口。
        latent_dim: latent 特征维。
        prefer_gpu: 是否优先使用 GPU ORT provider。
        start_method: multiprocessing 启动方式，默认 spawn。

    返回:
        初始化后得到可提交 latent patch 的 decoder 代理对象。

    边界异常:
        worker 进程启动失败或 READY 超时由 runtime 初始化阶段处理。
    """

    def __init__(
        self,
        *,
        decoder_stateful_onnx_path: str | Path,
        metadata_path: str | Path,
        patch_size: int,
        decode_chunk_size: int,
        latent_dim: int,
        prefer_gpu: bool,
        device_id: int | None = None,
        start_method: str = "spawn",
    ) -> None:
        self.decoder_stateful_onnx_path = Path(decoder_stateful_onnx_path)
        self.metadata_path = Path(metadata_path)
        self.patch_size = int(patch_size)
        self.decode_chunk_size = int(decode_chunk_size)
        self.latent_dim = int(latent_dim)
        self.prefer_gpu = bool(prefer_gpu)
        self.device_id = None if device_id is None else int(device_id)
        self.start_method = str(start_method)
        self.results: dict[str, list[DecoderResponse]] = {}
        # 诊断事件只记录当前进程可观测到的请求/响应时间点，用于 smoke 测试证明
        # latent patch 生成线程和 AudioVAE decoder worker 的确在重叠执行。
        self.task_timing_events: dict[str, list[dict[str, object]]] = {}
        # timing 只保留最近一小段任务，避免长期 SDK 进程反复合成时诊断信息无限增长。
        self.task_timing_order: list[str] = []
        self.stream_subscribers: dict[str, queue.Queue] = {}
        self.subscriber_lock = threading.Lock()
        self.ready = False
        self.stop_listener = False
        # shutdown 允许被 runtime 显式调用，也允许进程退出时由 atexit 再次调用；
        # 这里用显式状态位保证关闭流程幂等，避免重复 join/close 引发噪音。
        self._shutdown_started = False

        self.mp_ctx = mp.get_context(self.start_method)
        self.request_q = self.mp_ctx.Queue()
        self.response_q = self.mp_ctx.Queue()
        self.decoder_proc = None
        self.listener_thread: threading.Thread | None = None
        self.start()
        atexit.register(self.shutdown)

    def start(self) -> None:
        """启动 decoder worker 子进程和响应监听线程。"""

        from .worker.decode_worker import decoder_worker_proc

        self.decoder_proc = self.mp_ctx.Process(
            target=decoder_worker_proc,
            args=(
                self.request_q,
                self.response_q,
                self.decoder_stateful_onnx_path,
                self.metadata_path,
                self.patch_size,
                self.decode_chunk_size,
                self.latent_dim,
                self.prefer_gpu,
                self.device_id,
            ),
            daemon=True,
        )
        # spawn 场景下保护 __main__，降低 SDK 在交互环境和无 main guard 入口里的使用门槛。
        with _MainModuleSpawnGuard():
            self.decoder_proc.start()
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def subscribe_stream(self, task_id: str) -> queue.Queue:
        """为单次合成注册实时响应队列，前台生成器会从该队列消费 AUDIO/FINISH。"""

        subscriber: queue.Queue = queue.Queue()
        with self.subscriber_lock:
            if task_id in self.stream_subscribers:
                raise RuntimeError(f"voxcpm2 decoder stream subscriber already exists for task_id={task_id}")
            self.stream_subscribers[task_id] = subscriber
        return subscriber

    def unsubscribe_stream(self, task_id: str, subscriber: queue.Queue) -> None:
        """取消指定 task 的实时订阅，避免前台提前结束后队列继续累积消息。"""

        with self.subscriber_lock:
            if self.stream_subscribers.get(task_id) is subscriber:
                self.stream_subscribers.pop(task_id, None)

    def decode_patch(self, *, task_id: str, patch: np.ndarray | None, is_final: bool = False) -> None:
        """异步提交单个 latent patch；音频结果稍后由订阅队列返回。"""

        self._record_timing_event(
            task_id=str(task_id),
            direction="request",
            msg_type="FINISH" if is_final else "DECODE_PATCH",
            index=None,
        )
        self.request_q.put(
            DecodeRequest(
                task_id=str(task_id),
                msg_type="DECODE_PATCH",
                patch=None if patch is None else np.asarray(patch, dtype=np.float32),
                is_final=bool(is_final),
            )
        )

    def wait_until_ready(self, timeout: float = 60) -> bool:
        """等待 worker 完成 ORT decoder 初始化。"""

        started = time.time()
        while time.time() - started < timeout:
            if self.ready:
                return True
            time.sleep(0.1)
        return False

    def warmup(self, patch: np.ndarray | None = None, timeout: float = 30.0) -> None:
        """
        提前跑一轮最小 decoder 请求，把首个真实请求的初始化抖动前移。

        参数:
            patch: 可选的 warmup latent patch；为空时默认使用全零 patch。
            timeout: 等待 warmup 完成的超时时间。

        返回:
            无返回值；成功返回表示 decoder worker 已经真正执行过一次完整 decode 流程。

        边界异常:
            warmup 超时或 worker 返回 ERROR 时直接抛错，避免把问题拖到用户首个 stream 请求里。
        """

        warmup_task_id = f"decoder-warmup-{uuid.uuid4()}"
        subscriber = self.subscribe_stream(warmup_task_id)
        warmup_patch = (
            np.zeros((self.patch_size, self.latent_dim), dtype=np.float32)
            if patch is None
            else np.asarray(patch, dtype=np.float32)
        )
        started = time.time()
        try:
            self.decode_patch(task_id=warmup_task_id, patch=warmup_patch, is_final=False)
            self.decode_patch(task_id=warmup_task_id, patch=None, is_final=True)
            while time.time() - started < float(timeout):
                try:
                    message = subscriber.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not isinstance(message, DecoderResponse):
                    continue
                if message.msg_type == "ERROR":
                    raise RuntimeError(message.error_message or "voxcpm2-gguf decoder warmup failed")
                if message.msg_type == "FINISH":
                    return
            raise RuntimeError("voxcpm2-gguf decoder warmup timed out")
        finally:
            self.unsubscribe_stream(warmup_task_id, subscriber)
            with self.subscriber_lock:
                self.results.pop(warmup_task_id, None)

    def _listen_loop(self) -> None:
        """后台监听 worker 响应，并按 task_id 分发到订阅队列。"""

        while not self.stop_listener:
            try:
                message = self.response_q.get(timeout=0.1)
                if message is None:
                    break
                if not isinstance(message, DecoderResponse):
                    continue
                self._handle_response(message)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("VoxCPM2-GGUF decoder listener failed: %s", exc)
                break

    def _handle_response(self, message: DecoderResponse) -> None:
        """处理单条 worker 响应，READY 更新状态，其它消息 fan-out 给订阅者。"""

        if message.msg_type == "READY":
            logger.info("✅ [voxcpm2] [Decoder API] ready")
            self.ready = True
            return
        self._record_timing_event(
            task_id=message.task_id,
            direction="response",
            msg_type=message.msg_type,
            index=message.index,
        )
        with self.subscriber_lock:
            self.results.setdefault(message.task_id, []).append(message)
            subscriber = self.stream_subscribers.get(message.task_id)
        if subscriber is not None:
            subscriber.put(message)

    def get_task_timing_events(self, task_id: str) -> list[dict[str, object]]:
        """
        返回指定 task 的 decoder 请求/响应时间线快照。

        参数:
            task_id: 单次合成在 runtime 内部分配的任务编号。

        返回:
            返回按记录顺序排列的事件列表；调用方拿到的是深拷贝，不能反向修改 decoder 内部状态。

        边界异常:
            task_id 不存在时返回空列表，便于 smoke 脚本用统一逻辑处理。
        """

        with self.subscriber_lock:
            return deepcopy(self.task_timing_events.get(str(task_id), []))

    def _record_timing_event(
        self,
        *,
        task_id: str,
        direction: str,
        msg_type: str,
        index: int | None,
    ) -> None:
        """
        记录 decoder 主进程可观测的单个时序事件。

        参数:
            task_id: 单次合成任务编号。
            direction: 事件方向，`request` 表示主进程投递给 worker，`response` 表示 worker 返回主进程。
            msg_type: 事件类型，例如 `DECODE_PATCH`、`AUDIO`、`FINISH`。
            index: 音频 chunk 序号；请求或 final 事件没有序号时传 None。

        返回:
            本函数只更新内存诊断表，不返回业务值。

        边界异常:
            诊断记录不能影响合成主路径，因此这里只做内存追加，不做复杂序列化或磁盘 IO。
        """

        # 事件表和订阅表共用一把锁，保证 listener 线程和生成线程并发写入时顺序可见。
        with self.subscriber_lock:
            normalized_task_id = str(task_id)
            if normalized_task_id not in self.task_timing_events:
                self.task_timing_order.append(normalized_task_id)
                while len(self.task_timing_order) > 64:
                    expired_task_id = self.task_timing_order.pop(0)
                    self.task_timing_events.pop(expired_task_id, None)
            self.task_timing_events.setdefault(normalized_task_id, []).append(
                {
                    "time": time.perf_counter(),
                    "direction": str(direction),
                    "msg_type": str(msg_type),
                    "index": None if index is None else int(index),
                }
            )

    def shutdown(self) -> None:
        """关闭 worker、监听线程和 multiprocessing 队列，保证 SDK 退出时不残留子进程。"""

        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.stop_listener = True
        try:
            if self.decoder_proc and self.decoder_proc.is_alive():
                self.request_q.put(None)
        except Exception:
            pass
        # 先等子进程自行退出；只有超时后才升级为 terminate，避免把正常收尾也变成强杀。
        if self.decoder_proc and self.decoder_proc.is_alive():
            self.decoder_proc.join(timeout=0.3)
            if self.decoder_proc.is_alive():
                self.decoder_proc.terminate()
                self.decoder_proc.join(timeout=0.3)
        # 监听线程必须在队列关闭前退出，否则可能仍阻塞在 response_q.get() 上。
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)
        # 清理订阅表和诊断缓存，避免长生命周期进程在反复 init/shutdown 后保留旧任务引用。
        with self.subscriber_lock:
            self.stream_subscribers.clear()
            self.results.clear()
            self.task_timing_events.clear()
            self.task_timing_order.clear()
        for item in (self.request_q, self.response_q):
            try:
                item.close()
                item.join_thread()
            except Exception:
                pass
        self.decoder_proc = None
        self.listener_thread = None
        self.ready = False


class _MainModuleSpawnGuard:
    """
    VoxCPM2-GGUF decoder spawn 启动期间的主模块保护器。

    参数:
        本上下文管理器不接收显式参数，直接临时处理 `sys.modules["__main__"]`。

    返回:
        返回自身作为上下文管理器对象。

    边界异常:
        本类不吞掉子进程启动异常，只负责避免无 `__file__` 的交互入口被 spawn 重放。
    """

    def __enter__(self):
        # spawn worker 时不能重放用户入口脚本，否则 SDK 使用方必须自己加 main guard；
        # 这里在启动窗口临时隐藏 __main__.__file__，让 multiprocessing 跳过脚本重导入。
        self._main_module = sys.modules.get("__main__")
        self._restore_attrs = None
        if self._main_module is None:
            return self
        main_file = getattr(self._main_module, "__file__", None)
        main_spec = getattr(self._main_module, "__spec__", None)
        if main_file is None and main_spec is not None:
            return self
        self._restore_attrs = {"__file__": main_file, "__spec__": main_spec}
        self._main_module.__file__ = None
        self._main_module.__spec__ = types.SimpleNamespace(name=None)
        return self

    def __exit__(self, exc_type, exc, tb):
        # 启动窗口结束后恢复用户主模块，避免影响后续正常 introspection。
        if self._main_module is None or self._restore_attrs is None:
            return False
        self._main_module.__file__ = self._restore_attrs["__file__"]
        self._main_module.__spec__ = self._restore_attrs["__spec__"]
        return False
