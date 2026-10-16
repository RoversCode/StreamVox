#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   decoder.py
@Time    :   2026/03/06 15:37:20
@Author  :   RoversCode
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import multiprocessing as mp
import numpy as np
import traceback
import queue
import time
import atexit
import threading
import types
import sys
from pathlib import Path
from typing import Union, Optional
from .schema.item import DecodeResult
from .schema.decoder_protocol import DecoderState, DecoderResponse, DecodeRequest
from ...utils.log import logger


class DecoderAPI:
    """
    拉起Decoder Worker
    接收每一个chunk的code，放进队列让worker去解码
    """

    def __init__(
        self,
        onnx_path: Path,
        device_backend: str = "cpu",
        device_id: int | None = None,
        chunk_size: int = 12,
        start_method: str = "spawn",
    ):
        self.onnx_path = onnx_path
        self.device_backend = device_backend
        self.device_id = 0 if device_id is None else device_id
        self.chunk_size = chunk_size
        self.start_method = start_method

        # 任务控制
        self.task_counter = 0
        self.active_task_id = 0

        # 状态存储
        self.results = {}  # task_id -> list of np.ndarray
        self.events = {}  # task_id -> threading.Event
        # stream_subscribers 按 task_id 维护主进程内唯一的流式订阅队列；当前实现明确约束
        # “同一个 task 只允许一个实时消费者”，避免多订阅广播带来额外心智负担。
        self.stream_subscribers = {}  # task_id -> queue.Queue
        # subscriber_lock 保护 results 历史和订阅表，避免 listener 线程与业务线程并发修改时出现竞态。
        self.subscriber_lock = threading.Lock()
        self.ready_states = {"decoder": False}  # Decoder初始化完毕的标注位

        self.stop_listener = False

        # CUDA 解码器必须用全新的 Python 进程启动。
        # Linux 默认的 fork 会继承父进程里已经初始化过的 GPU/CUDA 状态，
        # 这会导致子进程中的 onnxruntime CUDA EP 出现 device_id=-1 等初始化异常。
        self.mp_ctx = mp.get_context(self.start_method)

        # 通讯队列
        self.codes_q = self.mp_ctx.Queue()  # 解码队列 -> Decoder (DecodeRequest)
        self.result_q = self.mp_ctx.Queue()  # Decoder -> Proxy (DecoderResponse)

        # 进程对象
        self.decoder_proc = None

        self.start()
        
        # 注册自动退出逻辑
        atexit.register(self.shutdown)


    def _handle_decoder_msg(self, msg: DecoderResponse):
        """处理来自解码器的消息"""
        if msg.msg_type == "READY":
            logger.info(f"✅ [qwen3] [Decoder API] ready")
            self.ready_states["decoder"] = True
            return

        task_id = msg.task_id

        # A1. 消息累积 (储蓄罐)
        if msg.msg_type == "AUDIO":
            msg.recv_time = time.time()  # 记录 Proxy 收到音频的具体时刻
            with self.subscriber_lock:
                if task_id not in self.results:
                    self.results[task_id] = []
                # 存储完整消息对象，以保留 compute_time
                self.results[task_id].append(msg)
                subscriber = self.stream_subscribers.get(task_id)
            if subscriber is not None:
                subscriber.put(msg)

        # A2. 收到结束信号 (闹钟)
        elif msg.msg_type == "FINISH":
            with self.subscriber_lock:
                if task_id not in self.results:
                    self.results[task_id] = []
                self.results[task_id].append(msg)
                subscriber = self.stream_subscribers.get(task_id)
            if subscriber is not None:
                subscriber.put(msg)

            if task_id in self.events:
                self.events[task_id].set()

    def subscribe_stream(self, task_id: Union[str, int]) -> queue.Queue:
        """
        为指定 task 注册一个主进程内的流式响应订阅队列。

        参数:
            task_id: 当前 decoder 任务标识。

        返回:
            返回一个线程安全队列；listener 会把该 task 的 AUDIO/FINISH 消息按到达顺序写入队列。

        异常:
            当同一个 task_id 已经存在订阅队列时抛出 RuntimeError，防止多个消费者意外共享同一条流。
        """
        subscriber: queue.Queue = queue.Queue()
        with self.subscriber_lock:
            if task_id in self.stream_subscribers:
                raise RuntimeError(f"qwen3 decoder stream subscriber already exists for task_id={task_id}")
            self.stream_subscribers[task_id] = subscriber
        return subscriber

    def unsubscribe_stream(self, task_id: Union[str, int], subscriber: queue.Queue) -> None:
        """
        取消指定 task 的流式响应订阅。

        参数:
            task_id: decoder 任务标识。
            subscriber: 之前通过 subscribe_stream 获取的队列对象。

        返回:
            本函数无返回值；队列会从订阅表中移除。

        异常:
            本函数不主动抛出异常；不存在的订阅关系会被静默忽略。
        """
        with self.subscriber_lock:
            current_subscriber = self.stream_subscribers.get(task_id)
            if current_subscriber is None:
                return
            # 只有持有当前 task 实际订阅队列的调用方才允许注销，避免错误移除其他请求的通道。
            if current_subscriber is not subscriber:
                return
            self.stream_subscribers.pop(task_id, None)

    def _listen_loop(self):
        """
        这个 _listen_loop 就是那个“搬运工”：
        它不断从 self.result_q 里取 Worker 发来的消息（Line 92）。
        如果是解码消息，调用 _handle_decoder_msg（Line 96）。
        在 _handle_decoder_msg 里（Line 117-146）：
        存数据：把音频数据 append 到 self.results[task_id]（Line 123）。
        发通知：当收到 FINISH 信号时，调用 self.events[task_id].set()（Line 146）。
        正是这一步 set()，唤醒了前面正在 wait() 的主线程。
        """
        """从结果队列中抓取数据并分类转发 (后台储蓄罐)"""
        while not self.stop_listener:
            try:
                msg = self.result_q.get(timeout=0.1)
                if msg is None:
                    logger.info(f"_listen_loop 收到 None，退出")
                    break

                if isinstance(msg, DecoderResponse):
                    self._handle_decoder_msg(msg)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"⚠️ [Proxy] 监听异常: {e}")
                traceback.print_exc()
                break

    def start(self):
        """启动工作进程"""
        from .worker.decode_worker import decoder_worker_proc

        # 1. 解调子进程 (Decoder)
        self.decoder_proc = self.mp_ctx.Process(
            target=decoder_worker_proc,
            args=(
                self.codes_q,
                self.result_q,
                self.onnx_path,
                self.chunk_size,
                self.device_backend,
                self.device_id,
            ),
            daemon=True,
        )
        with self._suppress_main_reimport_during_spawn():
            self.decoder_proc.start()

        # 2. 启动本地监听器
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _suppress_main_reimport_during_spawn(self):
        """
        在 spawn 启动窗口里临时屏蔽 multiprocessing 对用户 __main__ 的二次回放。

        背景：
        1. decoder worker 必须继续使用独立 Python 进程，不能破坏现有异步流水线；
        2. 但标准 spawn 会把用户脚本作为 __main__ 再执行一次，导致库把“主模块必须加 guard”的约束泄露给用户；
        3. 这里仅在拉起 decoder worker 的瞬间，把 __main__.__file__/__spec__ 暂时替换成一个空模块，
           让 multiprocessing 不再把用户入口脚本塞进 preparation_data。

        这样子进程仍然按原流程异步启动，只是不再重跑用户脚本顶层代码。
        """
        return _MainModuleSpawnGuard()

    def decode(
        self,
        codes: Union[np.ndarray],
        task_id="default",
        is_final: bool = False,
        stream: bool = True,
        state: Optional[DecoderState] = None,
    ) -> np.ndarray:
        """
        发送任务，接收结果
        """

        # 初始化状态位与储蓄罐
        if task_id not in self.results:
            self.results[task_id] = []
        if task_id not in self.events:
            self.events[task_id] = threading.Event()
        else:
            self.events[task_id].clear()

        # 构造并发送请求
        # 对于 stream=False (离线模式)，Worker 内部会根据 msg_type="DECODE" 自动判定 is_final
        msg_type = "DECODE_CHUNK" if stream else "DECODE"
        req = DecodeRequest(
            task_id=task_id,
            msg_type=msg_type,
            codes=codes,
            is_final=is_final,
            state=state,
        )
        self.codes_q.put(req)

        # 逻辑分支 A: 中间的流式包，无需等待
        if stream and not is_final:
            return np.array([], dtype=np.float32)

        # 逻辑分支 B: 离线模式或流式终包，开始同步等待
        self.events[task_id].wait()  # 阻塞，等待消费完毕
        # 从消息列表中提取音频碎片和统计信息
        responses = self.results.get(task_id, [])

        # 结果打包
        result = DecodeResult(responses=responses)

        # 清理该任务的残留资源
        if task_id in self.results:
            del self.results[task_id]
        if task_id in self.events:
            del self.events[task_id]

        return result

    def wait_until_ready(self, timeout=10):
        """阻塞直到所有工作进程就绪"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(self.ready_states.values()):
                return True
            time.sleep(0.1)
        return False


    def shutdown(self):
        """关闭所有子进程"""
        self.stop_listener = True

        # 1. 向子进程发送毒丸
        try:
            if self.decoder_proc and self.decoder_proc.is_alive():
                self.codes_q.put(None)
        except:
            pass

        # 2. 依次清理子进程
        for p in [self.decoder_proc]:
            if p and p.is_alive():
                p.join(timeout=0.3)
                if p.is_alive():
                    try:
                        p.terminate()
                    except:
                        pass

        # 3. 停止监听线程
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)

        # 4. 清理并销毁队列
        for q in [self.codes_q, self.result_q]:
            try:
                q.cancel_join_thread()
                while not q.empty():
                    q.get_nowait()
                q.close()
            except:
                pass


class _MainModuleSpawnGuard:
    """仅用于 decoder spawn 启动窗口的 __main__ 保护器。"""

    def __enter__(self):
        self._main_module = sys.modules.get("__main__")
        self._restore_attrs = None
        if self._main_module is None:
            return self

        # 只在脚本入口场景下做兼容处理；正常模块入口保持 multiprocessing 默认行为。
        main_file = getattr(self._main_module, "__file__", None)
        main_spec = getattr(self._main_module, "__spec__", None)
        if main_file is None and main_spec is not None:
            return self

        self._restore_attrs = {
            "__file__": main_file,
            "__spec__": main_spec,
        }
        self._main_module.__file__ = None
        self._main_module.__spec__ = types.SimpleNamespace(name=None)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._main_module is None or self._restore_attrs is None:
            return False
        self._main_module.__file__ = self._restore_attrs["__file__"]
        self._main_module.__spec__ = self._restore_attrs["__spec__"]
        return False
