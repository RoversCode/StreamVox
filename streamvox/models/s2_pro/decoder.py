from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

from ...utils.log import logger
from .schema.decoder_protocol import DecodeRequest, DecodeResult, DecoderResponse


class S2ProDecoderAPI:
    """
    S2-Pro true-streaming decoder 的主进程侧代理。

    参数:
        codec_dir: codec 模型和元数据所在目录。
        providers: ONNX Runtime provider 列表，用于控制 decoder worker 的执行后端。
        decode_name: decoder ONNX 文件名。
        meta_name: codec 元数据文件名。
        start_method: multiprocessing 启动方式；默认使用 spawn 以兼容 CUDA/ONNX 资源隔离。

    返回:
        初始化后得到一个可提交 acoustic code 解码请求的 decoder API 对象。

    异常:
        当多进程启动、队列创建或 worker 初始化失败时，会透出 multiprocessing / worker 侧异常。
    """

    def __init__(
        self,
        codec_dir: Path,
        providers: list[str] | None,
        decode_name: str,
        meta_name: str,
        start_method: str = "spawn",
    ) -> None:
        self.codec_dir = Path(codec_dir)
        self.providers = providers
        # decoder ONNX 文件名
        self.decode_name = decode_name
        # meta_name 是 codec 元数据文件名
        self.meta_name = meta_name
        # start_method 控制 multiprocessing 上下文；spawn 能避免子进程继承主进程里已初始化的运行时资源。
        self.start_method = start_method
        # results 按 task_id 缓存 worker 返回的消息；流式请求会逐步往同一个 task 的列表里追加 AUDIO/FINISH 等响应。
        self.results: dict[str | int, list[DecoderResponse]] = {}

        # events 按 task_id 保存等待信号；非流式请求、final 流式请求和 prewarm 会阻塞到 FINISH 或状态消息返回。
        self.events: dict[str | int, threading.Event] = {}

        # stream_subscribers 按 task_id 维护主进程内唯一的流式消息订阅队列；当前实现明确约束
        # “同一个 task 只允许一个实时消费者”，避免多订阅广播带来额外心智负担。
        self.stream_subscribers: dict[str | int, queue.Queue] = {}
        # subscriber_lock 保护 results 历史和订阅表的并发访问；listener 线程和业务线程会同时触达它们。
        self.subscriber_lock = threading.Lock()

        # ready_states 记录 worker 初始化状态；runtime 启动时会等待 decoder 发出 READY 后再继续。
        self.ready_states = {"decoder": False}
        # stop_listener 是监听线程的退出标志；shutdown 时设置为 True，避免线程继续阻塞读取队列。
        self.stop_listener = False

        # 使用指定 start_method 创建独立 multiprocessing 上下文，确保队列和进程属于同一种启动语义。
        self.mp_ctx = mp.get_context(self.start_method)
        # codes_q 是主进程发送 DecodeRequest 给 worker 的输入队列。
        self.codes_q = self.mp_ctx.Queue()
        # result_q 是 worker 返回 DecoderResponse 给主进程监听线程的输出队列。
        self.result_q = self.mp_ctx.Queue()
        # decoder_proc 保存实际 decoder worker 进程对象，后续用于健康检查、join 和 terminate。
        self.decoder_proc = None
        # listener_thread 在主进程后台消费 result_q，避免 decode 调用方直接阻塞轮询 worker 响应。
        self.listener_thread: threading.Thread | None = None
        # 初始化完成后立即启动 worker 和监听线程，使 runtime 后续可以直接调用 decode。
        self.start()
        # 注册进程退出清理钩子，防止 SDK 使用方忘记 shutdown 时残留 decoder 子进程和队列资源。
        atexit.register(self.shutdown)

    def start(self) -> None:
        """
        启动 decoder worker 子进程和主进程监听线程。

        参数:
            本函数不接收额外参数，使用 __init__ 中保存的 codec 路径、provider 和队列配置。

        返回:
            本函数无返回值；成功后 decoder_proc 和 listener_thread 会被设置为运行态对象。

        异常:
            当 worker 进程启动或监听线程创建失败时透出对应异常。
        """
        # 延迟导入 worker 入口，避免主模块导入 decoder.py 时立即触发 worker 依赖加载。
        from .worker.decode_worker import decoder_worker_proc

        # decoder worker 独立持有 ONNX decoder session，主进程只通过队列发送 acoustic code 和控制消息。
        self.decoder_proc = self.mp_ctx.Process(
            target=decoder_worker_proc,
            args=(
                self.codes_q,
                self.result_q,
                self.codec_dir,
                self.providers,
                self.decode_name,
                self.meta_name,
            ),
            daemon=True,
        )
        # spawn 模式会尝试重新导入 __main__；这里临时保护主模块，降低 SDK 在无 main guard 环境里的使用门槛。
        with self._suppress_main_reimport_during_spawn():
            self.decoder_proc.start()
        # 监听线程持续消费 worker 返回队列，把响应归档到 results，并在 FINISH 时唤醒等待中的 decode 调用。
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    def _suppress_main_reimport_during_spawn(self):
        """
        构造 spawn 启动期间的主模块保护上下文。

        参数:
            本函数不接收参数。

        返回:
            返回 _MainModuleSpawnGuard 上下文管理器。

        异常:
            本函数自身不主动抛出异常。
        """
        # spawn worker 时不能重放用户入口脚本，否则 SDK 使用方必须自己加 main guard。
        return _MainModuleSpawnGuard()

    def _handle_decoder_msg(self, msg: DecoderResponse) -> None:
        """
        处理 decoder worker 返回的一条响应消息。

        参数:
            msg: worker 通过 result_q 返回的 DecoderResponse。

        返回:
            本函数无返回值；会更新 ready_states、results 或 events。

        异常:
            本函数不主动抛出异常；调用方监听循环会捕获意外异常并记录日志。
        """
        # READY 是 worker 初始化完成的生命周期消息，只更新就绪状态，不归入某个 task 的结果列表。
        if msg.msg_type == "READY":  
            logger.info("✅ [s2-pro] [Decoder API] ready")
            self.ready_states["decoder"] = True
            return

        # 记录主进程接收时间，方便后续性能统计或排查 worker 到主进程之间的队列延迟。
        msg.recv_time = time.time()
        # 先写入结果历史，再读取当前 task 的实时订阅队列；这样主进程内的新旧消费路径都能看到同一份顺序。
        with self.subscriber_lock:
            self.results.setdefault(msg.task_id, []).append(msg)
            subscriber = self.stream_subscribers.get(msg.task_id)

        # 订阅队列用于实时流式消费；一旦 worker 解出音频，listener 线程立即把消息推给对应请求。
        if subscriber is not None:
            subscriber.put(msg)

        # FINISH 表示该 task 的 worker 侧解码已经结束；如果有同步等待方，就通过 Event 唤醒。
        if msg.msg_type == "FINISH" and msg.task_id in self.events:
            self.events[msg.task_id].set()

    def subscribe_stream(self, task_id: str | int) -> queue.Queue:
        """
        为指定 task 注册一个主进程内的流式响应订阅队列。

        参数:
            task_id: 当前 decoder 任务标识；同一个 synthesize 生命周期内应保持唯一。

        返回:
            返回一个线程安全队列；listener 线程会把该 task 的 AUDIO/FINISH 等消息按到达顺序写入队列。

        异常:
            当同一个 task_id 已经存在订阅队列时抛出 RuntimeError，防止多个消费者意外共享同一条流。
        """
        # 当前实现明确约束一个 task 只允许一个实时消费者；重复注册通常意味着上层生命周期管理存在问题。
        subscriber: queue.Queue = queue.Queue()
        with self.subscriber_lock:
            if task_id in self.stream_subscribers:
                raise RuntimeError(f"s2-pro decoder stream subscriber already exists for task_id={task_id}")
            self.stream_subscribers[task_id] = subscriber
        return subscriber

    def unsubscribe_stream(self, task_id: str | int, subscriber: queue.Queue) -> None:
        """
        取消指定 task 的流式响应订阅。

        参数:
            task_id: decoder 任务标识。
            subscriber: 之前通过 subscribe_stream 获取的队列对象。

        返回:
            本函数无返回值；队列会从订阅表中移除。

        异常:
            本函数不主动抛出异常；传入不存在的订阅队列时会静默忽略。
        """
        with self.subscriber_lock:
            current_subscriber = self.stream_subscribers.get(task_id)
            if current_subscriber is None:
                return
            # 只有持有当前 task 实际订阅队列的调用方才允许注销，避免错误移除其他请求的通道。
            if current_subscriber is not subscriber:
                return
            self.stream_subscribers.pop(task_id, None)

    def _listen_loop(self) -> None:
        """
        后台监听 worker 返回队列，并把消息转交给 _handle_decoder_msg。

        参数:
            本函数不接收参数，直接消费 self.result_q。

        返回:
            本函数在线程中运行，直到收到停止标志、None 哨兵或异常后退出。

        异常:
            队列超时会被忽略；其它异常会被记录日志并终止监听循环。
        """
        # 监听线程按短超时轮询 result_q，使 shutdown 设置 stop_listener 后最多等待一个超时周期即可退出。
        while not self.stop_listener:
            try:
                # 使用 timeout 而不是永久阻塞，确保关闭流程可以通过 stop_listener 及时打断监听线程。
                msg = self.result_q.get(timeout=0.1)
                # None 是关闭哨兵，表示 worker 或主进程希望监听线程立即退出。
                if msg is None:
                    break
                # 只处理协议定义的 DecoderResponse，避免队列里混入其它对象时破坏结果缓存结构。
                if isinstance(msg, DecoderResponse):
                    self._handle_decoder_msg(msg)
            except queue.Empty:
                # 正常轮询超时不代表错误，继续下一轮检查 stop_listener。
                continue
            except Exception as exc:
                # 监听线程一旦遇到非预期异常，继续运行可能导致结果错乱，因此记录日志后退出。
                logger.error("S2-Pro decoder listener failed: %s", exc)
                break

    def decode(
        self,
        codes: np.ndarray,
        task_id: str | int = "default",
        is_final: bool = False,
        stream: bool = True,
        decoder_state: dict[str, np.ndarray] | None = None,
        decoder_tail_codes: np.ndarray | None = None,
    ) -> DecodeResult:
        """
        向 decoder worker 提交一次 acoustic code 解码请求。

        参数:
            codes: 待解码的 acoustic code 矩阵，形状通常为 (num_codebooks, T)。
            task_id: 当前解码任务标识；同一个流式任务必须复用同一个 task_id 以保持 decoder 状态连续。
            is_final: 是否为当前 task 的最后一次请求；为 True 时会等待 worker 返回 FINISH。
            stream: 是否使用流式消息类型；True 时发送 DECODE_CHUNK，False 时发送 DECODE。
            decoder_state: prompt 预热得到的 decoder 初始状态；只应在当前 task 第一包解码时传入。
            decoder_tail_codes: prompt 预热后不足一个流式窗口的尾部 code；只应和 decoder_state 一起首次传入。

        返回:
            流式且非 final 请求立即返回空 DecodeResult；非流式或 final 请求会等待并返回该 task 的响应列表。

        异常:
            当队列写入、等待或 worker 侧处理失败时可能抛出异常；如果 worker 不返回 FINISH，调用会阻塞等待。
        """
        # 为当前 task 准备结果列表；流式多次调用会持续复用同一个 task_id 的缓存。
        self.results.setdefault(task_id, [])
        # 每次提交请求前清空等待事件，确保本轮等待只会被本轮或后续 FINISH 唤醒。
        self.events.setdefault(task_id, threading.Event()).clear()
        # 将主进程输入统一封装为 DecodeRequest，通过队列交给 worker 进程执行实际 ONNX decoder 推理。
        self.codes_q.put(
            DecodeRequest(
                task_id=task_id,
                # 流式模式使用 DECODE_CHUNK，让 worker 保留任务状态；非流式模式使用一次性 DECODE。
                msg_type="DECODE_CHUNK" if stream else "DECODE",
                # acoustic code 必须是 int64，避免跨进程序列化后 worker 侧出现 dtype 不一致。
                codes=np.asarray(codes, dtype=np.int64),
                is_final=is_final,
                decoder_state=decoder_state,
                # tail codes 和主 codes 一样统一成 int64；缺省时保持 None，表示没有 prompt 尾帧需要衔接。
                decoder_tail_codes=None if decoder_tail_codes is None else np.asarray(decoder_tail_codes, dtype=np.int64),
            )
        )
        # 普通流式 chunk 只负责投递，不阻塞等待音频；监听线程会异步把可用 AUDIO 放进 results。
        if stream and not is_final:
            return DecodeResult([])

        # 非流式请求或 final chunk 必须等待 FINISH，保证返回时该 task 的响应已经完整。
        self.events[task_id].wait()
        # 取出当前 task 累积的所有响应，交给调用方筛选 AUDIO 或其它消息。
        responses = self.results.get(task_id, [])
        # 同步完成的 task 清理结果缓存，避免长期合成后 results 持续增长。
        self.results.pop(task_id, None)
        # 同步完成的 task 清理事件对象，避免 events 持续持有历史 task_id。
        self.events.pop(task_id, None)
        return DecodeResult(responses=responses)

    def prewarm_decoder_state(self, codes: np.ndarray, task_id: str | int = "prompt") -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        """
        使用参考音频 code 预热 true-streaming decoder，并取回可持久化的 decoder state。

        参数:
            codes: prompt 参考音频对应的 acoustic code 矩阵。
            task_id: 预热任务标识；通常使用 prompt 角色相关的唯一值。

        返回:
            返回二元组：decoder state 字典，以及未凑满流式窗口的 tail codes。

        异常:
            当 worker 没有返回 DECODER_STATE 消息时抛出 RuntimeError；队列或 worker 异常会透出。
        """
        # 为预热 task 准备独立结果缓存，避免与真实合成 task 的 AUDIO/FINISH 响应混在一起。
        self.results.setdefault(task_id, [])
        # 清空等待事件，确保只等待本次 PREWARM_DECODER_STATE 请求的完成信号。
        self.events.setdefault(task_id, threading.Event()).clear()
        # 解码请求放入队列
        self.codes_q.put(
            DecodeRequest(
                task_id=task_id,
                # PREWARM_DECODER_STATE 告诉 worker 只推进 decoder 内部状态，不返回参考音频本身给用户。
                msg_type="PREWARM_DECODER_STATE",
                # prompt codes 同样统一成 int64，保证 worker 侧 decoder 输入类型稳定。
                codes=np.asarray(codes, dtype=np.int64),
                # 预热请求是一次性任务，is_final=True 让 worker 完成后发出结束信号。
                is_final=True,
            )
        )
        # 等待 worker 完成预热；该事件通常由 DECODER_STATE 或 FINISH 相关处理唤醒。
        self.events[task_id].wait()
        # 读取本次预热 task 的全部响应，用于查找 DECODER_STATE 消息。
        responses = self.results.get(task_id, [])
        # 预热完成后清理结果缓存，避免 prompt 构建任务占用长期内存。
        self.results.pop(task_id, None)
        # 预热完成后清理事件对象，避免 events 中保留一次性 prompt task。
        self.events.pop(task_id, None)
        # worker 可能返回多条消息；只接受包含 decoder_state 的 DECODER_STATE 作为有效预热结果。
        for msg in responses:
            if msg.msg_type == "DECODER_STATE" and msg.decoder_state is not None:
                # worker 返回的是 state 快照；主进程再拷贝一次，保证后续 prompt asset 持久化不受队列复用影响。
                state = {name: np.asarray(value).copy() for name, value in msg.decoder_state.items()}
                # tail codes 表示参考音频预热后不足一个流式窗口的残留 code，需要正式解码首包时一起送入。
                tail = np.asarray(msg.decoder_tail_codes, dtype=np.int64).copy() if msg.decoder_tail_codes is not None else None
                return state, tail
        # 没有拿到 decoder state 说明 worker 预热协议失败，继续合成会失去 prompt 音色连续性，因此直接报错。
        raise RuntimeError("s2-pro decoder prewarm did not return decoder state")

    def wait_until_ready(self, timeout: float = 60) -> bool:
        """
        等待 decoder worker 完成初始化。

        参数:
            timeout: 最大等待秒数。

        返回:
            worker 在超时前全部 ready 返回 True，否则返回 False。

        异常:
            本函数不主动抛出异常。
        """
        # 记录等待开始时间，用于实现简单的超时轮询。
        t0 = time.time()
        # 在超时时间内反复检查 ready_states；当前只有 decoder 一个状态，保留字典结构便于未来扩展。
        while time.time() - t0 < timeout:
            # 所有组件都 ready 后即可返回，runtime 初始化流程可以继续。
            if all(self.ready_states.values()):
                return True
            # 短暂睡眠避免忙等占用 CPU，同时保持初始化等待响应较快。
            time.sleep(0.1)
        # 超时仍未 ready，交给上层 runtime 决定是否抛出初始化失败。
        return False

    def shutdown(self) -> None:
        """
        关闭 decoder worker、监听线程和跨进程队列。

        参数:
            本函数不接收参数。

        返回:
            本函数无返回值；会尽力释放资源，并吞掉关闭阶段的非关键异常。

        异常:
            本函数尽量不向外抛出异常，避免解释器退出或重复 shutdown 时打断主流程。
        """
        # 先通知监听线程退出，避免关闭队列后线程仍在 get 导致异常刷日志。
        self.stop_listener = True
        try:
            # 如果 worker 仍然存活，向输入队列发送 None 作为停止哨兵，让 worker 走正常退出路径。
            if self.decoder_proc and self.decoder_proc.is_alive():
                self.codes_q.put(None)
        except Exception:
            # 关闭阶段队列可能已经损坏或被解释器清理，吞掉异常保证 shutdown 幂等。
            pass
        # 优先给 worker 一个短暂 join 时间，让它处理停止哨兵并自行退出。
        if self.decoder_proc and self.decoder_proc.is_alive():
            self.decoder_proc.join(timeout=0.3)
            # worker 未能及时退出时强制 terminate，避免 SDK 进程因为残留子进程无法结束。
            if self.decoder_proc.is_alive():
                self.decoder_proc.terminate()
        # 等待监听线程退出，避免主进程结束时后台线程还在访问 result_q。
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)
        # 两个 multiprocessing Queue 都需要 close 和 join_thread，释放底层管道和 feeder thread。
        for item in (self.codes_q, self.result_q):
            try:
                item.close()
                item.join_thread()
            except Exception:
                # 解释器退出阶段队列资源可能已经部分释放，关闭失败不应影响主进程退出。
                pass


class _MainModuleSpawnGuard:
    """仅用于 decoder spawn 启动窗口的 __main__ 保护器。"""

    def __enter__(self):
        '''
        解读：成功“骗”过了 Python 的多进程机制，防止程序在启动子进程时直接崩溃或卡死。
        
        1. 正常情况下的痛点（如果不加这段代码会怎样？）
        在 Windows 或 Mac 上，Python 创建多进程（子进程）的方式叫 spawn，你可以把它理解为**“老板（主进程）克隆了一个自己（子进程）”**。

        Python 的死规定： 克隆人一苏醒，第一件事就是必须要去重新读一遍老板当年读过的“入职手册”（也就是重新导入老板运行的 __main__ 脚本），以确保克隆人和老板拥有相同的知识背景。它怎么找这本手册呢？就是通过查看老板身上的 __file__（文件路径）属性。

        灾难发生： 如果你是在 Jupyter Notebook、某些复杂的打包环境、或者是特殊的交互式终端里运行代码，老板身上其实是没有这本“入职手册”的物理文件的（__file__ 可能不存在或路径不对）。

        崩溃后果： 克隆人一苏醒，死活找不到这本“入职手册”，于是当场报错罢工（抛出 ImportError 或 AttributeError），你的多进程程序直接崩溃。

        2. 加上这段代码后的效果（隐身术）
        这段代码就像是给老板施加了一个**“短暂的隐身术”**：

        老板准备克隆前（__enter__）： 老板偷偷把身上带有“入职手册路径”的铭牌（__file__ 和 __spec__）摘下来藏在口袋里，换上一个写着“无（None）”的假铭牌。

        开始克隆： 老板带着假铭牌克隆出了一个助手（子进程）。

        克隆人苏醒的瞬间： 克隆人按照 Python 的死规定，去查老板的铭牌，试图找入职手册。结果一看，“咦？文件路径是 None？”。

        神奇的效果产生了： 克隆人的底层逻辑会认为：“哦，看来老板不是通过普通脚本启动的，根本就没有什么入职手册需要读。那我就跳过这一步，直接去干老板分派给我的具体任务吧！”

        安全度过（__exit__）： 克隆人顺利去干活了。老板赶紧把口袋里的真铭牌重新挂回身上，一切恢复原样。
        
        '''
        self._main_module = sys.modules.get("__main__")  # 去系统模块表里把 __main__（主模块）揪出来。
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
        '''把之前备份好的 __file__ 和 __spec__ 原封不动地还给 __main__ 模块'''
        if self._main_module is None or self._restore_attrs is None:
            return False
        self._main_module.__file__ = self._restore_attrs["__file__"]  # 
        self._main_module.__spec__ = self._restore_attrs["__spec__"]
        return False
