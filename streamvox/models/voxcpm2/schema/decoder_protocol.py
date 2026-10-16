from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DecodeRequest:
    """
    AudioVAE decoder worker 的输入协议。

    参数:
        task_id: 单次合成请求的唯一标识，worker 用它维护独立 stateful decoder 状态。
        msg_type: 请求类型；当前支持 DECODE_PATCH。
        patch: 单个 latent patch；final 请求允许为空，用于刷新并释放任务状态。
        is_final: 当前请求是否为该 task 的最后一包。

    返回:
        本类只作为跨进程序列化载体，不主动返回业务值。

    边界异常:
        字段合法性由 worker 处理，避免主进程和子进程重复维护校验逻辑。
    """

    task_id: str
    msg_type: str
    patch: np.ndarray | None = None
    is_final: bool = False


@dataclass(slots=True)
class DecoderResponse:
    """
    AudioVAE decoder worker 的输出协议。

    参数:
        msg_type: 响应类型；READY 表示 worker 可用，AUDIO 表示音频块，ERROR 表示异常，FINISH 表示任务结束。
        task_id: 响应所属任务标识。
        index: 当前任务内音频块序号。
        audio: 解码得到的 float32 PCM 音频块。
        compute_time: worker 内部本次解码耗时。
        error_message: worker 捕获到的异常消息。

    返回:
        本类只作为跨线程和跨进程消息，不主动返回业务值。

    边界异常:
        本类不抛异常；调用方根据 msg_type 决定是否把 error_message 转成 RuntimeError。
    """

    msg_type: str
    task_id: str
    index: int = 0
    audio: np.ndarray | None = None
    compute_time: float = 0.0
    error_message: str | None = None
