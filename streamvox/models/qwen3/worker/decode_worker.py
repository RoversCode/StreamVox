#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   decode_worker.py
@Time    :   2026/03/07 16:19:46
@Author  :   RoversCode
@Version :   1.0
@Desc    :   None
"""

# here put the import lib
import os
import time
import traceback

import numpy as np

from ....runtime_compat import ensure_openmp_runtime_compat
from ....utils.log import logger
from ..schema.decoder_protocol import DecodeRequest, DecoderResponse, DecoderSession


def handle_decode_task(req: DecodeRequest, decoder, sessions, response_queue):
    """处理单次解码任务（顶层解耦函数）。"""
    codes_all = np.array(req.codes, dtype=np.int64)
    if codes_all.ndim == 1:
        codes_all = codes_all.reshape(-1, 16)

    # 获取或初始化会话信息。
    session = sessions.get(req.task_id, DecoderSession())
    current_state = session.state
    curr_index = session.index
    if current_state is None and req.state is not None:
        current_state = req.state

    is_task_final = req.is_final or (req.msg_type == "DECODE")

    try:
        t0 = time.time()
        # StatefulDecoder.decode 内部会自动处理分片和历史采样点抵消。
        audio, new_state = decoder.decode(
            codes_all,
            state=current_state,
            is_final=is_task_final,
        )
        dt = time.time() - t0

        # 更新会话状态。
        session.state = new_state
        session.index += 1
        sessions[req.task_id] = session

        # 回传音频。
        pcm_data = audio.copy() if len(audio) > 0 else np.array([], dtype=np.float32)
        response_queue.put(
            DecoderResponse(
                msg_type="AUDIO",
                task_id=req.task_id,
                index=curr_index,
                audio=pcm_data,
                compute_time=dt,
            )
        )

        if is_task_final:
            # FINISH 表示该 task 的最后一包已经处理完成，并把最终 state 一起带回。
            response_queue.put(
                DecoderResponse(
                    msg_type="FINISH",
                    task_id=req.task_id,
                    index=session.index,
                    state=new_state,
                )
            )
            sessions.pop(req.task_id, None)

    except Exception as exc:
        logger.error(f"⚠️ [DecoderWorker] 解码异常: {exc}")
        traceback.print_exc()
        sessions.pop(req.task_id, None)
        response_queue.put(DecoderResponse(msg_type="FINISH", task_id=req.task_id))


def decoder_worker_proc(
    codes_queue,
    pcm_queue,
    decoder_onnx_path,
    chunk_size=12,
    device_backend="cpu",
    device_id=0,
):
    """
    中文说明：
        解码子进程工人（DecoderWorker）。

    核心职责：
        1. 在独立进程中持有 StatefulDecoder，避免主进程堆积 ONNX 解码状态；
        2. 基于 task_id 管理解码会话，支持流式 chunk 连续解码；
        3. 把 READY / AUDIO / FINISH 等协议消息写回主进程。

    核心入参意义：
        codes_queue: 主进程投递 DecodeRequest 的输入队列。
        pcm_queue: worker 向主进程回传 DecoderResponse 的输出队列。
        decoder_onnx_path: 状态化 decoder ONNX 模型路径。
        chunk_size: 单次 decoder 处理的 code chunk 大小。
        device_backend: decoder 目标设备后端，支持 cpu / cuda / dml。
        device_id: 目标设备索引。

    预期输出：
        本函数本身无返回值；通过 `pcm_queue` 返回解码协议消息。

    可能的异常抛出：
        函数内部会捕获主要异常并打印日志，尽量避免 worker 直接静默崩溃。
    """
    # 业务原因：
    # worker 进程会在这里之后才导入 onnxruntime；必须在 import StatefulDecoder 之前完成兼容注入。
    # 否则 ORT 可能先行加载 OpenMP runtime，导致后续 llama/ORT 混用时直接触发 OMP Error #15。
    ensure_openmp_runtime_compat(source="qwen3.decoder_worker")
    from ..stateful_decoder import StatefulDecoder

    # 解码 worker 默认只开少量 OMP 线程，避免把 llama 那边好不容易调下来的 CPU 资源又抢回去。
    # 同时保留环境变量覆盖，方便继续针对具体机器做压测。
    os.environ["OMP_NUM_THREADS"] = os.getenv("STREAMVOX_DECODER_OMP_THREADS", "4")
    logger.debug(f"[Decoder Tune] OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    decoder = StatefulDecoder(
        decoder_onnx_path,
        device_backend=device_backend,
        device_id=device_id,
        chunk_size=chunk_size,
    )
    pcm_queue.put(DecoderResponse(msg_type="READY", task_id="decoder"))
    logger.debug("[DecoderWorker] ready (provider=%s)", decoder.active_provider)

    sessions = {}  # {task_id: DecoderState}

    try:
        while True:
            req: DecodeRequest = codes_queue.get()
            if req is None:
                pcm_queue.put(None)
                break

            if req.msg_type in ["DECODE", "DECODE_CHUNK"]:
                handle_decode_task(req, decoder, sessions, pcm_queue)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error(f"❌ [DecoderWorker] 崩溃: {exc}")
        traceback.print_exc()
