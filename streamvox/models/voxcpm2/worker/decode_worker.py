from __future__ import annotations

import os
import time
import traceback

import numpy as np

from ....runtime_compat import ensure_openmp_runtime_compat
from ....utils.log import logger
from ..audio_vae_streaming_ort import AudioVAEOrtStatefulDecoder
from ..schema.decoder_protocol import DecodeRequest, DecoderResponse


def decoder_worker_proc(
    request_queue,
    response_queue,
    decoder_stateful_onnx_path,
    metadata_path,
    patch_size,
    decode_chunk_size,
    latent_dim,
    prefer_gpu,
    device_id,
) -> None:
    """
    VoxCPM2-GGUF AudioVAE stateful decoder 子进程入口。

    参数:
        request_queue: 主进程投递 latent patch 的输入队列。
        response_queue: 子进程回传 READY、AUDIO、ERROR、FINISH 消息的输出队列。
        decoder_stateful_onnx_path: AudioVAE stateful decoder ONNX 文件路径。
        metadata_path: stateful decoder 导出元数据路径。
        patch_size: 单个 latent patch 的时间维长度。
        decode_chunk_size: 单个 patch 对应的输出采样窗口。
        latent_dim: latent 特征维度。
        prefer_gpu: 是否优先尝试 GPU ORT provider。
        device_id: ORT decoder 绑定的 CUDA 设备索引；为空时由 provider 默认选择。

    返回:
        本函数在子进程中常驻运行，不通过 return 返回业务值。

    边界异常:
        单个 task 的异常会转成 ERROR 响应并清理该 task；进程级异常会记录日志后退出。
    """

    # 子进程会首次加载 ORT/native 依赖，必须先注入 OpenMP 兼容策略，避免运行时库冲突。
    ensure_openmp_runtime_compat(source="voxcpm2_gguf.decoder_worker")
    # decoder worker 独立进程限制 OMP 线程，避免和 GGUF 生成线程抢占 CPU。
    os.environ["OMP_NUM_THREADS"] = os.getenv("STREAMVOX_VOXCPM2_DECODER_OMP_THREADS", "4")

    decoder = AudioVAEOrtStatefulDecoder.from_metadata(
        decoder_stateful_onnx_path=str(decoder_stateful_onnx_path),
        metadata_path=str(metadata_path),
        patch_size=int(patch_size),
        decode_chunk_size=int(decode_chunk_size),
        latent_dim=int(latent_dim),
        prefer_gpu=bool(prefer_gpu),
        device_id=None if device_id is None else int(device_id),
    )
    response_queue.put(DecoderResponse(msg_type="READY", task_id="decoder"))

    # sessions 按 task_id 保存 AudioVAE 显式 state，保证并发或连续请求之间互不污染。
    sessions: dict[str, dict[str, object]] = {}
    try:
        while True:
            request = request_queue.get()
            if request is None:
                response_queue.put(None)
                break
            if isinstance(request, DecodeRequest):
                _handle_decode_request(request, decoder, sessions, response_queue)
    except KeyboardInterrupt:
        return
    except Exception as exc:
        logger.error("VoxCPM2-GGUF decoder worker crashed: %s", exc)
        traceback.print_exc()


def _handle_decode_request(
    request: DecodeRequest,
    decoder: AudioVAEOrtStatefulDecoder,
    sessions: dict[str, dict[str, object]],
    response_queue,
) -> None:
    """处理单个 latent patch 解码请求，并把结果写回响应队列。"""

    task_id = str(request.task_id)
    try:
        # task 首次出现时创建独立 state；后续 patch 复用同一份 state 以保持流式卷积缓存连续。
        if task_id not in sessions:
            sessions[task_id] = {"state": decoder.init_state(), "index": 0}

        session = sessions[task_id]
        patch = request.patch
        if patch is not None:
            decode_start = time.time()
            waveform = decoder.decode_patch(session["state"], np.asarray(patch, dtype=np.float32))
            compute_time = time.time() - decode_start
            index = int(session["index"])
            session["index"] = index + 1
            response_queue.put(
                DecoderResponse(
                    msg_type="AUDIO",
                    task_id=task_id,
                    index=index,
                    audio=np.asarray(waveform.squeeze(0).squeeze(0), dtype=np.float32),
                    compute_time=compute_time,
                )
            )

        # final 请求负责释放 worker 内 task state，并通知前台生成器当前音频流已经结束。
        if request.is_final:
            response_queue.put(DecoderResponse(msg_type="FINISH", task_id=task_id, index=int(session["index"])))
            sessions.pop(task_id, None)
    except Exception as exc:
        logger.error("VoxCPM2-GGUF decoder task failed: %s", exc)
        traceback.print_exc()
        sessions.pop(task_id, None)
        response_queue.put(DecoderResponse(msg_type="ERROR", task_id=task_id, error_message=str(exc)))
        response_queue.put(DecoderResponse(msg_type="FINISH", task_id=task_id))
