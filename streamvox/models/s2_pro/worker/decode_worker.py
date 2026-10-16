from __future__ import annotations

import os
import time
import traceback

import numpy as np

from ....runtime_compat import ensure_openmp_runtime_compat
from ....utils.log import logger
from ..codec import TrueStreamingCodecDecoderOrtRuntime
from ..schema.decoder_protocol import DecodeRequest, DecoderResponse


def _empty_codes(codec: TrueStreamingCodecDecoderOrtRuntime) -> np.ndarray:
    return np.zeros((1, codec.config.num_codebooks, 0), dtype=np.int64)


def _normalize_codes(
    codes: np.ndarray | None,
    codec: TrueStreamingCodecDecoderOrtRuntime,
) -> np.ndarray:
    array = np.asarray(codes if codes is not None else _empty_codes(codec), dtype=np.int64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != codec.config.num_codebooks:
        raise ValueError(
            f"expected codes shape (1, {codec.config.num_codebooks}, T), got {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array)


def _drop_frames(
    audio: np.ndarray,
    frame_count: int,
    codec: TrueStreamingCodecDecoderOrtRuntime,
) -> np.ndarray:
    if frame_count <= 0:
        return audio
    # decoder 每个 code frame 对应 frame_length 个输出采样；prompt tail 只用于续上 state，不应直接返回给用户。
    drop_samples = min(int(frame_count) * int(codec.config.frame_length), int(audio.shape[-1]))
    return audio[..., drop_samples:]


def _handle_decode_task(
    req: DecodeRequest,
    codec: TrueStreamingCodecDecoderOrtRuntime,
    sessions: dict[str | int, dict[str, object]],
    response_queue,
) -> None:
    task_id = req.task_id
    try:
        # task 第一次出现时优先恢复 prompt 预热 state；没有缓存时才从零状态开始。
        if task_id not in sessions:
            if req.decoder_state is not None:
                codec.restore_decode_state(req.decoder_state)
            else:
                codec.reset_decode()
            tail_codes = _normalize_codes(req.decoder_tail_codes, codec)
            sessions[task_id] = {
                "index": 0,
                "pending_codes": tail_codes.copy(),
                "discard_frames": int(tail_codes.shape[-1]),
            }

        session = sessions[task_id]
        codes = _normalize_codes(req.codes, codec)
        pending_codes = np.asarray(session["pending_codes"], dtype=np.int64)
        if codes.shape[-1]:
            pending_codes = np.concatenate([pending_codes, codes], axis=-1)

        t0 = time.time()
        audio_chunks: list[np.ndarray] = []
        while pending_codes.shape[-1] >= codec.stream_code_frames:
            chunk = pending_codes[..., : codec.stream_code_frames]
            pending_codes = pending_codes[..., codec.stream_code_frames :]
            audio = codec.decode_codes_chunk(chunk, valid_frames=codec.stream_code_frames)
            discard_frames = int(session["discard_frames"])
            if discard_frames:
                drop = min(discard_frames, codec.stream_code_frames)
                audio = _drop_frames(audio, drop, codec)
                session["discard_frames"] = discard_frames - drop
            audio_chunks.append(audio)

        if req.is_final and pending_codes.shape[-1]:
            valid_frames = int(pending_codes.shape[-1])
            audio = codec.decode_codes_chunk(pending_codes, valid_frames=valid_frames)
            discard_frames = int(session["discard_frames"])
            if discard_frames:
                drop = min(discard_frames, valid_frames)
                audio = _drop_frames(audio, drop, codec)
                session["discard_frames"] = discard_frames - drop
            audio_chunks.append(audio)
            pending_codes = _empty_codes(codec)

        dt = time.time() - t0
        session["pending_codes"] = pending_codes
        if audio_chunks:
            audio = np.concatenate(audio_chunks, axis=-1)
            index = int(session["index"])
            session["index"] = index + 1
            response_queue.put(
                DecoderResponse(
                    msg_type="AUDIO",
                    task_id=task_id,
                    index=index,
                    audio=audio.reshape(-1).astype(np.float32, copy=False),
                    compute_time=dt,
                )
            )
        if req.is_final:
            response_queue.put(
                DecoderResponse(
                    msg_type="FINISH",
                    task_id=task_id,
                    index=int(session["index"]),
                )
            )
            sessions.pop(task_id, None)
    except Exception as exc:
        logger.error("S2-Pro decoder worker failed: %s", exc)
        traceback.print_exc()
        sessions.pop(task_id, None)
        response_queue.put(DecoderResponse(msg_type="FINISH", task_id=task_id))


def _handle_prewarm_task(
    req: DecodeRequest,
    codec: TrueStreamingCodecDecoderOrtRuntime,
    response_queue,
) -> None:
    # make_prompt 阶段只用 prompt speech token 推进 decoder state，不把参考音频本身返回给用户。
    task_id = req.task_id
    try:
        t0 = time.time()
        codes = _normalize_codes(req.codes, codec)
        full_frames = (int(codes.shape[-1]) // codec.stream_code_frames) * codec.stream_code_frames
        full_codes = codes[..., :full_frames]
        tail_codes = codes[..., full_frames:]
        codec.reset_decode()
        if full_codes.shape[-1]:
            codec.decode_codes(full_codes)
        response_queue.put(
            DecoderResponse(
                msg_type="DECODER_STATE",
                task_id=task_id,
                decoder_state=codec.snapshot_decode_state(),
                decoder_tail_codes=tail_codes[0].copy(),
                compute_time=time.time() - t0,
            )
        )
    except Exception as exc:
        logger.error("S2-Pro decoder prewarm failed: %s", exc)
        traceback.print_exc()
    finally:
        response_queue.put(DecoderResponse(msg_type="FINISH", task_id=task_id))


def decoder_worker_proc(
    codes_queue,
    pcm_queue,
    codec_dir,
    providers,
    decode_name,
    meta_name,
) -> None:
    """
    中文说明：
        S2-Pro true-streaming decoder 的子进程 worker。

    核心职责：
        1. 独立持有 ONNX decoder runtime，避免主进程堆积解码状态；
        2. 支持正式解码与 prompt 预热两类协议请求；
        3. 把 READY / AUDIO / DECODER_STATE / FINISH 等消息回传给主进程。

    核心入参意义：
        codes_queue: 主进程投递解码请求的输入队列。
        pcm_queue: 子进程回传解码结果的输出队列。
        codec_dir: codec 资源目录。
        providers: ORT provider 列表。
        decode_name: decoder ONNX 文件名。
        meta_name: codec 元数据文件名。

    预期输出：
        本函数本身无返回值；通过 `pcm_queue` 返回协议消息。

    可能的异常抛出：
        函数内部会尽量捕获主要异常并记录日志，避免 worker 无声退出。
    """
    # 业务原因：
    # S2-Pro worker 同样会在本函数内首次加载 ORT/native 依赖，必须在构造 runtime 之前
    # 先把 OpenMP 冲突兼容策略注入到子进程环境。
    ensure_openmp_runtime_compat(source="s2_pro.decoder_worker")

    # decoder 独立进程默认限制 OMP 线程，避免和 GGUF 主循环抢 CPU。
    os.environ["OMP_NUM_THREADS"] = os.getenv("STREAMVOX_S2_PRO_DECODER_OMP_THREADS", "4")
    codec = TrueStreamingCodecDecoderOrtRuntime(
        codec_dir,
        providers=providers,
        decode_name=decode_name,
        meta_name=meta_name,
    )
    pcm_queue.put(DecoderResponse(msg_type="READY", task_id="decoder"))
    sessions: dict[str | int, dict[str, object]] = {}
    try:
        while True:
            req = codes_queue.get()
            if req is None:
                pcm_queue.put(None)
                break
            if isinstance(req, DecodeRequest):
                if req.msg_type == "PREWARM_DECODER_STATE":
                    _handle_prewarm_task(req, codec, pcm_queue)
                else:
                    _handle_decode_task(req, codec, sessions, pcm_queue)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("S2-Pro decoder worker crashed: %s", exc)
        traceback.print_exc()
