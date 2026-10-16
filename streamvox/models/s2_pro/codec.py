from __future__ import annotations

from ..onnx_backend import verify_session_provider

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from streamvox_runtime import ensure_single_onnxruntime_distribution, initialize_nvidia_runtime
ensure_single_onnxruntime_distribution()
import onnxruntime as ort

from .codec_common import (
    CodecOnnxConfig,
    ceil_div,
    create_session_options,
    resolve_ort_providers,
)
from ...utils.log import logger


@dataclass(slots=True)
class TensorStateSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def zeros_numpy(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=getattr(np, self.dtype))


def load_true_streaming_meta(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_specs_from_meta(items: list[dict[str, object]]) -> list[TensorStateSpec]:
    # decoder history state
    return [
        TensorStateSpec(
            name=str(item["name"]),
            shape=tuple(int(value) for value in item["shape"]),
            dtype=str(item["dtype"]),
        )
        for item in items
    ]


def _to_numpy_audio(
    audio: np.ndarray, dtype: np.dtype = np.float32
) -> tuple[np.ndarray, int]:
    array = np.asarray(audio, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2 and array.shape[0] == 1:
        array = array.reshape(1, 1, -1)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != 1:
        raise ValueError(f"expected audio shape [1, 1, T], got {tuple(array.shape)}")
    return np.ascontiguousarray(array), int(array.shape[-1])


def _pad_audio_for_encode(audio: np.ndarray, frame_length: int) -> np.ndarray:
    # full-sequence encode 要求长度对齐 frame_length；在 runtime 侧补零比把动态 pad 写进 ONNX 更可控。
    sample_count = int(audio.shape[-1])
    padded_length = ceil_div(sample_count, frame_length) * frame_length
    if padded_length == sample_count:
        return audio
    padded = np.zeros(
        (audio.shape[0], audio.shape[1], padded_length), dtype=audio.dtype
    )
    padded[..., :sample_count] = audio
    return padded


def _to_numpy_indices(indices: np.ndarray) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"expected indices shape [1, N, T], got {tuple(array.shape)}")
    return np.ascontiguousarray(array)


def _ort_tensor_dtype_to_numpy(type_name: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
    }
    try:
        return mapping[type_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported ONNX Runtime tensor dtype: {type_name}") from exc


def codec_config_from_meta(meta: dict[str, object]) -> CodecOnnxConfig:
    # codec 的基础形状参数只依赖 meta/yaml，不需要为了读取它们提前创建 ONNX session。
    return CodecOnnxConfig(
        sample_rate=int(meta["sample_rate"]),
        frame_length=int(meta["frame_length"]),
        hop_length=int(meta["hop_length"]),
        num_codebooks=int(meta["num_codebooks"]),
        stream_code_frames=int(meta["stream_code_frames"]),
    )


class _TrueStreamingCodecBase:
    def _init_codec_base(
        self,
        model_dir: Path,
        providers: list[str] | None,
        meta_name: str,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.meta = load_true_streaming_meta(self.model_dir / meta_name)
        self.config = codec_config_from_meta(self.meta)
        self.providers = resolve_ort_providers(providers)
        if "CUDAExecutionProvider" in self.providers:
            initialize_nvidia_runtime(consumer="onnx_cuda", required=True)


class TrueStreamingCodecEncoderOrtRuntime(_TrueStreamingCodecBase):
    def __init__(
        self,
        model_dir: Path,
        providers: list[str] | None = None,
        *,
        encode_name: str = "codec_encode_fp16.onnx",
        meta_name: str = "codec_true_streaming_meta_fp16.json",
    ) -> None:
        self._init_codec_base(model_dir, providers, meta_name)
        session_options = create_session_options()
        self.encode_session = ort.InferenceSession(
            str(self.model_dir / encode_name),
            sess_options=session_options,
            providers=self.providers,
        )
        verify_session_provider(self.encode_session, self.providers)
        self.encode_input_dtype = _ort_tensor_dtype_to_numpy(
            self.encode_session.get_inputs()[0].type
        )
        active_provider = self.encode_session.get_providers()[0]
        device_label = "CPU"
        if "CUDA" in active_provider:
            device_label = "CUDA:0"
        elif "Dml" in active_provider:
            device_label = "DML:0"
        precision_label = "FP16" if self.encode_input_dtype == np.float16 else "FP32"
        logger.info("✅ [s2-pro] [Codec Encoder] [%s] [%s]", device_label, precision_label)

    def encode_audio(
        self,
        audio: np.ndarray,
        audio_lengths: np.ndarray | None = None,
    ) -> np.ndarray:
        audio_np, sample_count = _to_numpy_audio(audio, dtype=self.encode_input_dtype)
        if audio_lengths is None:
            lengths_np = np.asarray([sample_count], dtype=np.int64)
        else:
            lengths_np = np.asarray(audio_lengths, dtype=np.int64)
        audio_np = _pad_audio_for_encode(
            audio_np, self.config.frame_length
        )  # pad 到固定长度，再裁减
        outputs = self.encode_session.run(
            None, {"audio": audio_np, "audio_lengths": lengths_np}
        )
        indices = np.asarray(outputs[0], dtype=np.int64)
        valid_frames = int(outputs[1][0])
        return indices[..., :valid_frames]


class TrueStreamingCodecDecoderOrtRuntime(_TrueStreamingCodecBase):
    def __init__(
        self,
        model_dir: Path,
        providers: list[str] | None = None,
        *,
        decode_name: str = "codec_decode_stream_fp16.onnx",
        meta_name: str = "codec_true_streaming_meta_fp16.json",
    ) -> None:
        self._init_codec_base(model_dir, providers, meta_name)
        session_options = create_session_options()
        self.decode_session = ort.InferenceSession(
            str(self.model_dir / decode_name),
            sess_options=session_options,
            providers=self.providers,
        )
        verify_session_provider(self.decode_session, self.providers)
        # decoder的历史state
        self.decode_state_specs = state_specs_from_meta(self.meta["decode_state_specs"])
        # 每个解码chunk的帧数
        self.stream_code_frames = int(self.meta["stream_code_frames"])
        # decode feed buffer 复用能减少每个 chunk 的小对象分配，降低流式路径抖动。
        self._decode_feeds: dict[str, np.ndarray] = {
            "indices_chunk": np.zeros(
                (1, self.config.num_codebooks, self.stream_code_frames), dtype=np.int64
            ),
            "valid_frames": np.zeros((1,), dtype=np.int64),
        }
        self._decode_output_names = [
            output.name for output in self.decode_session.get_outputs()
        ]  # 提前把模型输出的名字缓存起来。
        self.reset_decode()
        active_provider = self.decode_session.get_providers()[0]
        device_label = "CPU"
        if "CUDA" in active_provider:
            device_label = "CUDA:0"
        elif "Dml" in active_provider:
            device_label = "DML:0"
        precision_label = "Unknown"
        for item in self.decode_session.get_inputs():
            input_type = item.type.lower()
            if "float16" in input_type:
                precision_label = "FP16"
                break
            if "float" in input_type:
                precision_label = "FP32"
        logger.info("✅ [s2-pro] [Codec Decoder] [%s] [%s]", device_label, precision_label)

    def reset_decode(self) -> None:
        # 每条新语音必须重置 true-streaming state；同一条语音的 chunk 之间不能重置。
        self.decode_states = {
            spec.name: spec.zeros_numpy() for spec in self.decode_state_specs
        }
        for spec in self.decode_state_specs:
            self._decode_feeds[spec.name] = self.decode_states[spec.name]

    def snapshot_decode_state(self) -> dict[str, np.ndarray]:
        # prompt state 需要持久化到 npz，必须拷贝出来，避免后续流式解码原地更新污染缓存。
        return {
            spec.name: np.asarray(self.decode_states[spec.name]).copy()
            for spec in self.decode_state_specs
        }

    def restore_decode_state(self, state: dict[str, np.ndarray]) -> None:
        # 只接受 meta 中声明过的完整 state，避免不同模型或不同 codec meta 的缓存被误用。
        missing = [
            spec.name for spec in self.decode_state_specs if spec.name not in state
        ]
        if missing:
            raise ValueError(f"missing decoder state tensors: {missing}")
        for spec in self.decode_state_specs:
            value = np.asarray(state[spec.name], dtype=getattr(np, spec.dtype))
            if tuple(value.shape) != spec.shape:
                raise ValueError(
                    f"decoder state {spec.name} shape mismatch: expected {spec.shape}, got {tuple(value.shape)}"
                )
            self.decode_states[spec.name][...] = value
            self._decode_feeds[spec.name] = self.decode_states[spec.name]

    def decode_codes_chunk(
        self, indices_chunk: np.ndarray, valid_frames: int | None = None
    ) -> np.ndarray:
        indices_np = _to_numpy_indices(indices_chunk)
        if indices_np.shape[0] != 1 or indices_np.shape[1] != self.config.num_codebooks:
            raise ValueError(
                f"Expected indices_chunk shape (1, {self.config.num_codebooks}, T), got {indices_np.shape}"
            )
        chunk_frames = int(indices_np.shape[-1])
        if chunk_frames > self.stream_code_frames:
            raise ValueError(
                f"indices_chunk length {chunk_frames} exceeds stream_code_frames={self.stream_code_frames}"
            )

        effective_frames = chunk_frames if valid_frames is None else int(valid_frames)
        if effective_frames < 0 or effective_frames > chunk_frames:
            raise ValueError(
                f"valid_frames must be in [0, {chunk_frames}], got {effective_frames}"
            )

        self._decode_feeds["indices_chunk"].fill(0)
        self._decode_feeds["indices_chunk"][..., :chunk_frames] = indices_np
        self._decode_feeds["valid_frames"][0] = effective_frames
        # 解码
        outputs = self.decode_session.run(self._decode_output_names, self._decode_feeds)
        audio = outputs[0].astype(np.float32, copy=False)
        valid_samples = int(outputs[1][0]) # 解码出来语音长度

        # 存储state
        for spec, output in zip(self.decode_state_specs, outputs[2:]):
            self.decode_states[spec.name][...] = output
    
        return audio[..., :valid_samples]

    def decode_codes(self, indices: np.ndarray) -> np.ndarray:
        indices_np = _to_numpy_indices(indices)
        chunks: list[np.ndarray] = []
        for start in range(0, indices_np.shape[-1], self.stream_code_frames):
            end = min(start + self.stream_code_frames, indices_np.shape[-1])
            chunks.append(
                self.decode_codes_chunk(indices_np[..., start:end], end - start)
            )
        if not chunks:
            return np.zeros((1, 1, 0), dtype=np.float32)
        return np.concatenate(chunks, axis=-1)
