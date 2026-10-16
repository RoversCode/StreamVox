"""`AudioVAE` ORT 流式解码工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ort_runtime import create_ort_session, run_single_input


@dataclass
class AudioVAEChunkDecoderState:
    """描述单 patch chunk decoder 的轻量状态。"""

    produced_patches: int = 0


@dataclass(frozen=True)
class AudioVAEStateTensorSpec:
    """描述显式 state decoder 的单个状态张量规格。"""

    name: str
    module_name: str
    module_type: str
    channels: int
    state_size: int
    stride: int
    trim: int

    @classmethod
    def from_dict(cls, payload: dict) -> "AudioVAEStateTensorSpec":
        """从 metadata 字典恢复状态规格。"""

        return cls(
            name=str(payload["name"]),
            module_name=str(payload["module_name"]),
            module_type=str(payload["module_type"]),
            channels=int(payload["channels"]),
            state_size=int(payload["state_size"]),
            stride=int(payload["stride"]),
            trim=int(payload["trim"]),
        )


@dataclass
class AudioVAEStatefulDecoderState:
    """描述显式 state ORT decoder 的运行时状态。"""

    state_tensors: list[np.ndarray]
    produced_patches: int = 0


class AudioVAEOrtChunkDecoder:
    """单 patch 的 ORT chunk decoder。

    业务意图：
    - 这里主要用于单步数值对拍或最小单元验证。
    - 真正的产线流式链路应优先使用下面的 `AudioVAEOrtStatefulDecoder`。
    """

    def __init__(
        self,
        *,
        decoder_chunk_onnx_path: str,
        patch_size: int,
        decode_chunk_size: int,
        latent_dim: int,
        prefer_gpu: bool = True,
        device_id: int | None = None,
    ) -> None:
        self.patch_size = int(patch_size)
        self.decode_chunk_size = int(decode_chunk_size)
        self.chunk_samples = int(self.patch_size * self.decode_chunk_size)
        self.latent_dim = int(latent_dim)
        self.session, self.bundle = create_ort_session(
            decoder_chunk_onnx_path,
            prefer_gpu=prefer_gpu,
            device_id=device_id,
        )

    def init_state(self) -> AudioVAEChunkDecoderState:
        """初始化一份可复用的 chunk decoder 状态。"""

        return AudioVAEChunkDecoderState()

    def reset_state(self, state: AudioVAEChunkDecoderState) -> None:
        """重置一份已有状态。"""

        state.produced_patches = 0

    def decode_patch(self, state: AudioVAEChunkDecoderState, patch: np.ndarray) -> np.ndarray:
        """输入一个 patch，并立即解码出当前 patch 对应的音频块。"""

        patch = np.asarray(patch, dtype=np.float32)
        if patch.shape != (self.patch_size, self.latent_dim):
            raise ValueError(
                f"expected patch shape {(self.patch_size, self.latent_dim)}, got {patch.shape}"
            )

        waveform_chunk = run_single_input(self.session, "latent_patch", patch[None, ...])
        state.produced_patches += 1
        return np.asarray(waveform_chunk, dtype=np.float32)


class AudioVAEOrtStatefulDecoder:
    """显式 state 的 ORT streaming decoder。

    业务意图：
    - 新导出的 `audio_vae_decoder_stateful.onnx` 已经把官方 `streaming_decode()`
      内部维护的卷积缓存显式提升为输入输出张量。
    - 编排层只需要严格按导出 metadata 给出的顺序维护这些 state，即可复现官方
      stateful 流式语义，同时避免再依赖 PyTorch monkeypatch。
    """

    def __init__(
        self,
        *,
        decoder_stateful_onnx_path: str,
        state_specs: list[AudioVAEStateTensorSpec],
        patch_size: int,
        decode_chunk_size: int,
        latent_dim: int,
        prefer_gpu: bool = True,
        device_id: int | None = None,
    ) -> None:
        self.patch_size = int(patch_size)
        self.decode_chunk_size = int(decode_chunk_size)
        self.chunk_samples = int(self.patch_size * self.decode_chunk_size)
        self.latent_dim = int(latent_dim)
        self.state_specs = list(state_specs)
        self.session, self.bundle = create_ort_session(
            decoder_stateful_onnx_path,
            prefer_gpu=prefer_gpu,
            device_id=device_id,
        )
        # 导出时约定输出顺序固定为 `waveform_chunk + next_states...`，这里提前缓存名字，
        # 避免运行时每一步都重复查询 ORT 元信息。
        self._output_names = [item.name for item in self.session.get_outputs()]

    @classmethod
    def from_metadata(
        cls,
        *,
        decoder_stateful_onnx_path: str,
        metadata_path: str,
        patch_size: int,
        decode_chunk_size: int,
        latent_dim: int,
        prefer_gpu: bool = True,
        device_id: int | None = None,
    ) -> "AudioVAEOrtStatefulDecoder":
        """根据导出 metadata 创建显式 state decoder。"""

        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        state_specs = [
            AudioVAEStateTensorSpec.from_dict(item)
            for item in metadata["decoder_stateful"]["state_specs"]
        ]
        return cls(
            decoder_stateful_onnx_path=decoder_stateful_onnx_path,
            state_specs=state_specs,
            patch_size=patch_size,
            decode_chunk_size=decode_chunk_size,
            latent_dim=latent_dim,
            prefer_gpu=prefer_gpu,
            device_id=device_id,
        )

    def init_state(self, batch_size: int = 1) -> AudioVAEStatefulDecoderState:
        """初始化一份可复用的显式 state。"""

        state_tensors = [
            np.zeros((int(batch_size), spec.channels, spec.state_size), dtype=np.float32)
            for spec in self.state_specs
        ]
        return AudioVAEStatefulDecoderState(state_tensors=state_tensors, produced_patches=0)

    def reset_state(self, state: AudioVAEStatefulDecoderState) -> None:
        """把已有 state 清零，准备开始一段新的流式解码。"""

        for index, spec in enumerate(self.state_specs):
            state.state_tensors[index] = np.zeros(
                (state.state_tensors[index].shape[0], spec.channels, spec.state_size),
                dtype=np.float32,
            )
        state.produced_patches = 0

    def decode_patch(self, state: AudioVAEStatefulDecoderState, patch: np.ndarray) -> np.ndarray:
        """输入一个 patch，并用显式 state 推进一次解码。"""

        patch = np.asarray(patch, dtype=np.float32)
        if patch.shape != (self.patch_size, self.latent_dim):
            raise ValueError(
                f"expected patch shape {(self.patch_size, self.latent_dim)}, got {patch.shape}"
            )
        if len(state.state_tensors) != len(self.state_specs):
            raise ValueError(
                f"expected {len(self.state_specs)} state tensors, got {len(state.state_tensors)}"
            )

        # 按导出时的稳定顺序组装 feed dict，确保 ONNX 图里的 state_in/state_out 一一对应。
        ort_inputs = {"latent_patch": patch[None, ...]}
        for spec, state_value in zip(self.state_specs, state.state_tensors):
            ort_inputs[spec.name] = np.asarray(state_value, dtype=np.float32)

        ort_outputs = self.session.run(self._output_names, ort_inputs)
        if len(ort_outputs) != len(self.state_specs) + 1:
            raise RuntimeError(
                f"expected {len(self.state_specs) + 1} outputs from stateful decoder, got {len(ort_outputs)}"
            )

        waveform_chunk = np.asarray(ort_outputs[0], dtype=np.float32)
        # 逐个替换 next_state，保持下一个 patch 调用继续沿着当前流式状态推进。
        for index, next_state in enumerate(ort_outputs[1:]):
            state.state_tensors[index] = np.asarray(next_state, dtype=np.float32)
        state.produced_patches += 1
        return waveform_chunk


def _compare_array(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """计算数值误差指标。"""

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    abs_diff = np.abs(actual - expected)
    return {
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
    }
