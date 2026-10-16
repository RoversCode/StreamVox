"""M6 输入组织与 ORT 音频编码工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np

from .ort_runtime import run_single_input


@dataclass(frozen=True)
class HybridPromptCache:
    """描述混合运行时可复用的 prompt cache。"""

    mode: str
    prompt_text: str | None
    audio_feat: np.ndarray | None
    prompt_locenc_hidden: np.ndarray | None
    ref_audio_feat: np.ndarray | None
    reference_locenc_hidden: np.ndarray | None


@dataclass(frozen=True)
class HybridPreparedInputs:
    """描述一次完整推理的标准输入张量。"""

    mode: str
    text_token: np.ndarray
    text_mask: np.ndarray
    audio_feat: np.ndarray
    feat_mask: np.ndarray
    locenc_hidden: np.ndarray | None
    text_length: int
    context_len: int


class HybridInputBuilder:
    """负责构造混合运行时的四模式标准输入。

    设计边界：
    - 这里只处理输入组织与 ORT AudioVAE encoder，不承担 LM / DiT / VAE decoder 推理。
    - 所有输出都固定为 patch-major 语义，避免后续编排层再猜布局。
    """

    def __init__(
        self,
        *,
        audio_encoder_session,
        locenc_session,
        tokenizer,
        audio_start_token: int,
        ref_audio_start_token: int,
        ref_audio_end_token: int,
        patch_size: int,
        latent_dim: int,
        chunk_size: int,
        sample_rate: int,
    ) -> None:
        """初始化输入构造器。"""

        self.audio_encoder_session = audio_encoder_session
        self.locenc_session = locenc_session
        self.tokenizer = tokenizer
        self.audio_start_token = int(audio_start_token)
        self.ref_audio_start_token = int(ref_audio_start_token)
        self.ref_audio_end_token = int(ref_audio_end_token)
        self.patch_size = int(patch_size)
        self.latent_dim = int(latent_dim)
        self.chunk_size = int(chunk_size)
        self.sample_rate = int(sample_rate)
        # 全零 patch 的 locenc 输出在 text padding 和 reference 边界 token 中会重复使用，
        # 这里做惰性缓存，避免每次 build_standard_inputs 都重新跑一遍 ORT。
        self._zero_patch_locenc_hidden: np.ndarray | None = None

    def build_prompt_cache(
        self,
        *,
        prompt_text: str | None = None,
        prompt_audio_feat: np.ndarray | None = None,
        prompt_locenc_hidden: np.ndarray | None = None,
        reference_audio_feat: np.ndarray | None = None,
        reference_locenc_hidden: np.ndarray | None = None,
    ) -> HybridPromptCache | None:
        """构造与官方语义对齐的 prompt cache。

        业务意图：
        - `zero_shot` 没有任何缓存，因此返回 `None`。
        - prompt 音频走 left padding，reference 音频走 right padding，保持官方对齐策略。
        """

        has_prompt_audio = prompt_audio_feat is not None or prompt_locenc_hidden is not None
        has_reference_audio = reference_audio_feat is not None or reference_locenc_hidden is not None
        if has_prompt_audio != (prompt_text is not None):
            raise ValueError("prompt audio and prompt_text must both be provided or both be None")

        if not has_prompt_audio and not has_reference_audio:
            return None

        audio_feat = None
        resolved_prompt_locenc_hidden = None
        ref_audio_feat = None
        resolved_reference_locenc_hidden = None
        if has_prompt_audio:
            # make_prompt 已经预编码过 prompt 资产，stream 阶段只消费缓存特征。
            if prompt_audio_feat is not None:
                audio_feat = np.asarray(prompt_audio_feat, dtype=np.float32)
            else:
                raise ValueError("prompt_audio_feat is required when prompt_locenc_hidden is provided")
            resolved_prompt_locenc_hidden = (
                np.asarray(prompt_locenc_hidden, dtype=np.float32)
                if prompt_locenc_hidden is not None
                else self.encode_audio_feat(audio_feat)
            )
        if has_reference_audio:
            # reference 分支同样只消费预编码缓存，避免 stream 再重复读取文件和跑 ORT。
            if reference_audio_feat is not None:
                ref_audio_feat = np.asarray(reference_audio_feat, dtype=np.float32)
            else:
                raise ValueError("reference_audio_feat is required when reference_locenc_hidden is provided")
            resolved_reference_locenc_hidden = (
                np.asarray(reference_locenc_hidden, dtype=np.float32)
                if reference_locenc_hidden is not None
                else self.encode_audio_feat(ref_audio_feat)
            )

        if audio_feat is not None and ref_audio_feat is not None:
            mode = "ref_continuation"
        elif ref_audio_feat is not None:
            mode = "reference"
        else:
            mode = "continuation"

        return HybridPromptCache(
            mode=mode,
            prompt_text=prompt_text,
            audio_feat=audio_feat,
            prompt_locenc_hidden=resolved_prompt_locenc_hidden,
            ref_audio_feat=ref_audio_feat,
            reference_locenc_hidden=resolved_reference_locenc_hidden,
        )

    def build_standard_inputs(
        self,
        *,
        target_text: str,
        prompt_cache: HybridPromptCache | None,
        control_text: str | None = None,
        streaming_prefix_len: int = 4,
    ) -> HybridPreparedInputs:
        """复刻官方 `_generate_with_prompt_cache` 的四模式输入组织逻辑。"""

        # control_text 是 VoxCPM2-GGUF 的显式控制文本，业务约定为直接放进模型 text：
        # 非空时格式固定为 `(control_text)+text`。这里不能 strip，因为首尾空格本身可能是控制语义的一部分。
        normalized_control_text = "" if control_text is None else str(control_text)
        effective_target_text = f"({normalized_control_text}){target_text}" if normalized_control_text != "" else target_text

        if prompt_cache is None:
            mode = "zero_shot"
            full_text = effective_target_text
        else:
            mode = prompt_cache.mode
            if mode in ("continuation", "ref_continuation"):
                full_text = (prompt_cache.prompt_text or "") + effective_target_text
            else:
                full_text = effective_target_text

        text_token = np.asarray(self.tokenizer(full_text), dtype=np.int64)
        text_token = np.concatenate([text_token, np.asarray([self.audio_start_token], dtype=np.int64)], axis=0)
        text_length = int(text_token.shape[0])

        if mode in ("zero_shot", "continuation"):
            prompt_audio_feat = (
                prompt_cache.audio_feat
                if prompt_cache is not None and prompt_cache.audio_feat is not None
                else np.empty((0, self.patch_size, self.latent_dim), dtype=np.float32)
            )
            audio_length = int(prompt_audio_feat.shape[0])
            text_pad_token = np.zeros((audio_length,), dtype=np.int64)
            text_pad_feat = np.zeros((text_length, self.patch_size, self.latent_dim), dtype=np.float32)
            text_token = np.concatenate([text_token, text_pad_token], axis=0)
            audio_feat = np.concatenate([text_pad_feat, prompt_audio_feat], axis=0)
            text_mask = np.concatenate(
                [np.ones((text_length,), dtype=np.int32), np.zeros((audio_length,), dtype=np.int32)],
                axis=0,
            )
            feat_mask = np.concatenate(
                [np.zeros((text_length,), dtype=np.int32), np.ones((audio_length,), dtype=np.int32)],
                axis=0,
            )
        elif mode == "reference":
            assert prompt_cache is not None and prompt_cache.ref_audio_feat is not None
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(prompt_cache.ref_audio_feat)
            text_pad_feat = np.zeros((text_length, self.patch_size, self.latent_dim), dtype=np.float32)
            text_token = np.concatenate([ref_tokens, text_token], axis=0)
            audio_feat = np.concatenate([ref_feats, text_pad_feat], axis=0)
            text_mask = np.concatenate([ref_t_mask, np.ones((text_length,), dtype=np.int32)], axis=0)
            feat_mask = np.concatenate([ref_a_mask, np.zeros((text_length,), dtype=np.int32)], axis=0)
        else:
            assert prompt_cache is not None
            assert prompt_cache.ref_audio_feat is not None and prompt_cache.audio_feat is not None
            prompt_audio_length = int(prompt_cache.audio_feat.shape[0])
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(prompt_cache.ref_audio_feat)
            prompt_pad_token = np.zeros((prompt_audio_length,), dtype=np.int64)
            text_pad_feat = np.zeros((text_length, self.patch_size, self.latent_dim), dtype=np.float32)
            text_token = np.concatenate([ref_tokens, text_token, prompt_pad_token], axis=0)
            audio_feat = np.concatenate([ref_feats, text_pad_feat, prompt_cache.audio_feat], axis=0)
            text_mask = np.concatenate(
                [
                    ref_t_mask,
                    np.ones((text_length,), dtype=np.int32),
                    np.zeros((prompt_audio_length,), dtype=np.int32),
                ],
                axis=0,
            )
            feat_mask = np.concatenate(
                [
                    ref_a_mask,
                    np.zeros((text_length,), dtype=np.int32),
                    np.ones((prompt_audio_length,), dtype=np.int32),
                ],
                axis=0,
            )

        context_len = 0
        if feat_mask.size > 0 and int(feat_mask[-1]) == 1:
            audio_indices = np.nonzero(feat_mask)[0]
            context_len = int(min(max(0, streaming_prefix_len - 1), len(audio_indices)))

        # prompt/reference 音频的 locenc 已在 make_prompt 阶段提前算好；这里按最终输入布局
        # 把缓存和 text padding 对应的零 patch locenc 拼起来，供后续 prefill 直接读取。
        locenc_hidden = self._build_locenc_hidden(
            mode=mode,
            text_length=text_length,
            prompt_cache=prompt_cache,
        )

        return HybridPreparedInputs(
            mode=mode,
            text_token=text_token,
            text_mask=text_mask,
            audio_feat=audio_feat.astype(np.float32, copy=False),
            feat_mask=feat_mask,
            locenc_hidden=locenc_hidden,
            text_length=text_length,
            context_len=context_len,
        )

    def encode_wav(self, wav_path: str, *, padding_mode: str) -> np.ndarray:
        """用 ORT `AudioVAE encoder` 把 wav 编成 patch-major `audio_feat`。"""

        audio, _ = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        return self.encode_audio(audio, padding_mode=padding_mode)

    def encode_audio(self, audio: np.ndarray, *, padding_mode: str) -> np.ndarray:
        """
        用 ORT `AudioVAE encoder` 把内存音频编成 patch-major `audio_feat`。

        参数:
            audio: 已对齐到 runtime 采样率的单声道波形。
            padding_mode: patch 对齐时的补零方向，`left/right` 分别对应 prompt/reference。

        返回:
            返回 `[patch_count, patch_size, latent_dim]` 形状的 `audio_feat`。

        边界异常:
            空数组会在这里被拒绝，避免后续 ORT 输入出现长度为零的非法波形。
        """

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("audio must not be empty")
        patch_len = int(self.patch_size * self.chunk_size)
        padding_size = int(math.ceil(max(1, len(audio)) / patch_len) * patch_len - len(audio))
        if padding_size > 0:
            pad_width = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = np.pad(audio, pad_width=pad_width, mode="constant")

        waveform = audio.astype(np.float32, copy=False)[None, None, :]
        audio_feat = run_single_input(self.audio_encoder_session, "waveform", waveform)
        return np.asarray(audio_feat.squeeze(0), dtype=np.float32)

    def encode_audio_feat(self, audio_feat: np.ndarray) -> np.ndarray:
        """
        用 ORT `locenc` 把 patch-major `audio_feat` 编成后续 LM prefill 可复用的隐藏表示。

        参数:
            audio_feat: `AudioVAE encoder` 产出的 `[patch_count, patch_size, latent_dim]`。

        返回:
            返回 `[patch_count, hidden_dim]` 形状的 `locenc_hidden`。

        边界异常:
            空 patch 序列会被拒绝，避免把无意义的 prompt cache 写入 prompt 资产。
        """

        audio_feat = np.asarray(audio_feat, dtype=np.float32)
        if audio_feat.ndim != 3 or audio_feat.shape[0] == 0:
            raise ValueError("audio_feat must have shape [patch_count, patch_size, latent_dim] with patch_count > 0")
        locenc_hidden = run_single_input(self.locenc_session, "patches", audio_feat[None, ...]).squeeze(0)
        return np.asarray(locenc_hidden, dtype=np.float32)

    def prepare_prompt_source_features(
        self,
        wav_path: str | None,
        audio: np.ndarray | None,
        *,
        padding_mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        统一为 prompt/reference 音频生成可缓存的 `audio_feat + locenc_hidden`。

        参数:
            wav_path: 磁盘音频路径。
            audio: 已对齐到 runtime 采样率的内存音频。
            padding_mode: prompt/reference 对应的补零方向。

        返回:
            返回 `(audio_feat, locenc_hidden)`，供 make_prompt 直接写入 prompt 资产。

        边界异常:
            两种来源都缺失时会由 `_encode_prompt_source` 抛错。
        """

        audio_feat = self._encode_prompt_source(wav_path, audio, padding_mode=padding_mode)
        return audio_feat, self.encode_audio_feat(audio_feat)

    def _encode_prompt_source(
        self,
        wav_path: str | None,
        audio: np.ndarray | None,
        *,
        padding_mode: str,
    ) -> np.ndarray:
        """
        按统一入口把路径音频或内存音频编码成 `audio_feat`。

        参数:
            wav_path: 磁盘音频路径。
            audio: 已经对齐采样率的内存音频。
            padding_mode: patch 对齐补零方向。

        返回:
            返回编码后的 `audio_feat`。

        边界异常:
            两种来源都缺失时抛出异常，避免上游把空 prompt 静默送入模型。
        """

        if wav_path is not None:
            return self.encode_wav(wav_path, padding_mode=padding_mode)
        if audio is not None:
            return self.encode_audio(audio, padding_mode=padding_mode)
        raise ValueError("wav_path or audio is required to build prompt cache")

    def _build_locenc_hidden(
        self,
        *,
        mode: str,
        text_length: int,
        prompt_cache: HybridPromptCache | None,
    ) -> np.ndarray:
        """
        按最终输入布局拼接整段 `locenc_hidden`。

        参数:
            mode: 当前标准输入模式。
            text_length: 文本 token 段长度。
            prompt_cache: 预编码好的 prompt/reference cache。

        返回:
            返回与 `audio_feat` 首维严格对齐的 `locenc_hidden`。

        边界异常:
            continuation/reference 需要的预编码缓存缺失时抛错，避免静默退回慢路径。
        """

        zero_patch_locenc = self._get_zero_patch_locenc_hidden()
        text_locenc_hidden = np.repeat(zero_patch_locenc[None, :], int(text_length), axis=0)

        if mode == "zero_shot":
            return np.asarray(text_locenc_hidden, dtype=np.float32)
        if mode == "continuation":
            assert prompt_cache is not None and prompt_cache.prompt_locenc_hidden is not None
            return np.concatenate([text_locenc_hidden, prompt_cache.prompt_locenc_hidden], axis=0).astype(
                np.float32,
                copy=False,
            )
        if mode == "reference":
            assert prompt_cache is not None and prompt_cache.reference_locenc_hidden is not None
            reference_prefix = np.concatenate(
                [
                    zero_patch_locenc[None, :],
                    prompt_cache.reference_locenc_hidden,
                    zero_patch_locenc[None, :],
                ],
                axis=0,
            )
            return np.concatenate([reference_prefix, text_locenc_hidden], axis=0).astype(np.float32, copy=False)

        assert prompt_cache is not None
        assert prompt_cache.reference_locenc_hidden is not None and prompt_cache.prompt_locenc_hidden is not None
        reference_prefix = np.concatenate(
            [
                zero_patch_locenc[None, :],
                prompt_cache.reference_locenc_hidden,
                zero_patch_locenc[None, :],
            ],
            axis=0,
        )
        return np.concatenate([reference_prefix, text_locenc_hidden, prompt_cache.prompt_locenc_hidden], axis=0).astype(
            np.float32,
            copy=False,
        )

    def _get_zero_patch_locenc_hidden(self) -> np.ndarray:
        """
        返回全零 patch 对应的单步 `locenc_hidden`。

        参数:
            本函数不接收参数。

        返回:
            返回 `[hidden_dim]` 形状的一维向量，可重复用于 text padding 与 reference 边界。

        边界异常:
            这里使用惰性初始化，避免 text-only 场景在 runtime 刚创建时额外做一次 ORT 前向。
        """

        if self._zero_patch_locenc_hidden is None:
            zero_patch = np.zeros((1, self.patch_size, self.latent_dim), dtype=np.float32)
            self._zero_patch_locenc_hidden = self.encode_audio_feat(zero_patch)[0]
        return np.asarray(self._zero_patch_locenc_hidden, dtype=np.float32)

    def _make_ref_prefix(self, ref_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构造 `[ref_start, ref_audio, ref_end]` 的隔离前缀。"""

        ref_len = int(ref_feat.shape[0])
        zero_patch = np.zeros((1, self.patch_size, self.latent_dim), dtype=np.float32)
        tokens = np.concatenate(
            [
                np.asarray([self.ref_audio_start_token], dtype=np.int64),
                np.zeros((ref_len,), dtype=np.int64),
                np.asarray([self.ref_audio_end_token], dtype=np.int64),
            ],
            axis=0,
        )
        feats = np.concatenate([zero_patch, ref_feat.astype(np.float32, copy=False), zero_patch], axis=0)
        text_mask = np.concatenate(
            [
                np.asarray([1], dtype=np.int32),
                np.zeros((ref_len,), dtype=np.int32),
                np.asarray([1], dtype=np.int32),
            ],
            axis=0,
        )
        feat_mask = np.concatenate(
            [
                np.asarray([0], dtype=np.int32),
                np.ones((ref_len,), dtype=np.int32),
                np.asarray([0], dtype=np.int32),
            ],
            axis=0,
        )
        return tokens, feats, text_mask, feat_mask
