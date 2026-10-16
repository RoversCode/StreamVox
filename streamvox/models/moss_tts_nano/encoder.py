from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import sentencepiece as spm

from .common import MossOnnxBundle, resample_and_match_channels


class MossAudioEncoder:
    """
    MOSS 参考音频编码器与文本 tokenizer 封装。

    参数:
        bundle: 已加载的 MOSS bundle 元信息。
        codec_encode_session: codec encode ONNX session。

    返回:
        初始化后持有 SentencePiece tokenizer 和 codec encode 推理入口。

    边界异常:
        tokenizer.model 缺失或 codec session 非法时会在初始化或推理阶段抛出异常。
    """

    def __init__(self, *, bundle: MossOnnxBundle, codec_encode_session) -> None:
        self.bundle = bundle
        self.codec_meta = bundle.codec_meta
        self.codec_encode_session = codec_encode_session
        self.sp_model = spm.SentencePieceProcessor(model_file=str(bundle.tokenizer_path))

    @property
    def sample_rate(self) -> int:
        """
        返回 codec 采样率。

        参数:
            本属性无参数。

        返回:
            返回整数 Hz 采样率。

        边界异常:
            codec meta 缺失字段时会自然抛出 KeyError。
        """

        return int(self.codec_meta["codec_config"]["sample_rate"])

    def encode_text(self, text: str) -> list[int]:
        """
        使用 SentencePiece 将文本编码为 token id。

        参数:
            text: 待编码文本。

        返回:
            返回整数 token id 列表。

        边界异常:
            tokenizer 内部错误会自然抛出异常。
        """

        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    def encode_reference_audio_array(self, audio: np.ndarray, sample_rate: int) -> list[list[int]]:
        """
        将参考音频数组编码为 MOSS audio codes。

        参数:
            audio: 输入参考音频。
            sample_rate: 输入音频采样率。

        返回:
            返回 `frame/n_vq` 的 int token 列表。

        边界异常:
            音频为空、采样率非法或 codec encode ONNX 执行失败时抛出异常。
        """

        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        waveform = resample_and_match_channels(audio, int(sample_rate), target_sample_rate, target_channels)
        waveform_length = int(waveform.shape[-1])
        outputs = self.codec_encode_session.run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.codec_encode_session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(named_outputs["audio_code_lengths"], dtype=np.int32)
        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])

        prompt_audio_codes: list[list[int]] = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(num_quantizers)]
            )
        return prompt_audio_codes

    def encode_reference_audio_path(self, audio_path: str | Path) -> list[list[int]]:
        """
        从音频文件读取并编码参考音频。

        参数:
            audio_path: 本地音频文件路径。

        返回:
            返回 `frame/n_vq` 的 int token 列表。

        边界异常:
            文件缺失、音频读取失败或 codec encode 失败时抛出异常。
        """

        resolved_path = Path(audio_path).expanduser().resolve()
        audio, sample_rate = librosa.load(str(resolved_path), sr=None, mono=False)
        return self.encode_reference_audio_array(np.asarray(audio, dtype=np.float32), int(sample_rate))


__all__ = ["MossAudioEncoder"]
