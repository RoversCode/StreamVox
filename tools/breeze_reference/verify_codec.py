"""用官方随机小模型验证动态音频编码与跨窗口流式解码。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

from .codec_export import AudioDecoderGraph, AudioDecoderStepGraph, export_codec


def verify_tiny_codec(output: Path) -> dict:
    """保留真实采样率与上采样结构，用小通道数降低验证成本。

    Args:
        output: 随机小模型与验证报告的保存目录。
    Returns:
        编码离散一致性及解码浮点误差报告。
    """
    torch.manual_seed(91)
    torch.set_num_threads(1)
    config = Qwen3TTSTokenizerV2Config(
        encoder_valid_num_quantizers=16, input_sample_rate=24000, output_sample_rate=24000,
        decode_upsample_rate=1920, encode_downsample_rate=1920,
        encoder_config=dict(
            hidden_size=8, num_filters=2, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, head_dim=4, intermediate_size=16, codebook_dim=8,
            vector_quantization_hidden_dimension=8, codebook_size=8, num_quantizers=32,
            num_semantic_quantizers=1, upsample_groups=8, upsampling_ratios=[8, 6, 5, 4],
            frame_rate=12.5, pad_mode="constant", _attn_implementation="eager",
        ),
        decoder_config=dict(
            codebook_size=8, codebook_dim=8, latent_dim=8, hidden_size=8,
            head_dim=4, num_attention_heads=2, num_key_value_heads=2, intermediate_size=16,
            num_hidden_layers=2, num_quantizers=16, sliding_window=5, decoder_dim=16,
            upsample_rates=[8, 5, 4, 3], upsampling_ratios=[2, 2], _attn_implementation="eager",
        ),
    )
    model = Qwen3TTSTokenizerV2Model(config).eval().float()
    # 官方随机初始化的多级卷积输出极小，放大测试信号以避免绝对容差掩盖错误。
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            torch.nn.init.normal_(module.weight, std=0.7 / (module.in_channels * module.kernel_size[0]) ** 0.5)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=0.01)
    for name, value in model.named_buffers():
        if name.endswith("embed_sum"):
            torch.nn.init.normal_(value, std=0.2)
    for name, value in model.named_parameters():
        if name.endswith("embedding_sum"):
            torch.nn.init.normal_(value, std=0.2)
    metadata = export_codec(model, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    encoder = ort.InferenceSession(str(output / "audio_encoder.onnx"), options, providers=["CPUExecutionProvider"])
    report = {"tiny": True, "codec": metadata["validation"], "encoder_lengths": []}
    with torch.inference_mode():
        for length in (1, 100, 1919, 1920, 1921, 5760, 12113, 480137):
            audio = torch.randn(1, length) * 0.03
            reference = model.encode(
                audio, padding_mask=torch.ones_like(audio, dtype=torch.bool), return_dict=True,
            ).audio_codes[0].unsqueeze(0).numpy()
            actual = encoder.run(None, {"audio": audio.numpy()})[0]
            np.testing.assert_array_equal(actual, reference)
            report["encoder_lengths"].append({"samples": length, "frames": actual.shape[1], "codes_equal": True})
        del encoder
        decoder = ort.InferenceSession(str(output / "decoder.onnx"), options, providers=["CPUExecutionProvider"])
        codes = torch.randint(0, config.decoder_config.codebook_size, (1, 19, 16))
        full = AudioDecoderGraph(model.decoder).eval()
        reference = model.decoder(codes.transpose(1, 2)).squeeze(1).numpy()
        maximum = 0.0
        for length in (1, 3, 11, 19):
            actual = decoder.run(None, {"codes": codes[:, :length].numpy()})[0]
            expected = full(codes[:, :length]).numpy()
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        del decoder
        decoder = ort.InferenceSession(str(output / "decoder_step.onnx"), options, providers=["CPUExecutionProvider"])
        step = AudioDecoderStepGraph(model.decoder).eval()
        torch_states = tuple(torch.zeros(item["shape"]) for item in step.state_specs)
        states = {item["name"]: value.numpy() for item, value in zip(step.state_specs, torch_states, strict=True)}
        pieces = []
        torch_pieces = []
        for position in range(codes.shape[1]):
            next_codes = codes[:, position : position + 1]
            result = step(next_codes, torch.tensor([position]), *torch_states)
            torch_pieces.append(result[0].numpy())
            torch_states = result[1:]
            values = decoder.run(None, {
                "codes": next_codes.numpy(), "position": np.asarray([position], dtype=np.int64), **states,
            })
            pieces.append(values[0])
            states = {item["name"]: value for item, value in zip(step.state_specs, values[1:], strict=True)}
        actual = np.concatenate(pieces, axis=-1)
        torch_actual = np.concatenate(torch_pieces, axis=-1)
        np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=3e-5)
        np.testing.assert_allclose(torch_actual, reference, atol=3e-6, rtol=3e-5)
        report["decoder_dynamic_max_abs_error"] = maximum
        report["decoder_rollover_frames"] = codes.shape[1]
        report["decoder_window"] = config.decoder_config.sliding_window
        report["decoder_step_max_abs_error"] = float(np.max(np.abs(actual - reference)))
        report["torch_step_max_abs_error"] = float(np.max(np.abs(torch_actual - reference)))
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """执行可复现的小模型声码器数值验证。

    Returns:
        无返回值；失败时抛出数值断言，成功时保存并打印报告。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="随机小模型及报告目录")
    arguments = parser.parse_args()
    print(json.dumps(verify_tiny_codec(arguments.output), indent=2))


if __name__ == "__main__":
    main()
