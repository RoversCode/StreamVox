"""导出原始精度的音频编码器、整段解码器及显式状态流式解码器。

音频模型及算子来源于阿里巴巴千问团队的 qwen-tts 0.1.1，许可为 Apache-2.0。
显式状态遵循 Breeze 上游提交 daa6cb2efe5471f09369229f322d20bb956cb3c3 的流式语义。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .bundle import export_isolated_graph


class AudioEncoderGraph(nn.Module):
    """将二十四千赫兹单声道波形转换为十六码本的离散音频帧。

    Args:
        tokenizer_model: 官方音频分词器的底层模型。
    Returns:
        可导出的波形编码模块。
    """

    def __init__(self, tokenizer_model: nn.Module) -> None:
        super().__init__()
        self.encoder = tokenizer_model.encoder
        self.num_codebooks = int(tokenizer_model.encoder_valid_num_quantizers)
        self.samples_per_frame = int(tokenizer_model.encode_downsample_rate)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """编码有效波形，保留末尾不足一帧的官方补零语义。

        Args:
            audio: 形状为一乘采样数的原始波形。
        Returns:
            形状为一乘帧数乘码本数的整数编码。
        """
        hidden = self.encoder.encoder(audio.unsqueeze(1)).transpose(1, 2)
        positions = torch.arange(hidden.shape[1], dtype=torch.int64, device=audio.device)
        # 官方编码器的预填充分支使用完整因果注意力，不能擅自替换为短窗重编码。
        valid = positions.view(1, -1) <= positions.view(-1, 1)
        mask = torch.where(valid, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[None, None]
        for layer in self.encoder.encoder_transformer.layers:
            hidden = layer(
                hidden, attention_mask=mask, position_ids=positions.unsqueeze(0), use_cache=False,
            )[0]
        hidden = self.encoder.downsample(hidden.transpose(1, 2))
        # 后续码本不会影响已得到的码本，省略官方最终丢弃的后十六码本运算。
        quantizer = self.encoder.quantizer
        semantic_count = int(quantizer.num_semantic_quantizers)
        encoded = self._quantize(hidden, quantizer.semantic_residual_vector_quantizer, semantic_count)
        encoded.extend(self._quantize(hidden, quantizer.acoustic_residual_vector_quantizer, self.num_codebooks - semantic_count))
        encoded = torch.stack(encoded, dim=1)
        frames = (audio.shape[-1] + self.samples_per_frame - 1) // self.samples_per_frame
        return encoded[:, : self.num_codebooks, :frames].transpose(1, 2).contiguous()

    def _quantize(self, hidden: torch.Tensor, quantizer: nn.Module, count: int) -> list[torch.Tensor]:
        """用等价的平方欧氏距离选择最近码字，支持动态帧数导出。

        Args:
            hidden: 卷积及注意力后的音频特征。
            quantizer: 官方语义或声学残差量化器。
            count: 本分支需要的码本数量。
        Returns:
            按残差量化顺序排列的整数码字。
        """
        residual = quantizer.input_proj(hidden) if quantizer.input_proj is not None else hidden
        result = []
        for layer in quantizer.layers[:count]:
            embeddings = layer.codebook.embed
            vectors = residual.transpose(1, 2).float()
            table = embeddings.float()
            # 平方根单调，最近邻编号保持不变；显式矩阵运算避开不支持动态行数的距离导出器。
            squared = vectors.square().sum(-1, keepdim=True) + table.square().sum(-1)
            squared = (squared - 2 * torch.matmul(vectors, table.transpose(0, 1))).clamp(min=0)
            indices = squared.argmin(dim=-1)
            result.append(indices)
            residual = residual - F.embedding(indices, embeddings).transpose(1, 2)
        return result


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    """交换旋转位置编码的两半特征。

    Args:
        value: 最后一维为偶数的查询或键。
    Returns:
        与输入同形的半维旋转结果。
    """
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class AudioDecoderGraph(nn.Module):
    """保持官方算子及权重的整段因果解码模块。

    Args:
        decoder: 官方十二赫兹音频解码器。
    Returns:
        不采用历史重叠截断的整段解码模块。
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.window = int(decoder.config.sliding_window)
        self.num_layers = int(decoder.config.num_hidden_layers)
        self.num_heads = int(decoder.config.num_attention_heads)
        self.num_kv_heads = int(decoder.config.num_key_value_heads)
        self.head_dim = int(decoder.config.head_dim)
        self.register_buffer(
            "inv_freq", decoder.pre_transformer.rotary_emb.inv_freq.detach().clone(),
            persistent=False,
        )
        self.rope_scale = float(decoder.pre_transformer.rotary_emb.attention_scaling)

    def _transformer(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        past: tuple[torch.Tensor, ...] = (),
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """用显式张量表达官方注意力，避免导出缓存容器及掩码构造器。

        Args:
            hidden: 卷积后的音频特征。
            positions: 当前音频帧的绝对位置。
            past: 按层交替排列的固定窗口键值状态；空元组表示整段计算。
        Returns:
            变换后的特征与更新的键值状态。
        """
        transformer = self.decoder.pre_transformer
        hidden = transformer.input_proj(hidden)
        frequencies = positions.float().view(-1, 1) * self.inv_freq.float().view(1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = (embedding.cos() * self.rope_scale).to(hidden.dtype)[None, None]
        sine = (embedding.sin() * self.rope_scale).to(hidden.dtype)[None, None]
        length = hidden.shape[1]
        if past:
            # 单帧流式缓存仅淘汰模型原生滑窗之外的键值，绝不缩短模型上下文。
            valid = torch.arange(self.window, device=hidden.device) >= self.window - 1 - positions[-1]
            mask = torch.where(valid, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)
            mask = mask.view(1, 1, 1, self.window)
        else:
            query_positions = positions.view(-1, 1)
            key_positions = positions.view(1, -1)
            valid = (key_positions <= query_positions) & (key_positions > query_positions - self.window)
            mask = torch.where(valid, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)
            mask = mask.view(1, 1, length, length)
        updated: list[torch.Tensor] = []
        for index, layer in enumerate(transformer.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            query = attention.q_norm(attention.q_proj(normalized).view(1, length, self.num_heads, self.head_dim)).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(normalized).view(1, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            value = attention.v_proj(normalized).view(1, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
            query = query * cosine + _rotate_half(query) * sine
            key = key * cosine + _rotate_half(key) * sine
            if past:
                key = torch.cat((past[2 * index], key), dim=2)[:, :, -self.window :]
                value = torch.cat((past[2 * index + 1], value), dim=2)[:, :, -self.window :]
                updated.extend((key, value))
            groups = self.num_heads // self.num_kv_heads
            if groups != 1:
                # 将查询头分组后广播键值，避免为分组查询注意力复制键值缓存。
                grouped_query = query.reshape(1, self.num_kv_heads, groups, length, self.head_dim)
                scores = torch.matmul(grouped_query, key.unsqueeze(2).transpose(-1, -2)) * attention.scaling
                probabilities = torch.softmax(scores + mask.unsqueeze(1), dim=-1, dtype=torch.float32).to(query.dtype)
                attended = torch.matmul(probabilities, value.unsqueeze(2)).reshape(1, self.num_heads, length, self.head_dim)
            else:
                scores = torch.matmul(query, key.transpose(-1, -2)) * attention.scaling
                probabilities = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(query.dtype)
                attended = torch.matmul(probabilities, value)
            attended = attended.transpose(1, 2).reshape(1, length, -1)
            hidden = residual + layer.self_attn_layer_scale(attention.o_proj(attended))
            hidden = hidden + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(hidden)))
        return transformer.output_proj(transformer.norm(hidden)), tuple(updated)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """一次解码全部音频帧。

        Args:
            codes: 形状为一乘帧数乘码本数的有效整数编码。
        Returns:
            形状为一乘采样数的单声道波形。
        """
        hidden = self.decoder.quantizer.decode(codes.transpose(1, 2))
        hidden = self._convolution(self.decoder.pre_conv, hidden).transpose(1, 2)
        positions = torch.arange(hidden.shape[1], dtype=torch.int64, device=hidden.device)
        hidden, _ = self._transformer(hidden, positions)
        return self._waveform(hidden.transpose(1, 2)).squeeze(1)

    def _convolution(
        self, module: nn.Module, hidden: torch.Tensor,
        states: tuple[torch.Tensor, ...] = (), updated: list[torch.Tensor] | None = None,
        *, transpose: bool = False,
    ) -> torch.Tensor:
        """以动态长度安全的方式执行官方解码侧因果卷积。

        Args:
            module: 官方卷积模块。
            hidden: 当前特征。
            states: 流式子类使用的输入历史。
            updated: 流式子类使用的后继状态容器。
            transpose: 是否为转置卷积。
        Returns:
            与官方卷积数学等价的输出。
        """
        if transpose:
            return module(hidden)
        if int(module.conv.stride[0]) != 1:
            raise ValueError("Codec decoder export requires unit-stride causal convolutions")
        # 步幅为一时额外右填充恒为零，避免上游浮点取整把动态长度冻结。
        return module.conv(F.pad(hidden, (int(module.padding), 0))).contiguous()

    def _waveform(
        self, hidden: torch.Tensor, states: tuple[torch.Tensor, ...] = (),
        updated: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """共享整段与流式解码的上采样和波形重建顺序。

        Args:
            hidden: 注意力后的音频特征。
            states: 可选的卷积历史。
            updated: 可选的后继状态容器。
        Returns:
            单声道波形。
        """
        for blocks in self.decoder.upsample:
            hidden = self._convolution(blocks[0], hidden, states, updated, transpose=True)
            block = blocks[1]
            residual = hidden
            hidden = self._convolution(block.dwconv, hidden, states, updated).transpose(1, 2)
            hidden = block.gamma * block.pwconv2(block.act(block.pwconv1(block.norm(hidden))))
            hidden = residual + hidden.transpose(1, 2)
        hidden = self._convolution(self.decoder.decoder[0], hidden, states, updated)
        for block in self.decoder.decoder[1:-2]:
            hidden = self._convolution(block.block[1], block.block[0](hidden), states, updated, transpose=True)
            for unit in block.block[2:]:
                residual = hidden
                hidden = self._convolution(unit.conv1, unit.act1(hidden), states, updated)
                hidden = residual + self._convolution(unit.conv2, unit.act2(hidden), states, updated)
        hidden = self.decoder.decoder[-2](hidden)
        hidden = self._convolution(self.decoder.decoder[-1], hidden, states, updated)
        return hidden.clamp(-1, 1)


class AudioDecoderStepGraph(AudioDecoderGraph):
    """按单帧推进全部卷积和注意力状态，避免重复解码历史音频。

    Args:
        decoder: 官方十二赫兹音频解码器。
    Returns:
        每步输出一千九百二十个采样及后继状态的模块。
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__(decoder)
        self.state_specs: list[dict[str, Any]] = []
        self._convolution_names: dict[int, int] = {}
        self._add_convolution_state("pre_conv", decoder.pre_conv)
        for index, blocks in enumerate(decoder.upsample):
            self._add_convolution_state(f"upsample_{index}_transpose", blocks[0], transpose=True)
            self._add_convolution_state(f"upsample_{index}_depthwise", blocks[1].dwconv)
        self._add_convolution_state("decoder_pre_conv", decoder.decoder[0])
        for index, block in enumerate(decoder.decoder[1:-2]):
            self._add_convolution_state(f"decoder_{index}_transpose", block.block[1], transpose=True)
            for residual_index, unit in enumerate(block.block[2:]):
                self._add_convolution_state(f"decoder_{index}_residual_{residual_index}", unit.conv1)
        self._add_convolution_state("final_conv", decoder.decoder[-1])
        self.convolution_states = len(self.state_specs)
        for index in range(self.num_layers):
            for kind in ("key", "value"):
                self.state_specs.append({
                    "name": f"{kind}_{index}",
                    "shape": [1, self.num_kv_heads, self.window, self.head_dim],
                    "dtype": "float32",
                })

    def _add_convolution_state(self, name: str, module: nn.Module, *, transpose: bool = False) -> None:
        """记录卷积所需的精确输入历史范围。

        Args:
            name: 状态的稳定名称。
            module: 官方因果卷积或转置卷积。
            transpose: 是否为转置卷积。
        Returns:
            无。
        """
        left = (int(module.conv.kernel_size[0]) - 1) // int(module.conv.stride[0]) if transpose else int(module.padding)
        if not transpose and int(module.conv.stride[0]) != 1:
            raise ValueError("Stateful codec export requires unit-stride causal convolutions")
        if left <= 0:
            return
        self._convolution_names[id(module)] = len(self.state_specs)
        self.state_specs.append({"name": name, "shape": [1, int(module.conv.in_channels), left], "dtype": "float32"})

    def _convolution(
        self, module: nn.Module, hidden: torch.Tensor,
        states: tuple[torch.Tensor, ...] = (), updated: list[torch.Tensor] | None = None,
        *, transpose: bool = False,
    ) -> torch.Tensor:
        """只计算新增输入对应的波形，并携带精确的卷积历史。

        Args:
            module: 官方因果卷积或转置卷积。
            hidden: 当前新增特征。
            states: 本步输入状态。
            updated: 按输入顺序收集后继状态的容器。
            transpose: 是否为转置卷积。
        Returns:
            当前新增特征对应的卷积结果。
        """
        state_index = self._convolution_names.get(id(module))
        if state_index is None:
            return super()._convolution(module, hidden, transpose=transpose)
        history = states[state_index]
        left = int(self.state_specs[state_index]["shape"][-1])
        combined = torch.cat((history, hidden), dim=-1)
        assert updated is not None
        updated.append(combined[..., -left:])
        if transpose:
            stride = int(module.conv.stride[0])
            result = module(combined)
            return result[..., left * stride : left * stride + hidden.shape[-1] * stride]
        # 所有解码侧因果卷积步幅均为一，拼接完整左历史后无需再次填充。
        return module.conv(combined).contiguous()

    def forward(self, codes: torch.Tensor, position: torch.Tensor, *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """解码一帧并返回所有显式后继状态。

        Args:
            codes: 形状为一乘一乘码本数的有效整数编码。
            position: 当前帧的绝对位置，首帧为零。
            states: 按状态规格排列的卷积历史和键值缓存。
        Returns:
            波形及顺序一致的后继状态。
        """
        updated: list[torch.Tensor] = []
        hidden = self.decoder.quantizer.decode(codes.transpose(1, 2))
        hidden = self._convolution(self.decoder.pre_conv, hidden, states, updated).transpose(1, 2)
        hidden, next_kv = self._transformer(hidden, position, tuple(states[self.convolution_states :]))
        audio = self._waveform(hidden.transpose(1, 2), states, updated)
        return (audio.squeeze(1), *updated, *next_kv)


def verify_codec_exports(
    model: nn.Module, output_dir: Path, state_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """将实际导出图与官方编码和整段解码逐项对照。

    Args:
        model: 官方原始精度音频模型。
        output_dir: 已生成模型图的目录。
        state_specs: 单帧解码图的显式状态规格。
    Returns:
        精确编码比较及浮点波形误差报告；超出容差时抛出异常。
    """
    import numpy as np
    import onnxruntime as ort

    generator = torch.Generator(device="cpu").manual_seed(71)
    audio = torch.randn(1, 2 * int(model.encode_downsample_rate) + 113, generator=generator) * 0.03
    num_codebooks = int(model.encoder_valid_num_quantizers)
    codes = torch.randint(0, int(model.decoder.config.codebook_size), (1, 3, num_codebooks), generator=generator)
    with torch.inference_mode():
        expected_codes = model.encode(audio, padding_mask=torch.ones_like(audio, dtype=torch.bool), return_dict=True).audio_codes[0].unsqueeze(0).numpy()
        expected_audio = model.decoder(codes.transpose(1, 2)).squeeze(1).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    report: dict[str, Any] = {"atol": 1e-4, "rtol": 1e-3}

    # 逐个释放会话，验收时也不同时常驻三份权重。
    session = ort.InferenceSession(str(output_dir / "audio_encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    actual_codes = session.run(None, {"audio": audio.numpy()})[0]
    np.testing.assert_array_equal(actual_codes, expected_codes)
    report["encoder_codes_equal"] = True
    report["encoder_input_samples"] = int(audio.shape[-1])
    del session

    session = ort.InferenceSession(str(output_dir / "decoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    actual_audio = session.run(None, {"codes": codes.numpy()})[0]
    np.testing.assert_allclose(actual_audio, expected_audio, atol=report["atol"], rtol=report["rtol"])
    report["decoder_max_abs_error"] = float(np.max(np.abs(actual_audio - expected_audio)))
    del session

    session = ort.InferenceSession(str(output_dir / "decoder_step.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    states = {spec["name"]: np.zeros(spec["shape"], dtype=np.float32) for spec in state_specs}
    pieces = []
    for index in range(codes.shape[1]):
        feed = {"codes": codes[:, index : index + 1].numpy(), "position": np.asarray([index], dtype=np.int64), **states}
        values = session.run(None, feed)
        pieces.append(values[0])
        states = {spec["name"]: value for spec, value in zip(state_specs, values[1:], strict=True)}
    actual_audio = np.concatenate(pieces, axis=-1)
    np.testing.assert_allclose(actual_audio, expected_audio, atol=report["atol"], rtol=report["rtol"])
    report["decoder_step_max_abs_error"] = float(np.max(np.abs(actual_audio - expected_audio)))
    report["decoder_frames"] = int(codes.shape[1])
    return report


def export_codec(
    audio_tokenizer: Any, output_dir: str | Path, *, opset_version: int = 17,
) -> dict[str, Any]:
    """导出音频编码、整段解码和常量内存流式解码图及状态契约。

    Args:
        audio_tokenizer: 官方音频分词器或其底层模型。
        output_dir: 导出文件目录。
        opset_version: 导出的算子集版本。
    Returns:
        已写入文件的编码器、解码器和状态元数据。
    """
    model = getattr(audio_tokenizer, "model", audio_tokenizer)
    if model is None or model.config.model_type != "qwen3_tts_tokenizer_12hz":
        raise ValueError("Breeze codec export requires qwen3_tts_tokenizer_12hz")
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Codec export preserves the official float32 precision")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    model.encoder.config._attn_implementation = "eager"
    encoder = AudioEncoderGraph(model).eval()
    decoder = AudioDecoderGraph(model.decoder).eval()
    step = AudioDecoderStepGraph(model.decoder).eval()
    codes = torch.zeros(1, 3, int(model.encoder_valid_num_quantizers), dtype=torch.int64)
    options = {"opset_version": opset_version, "dynamo": False, "do_constant_folding": True}
    with torch.inference_mode():
        export_isolated_graph(
            encoder, (torch.zeros(1, 5760),), output_dir / "audio_encoder.onnx",
            input_names=["audio"], output_names=["codes"],
            dynamic_axes={"audio": {1: "samples"}, "codes": {1: "frames"}}, **options,
        )
        export_isolated_graph(
            decoder, (codes,), output_dir / "decoder.onnx",
            input_names=["codes"], output_names=["audio"],
            dynamic_axes={"codes": {1: "frames"}, "audio": {1: "samples"}}, **options,
        )
        states = tuple(torch.zeros(spec["shape"], dtype=torch.float32) for spec in step.state_specs)
        input_names = ["codes", "position", *[spec["name"] for spec in step.state_specs]]
        output_names = ["audio", *["next_" + spec["name"] for spec in step.state_specs]]
        export_isolated_graph(
            step, (codes[:, :1], torch.zeros(1, dtype=torch.int64), *states),
            output_dir / "decoder_step.onnx", input_names=input_names,
            output_names=output_names, **options,
        )
    metadata = {
        "sample_rate": int(model.output_sample_rate),
        "frame_rate": float(model.output_sample_rate) / int(model.decode_upsample_rate),
        "samples_per_frame": int(model.decode_upsample_rate),
        "num_codebooks": int(model.encoder_valid_num_quantizers),
        "audio_encoder": "audio_encoder.onnx", "decoder": "decoder.onnx",
        "decoder_step": "decoder_step.onnx", "chunk_frames": 1,
        "state_specs": step.state_specs,
    }
    metadata["validation"] = verify_codec_exports(model, output_dir, step.state_specs)
    (output_dir / "codec_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
