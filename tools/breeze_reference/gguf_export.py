"""把 Breeze 的两个自回归主干无量化导出为官方 llama.cpp 可加载的 GGUF。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gguf
import numpy as np
from safetensors import safe_open
import torch


class _CheckpointWeights:
    """逐张量读取检查点并拒绝形状或原始精度不符的权重。

    Args:
        checkpoint: 含原始配置和 Safetensors 的目录。
        precision: 目标存储精度；保存 BF16 时禁止舍入原始 FP32 权重。

    Returns:
        不常驻模型参数的只读权重访问器。
    """

    def __init__(self, checkpoint: Path, precision: str) -> None:
        self.checkpoint = checkpoint.resolve(strict=True)
        self.precision = precision
        index = self.checkpoint / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            with safe_open(self.checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
                self.weight_map = {name: "model.safetensors" for name in handle.keys()}

    def read(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        """校验并把一张原始浮点权重无损读取为处理器上的 FP32。

        Args:
            name: 原始检查点张量名称。
            shape: 由实际组件配置推导出的完整形状。

        Returns:
            精确提升 BF16 或保留原始 FP32 的连续数组。
        """
        if name not in self.weight_map:
            raise ValueError(f"Breeze 检查点缺少权重：{name}")
        path = (self.checkpoint / self.weight_map[name]).resolve(strict=True)
        if not path.is_relative_to(self.checkpoint):
            raise ValueError(f"Breeze 权重路径越界：{name}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
            if tuple(value.shape) != shape:
                raise ValueError(f"Breeze 权重形状不符：{name}，要求 {shape}，实际 {tuple(value.shape)}")
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"Breeze 原始权重只允许 BF16 或 FP32：{name}，实际 {value.dtype}")
            if self.precision == "bfloat16" and value.dtype != torch.bfloat16:
                raise ValueError(f"保存 BF16 时禁止舍入原始 FP32 权重：{name}")
            result = value.float().numpy()
        if not np.isfinite(result).all():
            raise ValueError(f"Breeze 权重包含非有限值：{name}")
        return np.ascontiguousarray(result)


def _validate_config(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """核对官方图能完整表达的固定 Breeze 架构，避免静默改变公式。

    Args:
        raw: 本地检查点中的完整配置。

    Returns:
        已核对的 Qwen3 主干配置和深度解码器配置。
    """
    backbone = raw.get("backbone_config")
    depth = raw.get("depth_decoder_config")
    if not isinstance(backbone, dict) or not backbone or backbone.get("model_type") != "qwen3":
        raise ValueError("GGUF 导出要求完整的内嵌 Qwen3 backbone_config")
    if not isinstance(depth, dict) or depth.get("model_type") != "breeze_depth_decoder_model":
        raise ValueError("GGUF 导出要求完整的 depth_decoder_config")
    if int(raw.get("num_codebooks", 0)) != 16 or int(depth.get("num_codebooks", 0)) != 16:
        raise ValueError("Breeze GGUF 必须完整保留十六码本")
    if int(raw["vocab_size"]) != int(depth["vocab_size"]):
        raise ValueError("主干与深度码本词表配置不一致")
    if int(raw["hidden_size"]) != int(backbone["hidden_size"]):
        raise ValueError("主干隐藏维度与内嵌 Qwen3 配置不一致")
    if int(depth["backbone_hidden_size"]) != int(backbone["hidden_size"]):
        raise ValueError("当前固定检查点不允许额外的主干隐藏状态投影")
    if int(depth["audio_embed_size"]) != int(backbone["hidden_size"]):
        raise ValueError("音频嵌入维度与主干隐藏维度不一致")
    for name, config in (("主干", backbone), ("深度解码器", depth)):
        for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings"):
            if int(config.get(field, 0)) <= 0:
                raise ValueError(f"{name} 配置必须提供正整数：{field}")
        if config.get("hidden_act", "silu") != "silu" or config.get("attention_bias", False) or config.get("mlp_bias", False):
            raise ValueError(f"{name} 仅支持原始无偏置 SiLU 架构")
        if config["num_attention_heads"] % config["num_key_value_heads"] or config["head_dim"] % 2:
            raise ValueError(f"{name} 的分组查询或旋转维度无效")
        if not np.isfinite(float(config["rms_norm_eps"])) or float(config["rms_norm_eps"]) <= 0:
            raise ValueError(f"{name} 的归一化精度配置无效")
        if not np.isfinite(float(config["rope_theta"])) or float(config["rope_theta"]) <= 0:
            raise ValueError(f"{name} 的旋转基频配置无效")
    if backbone.get("rope_scaling") or backbone.get("use_sliding_window", False):
        raise ValueError("当前固定 Qwen3 主干不允许旋转缩放或滑窗注意力")
    scaling = depth.get("rope_scaling", {})
    if scaling.get("rope_type") != "llama3":
        raise ValueError("深度解码器要求检查点原始的 Llama3 旋转缩放")
    factor, low, high = (float(scaling[key]) for key in ("factor", "low_freq_factor", "high_freq_factor"))
    if not all(np.isfinite(value) for value in (factor, low, high)) or factor < 1 or not 0 < low < high:
        raise ValueError("深度解码器的旋转缩放参数无效")
    if int(scaling.get("original_max_position_embeddings", 0)) <= 0:
        raise ValueError("深度解码器缺少原始旋转上下文长度")
    return backbone, depth


def _rope_factors(config: dict[str, Any]) -> np.ndarray:
    """计算官方 GGUF 所需的逐频率除数，保留原始 Llama3 旋转公式。

    Args:
        config: 深度解码器配置及其完整旋转缩放参数。

    Returns:
        长度为头维度一半的 FP32 频率除数；全局频率缩放保持一。
    """
    scaling = config["rope_scaling"]
    dimension = int(config["head_dim"])
    inverse = 1.0 / (float(config["rope_theta"]) ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension))
    wavelength = np.float32(2.0 * np.pi) / inverse
    original = float(scaling["original_max_position_embeddings"])
    low, high, factor = (float(scaling[key]) for key in ("low_freq_factor", "high_freq_factor", "factor"))
    low_wavelength, high_wavelength = original / low, original / high
    adjusted = np.where(wavelength > low_wavelength, inverse / factor, inverse)
    smooth = (original / wavelength - low) / (high - low)
    medium = (wavelength <= low_wavelength) & (wavelength >= high_wavelength)
    adjusted = np.where(medium, (1.0 - smooth) * inverse / factor + smooth * inverse, adjusted)
    return np.asarray(inverse / adjusted, dtype=np.float32)


def _permute_llama(weight: np.ndarray, heads: int) -> np.ndarray:
    """按官方 Llama 转换器重排查询或键的旋转半区。

    Args:
        weight: 原始查询或键的二维线性权重。
        heads: 该权重实际输出的注意力头数。

    Returns:
        数值不变、行布局适用于官方 Llama 旋转实现的连续数组。
    """
    return np.ascontiguousarray(weight.reshape(heads, 2, weight.shape[0] // heads // 2, weight.shape[1]).swapaxes(1, 2).reshape(weight.shape))


def _add_tensor(
    writer: gguf.GGUFWriter, tensors: list[dict[str, Any]], name: str,
    value: np.ndarray, *, source: str, precision: str, transform: str = "identity",
) -> None:
    """逐张量写入临时流并记录精度与数学变换。

    Args:
        writer: 当前组件的 GGUF 写入器。
        tensors: 当前组件的张量审计记录。
        name: 官方 GGUF 张量名称。
        value: 完成布局变换的 FP32 数组。
        source: 原始检查点来源名称或推导来源。
        precision: 当前张量的存储精度。
        transform: 相对原始权重的数学或布局变换。

    Returns:
        无返回值；数组写入后由调用方释放。
    """
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"导出张量包含非有限值：{name}")
    dtype = gguf.GGMLQuantizationType.F32 if precision == "float32" else gguf.GGMLQuantizationType.BF16
    stored = value if precision == "float32" else gguf.quantize(value, dtype)
    writer.add_tensor(name, stored, raw_dtype=dtype)
    tensors.append({"name": name, "source": source, "shape": list(value.shape), "dtype": dtype.name, "transform": transform})


def _writer(path: Path, architecture: str, config: dict[str, Any], vocab_size: int, precision: str) -> gguf.GGUFWriter:
    """创建接收嵌入或码本编号的官方解码器，并写入完整架构元数据。

    Args:
        path: 尚未公开给运行时的临时输出路径。
        architecture: 官方架构名称，取值为 qwen3 或 llama。
        config: 实际组件配置，不能使用 Breeze 顶层的历史主干字段。
        vocab_size: 组件实际输出行数。
        precision: 主要二维权重的存储精度。

    Returns:
        使用临时文件缓冲权重的 GGUF 写入器。
    """
    writer = gguf.GGUFWriter(path, architecture, use_temp_file=True)
    writer.add_name(f"Breeze TTS 2 {architecture}")
    writer.add_file_type(gguf.LlamaFileType.ALL_F32 if precision == "float32" else gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_vocab_size(vocab_size)
    writer.add_tokenizer_model("none")
    writer.add_context_length(int(config["max_position_embeddings"]))
    writer.add_embedding_length(int(config["hidden_size"]))
    writer.add_block_count(int(config["num_hidden_layers"]))
    writer.add_feed_forward_length(int(config["intermediate_size"]))
    writer.add_head_count(int(config["num_attention_heads"]))
    writer.add_head_count_kv(int(config["num_key_value_heads"]))
    writer.add_key_length(int(config["head_dim"]))
    writer.add_value_length(int(config["head_dim"]))
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_dimension_count(int(config["head_dim"]))
    writer.add_rope_freq_base(float(config["rope_theta"]))
    # 深度解码器的分段缩放已折入逐频率除数，不能再应用一次全局缩放。
    writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    writer.add_rope_scaling_factor(1.0)
    writer.add_causal_attention(True)
    return writer


def _add_trunk(
    writer: gguf.GGUFWriter, tensors: list[dict[str, Any]], weights: _CheckpointWeights,
    prefix: str, config: dict[str, Any], precision: str, *, llama_layout: bool,
) -> None:
    """按实际配置逐层导出共享的因果注意力、前馈层和归一化。

    Args:
        writer: 当前组件的 GGUF 写入器。
        tensors: 当前组件的张量审计记录。
        weights: 只读原始权重访问器。
        prefix: 原始组件的参数前缀。
        config: 当前组件的完整配置。
        precision: 原始二维权重的存储精度。
        llama_layout: 是否应用官方 Llama 查询、键布局重排。

    Returns:
        无返回值；全部参数逐张量写入。
    """
    hidden, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    dimension, queries, keys = (int(config[key]) for key in ("head_dim", "num_attention_heads", "num_key_value_heads"))
    mapping = [
        ("input_layernorm", "attn_norm", (hidden,)),
        ("post_attention_layernorm", "ffn_norm", (hidden,)),
        ("self_attn.q_proj", "attn_q", (queries * dimension, hidden)),
        ("self_attn.k_proj", "attn_k", (keys * dimension, hidden)),
        ("self_attn.v_proj", "attn_v", (keys * dimension, hidden)),
        ("self_attn.o_proj", "attn_output", (hidden, queries * dimension)),
        ("mlp.gate_proj", "ffn_gate", (intermediate, hidden)),
        ("mlp.up_proj", "ffn_up", (intermediate, hidden)),
        ("mlp.down_proj", "ffn_down", (hidden, intermediate)),
    ]
    if not llama_layout:
        mapping.extend((("self_attn.q_norm", "attn_q_norm", (dimension,)), ("self_attn.k_norm", "attn_k_norm", (dimension,))))
    for layer in range(int(config["num_hidden_layers"])):
        for original, target, shape in mapping:
            source = f"{prefix}.layers.{layer}.{original}.weight"
            value = weights.read(source, shape)
            transform = "identity"
            if llama_layout and target in ("attn_q", "attn_k"):
                value = _permute_llama(value, queries if target == "attn_q" else keys)
                transform = "llama_rope_permutation"
            _add_tensor(writer, tensors, f"blk.{layer}.{target}.weight", value, source=source,
                        precision="float32" if len(shape) == 1 else precision, transform=transform)
            del value
    source = f"{prefix}.norm.weight"
    _add_tensor(writer, tensors, "output_norm.weight", weights.read(source, (hidden,)), source=source, precision="float32")


def _finish(writer: gguf.GGUFWriter, temporary: Path, target: Path) -> dict[str, Any]:
    """完成单组件写入并原子替换最终资产。

    Args:
        writer: 已写入配置与张量的 GGUF 写入器。
        temporary: 当前组件的临时输出路径。
        target: 验证写入完成后才能公开的最终路径。

    Returns:
        最终路径、文件大小和完整 SHA256。
    """
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(target.resolve()), "size": target.stat().st_size, "sha256": digest.hexdigest()}


def export_generation_gguf(checkpoint: Path, output_dir: Path, *, precision: str = "float32") -> dict[str, Any]:
    """导出完整 Qwen3 主干和保留十六码本的 Llama 深度解码器。

    Args:
        checkpoint: 固定 Breeze 原始配置与 BF16 或测试用 FP32 Safetensors 所在目录。
        output_dir: 接收 backbone.gguf 和 depth.gguf 的目录。
        precision: 原始二维权重保存为 float32 或 bfloat16；投影查表与归一化始终为 FP32。

    Returns:
        包含资产摘要、逐张量映射和深度条件投影来源的可序列化报告。
    """
    if precision not in ("float32", "bfloat16"):
        raise ValueError("GGUF 导出精度只能为 float32 或 bfloat16，不允许量化")
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    backbone, depth = _validate_config(raw)
    weights = _CheckpointWeights(checkpoint, precision)
    if any(name.startswith("depth_decoder.model.backbone_hidden_state_projector.") for name in weights.weight_map):
        raise ValueError("当前固定检查点不支持额外的 backbone_hidden_state_projector")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "precision": precision, "quantized": False, "num_codebooks": 16, "num_depth_heads": 15,
        "source_checkpoint": str(checkpoint.resolve()), "files": {}, "components": {},
        "depth_conditioner": {"source": "depth_decoder.model.inputs_embeds_projector.weight",
                              "shape": [int(depth["hidden_size"]), int(depth["audio_embed_size"])],
                              "formula": "backbone_hidden @ inputs_embeds_projector.weight.T", "precision": "float32"},
    }
    vocab, depth_hidden = int(raw["vocab_size"]), int(depth["hidden_size"])
    for component, architecture, config in (("backbone", "qwen3", backbone), ("depth", "llama", depth)):
        target = output_dir / f"{component}.gguf"
        temporary = output_dir / f".{component}.gguf.partial"
        output_vocab = vocab + 1 if component == "backbone" else 16 * vocab
        writer = _writer(temporary, architecture, config, output_vocab, precision)
        tensors: list[dict[str, Any]] = []
        try:
            if component == "backbone":
                # 官方图要求查表参数；嵌入输入绕过该表，输出头则复用同一份原始权重。
                _add_tensor(writer, tensors, "token_embd.weight", weights.read("lm_head.weight", (vocab + 1, int(backbone["hidden_size"]))),
                            source="lm_head.weight", precision=precision, transform="shared_output_head")
                _add_trunk(writer, tensors, weights, "backbone_model", backbone, precision, llama_layout=False)
            else:
                projection_source = "depth_decoder.model.inputs_embeds_projector.weight"
                projection = weights.read(projection_source, (depth_hidden, int(depth["audio_embed_size"])))
                embedding_source = "depth_decoder.model.embed_tokens.weight"
                embedding = weights.read(embedding_source, (16 * vocab, int(depth["audio_embed_size"])))
                projected = embedding @ projection.T
                del embedding, projection
                _add_tensor(writer, tensors, "token_embd.weight", projected, source=f"{embedding_source};{projection_source}",
                            precision="float32", transform="embedding @ projector.T")
                del projected
                source = "depth_decoder.codebooks_head.weight"
                heads = weights.read(source, (15, depth_hidden, vocab))
                output = np.zeros((16 * vocab, depth_hidden), dtype=np.float32)
                # 输出按码本位置顺序排列；最后一段零行只用于满足官方等长词表约束。
                for codebook in range(15):
                    output[codebook * vocab:(codebook + 1) * vocab] = heads[codebook].T
                del heads
                _add_tensor(writer, tensors, "output.weight", output, source=source, precision=precision,
                            transform="concatenate(heads.transpose(0,2,1)); append one zero codebook")
                del output
                _add_tensor(writer, tensors, "rope_freqs.weight", _rope_factors(depth), source="depth_decoder_config.rope_scaling",
                            precision="float32", transform="original_inverse_frequency / scaled_inverse_frequency")
                _add_trunk(writer, tensors, weights, "depth_decoder.model", depth, precision, llama_layout=True)
            report["files"][target.name] = _finish(writer, temporary, target)
            report["components"][component] = {"architecture": architecture, "config": config, "vocab_size": output_vocab,
                                                "tensor_count": len(tensors), "tensors": tensors}
        finally:
            writer.close()
            if writer.temp_file is not None and not writer.temp_file.closed:
                writer.temp_file.close()
            temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    """提供纯处理器的离线导出命令，并保存可复核的张量报告。

    Returns:
        无返回值；命令失败时保留原始异常以供诊断。
    """
    parser = argparse.ArgumentParser(description="无量化导出 Breeze Qwen3 和深度 Llama GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    report = export_generation_gguf(args.checkpoint, args.output_dir, precision=args.precision)
    destination = args.output_dir / "gguf-export-report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination.resolve()), "files": report["files"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
