"""验证完整码本 GGUF 的权重精度、官方布局和离线线性折叠。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
gguf = pytest.importorskip("gguf")

from tools.breeze_reference.gguf_export import _rope_factors, export_generation_gguf


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> tuple[Path, dict, dict]:
    """创建保留十六码本和分组注意力的原始 BF16 小检查点。

    Args:
        tmp_path: 当前用例隔离的临时目录。

    Returns:
        检查点目录、原始配置和原始张量，供独立数学对照使用。
    """
    backbone = {
        "model_type": "qwen3", "hidden_size": 8, "intermediate_size": 12,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000, "hidden_act": "silu", "attention_bias": False,
        "rope_scaling": None, "use_sliding_window": False,
    }
    depth = {
        "model_type": "breeze_depth_decoder_model", "hidden_size": 4,
        "intermediate_size": 12, "num_hidden_layers": 1, "num_attention_heads": 1,
        "num_key_value_heads": 1, "head_dim": 4, "max_position_embeddings": 33,
        "rms_norm_eps": 1e-5, "rope_theta": 500000, "hidden_act": "silu",
        "attention_bias": False, "audio_embed_size": 8, "backbone_hidden_size": 8,
        "num_codebooks": 16, "vocab_size": 5,
        "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
                         "original_max_position_embeddings": 16,
                         "high_freq_factor": 0.0078125, "low_freq_factor": 0.001953125},
    }
    raw = {"hidden_size": 8, "vocab_size": 5, "num_codebooks": 16,
           "backbone_config": backbone, "depth_decoder_config": depth,
           "rms_norm_eps": 0.3, "rope_theta": 10}
    generator = torch.Generator(device="cpu").manual_seed(817)
    tensors = {}
    for prefix, hidden, queries, keys in (("backbone_model", 8, 2, 1), ("depth_decoder.model", 4, 1, 1)):
        shapes = {
            "input_layernorm": (hidden,), "post_attention_layernorm": (hidden,),
            "self_attn.q_proj": (queries * 4, hidden), "self_attn.k_proj": (keys * 4, hidden),
            "self_attn.v_proj": (keys * 4, hidden), "self_attn.o_proj": (hidden, queries * 4),
            "mlp.gate_proj": (12, hidden), "mlp.up_proj": (12, hidden), "mlp.down_proj": (hidden, 12),
        }
        if prefix == "backbone_model":
            shapes.update({"self_attn.q_norm": (4,), "self_attn.k_norm": (4,)})
        for name, shape in shapes.items():
            value = torch.randn(shape, generator=generator) / 8
            if len(shape) == 1:
                value += 1
            tensors[f"{prefix}.layers.0.{name}.weight"] = value.to(torch.bfloat16)
        tensors[f"{prefix}.norm.weight"] = (torch.randn(hidden, generator=generator) / 8 + 1).to(torch.bfloat16)
    for name, shape in {
        "lm_head.weight": (6, 8), "depth_decoder.model.embed_tokens.weight": (80, 8),
        "depth_decoder.model.inputs_embeds_projector.weight": (4, 8),
        "depth_decoder.codebooks_head.weight": (15, 4, 5),
    }.items():
        tensors[name] = (torch.randn(shape, generator=generator) / 8).to(torch.bfloat16)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    safetensors.save_file(tensors, checkpoint / "model.safetensors")
    return checkpoint, raw, tensors


def _arrays(path: Path) -> tuple[object, dict[str, np.ndarray]]:
    """读取小 GGUF，恢复 BF16 的原始浮点值供对照。

    Args:
        path: 测试生成的 GGUF 路径。

    Returns:
        文件读取器及按名称索引的 FP32 张量。
    """
    reader = gguf.GGUFReader(path)
    arrays = {tensor.name: gguf.dequantize(tensor.data, tensor.tensor_type) for tensor in reader.tensors}
    return reader, arrays


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_generation_gguf_preserves_heads_embeddings_and_layout(tiny_checkpoint: tuple, tmp_path: Path, precision: str) -> None:
    """导出布局必须保持完整输出头、条件投影和官方旋转行序。

    Args:
        tiny_checkpoint: 原始小检查点、配置和张量。
        tmp_path: 当前用例的隔离目录。
        precision: 不涉及量化的二维权重存储精度。
    """
    checkpoint, raw, original = tiny_checkpoint
    output = tmp_path / precision
    report = export_generation_gguf(checkpoint, output, precision=precision)
    main_reader, main = _arrays(output / "backbone.gguf")
    depth_reader, depth = _arrays(output / "depth.gguf")
    assert report["quantized"] is False
    assert report["num_codebooks"] == 16 and report["num_depth_heads"] == 15
    assert report["depth_conditioner"]["shape"] == [4, 8]
    assert main_reader.get_field("general.architecture").contents() == "qwen3"
    assert main_reader.get_field("qwen3.vocab_size").contents() == 6
    assert main_reader.get_field("qwen3.attention.layer_norm_rms_epsilon").contents() == pytest.approx(1e-6)
    assert main_reader.get_field("qwen3.rope.freq_base").contents() == 1000000
    assert main_reader.get_field("tokenizer.ggml.model").contents() == "none"
    assert depth_reader.get_field("llama.vocab_size").contents() == 80
    assert depth_reader.get_field("llama.rope.scaling.factor").contents() == 1
    assert "output.weight" not in main
    np.testing.assert_array_equal(main["token_embd.weight"], original["lm_head.weight"].float().numpy())
    np.testing.assert_array_equal(main["blk.0.attn_q.weight"], original["backbone_model.layers.0.self_attn.q_proj.weight"].float().numpy())
    projection = original["depth_decoder.model.inputs_embeds_projector.weight"].float().numpy()
    embedding = original["depth_decoder.model.embed_tokens.weight"].float().numpy()
    np.testing.assert_allclose(depth["token_embd.weight"], embedding @ projection.T, rtol=1e-6, atol=1e-7)
    hidden = np.asarray([0.1, -0.5, 0.75, 1.25], dtype=np.float32)
    expected_heads = original["depth_decoder.codebooks_head.weight"].float().numpy()
    for codebook in range(15):
        actual = depth["output.weight"][codebook * 5:(codebook + 1) * 5] @ hidden
        np.testing.assert_allclose(actual, hidden @ expected_heads[codebook], rtol=1e-6, atol=1e-7)
    assert np.count_nonzero(depth["output.weight"][75:]) == 0
    for target, source in (("attn_q", "q_proj"), ("attn_k", "k_proj")):
        # 逆置换还原上游半区，防止仅凭张量名称和形状掩盖错误的旋转布局。
        restored = depth[f"blk.0.{target}.weight"].reshape(1, 2, 2, 4).swapaxes(1, 2).reshape(4, 4)
        np.testing.assert_array_equal(restored, original[f"depth_decoder.model.layers.0.self_attn.{source}.weight"].float().numpy())
    for component, reader in (("backbone", main_reader), ("depth", depth_reader)):
        tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
        assert set(tensor_types.values()) <= {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16}
        assert report["components"][component]["tensor_count"] == len(reader.tensors)
        for tensor in report["components"][component]["tensors"]:
            if len(tensor["shape"]) == 1 or component == "depth" and tensor["name"] == "token_embd.weight":
                assert tensor_types[tensor["name"]] == gguf.GGMLQuantizationType.F32
    assert report["files"]["backbone.gguf"]["size"] == (output / "backbone.gguf").stat().st_size


def test_depth_rope_factors_reproduce_all_three_original_frequency_regions(tiny_checkpoint: tuple) -> None:
    """逐频率除数必须覆盖未缩放、平滑过渡和完全缩放三个区间。

    Args:
        tiny_checkpoint: 提供与真实检查点一致的深度旋转参数。
    """
    _, raw, _ = tiny_checkpoint
    config = dict(raw["depth_decoder_config"], head_dim=128)
    factors = _rope_factors(config)
    expected = []
    for dimension in range(0, 128, 2):
        inverse = 1.0 / 500000 ** (dimension / 128)
        wavelength = 2 * math.pi / inverse
        if wavelength > 8192:
            scaled = inverse / 32
        elif wavelength < 2048:
            scaled = inverse
        else:
            blend = (16 / wavelength - 0.001953125) / (0.0078125 - 0.001953125)
            scaled = inverse / 32 * (1 - blend) + inverse * blend
        expected.append(inverse / scaled)
    np.testing.assert_allclose(factors, expected, rtol=5e-7)
    assert np.any(factors == 1) and np.any(factors == 32)
    assert np.any((factors > 1) & (factors < 32))


def test_float32_reference_weights_are_preserved_without_bfloat16_rounding(tiny_checkpoint: tuple, tmp_path: Path) -> None:
    """原始 FP32 小模型必须可对照，但不能通过 BF16 选项悄悄舍入。

    Args:
        tiny_checkpoint: 可修改精度的原始小检查点。
        tmp_path: 导出原始 FP32 参考资产的隔离目录。
    """
    checkpoint, raw, original = tiny_checkpoint
    tensors = {name: value.float() for name, value in original.items()}
    tensors["lm_head.weight"][0, 0] = 0.12345679
    # 原始小模型配置允许省略由上游默认值提供的激活名称。
    del raw["depth_decoder_config"]["hidden_act"]
    (checkpoint / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    safetensors.save_file(tensors, checkpoint / "model.safetensors")
    export_generation_gguf(checkpoint, tmp_path / "fp32", precision="float32")
    _, arrays = _arrays(tmp_path / "fp32" / "backbone.gguf")
    np.testing.assert_array_equal(arrays["token_embd.weight"], tensors["lm_head.weight"].numpy())
    with pytest.raises(ValueError, match="禁止舍入"):
        export_generation_gguf(checkpoint, tmp_path / "bf16", precision="bfloat16")


@pytest.mark.parametrize("precision", ["float16", "q8_0", "auto"])
def test_export_rejects_precision_reduction_before_writing(tiny_checkpoint: tuple, tmp_path: Path, precision: str) -> None:
    """禁止以存储选项暗中降精度或量化。

    Args:
        tiny_checkpoint: 原始小检查点。
        tmp_path: 用于确认失败前未创建资产的目录。
        precision: 不被允许的存储精度。
    """
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="不允许量化"):
        export_generation_gguf(tiny_checkpoint[0], output, precision=precision)
    assert not output.exists()


@pytest.mark.parametrize("invalid", ["shape", "dtype", "projector", "codebooks", "backbone"])
def test_export_rejects_incompatible_checkpoint(tiny_checkpoint: tuple, tmp_path: Path, invalid: str) -> None:
    """拒绝会改变原始结构的配置、权重和隐藏投影。

    Args:
        tiny_checkpoint: 可受控修改的原始小检查点。
        tmp_path: 用于确认失败时未发布错误主干的目录。
        invalid: 当前用例破坏的检查点约束。
    """
    checkpoint, raw, tensors = tiny_checkpoint
    if invalid == "shape":
        tensors["lm_head.weight"] = torch.zeros((5, 8), dtype=torch.bfloat16)
    elif invalid == "dtype":
        tensors["lm_head.weight"] = tensors["lm_head.weight"].half()
    elif invalid == "projector":
        tensors["depth_decoder.model.backbone_hidden_state_projector.weight"] = torch.eye(8, dtype=torch.bfloat16)
    elif invalid == "codebooks":
        raw["num_codebooks"] = 8
    else:
        raw["backbone_config"] = None
    safetensors.save_file(tensors, checkpoint / "model.safetensors")
    (checkpoint / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        export_generation_gguf(checkpoint, output)
    assert not (output / "backbone.gguf").exists()
    assert not list(output.glob("*.partial"))
