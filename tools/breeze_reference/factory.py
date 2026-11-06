"""构建固定来源的 Breeze 组件，声明其动态输入输出和参考缓存。"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch import nn

from .graphs import AudioEmbeddingGraph, BackboneGraph, DepthGraph, TextEncoderGraph


SOURCE_COMMIT = "daa6cb2efe5471f09369229f322d20bb956cb3c3"
CHECKPOINT_REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"
CHECKPOINT_REPOSITORY = "BreezeBlue/breeze-tts-2"
GRAPH_NAMES = ("text_encoder", "audio_embedding", "backbone", "depth")


def source_modules(source: Path) -> tuple[Any, Any, Any]:
    """校验固定上游版本后导入本地参考实现。

    Args:
        source: 上游 Git 仓库根目录。

    Returns:
        上游模型、配置、文本兼容模块。
    """
    source = source.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"上游源码提交必须为 {SOURCE_COMMIT}，实际为 {commit}")
    # 文本归档可能使用不同换行或未建立索引；直接核对实际导入文件的内容。
    imported_files = (
        "breeze.py", "breeze_config.py", "breeze_base_config.py",
        "t5gemma2_compat.py", "breeze_backbone_factory.py",
        "generation_breeze.py", "logits_process.py",
    )
    for filename in imported_files:
        relative = f"models/{filename}"
        expected = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{SOURCE_COMMIT}:{relative}"],
            text=True, encoding="utf-8",
        )
        if (source / relative).read_text(encoding="utf-8") != expected:
            raise ValueError(f"上游源码内容与固定提交不符：{relative}")
    existing = sys.modules.get("models")
    if existing is not None:
        locations = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
        if source / "models" not in locations:
            raise RuntimeError("当前进程已导入另一份 models 包；请在独立进程运行导出工具")
    sys.path.insert(0, str(source))
    return tuple(
        importlib.import_module(name)
        for name in ("models.breeze", "models.breeze_config", "models.t5gemma2_compat")
    )


def tiny_config() -> dict[str, Any]:
    """提供覆盖真实注意力、码本偏移和位置编码的随机小模型配置。

    Returns:
        保留十六码本、Qwen3 查询键归一化与双向局部文本层的配置。
    """
    backbone = {
        "model_type": "qwen3", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "max_position_embeddings": 128, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "vocab_size": 64, "attention_bias": False,
        "hidden_act": "silu", "use_sliding_window": False,
    }
    depth = {
        "model_type": "breeze_depth_decoder_model", "hidden_size": 24,
        "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 3,
        "num_key_value_heads": 1, "head_dim": 8, "num_codebooks": 16,
        "vocab_size": 31, "audio_embed_size": 32, "backbone_hidden_size": 32,
        "max_position_embeddings": 33, "rms_norm_eps": 1e-5, "rope_theta": 500000,
        "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
                         "high_freq_factor": 0.0078125, "low_freq_factor": 0.001953125,
                         "original_max_position_embeddings": 16},
    }
    return {
        "model_type": "breeze", "backbone_model_type": "qwen3",
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "num_codebooks": 16, "audio_num_codebooks": 16, "vocab_size": 31,
        "audio_vocab_size": 31, "audio_embed_size": 32, "text_vocab_size": 64,
        "codebook_eos_token_id": 0, "codebook_pad_token_id": 30,
        "audio_token_id": 60, "audio_eos_token_id": 61,
        "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1,
        "backbone_config": backbone, "depth_decoder_config": depth,
        "text_encoder_proj_type": "linear", "tie_codebooks_embeddings": True,
        "text_encoder_config": {
            "model_type": "t5gemma2_text", "vocab_size": 64, "hidden_size": 24,
            "intermediate_size": 48, "num_hidden_layers": 2,
            "num_attention_heads": 3, "num_key_value_heads": 1, "head_dim": 8,
            "query_pre_attn_scalar": 8, "sliding_window": 4,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 128, "dropout_rate": 0.0,
            "attention_dropout": 0.0, "rms_norm_eps": 1e-6, "eoi_token_index": 63,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "linear", "rope_theta": 1000000.0,
                                   "factor": 8.0},
            },
        },
    }


def build_graph(
    name: str, raw: dict[str, Any], modules: tuple[Any, Any, Any], *, meta: bool,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """只构建指定推理图需要的上游组件。

    Args:
        name: 图名称。
        raw: 原始 Breeze 配置。
        modules: 已校验来源的上游模块。
        meta: 是否先建立无存储参数以供逐张量加载。
        dtype: 计算精度；BF16 只用于主干与深度解码器。

    Returns:
        具有稳定输入输出接口的图模块。
    """
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("导出计算精度只允许 FP32 或 BF16")
    if dtype == torch.bfloat16 and name not in ("backbone", "depth"):
        raise ValueError("BF16 仅用于 CUDA 主干与深度解码图")
    if not isinstance(raw.get("backbone_config"), dict) or not raw["backbone_config"]:
        raise ValueError("离线导出要求检查点内嵌完整 backbone_config，不能从远端补取配置")
    breeze, configuration, text_module = modules
    config = configuration.BreezeConfig(**raw)
    config._attn_implementation = "eager"
    config.depth_decoder_config._attn_implementation = "eager"
    config.text_encoder_config._attn_implementation = "eager"
    if raw.get("text_encoder_proj_type", "linear") != "linear":
        raise ValueError("固定 Breeze2 检查点要求线性文本投影")
    if raw.get("backbone_model_type") != "qwen3":
        raise ValueError("本工具只支持固定 Breeze2 检查点的 Qwen3 主干")
    with torch.device("meta" if meta else "cpu"):
        if name == "text_encoder":
            encoder = text_module.T5Gemma2TextEncoder(config.text_encoder_config)
            projection = nn.Linear(encoder.config.hidden_size, config.hidden_size, bias=False)
            graph = TextEncoderGraph(encoder, projection)
        elif name == "audio_embedding":
            graph = AudioEmbeddingGraph(breeze.BreezeBackboneModelEmbeddings(config))
        elif name == "backbone":
            factory = importlib.import_module("models.breeze_backbone_factory")
            backbone = factory.BreezeBackboneFactory.create_backbone(config)
            # 主干图接收完成投影的嵌入，不保存重复的音频查表参数。
            del backbone.embed_tokens
            graph = BackboneGraph(backbone, nn.Linear(config.hidden_size, config.vocab_size + 1, bias=False))
        elif name == "depth":
            graph = DepthGraph(breeze.BreezeDepthDecoderForCausalLM(config.depth_decoder_config))
        else:
            raise ValueError(f"未知导出图：{name}")
    graph = graph.to(dtype=dtype).eval()
    if dtype == torch.bfloat16:
        model = graph.backbone_model if name == "backbone" else graph.depth_decoder.model
        if model.config.hidden_act != "silu":
            raise ValueError("固定 Breeze2 的 BF16 图要求原始 SiLU 激活")
    if meta or dtype == torch.bfloat16:
        # 非持久缓冲区不在检查点内，必须按原始配置重新计算，不能使用未初始化存储。
        if name == "text_encoder":
            graph.text_encoder.rotary_emb = text_module.T5Gemma2RotaryEmbedding(config.text_encoder_config)
            graph.text_encoder.embed_tokens.embed_scale = torch.tensor(config.text_encoder_config.hidden_size**0.5)
        elif name == "audio_embedding":
            graph.embedding.audio_tokens_offsets = torch.arange(config.num_codebooks) * config.vocab_size
        elif name == "backbone":
            rotary = graph.backbone_model.rotary_emb
            graph.backbone_model.rotary_emb = type(rotary)(config=rotary.config, device="cpu")
        else:
            graph.depth_decoder.model.rotary_emb = breeze.BreezeRotaryEmbedding(config.depth_decoder_config)
    return graph


def empty_cache(model: nn.Module, batch: int = 1, length: int = 0) -> tuple[torch.Tensor, ...]:
    """根据原始注意力层创建扁平键值缓存。

    Args:
        model: 原始主干或深度解码器。
        batch: 批次数。
        length: 历史长度；零用于无历史的预填充。

    Returns:
        按层交替排列的浮点键和值。
    """
    attention = model.layers[0].self_attn
    heads = attention.k_proj.out_features // attention.head_dim
    shape = (batch, heads, length, attention.head_dim)
    dtype = attention.k_proj.weight.dtype
    return tuple(torch.zeros(shape, dtype=dtype) for _ in range(len(model.layers) * 2))


def graph_inputs(name: str, graph: nn.Module, raw: dict[str, Any], batch: int = 1, length: int = 3, past_length: int = 2) -> tuple[torch.Tensor, ...]:
    """创建覆盖动态导出维度的图输入。

    Args:
        name: 图名称。
        graph: 构建完成的图。
        raw: 模型配置。
        batch: 批次数。
        length: 当前序列长度。
        past_length: 自回归历史长度。

    Returns:
        与稳定图接口顺序一致的输入元组。
    """
    if name == "text_encoder":
        ids = torch.randint(0, raw["text_encoder_config"]["vocab_size"], (batch, length))
        mask = torch.ones_like(ids)
        mask[:, -1] = 0
        return ids, mask
    if name == "audio_embedding":
        return (torch.randint(0, raw["vocab_size"], (batch, length, raw["num_codebooks"])),)
    if name == "backbone":
        return (torch.randn(batch, length, raw["hidden_size"]), *empty_cache(graph.backbone_model, batch, past_length))
    dtype = graph.depth_decoder.model.layers[0].self_attn.k_proj.weight.dtype
    return (
        torch.randint(0, raw["vocab_size"], (batch, 1)),
        torch.randn(batch, raw["hidden_size"], dtype=dtype), torch.tensor([past_length], dtype=torch.int64),
        *empty_cache(graph.depth_decoder.model, batch, past_length),
    )


def graph_signature(name: str, graph: nn.Module) -> tuple[list[str], list[str], dict[str, dict[int, str]]]:
    """声明 ONNX 接口名称和批次、序列、缓存动态维度。

    Args:
        name: 图名称。
        graph: 相应图模块。

    Returns:
        输入名称、输出名称与动态维度映射。
    """
    if name == "text_encoder":
        inputs, outputs = ["input_ids", "attention_mask"], ["embeddings"]
    elif name == "audio_embedding":
        inputs, outputs = ["codes"], ["embeddings"]
    else:
        inputs = ["embeddings"] if name == "backbone" else ["token_ids", "backbone_hidden", "position"]
        outputs = ["logits", "hidden"] if name == "backbone" else ["logits"]
        model = graph.backbone_model if name == "backbone" else graph.depth_decoder.model
        for layer in range(len(model.layers)):
            inputs.extend((f"past_key_{layer}", f"past_value_{layer}"))
            outputs.extend((f"present_key_{layer}", f"present_value_{layer}"))
    dynamic: dict[str, dict[int, str]] = {}
    for key in inputs + outputs:
        if key == "position":
            continue
        if key.startswith("past_"):
            dynamic[key] = {0: "batch", 2: "past_length"}
        elif key.startswith("present_"):
            dynamic[key] = {0: "batch", 2: "present_length"}
        elif key in ("input_ids", "attention_mask", "codes", "embeddings"):
            dynamic[key] = {0: "batch", 1: "sequence_length"}
        else:
            dynamic[key] = {0: "batch"}
    return inputs, outputs, dynamic
