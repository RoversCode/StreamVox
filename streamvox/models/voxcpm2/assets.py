from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedding_runtime import EmbeddingResourceMetadata
from .lightweight_ops import LinearLayerRuntime, ScalarQuantizationLayerRuntime, StopPredictorRuntime


@dataclass(frozen=True, slots=True)
class LinearAssetSpec:
    """
    numpy 线性层权重路径。

    参数:
        weight_path: 线性层权重矩阵路径。
        bias_path: 线性层偏置路径；无偏置层使用 None。

    返回:
        只作为资产定位结构，不主动返回推理结果。

    边界异常:
        路径存在性由加载函数统一检查，便于错误信息包含资产名称。
    """

    weight_path: Path
    bias_path: Path | None


@dataclass(frozen=True, slots=True)
class FsqAssetSpec:
    """FSQ 轻量层资产规格，包含输入投影、输出投影和量化缩放系数。"""

    in_proj: LinearAssetSpec
    out_proj: LinearAssetSpec
    scale: float


@dataclass(frozen=True, slots=True)
class StopPredictorAssetSpec:
    """停止预测头资产规格，包含前置投影和最终分类头。"""

    stop_proj: LinearAssetSpec
    stop_head: LinearAssetSpec


@dataclass(frozen=True, slots=True)
class VoxCPM2AssetManifest:
    """
    VoxCPM2-GGUF runtime 所需资产清单。

    参数:
        ckpt_root: 标准 bundle 的 ckpt 目录。
        tokenizer_path: tokenizer 文件目录。
        embedding_metadata: 本地 embedding 元数据。
        base_lm_gguf_path: base LM GGUF 权重。
        residual_lm_gguf_path: residual LM GGUF 权重。
        locenc_onnx_path: LocEnc ONNX 权重；当前 bundle 使用 FP32 导出版本。
        locdit_onnx_path: LocDiT estimator ONNX 权重；当前 bundle 使用 FP32 导出版本。
        audio_encoder_onnx_path: AudioVAE encoder ONNX 权重；当前 bundle 仍使用 FP16 导出版本。
        audio_decoder_stateful_onnx_path: AudioVAE stateful decoder ONNX 权重；当前 bundle 仍使用 FP16 导出版本。
        audio_vae_export_metadata_path: stateful decoder 导出元数据。

    返回:
        该结构只做路径和常量聚合，runtime 按字段加载实际模块。

    边界异常:
        缺失文件会在 `load_voxcpm2_asset_manifest` 中统一抛出 FileNotFoundError。
    """

    ckpt_root: Path
    tokenizer_path: Path
    embedding_metadata: EmbeddingResourceMetadata
    base_lm_gguf_path: Path
    residual_lm_gguf_path: Path
    locenc_onnx_path: Path
    locdit_onnx_path: Path
    audio_encoder_onnx_path: Path
    audio_decoder_stateful_onnx_path: Path
    audio_vae_export_metadata_path: Path
    patch_size: int
    latent_dim: int
    chunk_size: int
    decode_chunk_size: int
    audio_encode_sample_rate: int
    audio_output_sample_rate: int
    audio_start_token: int
    ref_audio_start_token: int
    ref_audio_end_token: int
    enc_to_lm_proj: LinearAssetSpec
    lm_to_dit_proj: LinearAssetSpec
    res_to_dit_proj: LinearAssetSpec
    fusion_concat_proj: LinearAssetSpec
    fsq_layer: FsqAssetSpec
    stop_predictor: StopPredictorAssetSpec


def load_voxcpm2_asset_manifest(ckpt_root: str | Path) -> VoxCPM2AssetManifest:
    """
    从标准 ckpt 目录解析 VoxCPM2-GGUF 运行资产。

    参数:
        ckpt_root: `ckpts/voxcpm2-gguf/ckpt` 目录。

    返回:
        返回 runtime 可直接消费的资产清单。

    边界异常:
        当任一运行期权重、tokenizer、metadata 缺失时抛出 FileNotFoundError。
    """

    root = Path(ckpt_root).resolve()
    tokenizer_dir = root / "tokenizer"
    config = json.loads((tokenizer_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_json = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    audio_metadata_path = root / "audio_vae_metadata.json"
    audio_metadata = json.loads(audio_metadata_path.read_text(encoding="utf-8"))

    # 关键变量：本地 embedding 元数据必须指向当前 bundle 内的 npy，不能保留参考工程绝对路径。
    embedding_json = root / "embedding" / "base_lm_embed_tokens.json"
    embedding_payload = json.loads(embedding_json.read_text(encoding="utf-8"))
    embedding_payload["weight_path"] = str((root / "embedding" / "base_lm_embed_tokens.npy").resolve())

    manifest = VoxCPM2AssetManifest(
        ckpt_root=root,
        tokenizer_path=tokenizer_dir,
        embedding_metadata=EmbeddingResourceMetadata(**embedding_payload),
        base_lm_gguf_path=root / "base_lm_q8_0.gguf",
        residual_lm_gguf_path=root / "residual_lm_q8_0.gguf",
        # LocEnc / LocDiT estimator 当前已经切回 FP32，但文件名本身不带精度后缀；
        # 这里继续按固定文件名解析，避免把“精度变化”错误建模成“路径协议变化”。
        locenc_onnx_path=root / "locenc.onnx",
        locdit_onnx_path=root / "locdit_estimator.onnx",
        audio_encoder_onnx_path=root / "audio_vae_encoder_fp16.onnx",
        audio_decoder_stateful_onnx_path=root / "audio_vae_decoder_stateful_fp16.onnx",
        audio_vae_export_metadata_path=audio_metadata_path,
        patch_size=int(config["patch_size"]),
        latent_dim=int(config["audio_vae_config"]["latent_dim"]),
        chunk_size=int(audio_metadata["chunk_size"]),
        decode_chunk_size=int(audio_metadata["decode_chunk_size"]),
        audio_encode_sample_rate=int(config["audio_vae_config"]["sample_rate"]),
        audio_output_sample_rate=int(config["audio_vae_config"]["out_sample_rate"]),
        audio_start_token=int(tokenizer_json["model"]["vocab"]["<|audio_start|>"]),
        ref_audio_start_token=int(tokenizer_json["model"]["vocab"]["<|ref_audio_start|>"]),
        ref_audio_end_token=int(tokenizer_json["model"]["vocab"]["<|ref_audio_end|>"]),
        enc_to_lm_proj=_linear(root, "enc_to_lm_proj"),
        lm_to_dit_proj=_linear(root, "lm_to_dit_proj"),
        res_to_dit_proj=_linear(root, "res_to_dit_proj"),
        fusion_concat_proj=_linear(root, "fusion_concat_proj"),
        fsq_layer=FsqAssetSpec(
            in_proj=_linear(root, "fsq_layer_in_proj"),
            out_proj=_linear(root, "fsq_layer_out_proj"),
            scale=float(config["scalar_quantization_scale"]),
        ),
        stop_predictor=StopPredictorAssetSpec(
            stop_proj=_linear(root, "stop_proj"),
            stop_head=_linear(root, "stop_head", has_bias=False),
        ),
    )
    _validate_manifest_files(manifest)
    return manifest


def load_linear_runtime(spec: LinearAssetSpec) -> LinearLayerRuntime:
    """
    加载 numpy 线性层运行时。

    参数:
        spec: 线性层权重和偏置路径。

    返回:
        返回可直接执行 forward 的 LinearLayerRuntime。

    边界异常:
        np.load 在文件缺失或损坏时会抛出对应异常。
    """

    weight = np.load(spec.weight_path, mmap_mode="r")
    bias = None if spec.bias_path is None else np.load(spec.bias_path, mmap_mode="r")
    return LinearLayerRuntime(weight=np.asarray(weight, dtype=np.float32), bias=None if bias is None else np.asarray(bias, dtype=np.float32))


def load_fsq_runtime(spec: FsqAssetSpec) -> ScalarQuantizationLayerRuntime:
    """加载 FSQ 轻量运行时，供 base hidden 转 residual 条件使用。"""

    return ScalarQuantizationLayerRuntime(
        in_proj=load_linear_runtime(spec.in_proj),
        out_proj=load_linear_runtime(spec.out_proj),
        scale=float(spec.scale),
    )


def load_stop_predictor_runtime(spec: StopPredictorAssetSpec) -> StopPredictorRuntime:
    """加载停止预测头运行时，用于控制 latent patch 生成终止。"""

    return StopPredictorRuntime(
        stop_proj=load_linear_runtime(spec.stop_proj),
        stop_head=load_linear_runtime(spec.stop_head),
    )


def _linear(root: Path, name: str, *, has_bias: bool = True) -> LinearAssetSpec:
    """按 `ckpt/asset/<name>.*.npy` 约定构造线性层资产路径。"""

    asset_root = root / "asset"
    return LinearAssetSpec(
        weight_path=asset_root / f"{name}.weight.npy",
        bias_path=(asset_root / f"{name}.bias.npy") if has_bias else None,
    )


def _validate_manifest_files(manifest: VoxCPM2AssetManifest) -> None:
    """集中校验运行期文件存在性，避免 runtime 初始化到一半才失败。"""

    required = [
        manifest.base_lm_gguf_path,
        manifest.residual_lm_gguf_path,
        manifest.locenc_onnx_path,
        manifest.locdit_onnx_path,
        manifest.audio_encoder_onnx_path,
        manifest.audio_decoder_stateful_onnx_path,
        manifest.audio_vae_export_metadata_path,
        Path(manifest.embedding_metadata.weight_path),
        manifest.tokenizer_path / "tokenizer.json",
    ]
    for linear in (
        manifest.enc_to_lm_proj,
        manifest.lm_to_dit_proj,
        manifest.res_to_dit_proj,
        manifest.fusion_concat_proj,
        manifest.fsq_layer.in_proj,
        manifest.fsq_layer.out_proj,
        manifest.stop_predictor.stop_proj,
        manifest.stop_predictor.stop_head,
    ):
        required.append(linear.weight_path)
        if linear.bias_path is not None:
            required.append(linear.bias_path)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing voxcpm2-gguf assets: {missing}")
