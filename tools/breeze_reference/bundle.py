"""核对本地模型资产并写入许可证、导出报告及可审计的 bundle 清单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from .factory import CHECKPOINT_REPOSITORY, CHECKPOINT_REVISION, SOURCE_COMMIT


def export_isolated_graph(
    graph: torch.nn.Module, arguments: tuple[torch.Tensor, ...], path: Path, **options: Any,
) -> None:
    """隔离每个 FP32 图的外部参数，同时保留根目录中的稳定 ONNX 路径。

    Args:
        graph: 已加载原始权重的导出模块。
        arguments: 当前图的参考输入。
        path: 最终 ONNX 本体路径。
        options: 传递给 Torch ONNX 导出器的算子集与图接口选项。

    Returns:
        无。
    """
    import onnx

    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in (*graph.parameters(), *graph.buffers(), *arguments)):
        raise ValueError("ONNX 组件只允许原始权重精确提升后的 FP32 计算")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = path.parent / f"{path.stem}.weights"
    # 只清理该图专属的生成目录；链接和越界路径不能成为递归删除目标。
    if parameters.is_symlink() or parameters.resolve() != parameters:
        raise ValueError(f"图参数目录必须直接位于导出目录内：{parameters}")
    if parameters.exists():
        if not parameters.is_dir():
            raise ValueError(f"图参数路径必须为目录：{parameters}")
        shutil.rmtree(parameters)
    parameters.mkdir()
    temporary = parameters / path.name
    try:
        # 大图导出器按张量名写文件，必须在独立目录内完成，不能先写到共享根目录。
        torch.onnx.export(graph, arguments, str(temporary), **options)
        model = onnx.load(str(temporary), load_external_data=False)
        for tensor in onnx.external_data_helper._get_all_tensors(model):
            for field in tensor.external_data:
                if field.key != "location":
                    continue
                external = (parameters / field.value).resolve(strict=True)
                if not external.is_relative_to(parameters):
                    raise ValueError(f"ONNX 外部参数文件越过图参数目录：{field.value}")
                field.value = external.relative_to(path.parent).as_posix()
        # 不加载外部参数，避免重写时把大模型权重重新读入内存或内联到本体。
        onnx.save_model(model, str(path))
    finally:
        temporary.unlink(missing_ok=True)
        if not any(parameters.iterdir()):
            parameters.rmdir()


def sha256(path: Path) -> str:
    """流式计算导出文件摘要。

    Args:
        path: 已存在的文件。

    Returns:
        小写十六进制 SHA256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_files(path: Path) -> set[Path]:
    """枚举模型本体及其引用的外部参数文件。

    Args:
        path: 已生成的 ONNX 文件。

    Returns:
        位于导出目录内的全部实际依赖文件。
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    files = {path}
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        for field in tensor.external_data:
            if field.key != "location":
                continue
            external = (path.parent / field.value).resolve(strict=True)
            if not external.is_relative_to(path.parent.resolve()):
                raise ValueError(f"ONNX 外部参数文件越过导出目录：{field.value}")
            files.add(external)
    return files


def validate_checkpoint_assets(checkpoint: Path, audio_tokenizer: Path | None) -> Path:
    """在分配模型前核对完整检查点的分词器与独立许可证。

    Args:
        checkpoint: 本地完整 Breeze 检查点目录。
        audio_tokenizer: 可选的本地音频分词器目录。

    Returns:
        通过完整性检查的本地音频分词器路径。
    """
    tokenizer_json = checkpoint / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ValueError(f"完整 bundle 必须包含文本分词器：{tokenizer_json}")
    if not (checkpoint / "LICENSE").is_file():
        raise ValueError("完整 bundle 必须保留检查点的权重许可证 LICENSE。")
    audio_tokenizer_path = audio_tokenizer or checkpoint / "audio_tokenizer"
    if not (audio_tokenizer_path / "config.json").is_file():
        raise ValueError(f"完整 bundle 必须包含本地音频分词器：{audio_tokenizer_path}")
    return audio_tokenizer_path


def copy_tokenizer_assets(checkpoint: Path, destination: Path) -> None:
    """复制原始文本分词器及其特殊符号配置。

    Args:
        checkpoint: 已检查完整性的本地检查点目录。
        destination: 模型 bundle 的 ckpt 目录。
    """
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = checkpoint / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)


def write_bundle(
    bundle_root: Path, raw: dict[str, Any], report: dict[str, Any], exported_files: set[Path],
    *, source_dir: Path, checkpoint: Path | None, tiny: bool,
) -> None:
    """在所有数值验证完成后收口可分发的模型资产与来源清单。

    Args:
        bundle_root: 模型 bundle 根目录。
        raw: 原始模型配置。
        report: 已验证的组件报告；将补充实际文件摘要。
        exported_files: 实际参与推理的 GGUF、ONNX、外部参数及分词器资产。
        source_dir: 通过固定提交检查的上游源码目录。
        checkpoint: 完整模型的本地检查点目录；随机小模型为无。
        tiny: 是否仅分发随机小模型的验证产物。
    """
    destination = bundle_root / "ckpt"
    if report.get("generation", {}).get("validated") is not True:
        raise ValueError("GGUF 生成组件必须完成数值验证后才能写入清单")
    if not tiny and checkpoint is None:
        raise ValueError("完整 bundle 必须提供原始模型检查点")
    if any(not path.resolve().is_relative_to(destination.resolve()) for path in exported_files):
        raise ValueError("导出资产必须位于当前 bundle 的 ckpt 目录内")
    # 只在全部组件验证成功后写入完成报告，避免部分产物被误认为完整导出。
    report["files"] = {
        path.relative_to(destination).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(exported_files)
    }
    (destination / "config.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "export-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    legal_files: list[Path] = []
    upstream_license = source_dir / "LICENSE"
    if not upstream_license.is_file():
        raise ValueError(f"上游源码缺少分发所需许可证：{upstream_license}")
    shutil.copy2(upstream_license, bundle_root / "LICENSE")
    legal_files.append(bundle_root / "LICENSE")
    if not tiny and (checkpoint / "LICENSE").is_file():
        shutil.copy2(checkpoint / "LICENSE", bundle_root / "MODEL_LICENSE")
        legal_files.append(bundle_root / "MODEL_LICENSE")
    readme_source = source_dir / "README.md"
    if not tiny and (checkpoint / "README.md").is_file():
        readme_source = checkpoint / "README.md"
    if readme_source.is_file():
        shutil.copy2(readme_source, bundle_root / "README.md")
        legal_files.append(bundle_root / "README.md")
    notices = [
        "Breeze 源码参考提交：" + SOURCE_COMMIT,
        "Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.",
        "Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.",
    ]
    for directory, prefix in ((source_dir, "SOURCE"), (checkpoint, "MODEL")):
        if directory is not None and (directory / "NOTICE").is_file():
            target = bundle_root / f"{prefix}_NOTICE"
            shutil.copy2(directory / "NOTICE", target)
            legal_files.append(target)
            notices.append(target.read_text(encoding="utf-8"))
    if not tiny:
        # 权重许可要求衍生文件明确保留独立许可、原始归属与转换事实。
        notices.append("Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial License Agreement. "
                       "Copyright (c) 2026 RESONIA, INC. All Rights Reserved.")
        attribution = "Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only."
        modifications = (
            "转换修改：主干与深度解码器导出为 GGUF，文本、音频嵌入、条件投影及音频编解码器使用 FP32 ONNX；"
            "生成权重精确提升到 FP32 或保留原始 BF16，深度嵌入采用等价投影 E'=E@P.T，完整保留十六码本与流式状态；"
            "未量化或微调。"
        )
        original_readme = (bundle_root / "README.md").read_text(encoding="utf-8") if (bundle_root / "README.md").is_file() else ""
        (bundle_root / "README.md").write_text(attribution + "\n\n" + modifications + "\n\n" + original_readme, encoding="utf-8")
        if bundle_root / "README.md" not in legal_files:
            legal_files.append(bundle_root / "README.md")
    (bundle_root / "NOTICE").write_text("\n\n".join(notices) + "\n", encoding="utf-8")
    legal_files.append(bundle_root / "NOTICE")
    # 清单只纳入本轮生成并校验的文件，避免已有目录中的旧后端资产混入。
    assets = sorted(exported_files | {destination / "config.json", destination / "export-report.json"}) + legal_files
    manifest = {
        "model": "breeze-tts-2-gguf", "adapter": "BreezeTTS2GGUFAdapter",
        "sources": [
            f"https://github.com/breezeblue-ai/breeze-tts/tree/{SOURCE_COMMIT}",
            f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/tree/{CHECKPOINT_REVISION}",
        ],
        "assets": [
            {"path": path.relative_to(bundle_root).as_posix(), "sha256": sha256(path)}
            for path in sorted(assets)
        ],
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
