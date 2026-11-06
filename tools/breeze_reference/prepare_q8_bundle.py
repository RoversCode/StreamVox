"""从已验收的原始 BF16 资产和独立 Q8 对照报告派生完整模型包。"""

from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import stat
from typing import Any

import gguf

from streamvox.model_bundle import is_runtime_manifest_asset, sha256_file, validate_local_bundle, write_local_manifest

from .q8_audit import LOSSY_TASK_AUDIT_SCOPE, validate_q8_audit


_MODEL = "breeze-tts-2-gguf"
_COMPONENTS = ("backbone.gguf", "depth.gguf")
_Q8_NOTICE = (
    "StreamVox Q8 派生说明。本包使用官方 llama.cpp 的 Q8_0 默认策略量化主干和深度解码器，"
    "二维生成权重及深度投影嵌入表保存为 Q8_0，归一化、位置频率及其余 ONNX 组件保留 FP32。"
    "完整保留十六码本、十五个深度输出头、快速文本资产和流式状态。"
    "量化误差的独立对照记录见 ckpt/q8-validation.json，原始 BF16 导出及验收来源见 "
    "ckpt/source-export-report.json。模型权重与派生模型继续遵守原始研究及非商业许可证。"
)


def _gguf_descriptor(path: Path) -> dict[str, Any]:
    """只读取模型元数据和张量描述，避免把权重载入内存。

    Args:
        path: 完整的本地 GGUF 文件。
    Returns:
        文件类型、架构元数据以及逐张量名称、形状和存储类型。
    """
    reader = gguf.GGUFReader(path)
    file_type = reader.get_field("general.file_type")
    if file_type is None:
        raise ValueError(f"GGUF 缺少文件精度声明：{path.name}")
    return {
        "file_type": int(file_type.contents()),
        "metadata": {
            name: field.contents() for name, field in reader.fields.items()
            if not name.startswith("GGUF.") and name not in {"general.file_type", "general.quantization_version"}
        },
        "tensors": [
            {"name": tensor.name, "shape": [int(size) for size in tensor.shape], "dtype": tensor.tensor_type.name}
            for tensor in reader.tensors
        ],
    }


def _portable_report(value: Any, label: str) -> Any:
    """为公开报告移除本机绝对路径，保留全部数值和文件摘要。

    Args:
        value: 报告中的字典、列表或标量。
        label: 用于说明路径所属来源的相对标签。
    Returns:
        使用相对来源标签的独立报告值。
    """
    if isinstance(value, dict):
        return {key: _portable_report(item, label) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_report(item, label) for item in value]
    if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
        return f"{label}/{PureWindowsPath(value).name}"
    return value


def _validated_q8_models(
    originals: dict[str, Path], quantized_dir: Path, validation_report: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """绑定完整对照报告与实际文件，并审计量化前后的模型布局。

    Args:
        originals: 通过完整源清单校验的资产路径。
        quantized_dir: 包含完整量化权重的独立目录。
        validation_report: 独立执行的完整量化对照报告。
    Returns:
        实际量化描述、候选文件和移除本机路径后的验证报告。
    """
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    validate_q8_audit(report)
    descriptors, candidate_files = {}, {}
    for filename in _COMPONENTS:
        original, candidate = originals[f"ckpt/{filename}"], quantized_dir / filename
        candidate = candidate.resolve(strict=True)
        if not candidate.is_relative_to(quantized_dir) or not candidate.is_file():
            raise ValueError("量化文件必须位于候选目录内。")
        for kind, path in (("reference", original), ("candidate", candidate)):
            record = report.get(kind, {}).get("files", {}).get(filename, {})
            if record.get("sha256") != sha256_file(path) or record.get("bytes") != path.stat().st_size:
                raise ValueError(f"Q8 对照报告文件摘要不符：{kind}/{filename}")
        baseline, quantized = _gguf_descriptor(original), _gguf_descriptor(candidate)
        architecture = "qwen3" if filename == "backbone.gguf" else "llama"
        if baseline["metadata"].get("general.architecture") != architecture:
            raise ValueError(f"源 GGUF 架构不符：{filename}")
        if baseline["file_type"] != int(gguf.LlamaFileType.MOSTLY_BF16):
            raise ValueError("源 GGUF 必须保存原始 BF16 权重。")
        if quantized["file_type"] != int(gguf.LlamaFileType.MOSTLY_Q8_0):
            raise ValueError("候选 GGUF 必须声明官方 Q8_0 文件类型。")
        baseline_shapes = {tensor["name"]: tensor["shape"] for tensor in baseline["tensors"]}
        candidate_shapes = {tensor["name"]: tensor["shape"] for tensor in quantized["tensors"]}
        if (
            not baseline_shapes or baseline_shapes != candidate_shapes or baseline["metadata"] != quantized["metadata"]
            or len(candidate_shapes) != len(quantized["tensors"])
        ):
            raise ValueError(f"量化改变了模型架构、张量名称或形状：{filename}")
        if any(tensor["dtype"] not in {"BF16", "F32"} for tensor in baseline["tensors"]):
            raise ValueError("源 GGUF 含有已经量化的权重。")
        if not any(tensor["dtype"] == "Q8_0" for tensor in quantized["tensors"]) or any(
            tensor["dtype"] != ("Q8_0" if len(tensor["shape"]) >= 2 else "F32") for tensor in quantized["tensors"]
        ):
            raise ValueError("候选 GGUF 必须使用默认二维 Q8_0 及一维 F32 策略。")
        descriptors[filename.removesuffix(".gguf")] = quantized
        candidate_files[filename] = candidate
    portable_validation = _portable_report(report, "validation")
    for kind, label in (("reference", "source-bf16"), ("candidate", "candidate-q8")):
        portable_validation[kind]["directory"] = label
        for filename, entry in portable_validation[kind]["files"].items():
            entry["path"] = f"{label}/{filename}"
    return descriptors, candidate_files, portable_validation


def _write_q8_metadata(
    output: Path, source_report: dict[str, Any], source_report_sha256: str,
    descriptors: dict[str, Any], validation: dict[str, Any],
) -> list[Path]:
    """写入准确的派生精度、来源、数值记录及许可声明。

    Args:
        output: 当前构建独占的输出目录。
        source_report: 原始 BF16 导出和验收报告。
        source_report_sha256: 已通过源清单校验的原始报告摘要。
        descriptors: 量化文件的实际张量描述。
        validation: 已移除本机路径的独立量化对照报告。
    Returns:
        需要加入新资产清单的来源和验收报告路径。
    """
    checkpoint = output / "ckpt"
    provenance_path = checkpoint / "source-export-report.json"
    validation_path = checkpoint / "q8-validation.json"
    derived = _portable_report(source_report, "source-bf16")
    provenance_path.write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    derived["precision"], derived["quantized"] = "q8_0", True
    derived["source_provenance"] = {
        "report": provenance_path.name, "original_report_sha256": source_report_sha256,
        "precision": "bfloat16", "validation_scope": "原始 BF16 模型的既有验收，仅用于记录派生来源。",
    }
    file_records = {
        name: {key: validation["candidate"]["files"][name][key] for key in ("bytes", "sha256")}
        for name in _COMPONENTS
    }
    conditioner = source_report.get("generation", {}).get("export", {}).get("depth_conditioner", {})
    scope = validation.get("basis", "q8_0_strict_numerical_v1")
    strict_passed = scope != LOSSY_TASK_AUDIT_SCOPE
    derived["quantized_generation_validation_scope"] = scope
    derived["numerical_equivalence_claimed"] = False
    derived["strict_numerical_gates_passed"] = strict_passed
    derived["numerical_error_bounds_passed"] = strict_passed
    derived["generation"] = {
        "precision": "q8_0", "validated": True,
        "export": {
            "precision": "q8_0", "quantized": True, "num_codebooks": 16, "num_depth_heads": 15,
            "files": {name: {"path": name, **record} for name, record in file_records.items()},
            "components": descriptors,
            "depth_conditioner": _portable_report(conditioner, "source-bf16"),
        },
        "validation": {
            "report": validation_path.name, "passed": True, "quantization": "q8_0", "scope": scope,
            "numerical_equivalence_claimed": False, "strict_numerical_gates_passed": strict_passed,
            "numerical_error_bounds_passed": strict_passed,
        },
    }
    derived["files"].update(file_records)
    for path in (provenance_path, validation_path):
        derived["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (checkpoint / "export-report.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = output / "README.md"
    original_readme = readme.read_text(encoding="utf-8")
    notice_text = _Q8_NOTICE
    if scope == LOSSY_TASK_AUDIT_SCOPE:
        notice_text += (
            " 本次发布采用有损任务验收，覆盖完整真实轨迹的结构检查、多种子自然结束、固定内容复核和用户试听。"
            "原始随机输入及真实轨迹的严格数值审计保留失败状态，全部超界指标和既有门限随包公开。"
        )
    updated_readme, replacements = re.subn(r"(?m)^转换修改：[^\r\n]*", notice_text, original_readme, count=1)
    if not replacements:
        updated_readme = notice_text + "\n\n" + original_readme
    readme.write_text(updated_readme, encoding="utf-8")
    notice = output / "NOTICE"
    notice.write_text(notice_text + "\n\n" + notice.read_text(encoding="utf-8"), encoding="utf-8")
    return [provenance_path, validation_path]


def prepare_q8_bundle(source: Path, output: Path, quantized_dir: Path, validation_report: Path) -> Path:
    """在完整校验来源及量化误差报告后，安全组装独立的 Q8 模型包。

    Args:
        source: 已通过原生数值验收的完整原始 BF16 模型包。
        output: 尚不存在、位于源模型之外的输出目录。
        quantized_dir: 包含官方量化后的两个完整 GGUF 文件的目录。
        validation_report: 对上述源文件与候选文件执行完整 Q8 对照的成功报告。
    Returns:
        已包含全部运行资产、许可和独立验收记录的新清单路径。
    """
    source, output = source.resolve(), output.resolve()
    quantized_dir = quantized_dir.resolve(strict=True)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Q8 模型包必须位于原模型目录之外。")
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    validate_local_bundle(source, _MODEL, True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    originals: dict[str, Path] = {}
    for item in manifest["assets"]:
        original = (source / item["path"]).resolve(strict=True)
        if not original.is_relative_to(source) or not original.is_file() or item["path"] in originals:
            raise ValueError("源清单包含重复、越界或非普通文件资产。")
        if not is_runtime_manifest_asset(item["path"]) and sha256_file(original) != item["sha256"]:
            raise ValueError(f"源许可或归属文件摘要不符：{item['path']}")
        originals[item["path"]] = original
    required = {
        "LICENSE", "MODEL_LICENSE", "NOTICE", "README.md", "ckpt/config.json", "ckpt/tokenizer.json",
        "ckpt/codec_metadata.json", "ckpt/export-report.json", "ckpt/backbone.gguf", "ckpt/depth.gguf",
        "ckpt/depth_conditioner.onnx", "ckpt/text_encoder.onnx", "ckpt/audio_embedding.onnx",
        "ckpt/audio_encoder.onnx", "ckpt/decoder_step.onnx",
        "ckpt/text_encoder_trunk.onnx", "ckpt/text_encoder_lookup.onnx", "ckpt/text_encoder_lookup.json",
    }
    if not required.issubset(originals):
        raise ValueError("源模型包缺少完整运行资产或原始许可证。")
    source_report = json.loads(originals["ckpt/export-report.json"].read_text(encoding="utf-8"))
    generation = source_report.get("generation", {})
    if (
        source_report.get("tiny") is not False or source_report.get("quantized") is not False
        or source_report.get("precision") != "bfloat16" or generation.get("precision") != "bfloat16"
        or generation.get("validated") is not True
    ):
        raise ValueError("Q8 模型包必须派生自已验收的完整原始 BF16 模型。")
    declared_hashes = {item["path"]: item["sha256"] for item in manifest["assets"]}
    if not set(_COMPONENTS).issubset(source_report.get("files", {})):
        raise ValueError("源导出报告缺少完整生成组件。")
    for name, entry in source_report.get("files", {}).items():
        if entry.get("sha256") != declared_hashes.get(f"ckpt/{name}"):
            raise ValueError(f"源导出报告与资产清单不一致：{name}")
    descriptors, candidate_files, portable_validation = _validated_q8_models(originals, quantized_dir, validation_report)
    output.mkdir(parents=True, exist_ok=False)
    owned_output = output.lstat()
    try:
        assets = []
        for name, original in originals.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            original = candidate_files.get(name.removeprefix("ckpt/"), original)
            # 所有待更新的文本独立复制，避免写入时影响源模型的硬链接。
            if destination.suffix == ".json" or name in {"README.md", "NOTICE"}:
                shutil.copy2(original, destination)
            else:
                try:
                    destination.hardlink_to(original)
                except OSError as error:
                    if (
                        error.errno not in {errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
                        and getattr(error, "winerror", None) not in {1, 17, 50}
                    ):
                        raise
                    shutil.copy2(original, destination)
            assets.append(destination)
        created_reports = _write_q8_metadata(
            output, source_report, declared_hashes["ckpt/export-report.json"], descriptors, portable_validation,
        )
        sources = list(manifest.get("sources", []))
        if "local" not in sources and "modelscope" not in sources:
            sources.append("local")
        manifest_path = write_local_manifest(
            _MODEL, output, assets + created_reports, sources=sources, adapter=manifest["adapter"],
        )
        written = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hashes = {item["path"]: item["sha256"] for item in written["assets"]}
        expected_hashes = dict(declared_hashes)
        for filename in _COMPONENTS:
            expected_hashes[f"ckpt/{filename}"] = portable_validation["candidate"]["files"][filename]["sha256"]
        # 清单生成期间再次核对来源绑定，避免并发权重变化形成看似完整的模型包。
        for name, digest in expected_hashes.items():
            if name not in {"README.md", "NOTICE", "ckpt/export-report.json"} and actual_hashes.get(name) != digest:
                raise ValueError(f"派生期间资产发生变化：{name}")
        return manifest_path
    except BaseException as error:
        try:
            current = output.lstat()
            # 只清理由本次创建且身份未变的目录，防止并发路径替换误删用户数据。
            if (
                output.resolve(strict=True) != output or output == source or output.is_relative_to(source)
                or source.is_relative_to(output) or output.is_symlink() or output.is_junction()
                or not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != (owned_output.st_dev, owned_output.st_ino)
            ):
                raise RuntimeError("输出目录路径或身份已变化，已停止半成品清理。")
            shutil.rmtree(output)
        except BaseException as cleanup_error:
            error.add_note(f"Q8 模型包半成品清理失败：{cleanup_error}")
        raise


def main() -> None:
    """读取源模型、量化权重和独立报告，组装可分发的 Q8 模型包。

    Returns:
        无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare_q8_bundle(arguments.source, arguments.output, arguments.quantized_dir, arguments.validation_report))


if __name__ == "__main__":
    main()
