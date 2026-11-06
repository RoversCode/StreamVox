"""验证 Q8 模型包的来源绑定、精度审计、许可保留及失败清理。"""

from __future__ import annotations

from copy import deepcopy
import errno
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from streamvox.model_bundle import sha256_file, validate_local_bundle, write_local_manifest
from tools.breeze_reference import prepare_q8_bundle as preparation
from tools.breeze_reference import q8_audit


@pytest.fixture
def bundle_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """准备小型完整清单，并用独立描述模拟真实模型头部。

    Args:
        tmp_path: 当前测试的独立目录。
        monkeypatch: 替换权重描述读取的工具。
    Returns:
        源模型、候选模型、报告、描述和原始内容的集合。
    """
    source, candidate, output = tmp_path / "source", tmp_path / "candidate", tmp_path / "output"
    checkpoint = source / "ckpt"
    checkpoint.mkdir(parents=True)
    candidate.mkdir()
    names = (
        "config.json", "tokenizer.json", "codec_metadata.json", "backbone.gguf", "depth.gguf",
        "depth_conditioner.onnx", "text_encoder.onnx", "audio_embedding.onnx", "audio_encoder.onnx",
        "decoder_step.onnx", "text_encoder_trunk.onnx", "text_encoder_lookup.onnx", "text_encoder_lookup.json",
        "text_encoder.weights/weight.bin",
    )
    for name in names:
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"原始资产 {name}".encode())
    for name in ("LICENSE", "MODEL_LICENSE", "NOTICE", "README.md"):
        text = f"原始许可与归属 {name}\n"
        if name == "README.md":
            text = "原始署名\n\n转换修改：原始 BF16；未量化或微调。\n\n上游模型说明\n"
        (source / name).write_text(text, encoding="utf-8")
    original_report = {
        "tiny": False, "precision": "bfloat16", "quantized": False,
        "files": {name: {"bytes": (checkpoint / name).stat().st_size, "sha256": sha256_file(checkpoint / name)} for name in names},
        "text_encoder_split": {"validated_structure": True, "precision": "float32"},
        "generation": {
            "validated": True, "precision": "bfloat16",
            "validation": {"cpu": {"weight_precision": "bfloat16", "maximum_error": 0.01}},
            "export": {"source_checkpoint": str(tmp_path / "private"), "components": {"old": "原始映射"}},
        },
    }
    (checkpoint / "export-report.json").write_text(json.dumps(original_report), encoding="utf-8")
    assets = [path for path in source.rglob("*") if path.is_file()]
    write_local_manifest("breeze-tts-2-gguf", source, assets, sources=["https://example.com/original"], adapter="BreezeTTS2GGUFAdapter")
    descriptors = {}
    metric = {
        "actual_finite": True, "reference_finite": True, "shape_match": True,
        "actual_shape": [1, 64], "reference_shape": [1, 64], "expected_shape": [1, 64],
        "max_abs": 0.002, "rms": 0.0001, "relative_l2": 0.001,
        "kl_reference_to_actual": 0.0001, "max_probability_difference": 0.001,
        "passed": True, "errors": [],
    }
    report = {
        "passed": True, "quantization": "q8_0", "errors": [], "policy": dict(q8_audit.Q8_POLICY),
        "cases": [
            {"name": f"case-{index}", "passed": True, "errors": [], "outputs": {"logits": deepcopy(metric)}}
            for index in range(57)
        ],
        "summary": {"passed": True, "cases": 57, "failed_cases": 0, "expected_cases": 57},
        "coverage": {
            "full_suite": True, "batch_sizes": [1, 2], "depth_positions": list(range(16)), "depth_heads": list(range(15)),
            "backbone_prefill_lengths": [3, 5, 7, 511, 512, 513], "backbone_scenarios": 6,
            "misaligned_positions": True, "incremental_steps": 3,
        },
    }
    for kind, directory in (("reference", checkpoint), ("candidate", candidate)):
        report[kind] = {"directory": str(directory), "files": {}}
        for name in ("backbone.gguf", "depth.gguf"):
            path = directory / name
            if kind == "candidate":
                path.write_bytes(f"量化资产 {name}".encode())
            report[kind]["files"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            descriptors[path] = {
                "file_type": 32 if kind == "reference" else 7,
                "metadata": {"general.architecture": "qwen3" if name == "backbone.gguf" else "llama"},
                "tensors": [
                    {"name": "token_embd.weight", "shape": [32, 64], "dtype": "BF16" if kind == "reference" else "Q8_0"},
                    {"name": "output_norm.weight", "shape": [32], "dtype": "F32"},
                ],
            }
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(preparation, "_gguf_descriptor", lambda path: deepcopy(descriptors[path]))
    return {
        "source": source, "candidate": candidate, "output": output, "report": report_path,
        "descriptors": descriptors, "before": {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()},
    }


def test_q8_bundle_preserves_assets_and_replaces_precision_evidence(bundle_case: dict) -> None:
    """派生包完整保留来源资产，并以独立 Q8 证据替换旧精度声明。

    Args:
        bundle_case: 源模型、候选模型与独立验证报告。
    Returns:
        全部来源保持完整且新清单通过校验时正常返回。
    """
    case = bundle_case
    manifest_path = preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    output = case["output"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["assets"]} == {
        path.as_posix() for path in case["before"] if path.name != "manifest.json"
    } | {"ckpt/q8-validation.json", "ckpt/source-export-report.json"}
    assert manifest["sources"] == ["https://example.com/original", "local"]
    for path, content in case["before"].items():
        assert (case["source"] / path).read_bytes() == content
    for name in ("LICENSE", "MODEL_LICENSE", "ckpt/text_encoder_trunk.onnx", "ckpt/text_encoder_lookup.onnx", "ckpt/text_encoder_lookup.json"):
        assert (output / name).read_bytes() == (case["source"] / name).read_bytes()
    assert "上游模型说明" in (output / "README.md").read_text(encoding="utf-8")
    assert "未量化或微调" not in (output / "README.md").read_text(encoding="utf-8")
    assert (case["source"] / "NOTICE").read_text(encoding="utf-8") in (output / "NOTICE").read_text(encoding="utf-8")
    for name in ("README.md", "NOTICE", "ckpt/export-report.json", "ckpt/text_encoder_lookup.json"):
        assert not (output / name).samefile(case["source"] / name)
    derived = json.loads((output / "ckpt/export-report.json").read_text(encoding="utf-8"))
    assert derived["precision"] == derived["generation"]["precision"] == "q8_0"
    assert derived["quantized"] is True and derived["generation"]["validated"] is True
    assert derived["generation"]["export"]["components"]["backbone"]["tensors"][0]["dtype"] == "Q8_0"
    assert "old" not in derived["generation"]["export"]["components"]
    assert "cpu" not in derived["generation"]["validation"]
    for filename in ("backbone.gguf", "depth.gguf"):
        assert derived["files"][filename]["sha256"] == sha256_file(output / "ckpt" / filename)
        assert (output / "ckpt" / filename).read_bytes() == (case["candidate"] / filename).read_bytes()
    provenance = json.loads((output / "ckpt/source-export-report.json").read_text(encoding="utf-8"))
    assert provenance["precision"] == "bfloat16"
    assert provenance["generation"]["validation"]["cpu"]["maximum_error"] == 0.01
    public_report_text = (output / "ckpt/q8-validation.json").read_text(encoding="utf-8")
    assert str(case["source"].parent) not in public_report_text
    public_report = json.loads(public_report_text)
    assert public_report["reference"]["directory"] == "source-bf16"
    assert public_report["candidate"]["files"]["depth.gguf"]["path"] == "candidate-q8/depth.gguf"
    assert public_report["cases"][0]["outputs"]["logits"]["relative_l2"] == 0.001
    validate_local_bundle(output, "breeze-tts-2-gguf", True)


@pytest.mark.parametrize("fault", ["failed", "bf16", "partial", "case", "coverage", "reference_hash", "candidate_hash", "bytes"])
def test_q8_bundle_rejects_failed_partial_or_unbound_reports(bundle_case: dict, fault: str) -> None:
    """旧精度、失败、缺项及文件不匹配的报告必须在创建输出前拒绝。

    Args:
        bundle_case: 完整的离线模型输入。
        fault: 本次注入的报告错误。
    Returns:
        错误被拒绝且未创建输出目录时正常返回。
    """
    case = bundle_case
    report = json.loads(case["report"].read_text(encoding="utf-8"))
    if fault == "failed":
        report["passed"] = False
    elif fault == "bf16":
        report["quantization"] = "bfloat16"
    elif fault == "partial":
        report["cases"].pop()
    elif fault == "case":
        report["cases"][0]["passed"] = False
    elif fault == "coverage":
        report["coverage"]["full_suite"] = False
    elif fault == "bytes":
        report["candidate"]["files"]["depth.gguf"]["bytes"] = 1
    else:
        report[fault.removesuffix("_hash")]["files"]["depth.gguf"]["sha256"] = "0" * 64
    case["report"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="对照报告"):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()


@pytest.mark.parametrize(("field_path", "value"), [
    (("policy", "relative_l2_max"), 0.5),
    (("cases", 0, "outputs", "logits", "relative_l2"), 0.1),
    (("cases", 0, "outputs", "logits", "kl_reference_to_actual"), 0.1),
    (("cases", 0, "outputs", "logits", "max_probability_difference"), 0.1),
    (("cases", 0, "outputs", "logits", "actual_finite"), False),
    (("cases", 0, "outputs", "logits", "shape_match"), False),
    (("cases", 0, "outputs", "logits", "actual_shape"), [1, 63]),
    (("cases", 0, "outputs"), {}),
])
def test_strict_scope_recomputes_output_metrics(bundle_case: dict, field_path: tuple, value: object) -> None:
    """严格验收必须重算指标及结构，不能信任错误的成功标记。

    Args:
        bundle_case: 具有真实报告层次和成功标记的输入。
        field_path: 注入策略、数值或结构错误的位置。
        value: 与成功标记矛盾的字段内容。
    Returns:
        标记伪装被拒绝且未生成模型包时正常返回。
    """
    case = bundle_case
    report = json.loads(case["report"].read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["summary"]["passed"] is True
    assert all(entry["passed"] and entry["outputs"]["logits"]["passed"] for entry in report["cases"])
    target = report
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    case["report"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()


@pytest.mark.parametrize("fault", ["source_type", "candidate_type", "shape", "metadata", "tensor_dtype"])
def test_q8_bundle_audits_real_storage_and_structure(bundle_case: dict, fault: str) -> None:
    """成功报告不能替代模型实际存储类型与完整布局的审计。

    Args:
        bundle_case: 完整的离线模型输入及可变描述。
        fault: 注入到源或候选描述中的错误。
    Returns:
        描述不合要求时正常拒绝派生。
    """
    case = bundle_case
    descriptor = case["descriptors"][case["candidate"] / "depth.gguf"]
    if fault == "source_type":
        case["descriptors"][case["source"] / "ckpt/depth.gguf"]["file_type"] = 7
    elif fault == "candidate_type":
        descriptor["file_type"] = 32
    elif fault == "shape":
        descriptor["tensors"][0]["shape"] = [32, 1]
    elif fault == "metadata":
        descriptor["metadata"]["general.architecture"] = "qwen3"
    else:
        descriptor["tensors"][0]["dtype"] = "F32"
    with pytest.raises(ValueError):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()


def test_q8_bundle_rejects_tampered_legal_asset(bundle_case: dict) -> None:
    """运行期校验忽略的许可证也必须严格核对摘要。

    Args:
        bundle_case: 带完整许可清单的源模型。
    Returns:
        许可证摘要不匹配时正常拒绝。
    """
    case = bundle_case
    (case["source"] / "LICENSE").write_text("变更内容", encoding="utf-8")
    with pytest.raises(ValueError, match="许可或归属"):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()


@pytest.mark.parametrize("fault", ["tiny", "quantized", "unvalidated"])
def test_q8_bundle_rejects_unsuitable_source(bundle_case: dict, fault: str) -> None:
    """小模型、重复量化及缺少原生验收的来源不能进入派生流程。

    Args:
        bundle_case: 可修改源报告的模型输入。
        fault: 本次注入的源模型状态。
    Returns:
        无效来源被拒绝时正常返回。
    """
    case = bundle_case
    path = case["source"] / "ckpt/export-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if fault == "unvalidated":
        report["generation"]["validated"] = False
    else:
        report[fault] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    write_local_manifest("breeze-tts-2-gguf", case["source"], [case["source"] / path for path in case["before"] if path.name != "manifest.json"])
    with pytest.raises(ValueError, match="已验收"):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()


def test_q8_bundle_preserves_existing_output(bundle_case: dict) -> None:
    """已有输出目录和其中的用户文件必须完整保留。

    Args:
        bundle_case: 本次派生的独立输入。
    Returns:
        已有目录被拒绝且内容保持完整时正常返回。
    """
    case = bundle_case
    case["output"].mkdir()
    marker = case["output"] / "user.txt"
    marker.write_text("保留内容", encoding="utf-8")
    with pytest.raises(FileExistsError):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert marker.read_text(encoding="utf-8") == "保留内容"


def test_q8_bundle_falls_back_to_cross_volume_copy(bundle_case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """跨卷无法链接时复制完整资产，并保持源文件内容。

    Args:
        bundle_case: 带小型权重的派生输入。
        monkeypatch: 注入跨卷硬链接错误的工具。
    Returns:
        回退复制成功且资产独立时正常返回。
    """
    case = bundle_case

    def cross_volume(destination: Path, target: Path) -> None:
        """模拟跨卷创建硬链接失败。

        Args:
            destination: 请求创建的目标文件。
            target: 原始文件。
        Returns:
            始终抛出跨卷错误。
        """
        raise OSError(errno.EXDEV, "跨卷", str(destination))

    monkeypatch.setattr(Path, "hardlink_to", cross_volume)
    preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not (case["output"] / "ckpt/text_encoder.onnx").samefile(case["source"] / "ckpt/text_encoder.onnx")
    for path, content in case["before"].items():
        assert (case["source"] / path).read_bytes() == content


def test_q8_bundle_cleans_only_new_directory_after_copy_failure(bundle_case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """复制失败仅清除本次半成品，并保留输入及原始错误。

    Args:
        bundle_case: 完整的离线派生输入。
        monkeypatch: 注入文件复制错误的工具。
    Returns:
        半成品被清除且源文件完整时正常返回。
    """
    case = bundle_case

    def fail_copy(source: Path, destination: Path) -> None:
        """模拟文件系统拒绝复制。

        Args:
            source: 原始资产路径。
            destination: 本次输出路径。
        Returns:
            始终抛出权限错误。
        """
        raise PermissionError("模拟复制失败")

    monkeypatch.setattr(preparation.shutil, "copy2", fail_copy)
    with pytest.raises(PermissionError, match="模拟复制失败"):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()
    for path, content in case["before"].items():
        assert (case["source"] / path).read_bytes() == content


def test_gguf_descriptor_reads_q8_metadata_without_materializing_weights(tmp_path: Path) -> None:
    """真实小 GGUF 的描述读取保持量化类型和原始张量维度。

    Args:
        tmp_path: 独立的小模型目录。
    Returns:
        原生描述与写入结构完全一致时正常返回。
    """
    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    values = np.zeros((2, 32), dtype=np.float32)
    writer.add_tensor("token_embd.weight", gguf.quantize(values, gguf.GGMLQuantizationType.Q8_0), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    descriptor = preparation._gguf_descriptor(path)
    assert descriptor["file_type"] == 7
    assert descriptor["metadata"]["general.architecture"] == "llama"
    assert descriptor["tensors"] == [{"name": "token_embd.weight", "shape": [32, 2], "dtype": "Q8_0"}]


def test_q8_bundle_rejects_weights_changed_during_assembly(bundle_case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """最终清单必须仍与验收时的权重摘要绑定。

    Args:
        bundle_case: 带固定验收摘要的模型输入。
        monkeypatch: 模拟其他进程在组装期间修改候选权重。
    Returns:
        漂移被发现、半成品被清理且 BF16 来源完整时正常返回。
    """
    case = bundle_case
    write_manifest = preparation.write_local_manifest

    def change_weights(*args: object, **kwargs: object) -> Path:
        """在最终摘要计算前模拟候选权重内容变化。

        Args:
            args: 清单构建的位置参数。
            kwargs: 清单构建的命名参数。
        Returns:
            记录变更后内容的清单路径。
        """
        (case["output"] / "ckpt/backbone.gguf").write_bytes(b"concurrent candidate mutation")
        return write_manifest(*args, **kwargs)

    monkeypatch.setattr(preparation, "write_local_manifest", change_weights)
    with pytest.raises(ValueError, match="派生期间资产发生变化"):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()
    for path, content in case["before"].items():
        assert (case["source"] / path).read_bytes() == content


def test_q8_bundle_keeps_directory_replaced_during_failure(bundle_case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """构建目录被替换时，异常清理不得触及替换后的用户目录。

    Args:
        bundle_case: 当前构建拥有的输出目录信息。
        monkeypatch: 注入并发目录替换的工具。
    Returns:
        用户目录与 BF16 来源保留、清理被拒绝时正常返回。
    """
    case = bundle_case
    displaced = case["output"].with_name("displaced")
    error = RuntimeError("模拟目录替换")

    def replace_directory(*args: object, **kwargs: object) -> list[Path]:
        """模拟其他进程将本次目录移走并建立同名用户目录。

        Args:
            args: 元数据写入的位置参数。
            kwargs: 元数据写入的命名参数。
        Returns:
            始终在完成替换后抛出原始错误。
        """
        assert displaced.resolve().parent == case["source"].parent.resolve()
        case["output"].rename(displaced)
        case["output"].mkdir()
        (case["output"] / "user.txt").write_text("用户内容", encoding="utf-8")
        raise error

    monkeypatch.setattr(preparation, "_write_q8_metadata", replace_directory)
    with pytest.raises(RuntimeError) as captured:
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert captured.value is error
    assert any("身份已变化" in note for note in error.__notes__)
    assert (case["output"] / "user.txt").read_text(encoding="utf-8") == "用户内容"
    assert displaced.is_dir()


@pytest.mark.parametrize("nested", [False, True])
def test_q8_bundle_rejects_output_inside_source(bundle_case: dict, nested: bool) -> None:
    """输出不得覆盖或嵌入已有 BF16 模型目录。

    Args:
        bundle_case: 带完整原始内容记录的模型输入。
        nested: 是否使用源目录内的子目录。
    Returns:
        输出被拒绝且原始内容保持完整时正常返回。
    """
    case = bundle_case
    output = case["source"] / "nested" if nested else case["source"]
    with pytest.raises(ValueError, match="原模型目录之外"):
        preparation.prepare_q8_bundle(case["source"], output, case["candidate"], case["report"])
    for path, content in case["before"].items():
        assert (case["source"] / path).read_bytes() == content


def _lossy_trajectory(strict: dict, names: list[str], failed_case: dict) -> dict:
    """建立具有完整冻结计划的六任务消费事件报告。

    Args:
        strict: 绑定源文件和候选文件的基础数值报告。
        names: 六个固定任务名称。
        failed_case: 保留原始数值失败的输出指标。
    Returns:
        覆盖主干、十六码本和引导分支的真实轨迹结构。
    """
    plan = {
        "schema_version": 1, "kind": "breeze_real_trajectory",
        "reference": {"files": {
            name: {key: value[key] for key in ("bytes", "sha256")}
            for name, value in strict["reference"]["files"].items()
        }},
        "arrays": {"bytes": 123, "sha256": "a" * 64},
        "samples": [
            {"name": name, "text": "甲" * 33, "frames": 1, "finite_audio": True, "termination": "eos"}
            for name in names
        ], "events": [],
    }
    cases = []
    for name in names:
        events = plan["events"]
        events.append({
            "id": len(events), "component": "backbone", "operation": "decode", "sample": name,
            "phase": "reference", "arguments": {"positions": [0], "embeddings": {"array": f"{name}_prefix"}},
            "expected": {},
        })
        for component, position in [("backbone", 513)] + [("depth", position) for position in range(16)]:
            width = 2052 if component == "backbone" else 2051
            event_id = len(events)
            arguments = {"positions": [position], "embeddings": {"array": f"e{event_id}_input"}}
            expected = {}
            if component == "backbone" or position:
                arguments.update({"logit_start": 0 if component == "backbone" else (position - 1) * width, "logit_count": width})
                expected = {"logits": f"e{event_id}_logits"}
                if component == "backbone":
                    expected["hidden"] = f"e{event_id}_hidden"
            event = {
                "id": event_id, "component": component, "operation": "decode", "sample": name,
                "phase": "synthesis", "arguments": arguments, "expected": expected,
            }
            events.append(event)
            if not expected:
                continue
            case = {key: event[key] for key in ("id", "component", "operation", "sample", "phase")}
            case.update({"event": event_id, "passed": False, "errors": [], "outputs": {}})
            for output_name in expected:
                metric = deepcopy(failed_case["outputs"]["logits"])
                for shape in ("actual_shape", "reference_shape", "expected_shape"):
                    metric[shape] = [1, width if output_name == "logits" else 64]
                case["outputs"][output_name] = metric
            cases.append(case)
            guided = {
                "id": len(events), "component": component, "operation": "guided", "sample": name, "phase": "synthesis",
                "positive_ref": {"event": event_id, "row": 0}, "negative_ref": {"event": event_id, "row": 0},
                "scale": 4.0, "expected": f"e{len(events)}_guided",
            }
            events.append(guided)
            cases.append({
                **{key: guided[key] for key in ("id", "component", "operation", "sample", "phase")},
                "event": guided["id"], "passed": False, "errors": [], "outputs": {"logits": deepcopy(case["outputs"]["logits"])},
            })
    trajectory = deepcopy(strict)
    trajectory.update({"passed": False, "cases": cases})
    trajectory["summary"] = {
        "passed": False, "cases": len(cases), "failed_cases": len(cases),
        "checked_outputs": sum(len(case["outputs"]) for case in cases), "events": len(plan["events"]),
    }
    canonical = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    trajectory["trace"] = {
        "plan": plan, "canonical_sha256": digest,
        "files": {"plan.json": {"bytes": len(canonical), "sha256": digest}, "arrays.npz": deepcopy(plan["arrays"])},
    }
    trajectory["coverage"] = {
        "complete": True, "all_consumed_outputs": True, "cfg": True, "sample_count": 6, "samples": names,
        "depth_positions": list(range(16)), "depth_heads": list(range(1, 16)), "long_reference_prefix_length": 513,
        "per_sample": {
            name: {
                "backbone": True, "depth_positions": list(range(16)), "depth_heads": list(range(1, 16)),
                "guided_components": ["backbone", "depth"],
            } for name in names
        },
    }
    return trajectory


@pytest.fixture
def lossy_case(bundle_case: dict) -> dict:
    """建立保留数值失败的独立有损任务验收报告。

    Args:
        bundle_case: 具有完整摘要绑定的模型输入。
    Returns:
        带合成、内容及试听证据的有损验收输入。
    """
    case = bundle_case
    strict = json.loads(case["report"].read_text(encoding="utf-8"))
    names = [f"task-{index}" for index in range(6)]
    metric = {
        "actual_finite": True, "reference_finite": True, "shape_match": True,
        "actual_shape": [1, 64], "reference_shape": [1, 64], "expected_shape": [1, 64],
        "relative_l2": 0.03, "kl_reference_to_actual": 0.0001, "max_probability_difference": 0.001,
        "passed": False, "errors": ["相对误差超过原门限"],
    }
    failed_case = {"passed": False, "errors": [], "outputs": {"logits": metric}}
    synthetic = deepcopy(strict)
    synthetic.update({"passed": False, "errors": [], "policy": dict(q8_audit.Q8_POLICY)})
    synthetic["cases"] = [deepcopy(failed_case) for _ in range(57)]
    for current in synthetic["cases"][37:]:
        current["passed"] = current["outputs"]["logits"]["passed"] = True
        current["outputs"]["logits"]["relative_l2"] = 0.01
        current["outputs"]["logits"]["errors"] = []
    synthetic["summary"] = {"passed": False, "cases": 57, "expected_cases": 57, "failed_cases": 37}
    trajectory = _lossy_trajectory(synthetic, names, failed_case)
    synthesis = {
        "synthesis_checks_passed": True, "candidate": deepcopy(strict["candidate"]),
        "samples": [
            {"name": f"task-{index}", "seed": 42 if index < 6 else 7, "frames": 10, "seconds": 0.8,
             "finite_audio": True, "termination": "eos", "audio_sha256": f"{index + 1:064x}",
             "wav_sha256": f"{index + 101:064x}"}
            for index in range(8)
        ],
    }
    report = {
        "schema_version": 1, "basis": "q8_0_lossy_task_audit_v1", "passed": True, "quantization": "q8_0",
        "reference": deepcopy(strict["reference"]), "candidate": deepcopy(strict["candidate"]),
        "numerical_equivalence_claimed": False, "strict_numerical_gates_passed": False,
        "summary": {
            "passed": True, "structural_failed_cases": 0, "numerical_failed_cases": trajectory["summary"]["failed_cases"],
            "synthetic_failed_cases": 37, "checked_outputs": trajectory["summary"]["checked_outputs"],
            "synthesis_samples": 8, "asr_samples": 6, "listened_samples": 2,
        },
        "reports": {
            "synthetic_stress": synthetic, "real_trajectory": trajectory, "synthesis": synthesis,
            "asr": {
                "model_revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
                "normalization": "NFKC+OpenCC-t2s+punctuation-whitespace",
                "samples": [
                    {"name": name, "reference_edits": 2, "candidate_edits": 1,
                     "reference_characters": 33, "candidate_characters": 33,
                     "candidate_audio_sha256": f"{index + 1:064x}", "candidate_wav_sha256": f"{index + 101:064x}"}
                    for index, name in enumerate(names)
                ],
                "aggregate": {"reference_edits": 12, "candidate_edits": 6, "characters": 198},
            },
            "listening": {
                "confirmed": True, "feedback": "声音正常",
                "samples": [{"name": name, "audio_sha256": f"{index + 1:064x}"} for index, name in enumerate(names[:2])],
            },
        },
    }
    for side, edits in (("baseline", 2), ("candidate", 1)):
        report["reports"]["asr"][side] = {
            "status": "complete", "model": {"id": "openai/whisper-small", "revision": report["reports"]["asr"]["model_revision"]},
            "settings": {"language": "zh", "normalization": {"opencc_configuration": "t2s"}},
            "samples": [
                {"name": name, "status": "complete", "wav_sha256": f"{index + (101 if side == 'candidate' else 1001):064x}",
                 "canonical_normalized_reference": "甲" * 33, "canonical_normalized_transcript": "甲" * (33 - edits) + "乙" * edits,
                 "canonical_reference_characters": 33, "canonical_edit_distance": edits}
                for index, name in enumerate(names)
            ],
        }
    case["report"].write_text(json.dumps(report), encoding="utf-8")
    return case


def test_lossy_scope_preserves_failed_audits_and_explicit_metadata(lossy_case: dict) -> None:
    """有损发布只能通过明确任务范围，同时保留原数值失败及指标。

    Args:
        lossy_case: 结构与任务证据完整的有损量化输入。
    Returns:
        有损任务通过和数值失败分别准确记录时正常返回。
    """
    case = lossy_case
    preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    exported = json.loads((case["output"] / "ckpt/export-report.json").read_text(encoding="utf-8"))
    assert exported["quantized_generation_validation_scope"] == "q8_0_lossy_task_audit_v1"
    assert exported["numerical_equivalence_claimed"] is False
    assert exported["numerical_error_bounds_passed"] is False
    assert exported["generation"]["validation"]["strict_numerical_gates_passed"] is False
    report = json.loads((case["output"] / "ckpt/q8-validation.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["reports"]["synthetic_stress"]["passed"] is False
    assert report["reports"]["synthetic_stress"]["summary"]["failed_cases"] == 37
    assert report["reports"]["real_trajectory"]["passed"] is False
    assert report["reports"]["real_trajectory"]["cases"][0]["outputs"]["logits"]["relative_l2"] == 0.03
    assert report["reports"]["real_trajectory"]["policy"] == q8_audit.Q8_POLICY
    assert "严格数值审计保留失败状态" in (case["output"] / "NOTICE").read_text(encoding="utf-8")
    for path, content in case["before"].items():
        assert (case["source"] / path).read_bytes() == content


@pytest.mark.parametrize(("field_path", "value"), [
    (("numerical_equivalence_claimed",), True),
    (("strict_numerical_gates_passed",), True),
    (("summary", "numerical_failed_cases"), 0),
    (("summary", "synthetic_failed_cases"), 0),
    (("summary", "checked_outputs"), 1),
    (("reports", "synthetic_stress", "errors"), ["缓存失败"]),
    (("reports", "synthetic_stress", "policy", "relative_l2_max"), 0.5),
    (("reports", "synthetic_stress", "candidate", "files", "depth.gguf", "sha256"), "0" * 64),
    (("reports", "synthetic_stress", "cases"), []),
    (("reports", "real_trajectory", "cases", 0, "errors"), ["执行失败"]),
    (("reports", "real_trajectory", "cases", 0, "outputs", "logits", "actual_finite"), False),
    (("reports", "real_trajectory", "cases", 0, "outputs", "logits", "shape_match"), False),
    (("reports", "real_trajectory", "cases", 0, "outputs", "logits", "passed"), True),
    (("reports", "real_trajectory", "coverage", "complete"), False),
    (("reports", "real_trajectory", "coverage", "per_sample", "task-0", "depth_heads"), [1]),
    (("reports", "synthesis", "samples", 0, "termination"), "max_new_tokens"),
    (("reports", "synthesis", "samples", 0, "finite_audio"), False),
    (("reports", "synthesis", "samples", 0, "audio_sha256"), "partial"),
    (("reports", "synthesis", "samples", 0, "wav_sha256"), "partial"),
    (("reports", "synthesis", "candidate", "files", "depth.gguf", "sha256"), "0" * 64),
    (("reports", "asr", "model_revision"), "unfrozen"),
    (("reports", "asr", "normalization"), "changed"),
    (("reports", "asr", "samples", 0, "candidate_edits"), 3),
    (("reports", "asr", "samples", 0, "candidate_characters"), 34),
    (("reports", "asr", "aggregate", "candidate_edits"), 0),
    (("reports", "asr", "samples", 0, "candidate_audio_sha256"), "0" * 64),
    (("reports", "asr", "samples", 0, "candidate_wav_sha256"), "0" * 64),
    (("reports", "asr", "candidate", "samples", 0, "wav_sha256"), "0" * 64),
    (("reports", "asr", "candidate", "samples", 0, "canonical_normalized_transcript"), "甲" * 33),
    (("reports", "asr", "baseline", "samples", 0, "canonical_reference_characters"), 34),
    (("reports", "asr", "candidate", "settings", "language"), "en"),
    (("reports", "asr", "baseline", "model", "revision"), "unfrozen"),
    (("reports", "real_trajectory", "trace", "plan"), {}),
    (("reports", "real_trajectory", "trace", "canonical_sha256"), "0" * 64),
    (("reports", "real_trajectory", "trace", "files", "arrays.npz", "bytes"), 124),
    (("reports", "real_trajectory", "cases", 0, "phase"), "reference"),
    (("reports", "listening", "confirmed"), False),
    (("reports", "listening", "samples", 0, "audio_sha256"), "0" * 64),
])
def test_lossy_scope_rejects_unbound_or_incomplete_evidence(lossy_case: dict, field_path: tuple, value: object) -> None:
    """有损范围仍严格拒绝证据缺失、门限改写和结构或内容错误。

    Args:
        lossy_case: 具有完整成功任务证据的输入。
        field_path: 需要注入错误的报告字段路径。
        value: 与验收契约不符的替换值。
    Returns:
        输入被拒绝且尚未创建输出目录时正常返回。
    """
    case = lossy_case
    report = json.loads(case["report"].read_text(encoding="utf-8"))
    target = report
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    case["report"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        preparation.prepare_q8_bundle(case["source"], case["output"], case["candidate"], case["report"])
    assert not case["output"].exists()


def test_lossy_scope_rejects_rewritten_asr_counts_even_when_totals_match(lossy_case: dict) -> None:
    """汇总自洽的手抄计数也必须与实际转录文本一致。

    Args:
        lossy_case: 保存原始转录和候选音频摘要的完整报告。
    Returns:
        错误计数被拒绝时正常返回。
    """
    report = json.loads(lossy_case["report"].read_text(encoding="utf-8"))
    asr = report["reports"]["asr"]
    asr["samples"][0]["candidate_edits"] = 0
    asr["aggregate"]["candidate_edits"] -= 1
    with pytest.raises(ValueError, match="原始转录"):
        q8_audit.validate_q8_audit(report)


def test_lossy_scope_rejects_truncated_cases_even_when_totals_match(lossy_case: dict) -> None:
    """完整覆盖标记和更新后的汇总不能掩盖绝大多数消费事件被删除。

    Args:
        lossy_case: 每个样本覆盖全部码本的完整报告。
    Returns:
        每样本仅保留一个事件的报告被拒绝时正常返回。
    """
    report = json.loads(lossy_case["report"].read_text(encoding="utf-8"))
    trajectory = report["reports"]["real_trajectory"]
    kept = {}
    for case in trajectory["cases"]:
        kept.setdefault(case["sample"], case)
    trajectory["cases"] = list(kept.values())
    checked = sum(len(case["outputs"]) for case in trajectory["cases"])
    trajectory["summary"].update(cases=6, failed_cases=6, checked_outputs=checked)
    report["summary"].update(numerical_failed_cases=6, checked_outputs=checked)
    with pytest.raises(ValueError, match="遗漏消费事件"):
        q8_audit.validate_q8_audit(report)


@pytest.mark.parametrize("fault", ["duplicate", "outputs", "head", "position_zero", "guided_reference"])
def test_lossy_scope_checks_actual_frozen_event_coverage(lossy_case: dict, fault: str) -> None:
    """事件集合、输出头和码本覆盖都按冻结计划逐项检查。

    Args:
        lossy_case: 带完整冻结计划与数值指标的报告。
        fault: 当前注入的事件或覆盖错误。
    Returns:
        即使重新计算计划摘要，错误覆盖仍被拒绝时正常返回。
    """
    report = json.loads(lossy_case["report"].read_text(encoding="utf-8"))
    trajectory = report["reports"]["real_trajectory"]
    plan = trajectory["trace"]["plan"]
    if fault == "duplicate":
        trajectory["cases"][1] = deepcopy(trajectory["cases"][0])
        trajectory["summary"]["checked_outputs"] += 1
        report["summary"]["checked_outputs"] += 1
    elif fault == "outputs":
        del trajectory["cases"][0]["outputs"]["hidden"]
        trajectory["summary"]["checked_outputs"] -= 1
        report["summary"]["checked_outputs"] -= 1
    elif fault == "guided_reference":
        event = next(event for event in plan["events"] if event["operation"] == "guided")
        event["positive_ref"]["row"] = 1
    else:
        position = 1 if fault == "head" else 0
        event = next(event for event in plan["events"] if event["component"] == "depth"
                     and event["operation"] == "decode" and event["arguments"]["positions"] == [position])
        if fault == "head":
            event["arguments"]["logit_start"] += 1
        else:
            event["arguments"]["positions"] = [1]
    canonical = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    trajectory["trace"]["canonical_sha256"] = digest
    trajectory["trace"]["files"]["plan.json"] = {"bytes": len(canonical), "sha256": digest}
    with pytest.raises(ValueError):
        q8_audit.validate_q8_audit(report)
