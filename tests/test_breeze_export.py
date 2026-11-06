"""验证离线导出的许可证边界、共享权重与外部参数资产。"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.breeze_reference import bundle, conditioner_export, export, factory, weights


@pytest.fixture(scope="module")
def reference_tokenizers() -> tuple[object, object]:
    """读取本地原始分词器与固定 Transformers 版本的参考分词器。

    Returns:
        直接读取 JSON 的分词器及原始高层分词器。
    """
    checkpoint_value = os.environ.get("BREEZE_REFERENCE_CHECKPOINT_DIR")
    if not checkpoint_value:
        pytest.skip("需要通过 BREEZE_REFERENCE_CHECKPOINT_DIR 指定本地参考检查点")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("分词参考固定为 Transformers 4.57.3")
    checkpoint = Path(checkpoint_value)
    raw = tokenizers.Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    reference = transformers.AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    return raw, reference


@pytest.mark.parametrize("special", [False, True], ids=["without-specials", "with-specials"])
@pytest.mark.parametrize("text", [
    "你好，今天的天气很适合散步。",
    "Hello, this is a StreamVox tokenizer check.",
    "[S0]这是参考音频里说的话。",
    "[S0]<ins_bos>年轻的成年女声，平静、清晰。<ins_eos>接下来是目标文本。",
    "[S0]你好[笑]，我们继续。[咳嗽][清嗓子][叹气]",
    "[S0]Hello (laugh), let us continue. (cough) (clears throat) (sigh)",
    "[S0]你好，StreamVox! 123 <ins_bos>清晰<ins_eos>",
    " [S0] Hello  world.\n你好。\tTest",
    "",
], ids=["chinese", "english", "reference", "instruction", "chinese-events", "english-events", "mixed", "whitespace", "empty"])
def test_tokenizer_ids_and_roundtrip_match_fixed_reference(
    reference_tokenizers: tuple[object, object], text: str, special: bool,
) -> None:
    """低层 JSON 分词必须保持原高层分词的特殊符号与解码重编码语义。

    Args:
        reference_tokenizers: 原始 JSON 与固定版本高层分词器。
        text: 覆盖目标、参考、指令及事件边界的文本。
        special: 是否自动添加起止特殊符号。
    """
    raw, reference = reference_tokenizers
    actual = raw.encode(text, add_special_tokens=special).ids
    expected = reference.encode(text, add_special_tokens=special)
    assert actual == expected
    rendered = raw.decode(actual, skip_special_tokens=False)
    assert rendered == reference.decode(expected, skip_special_tokens=False)
    assert raw.encode(rendered, add_special_tokens=False).ids == reference.encode(rendered, add_special_tokens=False)


def test_tokenizer_bos_eos_postprocessor_matches_fixed_reference(
    reference_tokenizers: tuple[object, object],
) -> None:
    """原始 JSON 必须自带相同的起始符号处理，且不能额外添加结束符号。

    Args:
        reference_tokenizers: 原始 JSON 与固定版本高层分词器。
    """
    raw, reference = reference_tokenizers
    actual_processor = json.loads(raw.to_str())["post_processor"]
    reference_processor = json.loads(reference.backend_tokenizer.to_str())["post_processor"]
    assert actual_processor == reference_processor
    assert reference.add_bos_token is True
    assert reference.add_eos_token is False
    assert raw.encode("", add_special_tokens=True).ids == [reference.bos_token_id]
    assert raw.encode("", add_special_tokens=False).ids == []


def test_full_export_rejects_missing_model_license_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 bundle 缺少模型独立许可证时，不得开始加载或导出权重。

    Args:
        tmp_path: 隔离的临时文件目录。
        monkeypatch: 命令行与本地源码导入的隔离工具。
    """
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", [
        "breeze-export", "--checkpoint", str(checkpoint),
        "--source-dir", str(tmp_path / "source"), "--output", str(output),
    ])
    monkeypatch.setattr(factory, "source_modules", lambda source: ())

    def reject_graph_build(*args: object, **kwargs: object) -> None:
        """阻止缺少许可证的任务进入模型分配阶段。

        Args:
            args: 模型构建的位置参数。
            kwargs: 模型构建的命名参数。
        """
        pytest.fail("缺少许可证时仍尝试构建模型")

    monkeypatch.setattr(factory, "build_graph", reject_graph_build)
    with pytest.raises(ValueError, match="权重许可证"):
        export.main()
    assert not (output / "manifest.json").exists()
    assert not (output / "ckpt" / "export-report.json").exists()


@pytest.mark.parametrize("raw", [
    {"backbone_model_name_or_path": "remote/model"},
    {"backbone_model_name_or_path": "remote/model", "backbone_config": None},
    {"backbone_model_name_or_path": "remote/model", "backbone_config": {}},
    {"backbone_model_name_or_path": "remote/model", "backbone_config": []},
], ids=["missing", "none", "empty", "list"])
def test_graph_build_rejects_missing_embedded_backbone_before_source_construction(raw: dict) -> None:
    """配置不完整时必须离线拒绝，不能进入上游构建或按远端名称补取。

    Args:
        raw: 含远端主干名称但缺少有效内嵌配置的输入。
    """
    # 空模块元组不能构建模型，确保拒绝发生在访问上游构造器之前。
    with pytest.raises(ValueError, match="backbone_config"):
        factory.build_graph("backbone", raw, (), meta=True)


def test_export_rejects_missing_runtime_python_before_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """数值验收解释器不存在时，不得生成看似完整的导出目录。

    Args:
        tmp_path: 尚未创建模型产物的隔离目录。
        monkeypatch: 命令行隔离工具。
        capsys: 捕获解释器参数的错误说明。
    """
    output = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", [
        "breeze-export", "--tiny", "--source-dir", str(tmp_path), "--output", str(output),
        "--runtime-python", str(tmp_path / "missing-python.exe"),
    ])
    with pytest.raises(SystemExit) as error:
        export.main()
    assert error.value.code == 2
    assert "验证解释器不存在" in capsys.readouterr().err
    assert not output.exists()


def test_bundle_rejects_unvalidated_generation_before_writing_manifest(tmp_path: Path) -> None:
    """生成组件没有通过数值验收时，不能写入完成报告或清单。

    Args:
        tmp_path: 不含任何导出资产的隔离目录。
    """
    with pytest.raises(ValueError, match="数值验证"):
        bundle.write_bundle(
            tmp_path, {}, {"generation": {"validated": False}}, set(),
            source_dir=tmp_path, checkpoint=None, tiny=True,
        )
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "ckpt/export-report.json").exists()


def test_onnx_export_rejects_reduced_precision_before_writing_files(tmp_path: Path) -> None:
    """辅助图必须使用 FP32，不能重新引入低精度 ONNX 执行路径。

    Args:
        tmp_path: 尚未创建图及参数目录的隔离目录。
    """
    graph = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        bundle.export_isolated_graph(graph, (torch.zeros(1, 4, dtype=torch.bfloat16),), tmp_path / "projector.onnx")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["original-bf16", "original-fp32"])
def test_depth_conditioner_preserves_source_weights_and_dynamic_batch(tmp_path: Path, dtype: torch.dtype) -> None:
    """原始投影必须精确提升到 FP32，并保留独立的双分支批次接口。

    Args:
        tmp_path: 本地原始参数与实际 ONNX 图的隔离目录。
        dtype: 检查点原始存储精度。
    """
    safetensors = pytest.importorskip("safetensors.torch")
    ort = pytest.importorskip("onnxruntime")
    checkpoint, destination = tmp_path / "checkpoint", tmp_path / "ckpt"
    checkpoint.mkdir()
    original = (torch.arange(8, dtype=torch.float32).reshape(2, 4) / 13).to(dtype)
    safetensors.save_file(
        {"depth_decoder.model.inputs_embeds_projector.weight": original},
        str(checkpoint / "model.safetensors"),
    )
    raw = {"hidden_size": 4, "depth_decoder_config": {"hidden_size": 2, "audio_embed_size": 4, "backbone_hidden_size": 4}}
    report = conditioner_export.export_depth_conditioner(checkpoint, destination, raw)
    assert report["validated"] is True
    assert report["source_dtype"] == str(dtype).removeprefix("torch.")
    assert report["precision"] == "float32"
    assert report["cases"] == 3
    assert report["inputs"] == [{"name": "backbone_hidden", "type": "tensor(float)", "shape": ["batch", 4]}]
    assert report["outputs"] == [{"name": "embeddings", "type": "tensor(float)", "shape": ["batch", 2]}]
    session = ort.InferenceSession(str(destination / "depth_conditioner.onnx"), providers=["CPUExecutionProvider"])
    hidden = np.arange(12, dtype=np.float32).reshape(3, 4) / 17
    actual = session.run(None, {"backbone_hidden": hidden})[0]
    np.testing.assert_allclose(actual, hidden @ original.float().numpy().T, rtol=1e-6, atol=1e-6)


def test_bundle_records_new_assets_and_preserves_model_legal_materials(tmp_path: Path) -> None:
    """混合后端清单必须保留来源及转换事实，并排除旧后端遗留文件。

    Args:
        tmp_path: 包含源码、检查点及导出目录的隔离根目录。
    """
    source, checkpoint, output = tmp_path / "source", tmp_path / "checkpoint", tmp_path / "bundle"
    source.mkdir()
    checkpoint.mkdir()
    destination = output / "ckpt"
    destination.mkdir(parents=True)
    (source / "LICENSE").write_text("source license", encoding="utf-8")
    (source / "NOTICE").write_text("source notice", encoding="utf-8")
    (checkpoint / "LICENSE").write_text("model license", encoding="utf-8")
    (checkpoint / "NOTICE").write_text("model notice", encoding="utf-8")
    (checkpoint / "README.md").write_text("original model attribution", encoding="utf-8")
    exported = set()
    for filename in ("backbone.gguf", "depth.gguf", "depth_conditioner.onnx"):
        path = destination / filename
        path.write_bytes(filename.encode("utf-8"))
        exported.add(path)
    (destination / "backbone.onnx").write_bytes(b"old backend")
    report = {"generation": {"validated": True, "precision": "float32"}}
    bundle.write_bundle(output, {}, report, exported, source_dir=source, checkpoint=checkpoint, tiny=False)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assets = {item["path"]: item["sha256"] for item in manifest["assets"]}
    assert manifest["model"] == "breeze-tts-2-gguf"
    assert manifest["adapter"] == "BreezeTTS2GGUFAdapter"
    assert "ckpt/backbone.onnx" not in assets
    assert {"backbone.gguf", "depth.gguf", "depth_conditioner.onnx"} == set(report["files"])
    for path in exported:
        assert report["files"][path.name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (output / "MODEL_LICENSE").read_text(encoding="utf-8") == "model license"
    assert (output / "SOURCE_NOTICE").read_text(encoding="utf-8") == "source notice"
    assert (output / "MODEL_NOTICE").read_text(encoding="utf-8") == "model notice"
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "original model attribution" in readme
    assert "GGUF" in readme and "FP32 ONNX" in readme
    assert "E'=E@P.T" in readme and "十六码本" in readme and "未量化" in readme


def test_shared_audio_embedding_loads_bfloat16_exactly_without_backbone_copy(tmp_path: Path) -> None:
    """允许检查点只存一份共享音频权重，并精确提升到 FP32。

    Args:
        tmp_path: 本地随机检查点目录。
    """
    safetensors = pytest.importorskip("safetensors.torch")
    original = (torch.arange(24, dtype=torch.float32).reshape(6, 4) / 13).bfloat16()
    safetensors.save_file(
        {"depth_decoder.model.embed_tokens.weight": original},
        str(tmp_path / "model.safetensors"),
    )
    graph = torch.nn.Module()
    graph.embedding = torch.nn.Module()
    graph.embedding.embed_audio_tokens = torch.nn.Embedding(6, 4, device="meta")
    weights.load_graph_weights(graph, "audio_embedding", tmp_path)
    loaded = graph.embedding.embed_audio_tokens.weight
    assert loaded.dtype == torch.float32
    assert not loaded.is_meta
    torch.testing.assert_close(loaded, original.float(), rtol=0, atol=0)


def test_checkpoint_index_cannot_resolve_weights_outside_checkpoint(tmp_path: Path) -> None:
    """索引中的实际参数路径必须仍属于指定检查点。

    Args:
        tmp_path: 隔离的本地检查点目录。
    """
    pytest.importorskip("safetensors")
    graph = torch.nn.Linear(3, 2, bias=False, device="meta")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="目录外文件"):
        weights.load_graph_weights(graph, "backbone", tmp_path)


def test_bfloat16_export_rejects_downcasting_float32_checkpoint(tmp_path: Path) -> None:
    """原始精度导出不得把 FP32 检查点降转为 BF16。

    Args:
        tmp_path: 保存原生 FP32 权重的检查点目录。
    """
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file({"weight": torch.randn(2, 3)}, str(tmp_path / "model.safetensors"))
    graph = torch.nn.Linear(3, 2, bias=False, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16|bfloat16|降转"):
        weights.load_graph_weights(graph, "backbone", tmp_path)


def test_empty_cache_preserves_decoder_bfloat16_dtype() -> None:
    """显式缓存必须采用对应解码器的原始推理精度。

    Returns:
        无。
    """
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    attention = torch.nn.Module()
    attention.head_dim = 4
    attention.k_proj = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    model.layers[0].self_attn = attention
    cache = factory.empty_cache(model, batch=2, length=0)
    assert len(cache) == 2
    assert all(value.dtype == torch.bfloat16 for value in cache)
    assert all(value.shape == (2, 2, 0, 4) for value in cache)


def test_external_parameter_file_is_discovered_and_used_by_runtime(tmp_path: Path) -> None:
    """真实 ONNX 外部参数可使用任意合法相对路径，且必须进入资产枚举。

    Args:
        tmp_path: ONNX 与外部参数文件的导出目录。
    """
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    helper = onnx.helper
    (tmp_path / "parameters").mkdir()
    weights = np.array([[1, 2], [3, 4]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "external-parameter-test",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
        initializer=[onnx.numpy_helper.from_array(weights, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "model.onnx"
    onnx.save_model(
        model, path, save_as_external_data=True, all_tensors_to_one_file=True,
        location="parameters/custom-weights.bin", size_threshold=0,
    )
    expected_files = {path.resolve(), (tmp_path / "parameters/custom-weights.bin").resolve()}
    assert bundle.graph_files(path) == expected_files
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": np.array([[2, 5]], dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.array([[17, 24]], dtype=np.float32))


def test_external_parameter_exports_isolate_graph_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同参数名的独立图必须独占外部参数，且重导出只清理自身资产。

    Args:
        tmp_path: 两个真实 ONNX 图及独立外部参数的输出目录。
        monkeypatch: 强制小模型也采用逐参数外部存储的隔离工具。
    """
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    original_export = torch.onnx.export

    class MatrixGraph(torch.nn.Module):
        """保留独立图的参数，并以 FP32 矩阵乘法执行实际验证。"""

        def __init__(self, value: torch.Tensor) -> None:
            """登记需要原样导出的共享名称参数。

            Args:
                value: 指定数值与存储精度的矩阵。
            """
            super().__init__()
            self.weight = torch.nn.Parameter(value)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """使用 FP32 原始参数验证外部存储不改变矩阵运算。

            Args:
                value: FP32 输入矩阵。

            Returns:
                FP32 矩阵乘积。
            """
            return value @ self.weight

    def export_with_external_weights(
        graph: torch.nn.Module, arguments: tuple[torch.Tensor, ...],
        path: str | Path, **options: object,
    ) -> None:
        """通过真实导出和重新保存，让小参数复现大型模型的外部资产布局。

        Args:
            graph: 参与真实导出的矩阵图。
            arguments: 实际前向输入。
            path: 隔离目录内的临时模型路径。
            options: 传给原始 Torch 导出器的选项。
        """
        original_export(graph, arguments, path, **options)
        model = onnx.load(str(path))
        onnx.save_model(
            model, str(path), save_as_external_data=True,
            all_tensors_to_one_file=False, size_threshold=0,
        )

    monkeypatch.setattr(torch.onnx, "export", export_with_external_weights)
    value = torch.tensor([[2, 5]], dtype=torch.float32)
    cases = [
        (tmp_path / "first.onnx", torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)),
        (tmp_path / "second.onnx", torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)),
    ]
    options = {
        "input_names": ["input"], "output_names": ["output"], "opset_version": 17,
        "dynamo": False, "external_data": True, "do_constant_folding": False,
    }
    for path, weight in cases:
        bundle.export_isolated_graph(MatrixGraph(weight), (value,), path, **options)

    dependencies: list[set[Path]] = []
    for path, weight in cases:
        model = onnx.load(str(path), load_external_data=False)
        initializer = next(tensor for tensor in model.graph.initializer if tensor.name == "weight")
        assert initializer.data_type == onnx.TensorProto.FLOAT
        location = next(field.value for field in initializer.external_data if field.key == "location")
        assert location == f"{path.stem}.weights/weight"
        files = bundle.graph_files(path)
        assert files == {path, tmp_path / location}
        dependencies.append(files)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input": value.numpy()})[0]
        np.testing.assert_array_equal(actual, (value @ weight.float()).numpy())
    assert dependencies[0].isdisjoint(dependencies[1])

    # 重导出可能减少参数文件，必须清掉自身旧资产，同时完整保留另一个图。
    other_files = {path: path.read_bytes() for path in dependencies[1]}
    stale = tmp_path / "first.weights" / "stale.bin"
    stale.write_bytes(b"obsolete weight")
    replacement = torch.tensor([[9, 10], [11, 12]], dtype=torch.float32)
    bundle.export_isolated_graph(MatrixGraph(replacement), (value,), cases[0][0], **options)
    assert not stale.exists()
    assert all(path.read_bytes() == content for path, content in other_files.items())
    session = ort.InferenceSession(str(cases[0][0]), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": value.numpy()})[0]
    np.testing.assert_array_equal(actual, (value @ replacement).numpy())
    assert set(tmp_path.rglob("*.onnx")) == {path for path, _ in cases}
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == set.union(*dependencies)


def test_external_parameter_reference_cannot_escape_export_directory(tmp_path: Path) -> None:
    """模型外部参数引用不能把 bundle 目录外文件纳入资产。

    Args:
        tmp_path: 同时存放导出目录和目录外文件的临时根目录。
    """
    onnx = pytest.importorskip("onnx")
    destination = tmp_path / "bundle"
    destination.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(np.array([1], dtype=np.float32).tobytes())
    tensor = onnx.TensorProto(name="weight", data_type=onnx.TensorProto.FLOAT, dims=[1])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../outside.bin")
    graph = onnx.helper.make_graph([], "outside-parameter-test", [], [], initializer=[tensor])
    model = onnx.helper.make_model(graph)
    path = destination / "model.onnx"
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="越过导出目录"):
        bundle.graph_files(path)


def test_tiny_export_manifest_matches_actual_files_and_fixed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实随机小模型导出必须记录实际资产、法律文件与固定参考来源。

    Args:
        tmp_path: 完整测试 bundle 的输出目录。
        monkeypatch: 命令行隔离工具。
    """
    source_value = os.environ.get("BREEZE_REFERENCE_SOURCE_DIR")
    if not source_value:
        pytest.skip("需要通过 BREEZE_REFERENCE_SOURCE_DIR 指定本地固定版本参考源码")
    pytest.importorskip("transformers")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(sys, "argv", [
        "breeze-export", "--tiny", "--source-dir", source_value,
        "--output", str(tmp_path), "--runtime-python", os.environ.get("BREEZE_RUNTIME_PYTHON", sys.executable),
    ])
    export.main()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "breeze-tts-2-gguf"
    assert manifest["adapter"] == "BreezeTTS2GGUFAdapter"
    assert manifest["sources"] == [
        "https://github.com/breezeblue-ai/breeze-tts/tree/daa6cb2efe5471f09369229f322d20bb956cb3c3",
        "https://huggingface.co/BreezeBlue/breeze-tts-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4",
    ]
    assets = {asset["path"]: asset["sha256"] for asset in manifest["assets"]}
    actual = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}
    assert set(assets) == actual - {"manifest.json"}
    assert {"LICENSE", "NOTICE"} <= set(assets)
    for relative, expected in assets.items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == expected
    report = json.loads((tmp_path / "ckpt/export-report.json").read_text(encoding="utf-8"))
    assert report["tiny"] is True
    assert report["quantized"] is False
    assert set(report["graphs"]) == {"text_encoder", "audio_embedding", "depth_conditioner"}
    assert report["generation"]["validated"] is True
    assert report["generation"]["precision"] == "float32"
    assert {"backbone.gguf", "depth.gguf", "depth_conditioner.onnx"} <= set(report["files"])
    assert not {"backbone.onnx", "depth.onnx", "cfg_pack.onnx"} & set(report["files"])
    assert not list((tmp_path / "ckpt").glob("*.bf16.onnx"))
