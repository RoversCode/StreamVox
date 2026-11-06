"""将本地 Breeze2 检查点导出为无量化的 GGUF 生成组件与 FP32 ONNX 编解码组件。

第三方参考：https://github.com/breezeblue-ai/breeze-tts ，
提交 daa6cb2efe5471f09369229f322d20bb956cb3c3，
模型配置版本 3e28c5151381a722f1d8661b4118c298caa77aa4。
只导入用户指定的本地源码，不下载源码、配置或模型权重。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import torch

from . import bundle, comparison, conditioner_export, factory, gguf_export, weights


def write_tiny_checkpoint(destination: Path, raw: dict[str, Any], modules: tuple[Any, ...]) -> Path:
    """把两个随机小型生成组件写成共享的 FP32 参考检查点。

    Args:
        destination: 仅用于当前导出任务的临时检查点目录。
        raw: 保留十六码本和真实注意力公式的小模型配置。
        modules: 已通过固定提交校验的上游模块。

    Returns:
        含配置与 Safetensors 的临时检查点目录。
    """
    from safetensors.torch import save_file

    parameters: dict[str, torch.Tensor] = {}
    for name in ("backbone", "depth"):
        graph = factory.build_graph(name, raw, modules, meta=False)
        parameters.update({key: value.detach().cpu().contiguous() for key, value in graph.state_dict().items()})
        del graph
    destination.mkdir(parents=True, exist_ok=True)
    save_file(parameters, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    """导出各组件，完成真实数值验证后再写入可加载的模型清单。

    Returns:
        无返回值；校验失败时不生成完成清单。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="本地完整 Breeze2 检查点目录")
    parser.add_argument("--source-dir", type=Path, required=True, help="固定提交的本地上游源码目录")
    parser.add_argument("--output", type=Path, required=True, help="模型 bundle 根目录")
    parser.add_argument("--audio-tokenizer", type=Path, help="本地音频分词器目录，默认使用检查点内 audio_tokenizer")
    parser.add_argument("--tiny", action="store_true", help="导出并校验随机小模型，不包含真实分词器与音频组件")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32", help="GGUF 权重精度，默认精确提升到 FP32")
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable), help="安装 StreamVox 运行依赖的数值验收解释器")
    parser.add_argument("--llama-backend", choices=("cpu", "cuda"), default="cpu", help="GGUF 数值验收使用的实际后端")
    arguments = parser.parse_args()
    if not arguments.tiny and arguments.checkpoint is None:
        parser.error("完整模型导出必须提供 --checkpoint")
    if arguments.tiny and arguments.checkpoint is not None:
        parser.error("--tiny 不读取检查点，请移除 --checkpoint")
    if arguments.tiny and arguments.audio_tokenizer is not None:
        parser.error("--tiny 不导出音频组件，请移除 --audio-tokenizer")
    if arguments.tiny and arguments.precision != "float32":
        parser.error("--tiny 使用原始 FP32 随机权重，仅支持 --precision float32")
    if not arguments.runtime_python.is_file():
        parser.error(f"GGUF 验证解释器不存在：{arguments.runtime_python}")
    audio_tokenizer_path = None
    if not arguments.tiny:
        audio_tokenizer_path = bundle.validate_checkpoint_assets(arguments.checkpoint, arguments.audio_tokenizer)
    torch.set_num_threads(2)
    torch.manual_seed(20261104)
    raw = factory.tiny_config() if arguments.tiny else json.loads((arguments.checkpoint / "config.json").read_text(encoding="utf-8"))
    modules = factory.source_modules(arguments.source_dir)
    bundle_root = arguments.output.resolve()
    destination = bundle_root / "ckpt"
    destination.mkdir(parents=True, exist_ok=True)
    (bundle_root / "manifest.json").unlink(missing_ok=True)
    (destination / "export-report.json").unlink(missing_ok=True)
    report: dict[str, Any] = {
        "source_commit": factory.SOURCE_COMMIT, "checkpoint_revision": factory.CHECKPOINT_REVISION,
        "tiny": arguments.tiny, "precision": arguments.precision, "onnx_precision": "float32", "quantized": False,
        "graphs": {}, "excluded_components": ["codec_model", "embed_text_tokens"],
    }
    exported_files: set[Path] = set()
    with TemporaryDirectory(prefix="breeze-gguf-reference-") as temporary, torch.inference_mode():
        checkpoint = arguments.checkpoint
        if arguments.tiny:
            checkpoint = write_tiny_checkpoint(Path(temporary), raw, modules)
        for name in ("text_encoder", "audio_embedding"):
            print(f"导出 {name}：构建与加载 FP32 组件", flush=True)
            graph = factory.build_graph(name, raw, modules, meta=not arguments.tiny)
            if not arguments.tiny or name == "audio_embedding":
                weights.load_graph_weights(graph, name, checkpoint)
            reference_error = comparison.verify_reference(name, graph, raw)
            inputs, outputs, dynamic = factory.graph_signature(name, graph)
            path = destination / f"{name}.onnx"
            bundle.export_isolated_graph(
                graph, factory.graph_inputs(name, graph, raw), path,
                input_names=inputs, output_names=outputs, dynamic_axes=dynamic,
                opset_version=17, dynamo=False, external_data=True,
            )
            verified = comparison.verify_onnx(path, name, graph, raw)
            verified["upstream_maximum_absolute_error"] = reference_error
            report["graphs"][name] = verified
            exported_files.update(bundle.graph_files(path))
            del graph
            gc.collect()
            if name == "text_encoder" and not arguments.tiny:
                from .text_encoder_split import split_text_encoder

                exported_files.update(split_text_encoder(destination))
                report["text_encoder_split"] = {
                    "validated_structure": True, "precision": "float32", "lookup_backend": "cpu", "trunk_backend": "cuda",
                }
        report["graphs"]["depth_conditioner"] = conditioner_export.export_depth_conditioner(checkpoint, destination, raw)
        exported_files.update(bundle.graph_files(destination / "depth_conditioner.onnx"))
        print("导出并校验 GGUF 主干与深度解码器", flush=True)
        generation_export = gguf_export.export_generation_gguf(checkpoint, destination, precision=arguments.precision)
        from .gguf_comparison import verify_gguf_generation

        generation_validation = verify_gguf_generation(
            destination, raw, modules, checkpoint, python=arguments.runtime_python,
            backend=arguments.llama_backend, precision=arguments.precision,
        )
        report["generation"] = {
            "validated": True, "precision": arguments.precision,
            "export": generation_export, "validation": generation_validation,
        }
        exported_files.update(destination / filename for filename in ("backbone.gguf", "depth.gguf"))
        if not arguments.tiny:
            from qwen_tts import Qwen3TTSTokenizer
            from .codec_export import export_codec

            print("导出音频组件：加载本地官方分词器", flush=True)
            audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                str(audio_tokenizer_path.resolve()), device_map="cpu", local_files_only=True,
            )
            report["codec"] = export_codec(audio_tokenizer, destination)
            del audio_tokenizer
            gc.collect()
            for filename in ("audio_encoder.onnx", "decoder.onnx", "decoder_step.onnx"):
                exported_files.update(bundle.graph_files(destination / filename))
            exported_files.add(destination / "codec_metadata.json")
            bundle.copy_tokenizer_assets(arguments.checkpoint, destination)
            exported_files.update(
                destination / filename for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                if (destination / filename).is_file()
            )
        bundle.write_bundle(
            bundle_root, raw, report, exported_files, source_dir=arguments.source_dir,
            checkpoint=arguments.checkpoint, tiny=arguments.tiny,
        )
    print(json.dumps({"output": str(bundle_root), "graphs": list(report["graphs"]), "tiny": arguments.tiny}, ensure_ascii=False))


if __name__ == "__main__":
    main()
