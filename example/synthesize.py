"""使用自行准备的参考音频，逐块合成并保存单声道 WAV。"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path


def main() -> int:
    """解析命令行输入，构建临时音色并保存合成结果。

    Returns:
        合成成功时返回零；输入无效或运行失败时抛出异常。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-tts-clone-0.6b-gguf", help="模型名或本地 bundle 目录。")
    parser.add_argument("--reference-audio", required=True, type=Path, help="自行准备且有权使用的参考音频。")
    parser.add_argument("--reference-text", required=True, help="与参考音频内容一致的转写文本。")
    parser.add_argument("--text", required=True, help="需要合成的文本。")
    parser.add_argument("--output", type=Path, default=Path("output.wav"), help="输出 WAV 文件路径。")
    parser.add_argument("--device", default="cpu", help="auto、cpu、gpu 或 gpu:设备编号。")
    parser.add_argument("--llama-backend", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--onnx-provider", choices=("cpu", "directml", "cuda"), default="cpu")
    parser.add_argument("--stream-kwargs-json", default="{}", help="模型私有合成参数的 JSON 对象。")
    args = parser.parse_args()

    if not args.reference_audio.is_file():
        parser.error("参考音频不存在，请通过 --reference-audio 指定自己的音频文件。")
    if not args.reference_text.strip() or not args.text.strip():
        parser.error("参考转写和目标文本都不能为空。")
    if args.output.suffix.lower() != ".wav":
        parser.error("输出文件必须使用 .wav 扩展名。")
    if args.reference_audio.resolve() == args.output.resolve():
        parser.error("输出路径不能覆盖参考音频。")
    try:
        stream_kwargs = json.loads(args.stream_kwargs_json)
    except json.JSONDecodeError as exc:
        parser.error(f"合成参数不是有效的 JSON：{exc}")
    if not isinstance(stream_kwargs, dict):
        parser.error("--stream-kwargs-json 必须是 JSON 对象。")
    if {"text", "role_name"}.intersection(stream_kwargs):
        parser.error("text 和 role_name 由示例本身提供，请勿在 JSON 参数中重复指定。")

    # 帮助和输入检查不需要加载模型依赖，避免查看用法时触发运行环境初始化。
    import soundfile as sf
    from streamvox import TTSEngine

    engine = TTSEngine(
        model=args.model,
        device=args.device,
        llama_backend=args.llama_backend,
        onnx_provider=args.onnx_provider,
    )
    try:
        prompt = engine.make_prompt(
            role_name="example_voice",
            audio_path=str(args.reference_audio),
            prompt_text=args.reference_text,
            persist=False,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        samples_written = 0
        # 逐块写入，避免为较长文本在内存中保存全部音频。
        with closing(engine.stream(text=args.text, role_name=prompt, **stream_kwargs)) as chunks:
            with sf.SoundFile(args.output, mode="w", samplerate=engine.runtime.sample_rate,
                              channels=1, subtype="PCM_16") as output:
                for chunk in chunks:
                    output.write(chunk.reshape(-1))
                    samples_written += chunk.size
        if samples_written == 0:
            raise RuntimeError("本次合成没有生成音频，请检查模型日志和输入。")
        print(f"已保存：{args.output.resolve()}，采样点数：{samples_written}")
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
