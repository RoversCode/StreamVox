from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# 关键启动修复：
# 该脚本通常通过 `python tools/benchmark_streamvox_models.py` 直接执行，
# 此时 Python 会优先把 `tools/` 作为脚本目录放进 `sys.path`，
# 仓库根目录不一定稳定排在可导入路径里。后续子流程在 `create_engine`
# 中需要 `import streamvox`，如果这里不显式补上 repo root，就会出现
# `ModuleNotFoundError: No module named 'streamvox'`。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resource_probe import ResourceMonitor, ResourceSummary


DEFAULT_PROMPT_TEXT = "所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。"
DEFAULT_SYNTH_TEXT = "我知道，这次更新解决了一个关键痛点。我们用同一段短文本测试模型加载、构建音色提示和流式合成时的资源峰值。" * 20


@dataclass(frozen=True, slots=True)
class ModelCase:
    """
    单个 StreamVox 模型测试用例配置。

    参数:
        key: 命令行和结果文件中使用的稳定短名称。
        display_name: 面向测试报告展示的模型名称。
        model_dir: 传给 TTSEngine 的 model 参数；既可以是本地 bundle 路径，也可以是 registry 中的模型名。
        synthesize_kwargs: 传给 TTSEngine.stream 的模型私有参数。

    返回:
        数据类用于表达一次通用 benchmark case，避免脚本把模型列表和私有参数写死在代码里。

    异常:
        本数据类不主动抛出异常；路径存在性由运行阶段统一校验。
    """

    key: str
    display_name: str
    model_dir: str
    synthesize_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhaseTiming:
    """
    单个业务阶段耗时记录。

    参数:
        label: 阶段名称。
        seconds: 阶段耗时，单位为秒。

    返回:
        数据类用于输出 JSON 和 Markdown 报告。

    异常:
        本数据类不主动抛出异常。
    """

    label: str
    seconds: float


@dataclass(slots=True)
class ModelRunResult:
    """
    单个模型完整资源测试结果。

    参数:
        model_key: 测试用例短名称。
        display_name: 模型展示名称。
        model_dir: 本地模型 bundle 路径。
        physical_gpu_index: nvidia-smi 视角下的物理 GPU 编号。
        cuda_visible_devices: 子进程实际设置的 CUDA_VISIBLE_DEVICES。
        engine_device: 传给 TTSEngine 的 device 参数。
        success: 测试是否成功跑完整个流程。
        error: 失败时的异常描述。
        timings: 初始化、prompt、首包和完整合成等阶段耗时。
        audio_samples: 合成出的音频采样点数量。
        audio_seconds: 合成音频时长。
        resource_summary: 资源监控摘要。

    返回:
        数据类最终序列化为 JSON，供后续对比和归档。

    异常:
        本数据类不主动抛出异常。
    """

    model_key: str
    display_name: str
    model_dir: str
    physical_gpu_index: int
    cuda_visible_devices: str
    engine_device: str
    success: bool
    error: str | None
    timings: list[PhaseTiming]
    audio_samples: int
    audio_seconds: float
    resource_summary: ResourceSummary


PRESET_MODEL_CASES = {
    "qwen3-0.6b": ModelCase(
        key="qwen3-0.6b",
        display_name="Qwen3 TTS Clone 0.6B GGUF",
        model_dir="ckpts/qwen3-tts-clone-0.6b-gguf",
        synthesize_kwargs={"language": "chinese", "stream": True},
    ),
    "qwen3-1.7b": ModelCase(
        key="qwen3-1.7b",
        display_name="Qwen3 TTS Clone 1.7B GGUF",
        model_dir="ckpts/qwen3-tts-clone-1.7b-gguf",
        synthesize_kwargs={"language": "chinese", "stream": True},
    ),
    "s2-pro": ModelCase(
        key="s2-pro",
        display_name="S2-Pro 4B GGUF",
        model_dir="ckpts/s2-pro-4b-gguf",
        synthesize_kwargs={},
    ),
}


def parse_json_argument(raw_value: str) -> Any:
    """
    解析命令行 JSON 参数，支持直接传 JSON 字符串或 `@path/to/file.json`。

    参数:
        raw_value: 原始命令行字符串。

    返回:
        返回解析后的 Python 对象。

    异常:
        JSON 非法、文件不存在或内容不可读时抛出 ValueError/FileNotFoundError。
    """
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("json argument must be a non-empty string")
    if raw_value.startswith("@"):
        json_path = Path(raw_value[1:]).expanduser()
        if not json_path.exists():
            raise FileNotFoundError(f"json file not found: {json_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))
    return json.loads(raw_value)


def derive_case_key(model_ref: str) -> str:
    """
    从模型路径或模型名推导稳定 case key。

    参数:
        model_ref: 传给 TTSEngine 的模型标识。

    返回:
        返回仅包含字母、数字、点、下划线和中划线的短 key。

    异常:
        输入为空或清洗后为空时抛出 ValueError，避免输出文件名不可读。
    """
    raw_text = str(model_ref).strip()
    if not raw_text:
        raise ValueError("model reference must not be empty")
    candidate = Path(raw_text).name or raw_text
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-._")
    if not normalized:
        raise ValueError(f"cannot derive a valid case key from model reference: {model_ref}")
    return normalized


def derive_display_name(model_ref: str) -> str:
    """
    从模型路径或模型名推导默认展示名称。

    参数:
        model_ref: 传给 TTSEngine 的模型标识。

    返回:
        优先返回路径 basename；否则返回原始模型名。

    异常:
        本函数只做字符串规整，不主动抛出异常。
    """
    raw_text = str(model_ref).strip()
    if not raw_text:
        return raw_text
    return Path(raw_text).name or raw_text


def parse_stream_kwargs_json(raw_value: str | None) -> dict[str, Any]:
    """
    解析 `TTSEngine.stream` 的模型私有参数 JSON。

    参数:
        raw_value: JSON 字符串或 `@json-file`；为空时返回空字典。

    返回:
        返回字典，供 benchmark 原样透传给 `TTSEngine.stream`。

    异常:
        JSON 不是对象时抛出 TypeError，避免把数组或标量误当成关键字参数。
    """
    if raw_value in (None, ""):
        return {}
    payload = parse_json_argument(raw_value)
    if not isinstance(payload, dict):
        raise TypeError("stream kwargs json must decode to an object")
    return payload


def case_from_payload(payload: dict[str, Any]) -> ModelCase:
    """
    从 JSON payload 构造通用 benchmark case。

    参数:
        payload: 用户通过 `--case-json` 或隐藏子进程参数传入的对象。

    返回:
        返回标准化后的 `ModelCase`。

    异常:
        缺少模型标识、字段类型不合法或 stream kwargs 不是对象时抛出异常。
    """
    if not isinstance(payload, dict):
        raise TypeError("case payload must be an object")
    model_ref = payload.get("model")
    if model_ref is None:
        model_ref = payload.get("model_dir")
    if not isinstance(model_ref, str) or not model_ref.strip():
        raise ValueError("case payload must include non-empty 'model' or 'model_dir'")
    raw_stream_kwargs = payload.get("stream_kwargs", payload.get("synthesize_kwargs", {}))
    if raw_stream_kwargs is None:
        raw_stream_kwargs = {}
    if not isinstance(raw_stream_kwargs, dict):
        raise TypeError("case payload field 'stream_kwargs' must be an object")
    case_key = payload.get("key")
    if case_key is None:
        case_key = derive_case_key(model_ref)
    if not isinstance(case_key, str) or not case_key.strip():
        raise ValueError("case key must be a non-empty string")
    display_name = payload.get("display_name")
    if display_name is None:
        display_name = derive_display_name(model_ref)
    if not isinstance(display_name, str):
        raise TypeError("display_name must be a string")
    return ModelCase(
        key=case_key.strip(),
        display_name=display_name.strip() or derive_display_name(model_ref),
        model_dir=model_ref.strip(),
        synthesize_kwargs=raw_stream_kwargs,
    )


def serialize_case_payload(case: ModelCase) -> dict[str, Any]:
    """
    将 `ModelCase` 转成可跨进程传递的 JSON 对象。

    参数:
        case: 当前 benchmark case。

    返回:
        返回结构稳定的字典，供子进程反序列化。

    异常:
        本函数不主动抛出异常。
    """
    return {
        "key": case.key,
        "display_name": case.display_name,
        "model": case.model_dir,
        "stream_kwargs": case.synthesize_kwargs,
    }


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数:
        无。

    返回:
        返回 argparse.Namespace，包含测试模型、GPU、输入文本和输出路径等配置。

    异常:
        argparse 会在参数非法时打印错误并退出进程。
    """
    parser = argparse.ArgumentParser(description="Benchmark StreamVox model RAM and VRAM usage.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(PRESET_MODEL_CASES),
        help="内置模型预设 key；不传且未指定自定义 case 时，默认测试全部内置预设。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="自定义单模型的 model 参数；可传本地 bundle 路径或 registry 模型名。传入后无需修改脚本即可测试新模型。",
    )
    parser.add_argument("--model-key", default=None, help="自定义单模型的稳定短 key；不传则根据 --model 自动推导。")
    parser.add_argument("--display-name", default=None, help="自定义单模型的展示名称；不传则根据 --model 自动推导。")
    parser.add_argument(
        "--stream-kwargs-json",
        default=None,
        help="自定义单模型的 stream 私有参数 JSON，支持直接传 JSON 字符串或 @json-file。",
    )
    parser.add_argument(
        "--case-json",
        action="append",
        default=[],
        help=(
            "追加一个通用 benchmark case，支持 JSON 字符串或 @json-file。"
            "对象字段支持 key/display_name/model/stream_kwargs。"
        ),
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=1,
        help="nvidia-smi 视角下的物理 GPU 编号；当前任务固定使用 cuda:1。",
    )
    parser.add_argument(
        "--engine-device",
        default=None,
        help="传给 TTSEngine 的 device。默认跟随 --physical-gpu-index 生成 gpu:<index>，避免 Vulkan 后端绕过 CUDA_VISIBLE_DEVICES 后误用 0 卡。",
    )
    parser.add_argument(
        "--prompt-audio",
        default="reference.wav",
        help="参考音频路径；默认沿用现有样例音频。若要测试支持默认/空 prompt 的模型，请配合 --skip-make-prompt。",
    )
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT, help="参考音频对应文本；仅在传入 --prompt-audio 时使用。")
    parser.add_argument(
        "--skip-make-prompt",
        action="store_true",
        help="跳过 make_prompt，直接调用 stream；用于测试支持默认/空 prompt 的模型，例如 voxcpm2 text 模式。",
    )
    parser.add_argument("--text", default=DEFAULT_SYNTH_TEXT, help="用于合成测试的目标文本。")
    parser.add_argument("--sample-interval", type=float, default=0.2, help="资源采样间隔，单位秒。")
    parser.add_argument("--output-dir", default="tests/output/resource_benchmark", help="单模型 JSON 输出目录。")
    parser.add_argument(
        "--summary-json",
        default="tests/output/streamvox_resource_benchmark.json",
        help="多模型汇总 JSON 路径。",
    )
    parser.add_argument(
        "--summary-md",
        default="tests/output/streamvox_resource_benchmark.md",
        help="多模型汇总 Markdown 路径。",
    )
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    parser.add_argument("--child-output-json", help=argparse.SUPPRESS)
    parser.add_argument("--child-case-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # 默认让 TTSEngine 的设备索引与 nvidia-smi 物理卡编号一致；如果要兜底跑 0 卡，只需传 --physical-gpu-index 0。
    if args.engine_device is None:
        args.engine_device = f"gpu:{args.physical_gpu_index}"
    return args


def resolve_cases(args: argparse.Namespace) -> list[ModelCase]:
    """
    根据命令行参数解析本次要执行的 benchmark cases。

    参数:
        args: 解析后的命令行参数。

    返回:
        返回标准化后的 `ModelCase` 列表。

    异常:
        case key 重复、JSON 非法或字段冲突时抛出异常，避免多个模型覆盖同一输出文件。
    """
    cases: list[ModelCase] = []

    # 兼容旧用法：既没有指定内置 key，也没有自定义模型/case 时，默认跑全部预设。
    if args.models is None and args.model is None and not args.case_json:
        cases.extend(PRESET_MODEL_CASES.values())
    else:
        if args.models:
            for model_key in args.models:
                cases.append(PRESET_MODEL_CASES[model_key])
        if args.model is not None:
            cases.append(
                ModelCase(
                    key=args.model_key or derive_case_key(args.model),
                    display_name=args.display_name or derive_display_name(args.model),
                    model_dir=str(args.model),
                    synthesize_kwargs=parse_stream_kwargs_json(args.stream_kwargs_json),
                )
            )
        for raw_case_json in args.case_json:
            payload = parse_json_argument(raw_case_json)
            cases.append(case_from_payload(payload))

    if not cases:
        raise ValueError("no benchmark cases resolved; provide --models, --model or --case-json")

    seen_keys: set[str] = set()
    for case in cases:
        if case.key in seen_keys:
            raise ValueError(f"duplicate benchmark case key: {case.key}")
        seen_keys.add(case.key)
    return cases


def main() -> None:
    """
    脚本入口。

    参数:
        无；所有配置来自命令行参数。

    返回:
        无返回值；结果写入 JSON 和 Markdown 文件。

    异常:
        子进程模式会把单模型异常写入结果 JSON；父进程在任一模型失败时以非零状态退出。
    """
    args = parse_args()
    # 子进程只负责一个模型，保证每个模型的内存和显存基线互不污染。
    if args.child_case_json:
        result = run_single_model(args, case_from_payload(parse_json_argument(args.child_case_json)))
        write_json(Path(args.child_output_json), asdict(result))
        if not result.success:
            raise SystemExit(1)
        return

    resolved_cases = resolve_cases(args)
    print(
        f"[benchmark] starting {len(resolved_cases)} case(s), gpu={args.physical_gpu_index}, output_dir={args.output_dir}",
        flush=True,
    )
    results = run_all_models(args, resolved_cases)
    write_json(Path(args.summary_json), results)
    write_markdown_report(Path(args.summary_md), results)
    print(f"[benchmark] summary json: {args.summary_json}", flush=True)
    print(f"[benchmark] summary md:   {args.summary_md}", flush=True)
    # 只要模型测量失败或子进程退出异常，父进程就返回失败，方便 CI 或人工测试一眼看到问题。
    if any((not item["success"]) or (not item.get("process_exited_cleanly", True)) for item in results):
        raise SystemExit(1)


def run_all_models(args: argparse.Namespace, cases: list[ModelCase]) -> list[dict[str, Any]]:
    """
    逐模型拉起独立子进程执行资源测试。

    参数:
        args: 父进程命令行参数。

    返回:
        返回所有模型的结果字典列表。

    异常:
        子进程崩溃且没有写出 JSON 时，会生成一个失败结果继续测试后续模型。
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for case in cases:
        model_key = case.key
        child_json = output_dir / f"{model_key}.json"
        stdout_log = output_dir / f"{model_key}.stdout.log"
        stderr_log = output_dir / f"{model_key}.stderr.log"
        # 终端显式打印当前测试项，避免父进程在长时间执行时看起来像“没有反应”。
        print(
            f"[benchmark] running case={case.key} model={case.model_dir} log={stdout_log.name}",
            flush=True,
        )
        command = build_child_command(args, case, child_json)
        env = build_child_environment(args.physical_gpu_index)
        start_time = time.monotonic()
        # 每个模型独立进程执行，避免前一个 runtime 释放不彻底导致峰值串台。
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        elapsed_seconds = time.monotonic() - start_time

        if child_json.exists():
            payload = json.loads(child_json.read_text(encoding="utf-8"))
            payload["process_returncode"] = completed.returncode
            payload["process_seconds"] = elapsed_seconds
            # 子进程可能已经完成测量并写出 JSON，但在 Python/native runtime 析构阶段崩溃。
            # 这里保留测量成功状态，同时单独记录进程退出是否干净，避免报告把 teardown 崩溃误写成完全成功。
            payload["process_exited_cleanly"] = completed.returncode == 0
            payload["stdout_log"] = str(stdout_log)
            payload["stderr_log"] = str(stderr_log)
            print(
                "[benchmark] finished case={key} success={success} returncode={code} json={json_path}".format(
                    key=case.key,
                    success=payload.get("success"),
                    code=completed.returncode,
                    json_path=child_json,
                ),
                flush=True,
            )
            results.append(payload)
            continue

        # 子进程如果在导入阶段就崩溃，可能来不及写 JSON；这里补一个可读失败项。
        print(
            f"[benchmark] failed case={case.key} returncode={completed.returncode} without result json; see {stderr_log}",
            flush=True,
        )
        results.append(
            {
                "model_key": case.key,
                "display_name": case.display_name,
                "model_dir": case.model_dir,
                "physical_gpu_index": args.physical_gpu_index,
                "cuda_visible_devices": str(args.physical_gpu_index),
                "engine_device": args.engine_device,
                "success": False,
                "error": f"child process exited with {completed.returncode} before writing result json",
                "timings": [{"label": "process", "seconds": elapsed_seconds}],
                "audio_samples": 0,
                "audio_seconds": 0.0,
                "resource_summary": {},
                "process_returncode": completed.returncode,
                "process_exited_cleanly": completed.returncode == 0,
                "process_seconds": elapsed_seconds,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        )

    return results


def build_child_command(args: argparse.Namespace, case: ModelCase, child_json: Path) -> list[str]:
    """
    构建单模型子进程命令。

    参数:
        args: 父进程命令行参数。
        case: 需要运行的通用 benchmark case。
        child_json: 子进程结果 JSON 输出路径。

    返回:
        返回 subprocess.run 可直接使用的命令列表。

    异常:
        本函数不执行命令，不主动抛出异常。
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-case-json",
        json.dumps(serialize_case_payload(case), ensure_ascii=False, separators=(",", ":")),
        "--child-output-json",
        str(child_json),
        "--physical-gpu-index",
        str(args.physical_gpu_index),
        "--engine-device",
        args.engine_device,
        "--prompt-text",
        args.prompt_text,
        "--text",
        args.text,
        "--sample-interval",
        str(args.sample_interval),
    ]
    if args.skip_make_prompt:
        command.append("--skip-make-prompt")
    elif args.prompt_audio is not None:
        command.extend(["--prompt-audio", args.prompt_audio])
    return command


def build_child_environment(physical_gpu_index: int) -> dict[str, str]:
    """
    构建子进程环境变量。

    参数:
        physical_gpu_index: 需要暴露给子进程的物理 GPU 编号。

    返回:
        返回环境变量字典；默认清理 CUDA_VISIBLE_DEVICES，让 device=gpu:<index> 使用物理卡编号。

    异常:
        本函数不访问外部系统，不主动抛出异常。
    """
    env = os.environ.copy()
    # 不使用 CUDA_VISIBLE_DEVICES 重映射。
    # 业务原因：当前 GGUF 侧可能走 llama.cpp Vulkan 后端，Vulkan 的 main_gpu 不遵循 CUDA_VISIBLE_DEVICES；
    # 如果把物理 1 卡映射成 CUDA 0，再给 TTSEngine 传 gpu:0，就会出现 ONNX 跑 1 卡、Vulkan 跑 0 卡的混用。
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    return env


def run_single_model(args: argparse.Namespace, case: ModelCase) -> ModelRunResult:
    """
    在当前进程内运行一个模型的完整资源测试。

    参数:
        args: 子进程命令行参数。
        case: 当前模型测试配置。

    返回:
        返回单模型结果，包含成功状态、耗时、音频长度和资源峰值。

    异常:
        捕获模型执行异常并写入结果对象，确保父进程能得到结构化失败原因。
    """
    # 子进程再次清理可见卡重映射，确保 TTSEngine 的 gpu:1 被所有后端理解为物理 1 卡。
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("NVIDIA_VISIBLE_DEVICES", None)

    timings: list[PhaseTiming] = []
    monitor = ResourceMonitor(
        physical_gpu_index=args.physical_gpu_index,
        interval_seconds=args.sample_interval,
    )
    monitor.start()
    audio_samples = 0
    audio_seconds = 0.0
    error: str | None = None
    success = False
    engine = None

    try:
        validate_inputs(case, args)
        monitor.set_label("engine_init")
        engine, seconds = timed_call(
            lambda: create_engine(
                model_dir=case.model_dir,
                engine_device=args.engine_device,
            )
        )
        timings.append(PhaseTiming("engine_init", seconds))

        prompt = None
        # 显式要求跳过 make_prompt 时，直接走 engine.stream；
        # 否则沿用旧逻辑，用参考音频构建 prompt。
        if not args.skip_make_prompt:
            monitor.set_label("make_prompt")
            prompt, seconds = timed_call(
                lambda: engine.make_prompt(
                    role_name=f"resource_probe_{case.key}",
                    audio_path=args.prompt_audio,
                    prompt_text=args.prompt_text,
                    persist=False,
                )
            )
            timings.append(PhaseTiming("make_prompt", seconds))

        monitor.set_label("synthesize")
        chunks, first_chunk_seconds, synthesize_seconds = collect_stream(
            engine=engine,
            text=args.text,
            prompt=prompt,
            synthesize_kwargs=case.synthesize_kwargs,
            monitor=monitor,
        )
        timings.append(PhaseTiming("first_chunk", first_chunk_seconds))
        timings.append(PhaseTiming("synthesize", synthesize_seconds))
        audio_samples = sum(int(np.asarray(chunk).size) for chunk in chunks)
        audio_seconds = audio_samples / float(engine.runtime.sample_rate)
        success = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        monitor.set_label("shutdown")
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as shutdown_exc:
                if error is None:
                    error = f"shutdown failed: {type(shutdown_exc).__name__}: {shutdown_exc}"
                    success = False
        # 显式触发垃圾回收，帮助采样线程在 stop 前捕捉 shutdown 后的资源状态。
        gc.collect()

    resource_summary = monitor.stop()
    return ModelRunResult(
        model_key=case.key,
        display_name=case.display_name,
        model_dir=case.model_dir,
        physical_gpu_index=args.physical_gpu_index,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        engine_device=args.engine_device,
        success=success,
        error=error,
        timings=timings,
        audio_samples=audio_samples,
        audio_seconds=audio_seconds,
        resource_summary=resource_summary,
    )


def validate_inputs(case: ModelCase, args: argparse.Namespace) -> None:
    """
    校验本次测试需要的本地输入。

    参数:
        case: 当前模型测试配置。
        args: 命令行参数。

    返回:
        校验通过时无返回值。

    异常:
        模型标识无法解析或参考音频不存在时抛出异常，避免 runtime 初始化后才失败。
    """
    # model 参数既可能是本地 bundle 路径，也可能是 registry 已注册模型名；
    # 两者都不成立时，再提前失败，避免把输入错误混淆成 runtime 初始化异常。
    if not Path(case.model_dir).exists():
        from streamvox.models import default_registry

        if case.model_dir not in default_registry.list_models():
            raise FileNotFoundError(f"model path not found and registry model is unsupported: {case.model_dir}")
    # prompt 构建依赖参考音频；只有未显式跳过 make_prompt 时，才要求样例音频存在。
    if not args.skip_make_prompt and not Path(args.prompt_audio).exists():
        raise FileNotFoundError(f"prompt audio not found: {args.prompt_audio}")


def create_engine(*, model_dir: str, engine_device: str):
    """
    创建 StreamVox TTS 引擎。

    参数:
        model_dir: 传给 TTSEngine 的 model 参数；可为本地 bundle 路径或 registry 模型名。
        engine_device: TTSEngine 设备参数；本测试默认传 gpu:1，让 CUDA 和 Vulkan 后端都使用物理 1 卡。

    返回:
        返回已加载 runtime 的 TTSEngine 实例。

    异常:
        模型文件不完整、设备不可用或 runtime 初始化失败时由 TTSEngine 抛出异常。
    """
    from streamvox import TTSEngine

    return TTSEngine(
        model=model_dir,
        verify_model_sha256=False,
        device=engine_device,
    )


def collect_stream(
    *,
    engine: Any,
    text: str,
    prompt: Any,
    synthesize_kwargs: dict[str, Any],
    monitor: ResourceMonitor,
) -> tuple[list[np.ndarray], float, float]:
    """
    拉取一次完整流式合成结果。

    参数:
        engine: 已初始化的 TTSEngine。
        text: 目标合成文本。
        prompt: 可选的 prompt 资产；为空时直接让 engine 按模型默认语义处理。
        synthesize_kwargs: 模型私有合成参数。
        monitor: 资源监控器，用于在首包后切换阶段标签。

    返回:
        返回音频分片列表、首包耗时和完整合成耗时。

    异常:
        底层模型合成失败时向上抛出，外层会记录为结构化错误。
    """
    chunks: list[np.ndarray] = []
    start_time = time.monotonic()
    first_chunk_seconds = 0.0
    stream = engine.stream(
        text,
        role_name=prompt,
        track_performance=False,
        remove_meaningless_chars=True,
        **synthesize_kwargs,
    )
    for index, chunk in enumerate(stream):
        # 首包耗时单独记录，便于区分模型预热与后续持续生成开销。
        if index == 0:
            first_chunk_seconds = time.monotonic() - start_time
            monitor.set_label("synthesize_after_first_chunk")
        chunks.append(np.asarray(chunk, dtype=np.float32))
    total_seconds = time.monotonic() - start_time
    return chunks, first_chunk_seconds, total_seconds


def timed_call(callback: Any) -> tuple[Any, float]:
    """
    执行回调并记录耗时。

    参数:
        callback: 无参可调用对象。

    返回:
        返回回调结果和耗时秒数。

    异常:
        回调内部异常会原样向上抛出，避免吞掉真实失败原因。
    """
    start_time = time.monotonic()
    result = callback()
    return result, time.monotonic() - start_time


def write_json(path: Path, payload: Any) -> None:
    """
    写入 JSON 文件。

    参数:
        path: 输出路径。
        payload: 可 JSON 序列化的数据。

    返回:
        无返回值。

    异常:
        路径不可写时由 pathlib 抛出 OSError。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(path: Path, results: list[dict[str, Any]]) -> None:
    """
    写入 Markdown 汇总报告。

    参数:
        path: 输出路径。
        results: 所有模型的结果字典。

    返回:
        无返回值。

    异常:
        路径不可写时由 pathlib 抛出 OSError。
    """
    lines = [
        "# StreamVox 模型资源占用测试",
        "",
        "测试口径：每个模型独立子进程运行，子进程清理 `CUDA_VISIBLE_DEVICES`，引擎直接使用 `device=gpu:1`，让 CUDA/ONNX 与 llama.cpp Vulkan 都按物理卡编号命中 `cuda:1`。",
        "",
        "| 模型 | 状态 | 峰值内存 MiB | 峰值进程显存 MiB | 峰值整卡显存增量 MiB | 初始化 s | Prompt s | 首包 s | 合成 s | 音频 s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        summary = item.get("resource_summary", {})
        timings = {timing["label"]: timing["seconds"] for timing in item.get("timings", [])}
        lines.append(
            "| {model} | {status} | {rss} | {proc_gpu} | {delta_gpu} | {init} | {prompt} | {first} | {synth} | {audio} |".format(
                model=item.get("display_name", item.get("model_key", "")),
                status=format_status(item),
                rss=format_number(summary.get("peak_rss_mib")),
                proc_gpu=format_number(summary.get("peak_gpu_process_used_mib")),
                delta_gpu=format_number(summary.get("peak_gpu_total_delta_mib")),
                init=format_number(timings.get("engine_init")),
                prompt=format_number(timings.get("make_prompt")),
                first=format_number(timings.get("first_chunk")),
                synth=format_number(timings.get("synthesize")),
                audio=format_number(item.get("audio_seconds")),
            )
        )

    lines.extend(
        [
            "",
            "说明：`峰值进程显存` 来自 `nvidia-smi --query-compute-apps` 对当前进程树的归因；`峰值整卡显存增量` 来自物理 GPU 1 的整卡已用显存减测试前基线，可覆盖部分进程归因不完整的后端。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_status(item: dict[str, Any]) -> str:
    """
    格式化单模型测试状态。

    参数:
        item: 单模型结果字典，包含测量状态和子进程退出码。

    返回:
        返回 Markdown 表格中的状态文本。

    异常:
        本函数只读取字典字段，不主动抛出异常。
    """
    # 模型执行本身失败时优先展示业务异常，退出码只是附加线索。
    if not item.get("success"):
        return f"失败：{item.get('error')}"
    # 测量已完成但进程退出异常时必须显式写入报告，避免掩盖 native teardown 问题。
    if not item.get("process_exited_cleanly", True):
        return f"测量完成，退出异常({item.get('process_returncode')})"
    return "成功"


def format_number(value: Any) -> str:
    """
    格式化报告中的数字。

    参数:
        value: 任意待展示值。

    返回:
        数字返回两位小数字符串；空值返回短横线。

    异常:
        不可转为浮点数的值会原样转成字符串，避免报告生成失败。
    """
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
