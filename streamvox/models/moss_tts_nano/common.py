from __future__ import annotations

from ..onnx_backend import verify_session_provider

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from streamvox_runtime import ensure_single_onnxruntime_distribution, initialize_nvidia_runtime
ensure_single_onnxruntime_distribution()
import onnxruntime as ort

from ...utils.log import logger


SAMPLE_MODE_GREEDY = "greedy"
SAMPLE_MODE_FIXED = "fixed"
SAMPLE_MODE_FULL = "full"

MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
)



@dataclass(frozen=True, slots=True)
class MossOnnxBundle:
    """
    MOSS ONNX bundle 元信息。

    参数:
        asset_root: StreamVox bundle 的 `ckpt` 根目录。
        manifest_path: browser_poc_manifest.json 的绝对路径。
        manifest_dir: manifest 所在目录，用于解析相对路径。
        manifest: browser ONNX manifest 内容。
        tts_meta_path: TTS meta 文件绝对路径。
        codec_meta_path: codec meta 文件绝对路径。
        tts_meta: TTS meta 内容。
        codec_meta: codec meta 内容。
        tokenizer_path: tokenizer.model 绝对路径。

    返回:
        该对象只承载路径和元数据，不持有 ONNX session。

    边界异常:
        字段合法性由 `load_moss_onnx_bundle` 保证，本数据结构不主动校验。
    """

    asset_root: Path
    manifest_path: Path
    manifest_dir: Path
    manifest: dict[str, Any]
    tts_meta_path: Path
    codec_meta_path: Path
    tts_meta: dict[str, Any]
    codec_meta: dict[str, Any]
    tokenizer_path: Path


@dataclass(frozen=True, slots=True)
class MossOrtSessionBundle:
    """
    MOSS ONNX session 只读摘要。
    参数:
        model_path: 当前 session 绑定的 ONNX 文件绝对路径。
        providers: session 实际激活的 provider 列表。
        input_types: session 输入 dtype 摘要。
        output_types: session 输出 dtype 摘要。
    返回:
        该对象仅用于初始化日志与调试摘要，不参与推理流程。
    边界异常:
        字段合法性由 `summarize_onnx_session` 保证，本数据结构不主动校验。
    """

    model_path: str
    providers: list[str]
    input_types: list[str]
    output_types: list[str]




def resolve_manifest_path(asset_root: Path) -> Path:
    """
    在 `ckpt` 根目录下定位 browser_poc_manifest.json。

    参数:
        asset_root: StreamVox bundle 的 `ckpt` 根目录。

    返回:
        返回找到的 manifest 路径。

    边界异常:
        所有候选路径都不存在时抛出 FileNotFoundError。
    """

    tried_paths: list[Path] = []
    for relative_path in MANIFEST_CANDIDATE_RELATIVE_PATHS:
        candidate = (asset_root / relative_path).resolve()
        tried_paths.append(candidate)
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path_value) for path_value in tried_paths)
    raise FileNotFoundError(f"MOSS browser_poc_manifest.json not found. tried: {joined}")


def resolve_manifest_relative_path(manifest_dir: Path, relative_path: str | Path) -> Path:
    """
    将 MOSS manifest 中的相对路径解析为绝对路径。

    参数:
        manifest_dir: browser_poc_manifest.json 所在目录。
        relative_path: manifest 或 meta 中声明的相对路径。

    返回:
        返回绝对路径。

    边界异常:
        本函数不主动检查路径存在性。
    """

    return (manifest_dir / Path(relative_path)).resolve()


def load_moss_onnx_bundle(asset_root: Path) -> MossOnnxBundle:
    """
    加载 MOSS ONNX bundle 的 manifest、meta 和 tokenizer 路径。

    参数:
        asset_root: StreamVox bundle 的 `ckpt` 根目录。

    返回:
        返回 MossOnnxBundle。

    边界异常:
        manifest、meta 或 tokenizer 缺失时抛出 FileNotFoundError。
    """

    manifest_path = resolve_manifest_path(asset_root)
    manifest_dir = manifest_path.parent
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    tts_meta_path = resolve_manifest_relative_path(manifest_dir, manifest["model_files"]["tts_meta"])
    codec_meta_path = resolve_manifest_relative_path(manifest_dir, manifest["model_files"]["codec_meta"])
    tts_meta = __import__("json").loads(tts_meta_path.read_text(encoding="utf-8"))
    codec_meta = __import__("json").loads(codec_meta_path.read_text(encoding="utf-8"))
    tokenizer_path = resolve_manifest_relative_path(
        manifest_dir,
        str(manifest["model_files"].get("tokenizer_model", "tokenizer.model")),
    )
    return MossOnnxBundle(
        asset_root=asset_root,
        manifest_path=manifest_path,
        manifest_dir=manifest_dir,
        manifest=manifest,
        tts_meta_path=tts_meta_path,
        codec_meta_path=codec_meta_path,
        tts_meta=tts_meta,
        codec_meta=codec_meta,
        tokenizer_path=tokenizer_path,
    )


def build_onnx_session(path_value: Path, providers: list[str]) -> ort.InferenceSession:
    """
    创建单个 ONNX Runtime session。

    参数:
        path_value: ONNX 图文件路径。
        providers: 期望使用的 provider 列表。

    返回:
        返回已初始化的 InferenceSession。

    边界异常:
        ONNX 文件缺失、外部 data 缺失或 provider 不可用时抛出 ONNX Runtime 异常。
    """

    if "CUDAExecutionProvider" in providers:
        initialize_nvidia_runtime(consumer="onnx_cuda", required=True)

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(path_value), sess_options=options, providers=providers)
    verify_session_provider(session, providers)
    return session


def summarize_onnx_session(session: ort.InferenceSession, model_path: Path) -> MossOrtSessionBundle:
    """
    提取 MOSS ONNX session 的只读摘要。
    参数:
        session: 已初始化完成的 ORT session。
        model_path: 当前 session 对应的 ONNX 文件路径。
    返回:
        返回 providers 与 I/O dtype 摘要，供初始化日志复用。
    边界异常:
        若 ORT introspection 接口异常，会向外抛出异常，由调用方决定是否降级日志。
    """

    return MossOrtSessionBundle(
        model_path=str(Path(model_path).resolve()),
        providers=list(session.get_providers()),
        input_types=[str(item.type) for item in session.get_inputs()],
        output_types=[str(item.type) for item in session.get_outputs()],
    )


def ort_device_label(providers: list[str]) -> str:
    """
    统一格式化 ORT provider 的设备标签。
    参数:
        providers: session 实际激活的 provider 列表。
    返回:
        返回 `CUDA:0`、`DML:0`、`CPU` 等简洁设备标签。
    边界异常:
        provider 列表为空时返回 `Unknown`，避免初始化日志再次失败。
    """

    primary_provider = providers[0] if providers else ""
    if primary_provider == "DmlExecutionProvider":
        return "DML:0"
    if primary_provider == "CUDAExecutionProvider":
        return "CUDA:0"
    if primary_provider == "CPUExecutionProvider":
        return "CPU"
    if primary_provider:
        return primary_provider.replace("ExecutionProvider", "")
    return "Unknown"


def ort_precision_label(bundle: MossOrtSessionBundle) -> str:
    """
    根据 ONNX 文件名与 I/O dtype 推断精度标签。
    参数:
        bundle: session 的只读摘要。
    返回:
        返回 `FP16`、`FP32` 或 `Unknown`。
    边界异常:
        推断失败时返回 `Unknown`，不影响主链路初始化。
    """

    normalized_path = Path(bundle.model_path).stem.lower()
    if "fp16" in normalized_path:
        return "FP16"
    if "fp32" in normalized_path:
        return "FP32"

    joined_types = " ".join([*bundle.input_types, *bundle.output_types]).lower()
    if "float16" in joined_types:
        return "FP16"
    if "float)" in joined_types or "float32" in joined_types:
        return "FP32"
    return "Unknown"


def log_onnx_session_summary(*, model_name: str, module_name: str, session: ort.InferenceSession, model_path: Path) -> None:
    """
    记录 MOSS ONNX 模块初始化摘要。
    参数:
        model_name: 对外展示的模型名，例如 `moss-tts-nano-onnx`。
        module_name: 语义化模块名，例如 `Prefill`、`Codec Encode`。
        session: 已创建完成的 ORT session。
        model_path: 当前模块对应的 ONNX 文件路径。
    返回:
        无返回值；仅输出统一格式的初始化日志。
    边界异常:
        日志摘要生成失败时不应影响主链路，因此内部吞掉异常并降级为 warning。
    """

    try:
        bundle = summarize_onnx_session(session, model_path)
        logger.info(
            "✅ [%s] [%s] [%s] [%s]",
            model_name,
            module_name,
            ort_device_label(bundle.providers),
            ort_precision_label(bundle),
        )
    except Exception as exc:
        logger.warning("MOSS-TTS-Nano failed to summarize ORT session %s: %s", module_name, exc)


def flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    """
    将三维整数列表压平成 ONNX 可用的 int32 数据。

    参数:
        nested: 三维 Python 列表，通常是 batch/rows/row_width。

    返回:
        返回一维 int32 数据和原始维度。

    边界异常:
        输入维度不规则时会自然抛出异常。
    """

    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.zeros((dim0 * dim1 * dim2,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                data[offset] = int(nested[i][j][k])
                offset += 1
    return data, [dim0, dim1, dim2]


def flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    """
    将二维整数列表压平成 ONNX 可用的 int32 数据。

    参数:
        nested: 二维 Python 列表，通常是 attention mask。

    返回:
        返回一维 int32 数据和原始维度。

    边界异常:
        输入维度不规则时会自然抛出异常。
    """

    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


def extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    """
    从 global transformer 输出中提取最后一步 hidden state。

    参数:
        hidden_states: ONNX prefill/decode 返回的 `global_hidden`。

    返回:
        返回 float32 hidden state，形状为 batch/hidden。

    边界异常:
        hidden_states 不是支持的二维或三维 batch 形态时抛出 ValueError。
    """

    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


def softmax(values: np.ndarray) -> np.ndarray:
    """
    计算一维分数的 softmax 概率。

    参数:
        values: 待归一化的 logits 或分数。

    返回:
        返回 float64 概率数组。

    边界异常:
        输入包含非法数值时 numpy 会自然抛出异常或产生 NaN。
    """

    max_value = float(np.max(values))
    shifted = np.asarray(values - max_value, dtype=np.float64)
    exps = np.exp(shifted)
    return exps / np.sum(exps, dtype=np.float64)


def sample_from_scores(
    values: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """
    按 greedy、temperature、top-k 和 top-p 从分数中选择 token。

    参数:
        values: 一维候选分数。
        do_sample: 是否启用随机采样。
        temperature: 采样温度。
        top_k: 只保留分数最高的 k 个候选。
        top_p: nucleus sampling 阈值。
        rng: 当前合成会话使用的随机数生成器。

    返回:
        返回候选数组中的 token 索引。

    边界异常:
        `do_sample=True` 且 `temperature<=0` 时抛出 ValueError。
    """

    if not do_sample:
        return int(np.argmax(values))
    if not (temperature > 0):
        raise ValueError("temperature must be positive when do_sample=True")

    scores = np.asarray(values, dtype=np.float32).copy() / float(temperature)
    if top_k > 0 and top_k < scores.shape[0]:
        sorted_desc = np.sort(scores)[::-1]
        threshold = float(sorted_desc[top_k - 1])
        scores[scores < threshold] = float("-inf")

    if 0 < top_p < 1:
        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda item: item[1], reverse=True)
        sorted_scores = np.asarray([item[1] for item in indexed], dtype=np.float32)
        sorted_probs = softmax(sorted_scores)
        remove_mask = [False] * len(indexed)
        cumulative = 0.0
        for index, probability in enumerate(sorted_probs):
            cumulative += float(probability)
            if cumulative > float(top_p):
                remove_mask[index] = True
        for index in range(len(remove_mask) - 1, 0, -1):
            remove_mask[index] = remove_mask[index - 1]
        if remove_mask:
            remove_mask[0] = False
        for index, should_remove in enumerate(remove_mask):
            if should_remove:
                scores[indexed[index][0]] = float("-inf")

    probabilities = softmax(scores)
    random_value = float(rng.random())
    for index, probability in enumerate(probabilities):
        random_value -= float(probability)
        if random_value <= 0:
            return int(index)
    return int(np.argmax(scores))


def apply_repetition_penalty(values: np.ndarray, previous_token_ids: list[int], repetition_penalty: float) -> np.ndarray:
    """
    对已经出现过的音频 token 应用重复惩罚。

    参数:
        values: 当前 codebook 的 logits。
        previous_token_ids: 当前 codebook 历史 token。
        repetition_penalty: 重复惩罚系数。

    返回:
        返回应用惩罚后的新 logits。

    边界异常:
        非法 token id 会被跳过。
    """

    if not previous_token_ids or repetition_penalty == 1.0:
        return values
    result = values.copy()
    for token_id in set(int(item) for item in previous_token_ids):
        if token_id < 0 or token_id >= result.shape[0]:
            continue
        result[token_id] = result[token_id] * repetition_penalty if result[token_id] < 0 else result[token_id] / repetition_penalty
    return result


def argmax_with_repetition_penalty(values: np.ndarray, previous_token_set: set[int], repetition_penalty: float) -> int:
    """
    在贪心模式下应用重复惩罚后选择 token。

    参数:
        values: 当前 codebook 的 logits。
        previous_token_set: 当前 codebook 已出现 token 集合。
        repetition_penalty: 重复惩罚系数。

    返回:
        返回得分最高的 token id。

    边界异常:
        本函数不主动抛出异常。
    """

    best_index = 0
    best_value = float("-inf")
    apply_penalty = bool(previous_token_set) and repetition_penalty != 1.0
    for index, value in enumerate(values):
        score = float(value)
        if apply_penalty and index in previous_token_set:
            score = score * repetition_penalty if score < 0 else score / repetition_penalty
        if score > best_value:
            best_value = score
            best_index = index
    return int(best_index)


def normalize_sample_mode(raw_sample_mode: str | None, raw_do_sample: bool = True) -> str:
    """
    归一化 MOSS ONNX 的采样模式。

    参数:
        raw_sample_mode: 用户传入的模式。
        raw_do_sample: 用户是否要求采样。

    返回:
        返回标准采样模式字符串。

    边界异常:
        本函数不主动抛出异常；未知模式会按 do_sample 选择保守默认值。
    """

    normalized = str(raw_sample_mode or "").strip()
    if normalized in {SAMPLE_MODE_GREEDY, SAMPLE_MODE_FIXED, SAMPLE_MODE_FULL}:
        return normalized
    if normalized == "mixed3":
        return SAMPLE_MODE_FIXED if raw_do_sample else SAMPLE_MODE_GREEDY
    return SAMPLE_MODE_GREEDY if not raw_do_sample else SAMPLE_MODE_FIXED


def resolve_stream_decode_frame_budget(
    emitted_samples_total: int,
    sample_rate: int,
    first_audio_emitted_at_seconds: float | None,
) -> int:
    """
    根据当前音频领先实时播放的程度决定 codec 增量解码帧数。

    参数:
        emitted_samples_total: 已经输出给上游的总采样点数量。
        sample_rate: 输出采样率。
        first_audio_emitted_at_seconds: 首个音频块输出时间。

    返回:
        返回本次 codec step decode 建议消费的 frame 数。

    边界异常:
        `sample_rate<=0` 时按首包策略返回 1。
    """

    if not first_audio_emitted_at_seconds or sample_rate <= 0:
        return 1
    elapsed_seconds = max(0.0, time.perf_counter() - first_audio_emitted_at_seconds)
    emitted_seconds = emitted_samples_total / float(sample_rate)
    lead_seconds = emitted_seconds - elapsed_seconds
    if lead_seconds < 0.20:
        return 1
    if lead_seconds < 0.55:
        return 2
    if lead_seconds < 1.10:
        return 4
    return 8


def merge_audio_channels_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    将 codec 输出规整为 StreamVox 使用的一维 float32 PCM。

    参数:
        audio: codec 输出音频。

    返回:
        返回一维 float32 PCM。

    边界异常:
        维度超过二维时抛出 ValueError。
    """

    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1).astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"unexpected audio shape: {array.shape}")
    if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
        return np.mean(array, axis=0, dtype=np.float32).reshape(-1)
    return np.mean(array, axis=1, dtype=np.float32).reshape(-1)


def resample_and_match_channels(audio: np.ndarray, input_sample_rate: int, target_sample_rate: int, target_channels: int) -> np.ndarray:
    """
    将参考音频规整为 codec encoder 需要的 batch/channel/time 形态。

    参数:
        audio: 输入音频。
        input_sample_rate: 输入音频采样率。
        target_sample_rate: codec encoder 采样率。
        target_channels: codec encoder 声道数。

    返回:
        返回形状为 `1/channel/time` 的 float32 ndarray。

    边界异常:
        音频维度非法、采样率非法或声道无法转换时抛出 ValueError。
    """

    if input_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rate must be greater than 0")

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    elif waveform.ndim == 2:
        if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 8:
            waveform = waveform.T
    else:
        raise ValueError(f"unsupported reference audio shape: {waveform.shape}")

    if waveform.shape[-1] <= 0:
        raise ValueError("reference audio must not be empty")

    if int(input_sample_rate) != int(target_sample_rate):
        resampled_channels = [
            librosa.resample(channel, orig_sr=int(input_sample_rate), target_sr=int(target_sample_rate))
            for channel in waveform
        ]
        waveform = np.stack(resampled_channels, axis=0).astype(np.float32, copy=False)

    current_channels = int(waveform.shape[0])
    if current_channels == target_channels:
        resolved = waveform
    elif current_channels == 1 and target_channels > 1:
        resolved = np.repeat(waveform, target_channels, axis=0)
    elif current_channels > 1 and target_channels == 1:
        resolved = np.mean(waveform, axis=0, keepdims=True, dtype=np.float32)
    else:
        raise ValueError(f"unsupported reference audio channel conversion: {current_channels} -> {target_channels}")
    return resolved.reshape(1, target_channels, -1).astype(np.float32, copy=False)


def sample_assistant_text_token(text_logits: np.ndarray, manifest: dict[str, Any]) -> int:
    """
    从 local transformer 的文本 logits 中决定是否继续生成音频帧。

    参数:
        text_logits: local decoder 输出的文本/control logits。
        manifest: browser ONNX manifest。

    返回:
        返回贪心选择出的控制 token id。

    边界异常:
        manifest 缺少必要字段时会自然抛出 KeyError。
    """

    candidate_ids = np.asarray(
        [
            int(manifest["tts_config"]["audio_assistant_slot_token_id"]),
            int(manifest["tts_config"]["audio_end_token_id"]),
        ],
        dtype=np.int32,
    )
    candidate_scores = text_logits[candidate_ids]
    return int(candidate_ids[int(np.argmax(candidate_scores))])


def sample_audio_token(
    audio_logits: np.ndarray,
    previous_token_ids: list[int],
    previous_token_set: set[int],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    """
    从某个 audio codebook 的 logits 中采样一个音频 token。

    参数:
        audio_logits: 当前 codebook 的 logits。
        previous_token_ids: 当前 codebook 历史 token。
        previous_token_set: 当前 codebook 历史 token 集合。
        generation_defaults: 当前采样参数。
        rng: 当前会话随机数生成器。

    返回:
        返回采样到的 audio token id。

    边界异常:
        采样参数非法时由 `sample_from_scores` 抛出 ValueError。
    """

    repetition_penalty = float(generation_defaults["audio_repetition_penalty"])
    if not bool(generation_defaults["do_sample"]):
        return argmax_with_repetition_penalty(audio_logits, previous_token_set, repetition_penalty)
    penalized_scores = apply_repetition_penalty(audio_logits, previous_token_ids, repetition_penalty)
    return sample_from_scores(
        penalized_scores,
        do_sample=True,
        temperature=float(generation_defaults["audio_temperature"]),
        top_k=int(generation_defaults["audio_top_k"]),
        top_p=float(generation_defaults["audio_top_p"]),
        rng=rng,
    )


__all__ = [
    "MANIFEST_CANDIDATE_RELATIVE_PATHS",
    "MossOnnxBundle",
    "SAMPLE_MODE_FIXED",
    "SAMPLE_MODE_FULL",
    "SAMPLE_MODE_GREEDY",
    "apply_repetition_penalty",
    "argmax_with_repetition_penalty",
    "build_onnx_session",
    "extract_last_hidden",
    "flatten2d_int32",
    "flatten3d_int32",
    "load_moss_onnx_bundle",
    "merge_audio_channels_to_mono",
    "normalize_sample_mode",
    "log_onnx_session_summary",
    "ort_device_label",
    "ort_precision_label",
    "resample_and_match_channels",
    "resolve_manifest_path",
    "resolve_manifest_relative_path",
    "resolve_stream_decode_frame_budget",
    "summarize_onnx_session",
    "sample_assistant_text_token",
    "sample_audio_token",
    "sample_from_scores",
    "softmax",
]
