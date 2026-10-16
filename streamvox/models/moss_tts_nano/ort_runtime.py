from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterator

from ...utils.log import logger
from .common import (
    SAMPLE_MODE_FIXED,
    SAMPLE_MODE_FULL,
    SAMPLE_MODE_GREEDY,
    build_onnx_session,
    log_onnx_session_summary,
    load_moss_onnx_bundle,
)
from .decoder import MossAudioDecoder
from .encoder import MossAudioEncoder
from .llm import MossOnnxLlm
from .schema.record import MossTiming


def _resolve_codec_decode_step_providers(providers: list[str]) -> list[str]:
    """
    为 MOSS streaming codec decode step 选择稳定的 ONNX Runtime provider。

    参数:
        providers: 由上层设备解析得到的全局 provider 优先级列表。

    返回:
        返回只用于 `codec_decode_step.onnx` 的 provider 列表。

    边界异常:
        本函数不主动抛出异常；调用方仍需处理 ONNX Runtime session 创建失败。
    """

    # 业务意图：
    # codec decode step 在 Windows 上的已知兼容性问题只发生在 DML provider，
    # CUDA provider 实测可正常运行，因此这里只对包含 DML 的 provider 列表做定向回退，
    # 避免把本来可用的 CUDA 路径误降级到 CPU，平白损失推理性能。
    if os.name == "nt" and "DmlExecutionProvider" in providers:
        return ["CPUExecutionProvider"]
    return list(providers)


class MossOnnxCoreRuntime:
    """
    MOSS-TTS-Nano ONNX 低层执行器门面。

    参数:
        asset_root: StreamVox bundle 的 `ckpt` 根目录。
        providers: ONNX Runtime provider 列表。
        max_new_frames: 默认最大生成音频帧数。
        do_sample: 是否默认采样。
        sample_mode: 默认采样模式。

    返回:
        初始化后持有 `encoder`、`llm`、`decoder` 三个清晰职责组件，并保持旧接口兼容。

    边界异常:
        manifest、meta、ONNX 权重或 tokenizer 缺失时抛出 FileNotFoundError 或 ONNX Runtime 异常。
    """

    def __init__(
        self,
        *,
        asset_root: Path,
        providers: list[str],
        max_new_frames: int,
        do_sample: bool,
        sample_mode: str,
    ) -> None:
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.providers = list(providers)
        self.bundle = load_moss_onnx_bundle(self.asset_root)

        # 对外保留旧属性名，避免 runtime.py 和测试脚本继续改动。
        self.manifest_path = self.bundle.manifest_path
        self.manifest_dir = self.bundle.manifest_dir
        self.manifest = self.bundle.manifest
        self.tts_meta_path = self.bundle.tts_meta_path
        self.codec_meta_path = self.bundle.codec_meta_path
        self.tts_meta = self.bundle.tts_meta
        self.codec_meta = self.bundle.codec_meta

        # session 创建集中放在门面层，组件只消费自己负责的那部分 session。
        tts_dir = self.bundle.tts_meta_path.parent
        codec_dir = self.bundle.codec_meta_path.parent
        def build_optional_session(parent_dir: Path, filename: str | None, session_name: str):
            if not filename:
                return None
            path_value = parent_dir / filename
            if not path_value.is_file():
                logger.warning("[Moss Nano] optional session %s not found, skip loading: %s", session_name, path_value)
                return None
            return build_onnx_session(path_value, self.providers)

        tts_sessions: dict[str, Any] = {
            "prefill": build_onnx_session(tts_dir / self.tts_meta["files"]["prefill"], self.providers),
            "decode": build_onnx_session(tts_dir / self.tts_meta["files"]["decode_step"], self.providers),
            "local_fixed_sampled_frame": build_onnx_session(
                tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"],
                self.providers,
            ),
        }
        log_onnx_session_summary(
            model_name="moss-tts-nano-onnx",
            module_name="Prefill",
            session=tts_sessions["prefill"],
            model_path=tts_dir / self.tts_meta["files"]["prefill"],
        )
        log_onnx_session_summary(
            model_name="moss-tts-nano-onnx",
            module_name="Decode Step",
            session=tts_sessions["decode"],
            model_path=tts_dir / self.tts_meta["files"]["decode_step"],
        )
        log_onnx_session_summary(
            model_name="moss-tts-nano-onnx",
            module_name="Local Fixed Sampled Frame",
            session=tts_sessions["local_fixed_sampled_frame"],
            model_path=tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"],
        )
        local_greedy_frame_session = build_optional_session(
            tts_dir,
            self.tts_meta["files"].get("local_greedy_frame"),
            "local_greedy_frame",
        )
        if local_greedy_frame_session is not None:
            tts_sessions["local_greedy_frame"] = local_greedy_frame_session
            log_onnx_session_summary(
                model_name="moss-tts-nano-onnx",
                module_name="Local Greedy Frame",
                session=local_greedy_frame_session,
                model_path=tts_dir / self.tts_meta["files"]["local_greedy_frame"],
            )
        codec_encode_session = build_onnx_session(codec_dir / self.codec_meta["files"]["encode"], self.providers)
        log_onnx_session_summary(
            model_name="moss-tts-nano-onnx",
            module_name="Codec Encode",
            session=codec_encode_session,
            model_path=codec_dir / self.codec_meta["files"]["encode"],
        )
        codec_decode_step_providers = _resolve_codec_decode_step_providers(self.providers)
        if codec_decode_step_providers != self.providers:
            logger.info(
                "[Moss Nano] codec decode step provider fallback: %s -> %s",
                self.providers,
                codec_decode_step_providers,
            )
        codec_decode_step_session = build_onnx_session(
            codec_dir / self.codec_meta["files"]["decode_step"],
            codec_decode_step_providers,
        )
        log_onnx_session_summary(
            model_name="moss-tts-nano-onnx",
            module_name="Codec Decode Step",
            session=codec_decode_step_session,
            model_path=codec_dir / self.codec_meta["files"]["decode_step"],
        )

        self.encoder = MossAudioEncoder(bundle=self.bundle, codec_encode_session=codec_encode_session)
        self.llm = MossOnnxLlm(
            bundle=self.bundle,
            encoder=self.encoder,
            sessions=tts_sessions,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            sample_mode=sample_mode,
        )
        self.decoder = MossAudioDecoder(
            bundle=self.bundle,
            codec_decode_step_session=codec_decode_step_session,
        )

        # 兼容 runtime.py 当前直接访问 `core.sp_model` 的调用方式。
        self.sp_model = self.encoder.sp_model

    @property
    def sample_rate(self) -> int:
        """
        返回 codec 输出采样率。

        参数:
            本属性无参数。

        返回:
            返回整数 Hz 采样率。

        边界异常:
            decoder 未初始化成功时会自然暴露 AttributeError。
        """

        return int(self.decoder.sample_rate)

    def apply_generation_options(self, *, seed: int | None = None) -> None:
        """
        更新当前请求的生成参数。

        参数:
            seed: 随机种子；为空时沿用当前 RNG 状态。

        返回:
            无返回值。

        边界异常:
            seed 类型合法性由上层 adapter 负责。
        """

        self.llm.reset_seed(seed)

    def encode_text(self, text: str) -> list[int]:
        """
        对外暴露文本 token 化入口。

        参数:
            text: 待编码文本。

        返回:
            返回 token id 列表。

        边界异常:
            tokenizer 内部错误会自然抛出异常。
        """

        return self.encoder.encode_text(text)

    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        """
        对外暴露 `voice_clone` prompt 构造入口。

        参数:
            prompt_audio_codes: 参考音频 codec token。
            text_token_ids: 目标文本 token。

        返回:
            返回 `inputIds` 和 `attentionMask`。

        边界异常:
            参数合法性由 llm 组件内部保证。
        """

        return self.llm.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)

    def build_inference_request_rows(
        self,
        *,
        text: str,
        mode: str,
        prompt_text: str | None = None,
        prompt_audio_codes: list[list[int]] | None = None,
    ) -> dict[str, list[list[int]]]:
        """
        对外暴露统一的 TTS prefill 输入构造入口。

        参数:
            text: 当前文本分段。
            mode: `voice_clone` 或 `continuation`。
            prompt_text: continuation 可选参考文本。
            prompt_audio_codes: 可选参考音频 token。

        返回:
            返回 `inputIds` 和 `attentionMask`。

        边界异常:
            参数组合非法时抛出 ValueError。
        """

        return self.llm.build_inference_request_rows(
            text=text,
            mode=mode,
            prompt_text=prompt_text,
            prompt_audio_codes=prompt_audio_codes,
        )

    def encode_reference_audio_array(self, audio, sample_rate: int) -> list[list[int]]:
        """
        对外暴露参考音频数组编码入口。

        参数:
            audio: 输入参考音频。
            sample_rate: 输入音频采样率。

        返回:
            返回 `frame/n_vq` 的 int token 列表。

        边界异常:
            音频非法或 codec encode 失败时抛出异常。
        """

        return self.encoder.encode_reference_audio_array(audio, sample_rate)

    def encode_reference_audio_path(self, audio_path: str | Path) -> list[list[int]]:
        """
        对外暴露参考音频文件编码入口。

        参数:
            audio_path: 本地音频文件路径。

        返回:
            返回 `frame/n_vq` 的 int token 列表。

        边界异常:
            音频文件缺失或 codec encode 失败时抛出异常。
        """

        return self.encoder.encode_reference_audio_path(audio_path)

    def iter_generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        *,
        timing: MossTiming | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        对外暴露 audio frame 增量生成入口。

        参数:
            request_rows: TTS prefill 输入。
            timing: 可选性能统计对象；传入后会记录 prefill 和逐帧耗时。

        返回:
            逐步 yield frame 事件。

        边界异常:
            TTS prefill、decode 或 local step 失败时抛出异常。
        """

        yield from self.llm.iter_generate_audio_frames(request_rows, timing=timing)

    def iter_synthesize_mono(
        self,
        *,
        request_rows: dict[str, list[list[int]]],
        timing: MossTiming | None = None,
    ) -> Iterator:
        """
        对外暴露音频块级流式合成入口。

        参数:
            request_rows: 当前文本段的 TTS prefill 输入。
            timing: 可选性能统计对象；传入后会记录每个音频 chunk 的耗时。

        返回:
            逐块 yield float32 mono PCM。

        边界异常:
            LLM 生成或 codec 解码失败时抛出异常。
        """

        yield from self.decoder.iter_synthesize_mono(
            request_rows=request_rows,
            frame_iterator_factory=lambda rows: self.llm.iter_generate_audio_frames(rows, timing=timing),
            timing=timing,
        )

    def warmup(self) -> None:
        """
        预热 MOSS ONNX 的关键执行路径。

        参数:
            本函数无显式参数。

        返回:
            无返回值；成功表示正式 stream 主路径所需的 tokenizer、LLM frame 生成和 codec streaming decode 可执行。

        边界异常:
            任一路 ONNX 图执行失败时抛出异常。
        """

        logger.info("[Moss Nano] 开始预热")
        saved_generation_defaults = copy.deepcopy(self.manifest["generation_defaults"])
        saved_rng_state = copy.deepcopy(self.llm.rng.bit_generator.state)
        try:
            # warmup 不需要覆盖参考音频编码，只需要覆盖正式 stream 的首包和流式解码主链。
            # 这里改用最小 continuation 请求，避免初始化阶段额外走一次 make_prompt 相关路径。
            self.manifest["generation_defaults"]["max_new_frames"] = 1
            request_rows = self.llm.build_inference_request_rows(
                text="。",
                mode="continuation",
                prompt_text=None,
                prompt_audio_codes=None,
            )
            generated_frames: list[list[int]] = []
            for event in self.llm.iter_generate_audio_frames(request_rows):
                generated_frames = event["generated_frames"]
            warmup_frames = generated_frames or [[0] * int(self.manifest["tts_config"]["n_vq"])]
            self.decoder.warmup(build_minimal_frames=lambda: warmup_frames)
        finally:
            self.manifest["generation_defaults"] = saved_generation_defaults
            self.llm.rng.bit_generator.state = saved_rng_state
            self.decoder.codec_streaming_session.reset()
        logger.info("[Moss Nano] 预热完毕")

    def shutdown(self) -> None:
        """
        释放低层 ONNX session 引用。

        参数:
            本函数无显式参数。

        返回:
            无返回值；重复调用安全返回。

        边界异常:
            资源释放由 Python 对象析构负责，本函数不主动抛出异常。
        """

        self.decoder.shutdown()
        self.encoder = None
        self.llm = None


__all__ = [
    "MossOnnxCoreRuntime",
    "SAMPLE_MODE_FIXED",
    "SAMPLE_MODE_FULL",
    "SAMPLE_MODE_GREEDY",
]
