from __future__ import annotations

import math
import time
from typing import Any, Iterator

import numpy as np

from .common import (
    MossOnnxBundle,
    extract_last_hidden,
    flatten2d_int32,
    flatten3d_int32,
    normalize_sample_mode,
)
from .encoder import MossAudioEncoder
from .schema.record import MossTiming


class MossOnnxLlm:
    """
    MOSS TTS LLM 与 local transformer 执行器。

    参数:
        bundle: 已加载的 MOSS bundle 元信息。
        encoder: 文本 tokenizer 与参考音频编码器，用于文本 token 化。
        sessions: TTS 相关 ONNX session 映射。
        max_new_frames: 默认最大生成音频帧数。
        do_sample: 是否默认采样。
        sample_mode: 默认采样模式。

    返回:
        初始化后持有 TTS prompt 构造、prefill、decode 和单帧生成逻辑。

    边界异常:
        session 缺失或 meta 与导出图不匹配时会在推理阶段抛出异常。
    """

    def __init__(
        self,
        *,
        bundle: MossOnnxBundle,
        encoder: MossAudioEncoder,
        sessions: dict[str, Any],
        max_new_frames: int,
        do_sample: bool,
        sample_mode: str,
    ) -> None:
        self.bundle = bundle
        self.encoder = encoder
        self.manifest = bundle.manifest
        self.tts_meta = bundle.tts_meta
        self.sessions = dict(sessions)
        self.rng = np.random.default_rng(1234)
        self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        self.manifest["generation_defaults"]["do_sample"] = bool(do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = normalize_sample_mode(sample_mode, bool(do_sample))
        self.manifest["generation_defaults"]["do_sample"] = (
            self.manifest["generation_defaults"]["sample_mode"] != "greedy"
        )

    def reset_seed(self, seed: int | None) -> None:
        """
        重置 LLM 采样随机数状态。

        参数:
            seed: 随机种子；为空时保持当前状态。

        返回:
            无返回值。

        边界异常:
            seed 类型合法性由上层保证。
        """

        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        """
        构造 MOSS 输入序列中的文本行。

        参数:
            token_ids: 文本或控制 token id 列表。

        返回:
            返回 `row_width=n_vq+1` 的二维行列表。

        边界异常:
            manifest 缺少 token 配置时会自然抛出 KeyError。
        """

        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]], slot_token_id: int | None = None) -> list[list[int]]:
        """
        构造 MOSS 输入序列中的参考音频行。

        参数:
            prompt_audio_codes: 参考音频 codec token。
            slot_token_id: 第 0 列使用的音频槽位 token。

        返回:
            返回 `row_width=n_vq+1` 的二维行列表。

        边界异常:
            prompt_audio_codes 超出 `n_vq` 的列会被截断。
        """

        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        resolved_slot_token_id = int(
            self.manifest["tts_config"]["audio_user_slot_token_id"] if slot_token_id is None else slot_token_id
        )
        for code_row in prompt_audio_codes:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = resolved_slot_token_id
            for index in range(min(len(code_row), int(self.manifest["tts_config"]["n_vq"]))):
                row[index + 1] = int(code_row[index])
            rows.append(row)
        return rows

    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        """
        构造 `voice_clone` 模式的 TTS prefill 输入。

        参数:
            prompt_audio_codes: 参考音频 codec token。
            text_token_ids: 目标文本 token。

        返回:
            返回 `inputIds` 和 `attentionMask`。

        边界异常:
            上层会在进入本函数前保证 `voice_clone` 必然带参考音频。
        """

        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {"inputIds": rows, "attentionMask": [[1 for _ in rows]]}

    def build_continuation_request_rows(
        self,
        effective_text_token_ids: list[int],
        *,
        prompt_audio_codes: list[list[int]] | None = None,
    ) -> dict[str, list[list[int]]]:
        """
        构造 `continuation` 模式的 TTS prefill 输入。

        参数:
            effective_text_token_ids: `prompt_text + text` 的 token。
            prompt_audio_codes: 可选参考音频前缀。

        返回:
            返回 `inputIds` 和 `attentionMask`。

        边界异常:
            字段缺失时会由 manifest 访问自然抛出异常。
        """

        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            *self.encoder.encode_text("None"),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *effective_text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_text_rows([int(self.manifest["tts_config"]["audio_start_token_id"])]),
        ]
        if prompt_audio_codes:
            rows.extend(
                self.build_audio_prefix_rows(
                    prompt_audio_codes,
                    slot_token_id=int(self.manifest["tts_config"]["audio_assistant_slot_token_id"]),
                )
            )
        return {"inputIds": rows, "attentionMask": [[1 for _ in rows]]}

    def build_inference_request_rows(
        self,
        *,
        text: str,
        mode: str,
        prompt_text: str | None = None,
        prompt_audio_codes: list[list[int]] | None = None,
    ) -> dict[str, list[list[int]]]:
        """
        按合成模式构造 TTS prefill 输入。

        参数:
            text: 当前文本分段。
            mode: `voice_clone` 或 `continuation`。
            prompt_text: continuation 可选参考文本。
            prompt_audio_codes: 可选参考音频 codec token。

        返回:
            返回 `inputIds` 和 `attentionMask`。

        边界异常:
            mode 不支持或参数组合非法时抛出 ValueError。
        """

        resolved_mode = str(mode or "continuation").strip().lower() or "continuation"
        has_prompt_text = bool(str(prompt_text or "").strip())
        has_prompt_audio = prompt_audio_codes is not None and len(prompt_audio_codes) > 0
        if resolved_mode not in {"continuation", "voice_clone"}:
            raise ValueError("mode must be continuation or voice_clone")
        if resolved_mode == "voice_clone":
            if not has_prompt_audio:
                raise ValueError("voice_clone mode requires prompt audio")
            return self.build_voice_clone_request_rows(prompt_audio_codes or [], self.encoder.encode_text(text))
        if has_prompt_text != has_prompt_audio:
            raise ValueError("continuation mode accepts either target text only, or prompt_text and prompt_audio together")
        effective_text = str(prompt_text or "") + str(text or "")
        return self.build_continuation_request_rows(
            self.encoder.encode_text(effective_text),
            prompt_audio_codes=prompt_audio_codes,
        )

    def can_use_local_fixed_sampled_frame(self) -> bool:
        """
        判断当前参数是否能使用导出的 `fixed sampled frame` 快速图。

        参数:
            本函数无显式参数。

        返回:
            满足导出图常量约束时返回 True，否则返回 False。

        边界异常:
            本函数不主动抛出异常。
        """

        if "local_fixed_sampled_frame" not in self.sessions:
            return False
        generation_defaults = self.manifest["generation_defaults"]
        if generation_defaults.get("sample_mode") != "fixed":
            return False
        fixed_constants = dict(self.tts_meta.get("onnx", {}).get("fixed_sampled_frame_constants", {}))
        if not fixed_constants:
            return True

        for key, constant_value in fixed_constants.items():
            current_value = generation_defaults.get(key)

            # 导出图把部分采样参数固化成常量，因此运行时参数必须逐项完全对齐。
            if isinstance(constant_value, float):
                if current_value is None or not math.isclose(
                    float(current_value),
                    float(constant_value),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    return False
                continue
            if current_value != constant_value:
                return False
        return True

    def run_local_greedy_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
        repetition_penalty: float,
    ) -> tuple[bool, list[int]]:
        """
        使用 `local_greedy_frame` 图一次生成完整 audio frame。

        参数:
            global_hidden: 当前全局 hidden。
            previous_token_sets_by_channel: 每个 codebook 已出现 token 集合。
            repetition_penalty: 当前音频重复惩罚系数。

        返回:
            返回是否继续生成，以及当前帧 `n_vq` 个 token。

        边界异常:
            greedy frame 图缺失或输入输出不匹配时抛出异常。
        """

        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        outputs = self.sessions["local_greedy_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "repetition_penalty": np.asarray([float(repetition_penalty)], dtype=np.float32),
            },
        )
        output_names = [output.name for output in self.sessions["local_greedy_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        return should_continue, [int(item) for item in frame_token_ids]

    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        """
        使用 `fixed sampled frame` 图一次生成完整 audio frame。

        参数:
            global_hidden: 当前全局 hidden。
            previous_token_sets_by_channel: 每个 codebook 已出现 token 集合。

        返回:
            返回是否继续生成，以及当前帧 `n_vq` 个 token。

        边界异常:
            fixed sampled frame 图缺失或输入输出不匹配时抛出异常。
        """

        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        assistant_random_u = np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32)
        audio_random_u = np.asarray(
            [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
            dtype=np.float32,
        )
        outputs = self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = [output.name for output in self.sessions["local_fixed_sampled_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        return should_continue, [int(item) for item in frame_token_ids]

    def iter_generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        *,
        timing: MossTiming | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        按 frame 增量生成 MOSS audio token。

        参数:
            request_rows: `build_inference_request_rows` 生成的 `inputIds` 和 `attentionMask`。
            timing: 可选性能统计对象；传入后会记录 prefill 和逐帧耗时。

        返回:
            逐步 yield frame 事件，结束时 yield final 事件。

        边界异常:
            ONNX prefill、decode 或 local step 失败时直接抛出异常。
        """

        generation_defaults = self.manifest["generation_defaults"]
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        prefill_ids, prefill_dims = flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = flatten2d_int32(request_rows["attentionMask"])

        # prefill 是首包前最重的一次全局计算，需要单独统计，便于拆出首包延迟组成。
        prefill_start_time = time.perf_counter() if timing is not None else 0.0
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        if timing is not None:
            timing.prefill_times.append(time.perf_counter() - prefill_start_time)
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = extract_last_hidden(named_outputs["global_hidden"])
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }

        generated_frames: list[list[int]] = []
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        previous_tokens_by_channel = [[] for _ in range(n_vq)]
        previous_token_sets_by_channel = [set() for _ in range(n_vq)]
        can_use_fixed_sampled_frame = self.can_use_local_fixed_sampled_frame()

        for step_index in range(int(generation_defaults["max_new_frames"])):
            frame: list[int] = []
            local_frame_compute_time = 0.0

            # 当导出 bundle 自带整帧 greedy 图时，优先直接走该图，避免回退到逐 channel host 采样。
            if "local_greedy_frame" in self.sessions and not bool(generation_defaults["do_sample"]):
                local_frame_start_time = time.perf_counter() if timing is not None else 0.0
                should_continue, frame = self.run_local_greedy_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                    repetition_penalty=float(generation_defaults["audio_repetition_penalty"]),
                )
                if timing is not None:
                    local_frame_compute_time = time.perf_counter() - local_frame_start_time
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)

            # fixed sampled frame 是当前 StreamVox 实际保留的 MOSS 主采样路径。
            elif can_use_fixed_sampled_frame:
                local_frame_start_time = time.perf_counter() if timing is not None else 0.0
                should_continue, frame = self.run_local_fixed_sampled_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                )
                if timing is not None:
                    local_frame_compute_time = time.perf_counter() - local_frame_start_time
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            else:
                raise RuntimeError("moss nano fixed sampled frame graph is unavailable or generation defaults do not match")

            generated_frames.append(frame)
            next_row = np.full((1, 1, row_width), int(self.manifest["tts_config"]["audio_pad_token_id"]), dtype=np.int32)
            next_row[0, 0, 0] = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
            for index, token in enumerate(frame):
                next_row[0, 0, index + 1] = int(token)
            decode_feeds: dict[str, np.ndarray] = {
                "input_ids": next_row,
                "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
            }
            for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                decode_feeds[input_name] = past_by_name[input_name]
            # 每生成一帧后都要走一次 global decode，把这次写回耗时与 local frame 计算拆开统计。
            global_decode_step_start_time = time.perf_counter() if timing is not None else 0.0
            decode_outputs = self.sessions["decode"].run(None, decode_feeds)
            global_decode_step_time = (
                time.perf_counter() - global_decode_step_start_time if timing is not None else 0.0
            )
            decode_output_names = [output.name for output in self.sessions["decode"].get_outputs()]
            named_decode_outputs = dict(zip(decode_output_names, decode_outputs, strict=True))
            global_hidden = extract_last_hidden(named_decode_outputs["global_hidden"])
            past_valid_length += 1
            past_by_name = {
                output_name.replace("present_", "past_"): named_decode_outputs[output_name]
                for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
            }
            if timing is not None:
                timing.total_frames += 1
            yield {
                "type": "frame",
                "step_index": int(step_index),
                "frame": list(frame),
                "generated_frames": generated_frames,
                "local_frame_compute_time": float(local_frame_compute_time),
                "global_decode_step_time": float(global_decode_step_time),
                "frame_generation_time": float(local_frame_compute_time + global_decode_step_time),
            }
        yield {"type": "final", "generated_frames": generated_frames}


__all__ = ["MossOnnxLlm"]
