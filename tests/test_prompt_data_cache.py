from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from streamvox.engine import TTSEngine
from streamvox.models.base import BasePromptAsset, PromptCacheRecord, PromptReference, SerializedPromptAsset
from streamvox.models.s2_pro.adapter import S2Pro4BGGUFAdapter
from streamvox.models.voxcpm2.adapter import VoxCPM2GGUFAdapter
from streamvox.models.voxcpm2.runtime import VoxCPM2PromptAsset, serialize_prompt_asset
from streamvox.prompt_audio import build_prompt_reference, compute_prompt_audio_sha256, normalize_prompt_audio
from streamvox.prompt_cache import PromptCache


@dataclass(slots=True)
class FakePromptAsset(BasePromptAsset):
    marker: str = "fake"


class FakeAdapter:
    model_name = "fake-model"

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        return SerializedPromptAsset(
            role_name=asset.role_name,
            model=asset.model,
            sample_rate=asset.sample_rate,
            metadata=dict(asset.metadata),
            arrays={"codes": np.asarray([1, 2, 3], dtype=np.int64)},
            json_data={"adapter_text": "模型私有文本"},
        )

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BasePromptAsset:
        return FakePromptAsset(
            role_name=payload.role_name,
            model=payload.model,
            sample_rate=payload.sample_rate,
            metadata=dict(payload.metadata),
        )


class FakeCodecEncoder:
    def encode_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        frames = max(1, int(audio.shape[-1]))
        return [np.ones((10, frames), dtype=np.int64)]


class FakeS2Decoder:
    def prewarm_decoder_state(self, _encoded: np.ndarray, task_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return {"task": np.asarray([len(task_id)], dtype=np.int64)}, np.zeros((0, 0), dtype=np.int64)


class FakeS2Runtime:
    sample_rate = 24000
    codec_encoder = FakeCodecEncoder()
    decoder = FakeS2Decoder()
    num_codebooks = 10


class FakeVoxInputBuilder:
    def prepare_prompt_source_features(
        self,
        wav_path: str | None,
        audio: np.ndarray | None,
        *,
        padding_mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = wav_path, padding_mode
        if audio is None:
            raise ValueError("audio is required in this fake runtime")
        length = max(1, int(np.asarray(audio).size))
        feat = np.ones((length, 2, 3), dtype=np.float32)
        hidden = np.ones((length, 2, 3), dtype=np.float32)
        return feat, hidden


class FakeVoxRuntime:
    sample_rate = 16000
    input_builder = FakeVoxInputBuilder()


class PromptReferenceTests(unittest.TestCase):
    def test_build_prompt_reference_uses_target_sample_rate_for_hash_only(self) -> None:
        audio = np.asarray([[0.2, -0.2], [0.4, -0.4]], dtype=np.float32)

        reference = build_prompt_reference(audio=audio, sample_rate=24000, index=3)
        same = build_prompt_reference(audio=audio.copy(), sample_rate=24000, index=3)
        different_rate = build_prompt_reference(audio=audio.copy(), sample_rate=48000, index=3)

        self.assertEqual(reference.index, 3)
        self.assertEqual(reference.audio.dtype, np.float32)
        self.assertEqual(reference.audio.ndim, 1)
        self.assertFalse(hasattr(reference, "sample_rate"))
        self.assertFalse(hasattr(reference, "source"))
        self.assertFalse(hasattr(reference, "source_path"))
        self.assertEqual(reference.audio_sha256, same.audio_sha256)
        self.assertNotEqual(reference.audio_sha256, different_rate.audio_sha256)

    def test_compute_prompt_audio_sha256_is_stable(self) -> None:
        audio = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)

        first = compute_prompt_audio_sha256(audio, 24000)
        second = compute_prompt_audio_sha256(audio.copy(), 24000)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_normalize_prompt_audio_preserves_time_axis_for_common_stereo_layouts(self) -> None:
        channel_first = np.asarray([[0.0, 0.2, 0.4], [1.0, 1.2, 1.4]], dtype=np.float32)
        channel_last = channel_first.T

        first = normalize_prompt_audio(channel_first)
        last = normalize_prompt_audio(channel_last)

        np.testing.assert_allclose(first, np.asarray([0.5, 0.7, 0.9], dtype=np.float32))
        np.testing.assert_allclose(last, np.asarray([0.5, 0.7, 0.9], dtype=np.float32))


class PromptCachePromptDataTests(unittest.TestCase):
    def test_store_keeps_public_prompt_data_separate_from_model_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("streamvox.prompt_cache.get_streamvox_home", return_value=Path(temp_dir)):
                cache = PromptCache(model="fake-model")
                prompt = FakePromptAsset(
                    role_name="demo",
                    model="fake-model",
                    sample_rate=24000,
                    metadata={"speaker": "demo"},
                )
                record = PromptCacheRecord(
                    prompt_text=["第一句", "第二句"],
                    references=(
                        build_prompt_reference(audio=np.asarray([0.1, -0.2], dtype=np.float32), sample_rate=24000, index=0),
                        build_prompt_reference(audio=np.asarray([0.3, -0.4], dtype=np.float32), sample_rate=24000, index=1),
                    ),
                )

                cache.store(FakeAdapter(), prompt, record)
                serialized = cache.load_serialized("fake-model", "demo")
                restored_prompt = cache.load(FakeAdapter(), "fake-model", "demo")
                restored_record = cache.load_prompt_data_record("fake-model", "demo")

        self.assertIsNotNone(serialized)
        assert serialized is not None
        self.assertEqual(serialized.json_data["adapter_text"], "模型私有文本")
        self.assertFalse(hasattr(serialized, "prompt_references"))
        self.assertIsNotNone(restored_prompt)
        assert restored_prompt is not None
        self.assertFalse(hasattr(restored_prompt, "prompt_references"))
        self.assertIsNotNone(restored_record)
        assert restored_record is not None
        self.assertEqual(restored_record.prompt_text, ["第一句", "第二句"])
        self.assertEqual(len(restored_record.references), 2)
        np.testing.assert_allclose(restored_record.references[0].audio, np.asarray([0.1, -0.2], dtype=np.float32))

    def test_store_preserves_none_prompt_text_as_position_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("streamvox.prompt_cache.get_streamvox_home", return_value=Path(temp_dir)):
                cache = PromptCache(model="fake-model")
                prompt = FakePromptAsset(role_name="audio-only", model="fake-model", sample_rate=24000)
                record = PromptCacheRecord(
                    prompt_text=[None],
                    references=(
                        build_prompt_reference(audio=np.asarray([0.1, -0.2], dtype=np.float32), sample_rate=24000, index=0),
                    ),
                )

                cache.store(FakeAdapter(), prompt, record)
                restored_record = cache.load_prompt_data_record("fake-model", "audio-only")

        self.assertIsNotNone(restored_record)
        assert restored_record is not None
        self.assertEqual(restored_record.prompt_text, [None])
        self.assertEqual(len(restored_record.references), 1)

    def test_load_prompt_data_record_falls_back_to_legacy_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("streamvox.prompt_cache.get_streamvox_home", return_value=Path(temp_dir)):
                cache = PromptCache(model="fake-model")
                target = cache._path_for("fake-model", "legacy")
                np.savez(
                    target,
                    role_name="legacy",
                    model="fake-model",
                    sample_rate=24000,
                    metadata="{}",
                    json_data='{"prompt_text": "旧文本"}',
                    payload_index="[]",
                )

                record = cache.load_prompt_data_record("fake-model", "legacy")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.prompt_text, [])
        self.assertEqual(record.references, ())

    def test_load_prompt_data_record_drops_mismatched_public_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("streamvox.prompt_cache.get_streamvox_home", return_value=Path(temp_dir)):
                cache = PromptCache(model="fake-model")
                prompt = FakePromptAsset(role_name="bad", model="fake-model", sample_rate=24000)
                record = PromptCacheRecord(
                    prompt_text=["只有文本"],
                    references=(
                        build_prompt_reference(audio=np.asarray([0.1], dtype=np.float32), sample_rate=24000, index=0),
                    ),
                )
                target = cache.store(FakeAdapter(), prompt, record)
                with np.load(target, allow_pickle=False) as existing:
                    archive = {key: existing[key] for key in existing.files}
                archive["prompt_text_index"] = np.asarray('["只有文本", "多余文本"]')
                np.savez(target, **archive)

                restored = cache.load_prompt_data_record("fake-model", "bad")

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.prompt_text, [])
        self.assertEqual(restored.references, ())


class TTSEnginePromptDataTests(unittest.TestCase):
    def test_get_prompt_data_returns_none_for_missing_role(self) -> None:
        engine = TTSEngine.__new__(TTSEngine)
        engine.model = "fake-model"
        engine.adapter = FakeAdapter()
        engine.prompt_cache = types.SimpleNamespace(load_serialized=lambda *_args: None)

        self.assertIsNone(engine.get_prompt_data("missing"))

    def test_get_prompt_data_returns_public_view_without_polluting_prompt_asset(self) -> None:
        serialized = SerializedPromptAsset(
            role_name="demo",
            model="fake-model",
            sample_rate=24000,
            metadata={"speaker": "demo"},
            arrays={"codes": np.asarray([1], dtype=np.int64)},
            json_data={"adapter_text": "模型私有文本"},
        )
        record = PromptCacheRecord(
            prompt_text=["参考文本"],
            references=(build_prompt_reference(audio=np.asarray([0.1], dtype=np.float32), sample_rate=24000, index=0),),
        )
        engine = TTSEngine.__new__(TTSEngine)
        engine.model = "fake-model"
        engine.adapter = FakeAdapter()
        engine.prompt_cache = types.SimpleNamespace(
            load_serialized=lambda *_args: serialized,
            load_prompt_data_record=lambda *_args: record,
        )

        prompt_data = engine.get_prompt_data("demo")

        self.assertIsNotNone(prompt_data)
        assert prompt_data is not None
        self.assertEqual(prompt_data.role_name, "demo")
        self.assertEqual(prompt_data.prompt_text, ["参考文本"])
        self.assertEqual(len(prompt_data.references), 1)
        self.assertIsInstance(prompt_data.prompt, FakePromptAsset)
        self.assertFalse(hasattr(prompt_data.prompt, "prompt_references"))


class VoxCPM2PromptAssetShapeTests(unittest.TestCase):
    def test_voxcpm2_prompt_asset_no_longer_serializes_paths_or_audio_waveforms(self) -> None:
        self.assertFalse(hasattr(VoxCPM2PromptAsset, "prompt_wav_path"))
        self.assertFalse(hasattr(VoxCPM2PromptAsset, "prompt_audio"))
        self.assertFalse(hasattr(VoxCPM2PromptAsset, "reference_wav_path"))
        self.assertFalse(hasattr(VoxCPM2PromptAsset, "reference_audio"))

        asset = VoxCPM2PromptAsset(
            role_name="demo",
            model="voxcpm2-gguf",
            sample_rate=24000,
            prompt_audio_feat=np.ones((2, 3), dtype=np.float32),
            prompt_locenc_hidden=np.ones((2, 3), dtype=np.float32),
            prompt_text="提示文本",
            reference_audio_feat=np.ones((2, 3), dtype=np.float32),
            reference_locenc_hidden=np.ones((2, 3), dtype=np.float32),
        )

        serialized = serialize_prompt_asset(asset)

        self.assertNotIn("prompt_audio", serialized.arrays)
        self.assertNotIn("reference_audio", serialized.arrays)
        self.assertNotIn("prompt_wav_path", serialized.json_data)
        self.assertNotIn("reference_wav_path", serialized.json_data)
        self.assertIn("prompt_audio_feat", serialized.arrays)
        self.assertIn("reference_audio_feat", serialized.arrays)


class AdapterPromptCacheRecordTests(unittest.TestCase):
    def test_s2_pro_preserves_model_text_semantics_while_aligning_public_record(self) -> None:
        adapter = S2Pro4BGGUFAdapter()

        with patch("streamvox.models.s2_pro.runtime.build_s2_pro_role_prefix_state", return_value=(b"state", 12)):
            prompt = adapter.make_prompt(
                FakeS2Runtime(),
                role_name="demo",
                sample_rate=[24000, 24000],
                audio_data=[
                    np.asarray([0.1, 0.2], dtype=np.float32),
                    np.asarray([0.3, 0.4], dtype=np.float32),
                ],
                prompt_text="单条文本",
            )

        self.assertIsNotNone(prompt.prompt_cache_record)
        assert prompt.prompt_cache_record is not None
        self.assertEqual(prompt.prompt_text, "单条文本")
        self.assertEqual(prompt.prompt_cache_record.prompt_text, ["单条文本", None])
        self.assertEqual(len(prompt.prompt_cache_record.references), 2)

    def test_voxcpm2_make_prompt_attaches_public_prompt_cache_record(self) -> None:
        adapter = VoxCPM2GGUFAdapter()
        audio = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

        prompt = adapter.make_prompt(
            FakeVoxRuntime(),
            role_name="demo",
            sample_rate=16000,
            audio_data=audio,
            prompt_text="提示文本",
        )

        self.assertIsInstance(prompt, VoxCPM2PromptAsset)
        self.assertIsNotNone(prompt.prompt_cache_record)
        assert prompt.prompt_cache_record is not None
        self.assertEqual(prompt.prompt_cache_record.prompt_text, ["提示文本"])
        self.assertEqual(len(prompt.prompt_cache_record.references), 1)
        np.testing.assert_allclose(prompt.prompt_cache_record.references[0].audio, audio)

    def test_voxcpm2_make_prompt_uses_none_text_placeholder_when_model_allows_audio_only(self) -> None:
        adapter = VoxCPM2GGUFAdapter()
        audio = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

        prompt = adapter.make_prompt(
            FakeVoxRuntime(),
            role_name="audio-only",
            sample_rate=16000,
            audio_data=audio,
            prompt_text=None,
        )

        self.assertIsNotNone(prompt.prompt_cache_record)
        assert prompt.prompt_cache_record is not None
        self.assertEqual(prompt.prompt_cache_record.prompt_text, [None])
        self.assertEqual(len(prompt.prompt_cache_record.references), 1)


if __name__ == "__main__":
    unittest.main()
