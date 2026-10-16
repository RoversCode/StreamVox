from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

from streamvox.engine import TTSEngine
from streamvox.models.base import ExecutionPlan
from streamvox.models.qwen3.adapter import Qwen3TTSClone17BAdapter
from streamvox.models.registry import ModelRegistry
from streamvox.models.s2_pro.adapter import S2Pro4BGGUFAdapter
from streamvox.models.voxcpm2.adapter import VoxCPM2GGUFAdapter


class ExecutionPlanLlamaBackendTests(unittest.TestCase):
    """验证 GGUF 模型执行计划中的 llama_backend 传播行为。"""

    def test_qwen3_plan_only_sets_llama_backend_preference(self) -> None:
        plan = Qwen3TTSClone17BAdapter().resolve_execution_plan("gpu:2", llama_backend="vulkan")

        self.assertEqual(plan.get("llama").backend_preference, "vulkan")
        self.assertIsNone(plan.get("onnx").backend_preference)
        self.assertEqual(plan.get("llama").device_index, 2)

    def test_s2_pro_plan_only_sets_llama_backend_preference(self) -> None:
        plan = S2Pro4BGGUFAdapter().resolve_execution_plan("gpu:1", llama_backend="cuda")

        self.assertEqual(plan.get("llama").backend_preference, "cuda")
        self.assertIsNone(plan.get("onnx").backend_preference)
        self.assertEqual(plan.get("llama").device_index, 1)

    def test_voxcpm2_plan_only_sets_llama_backend_preference(self) -> None:
        plan = VoxCPM2GGUFAdapter().resolve_execution_plan("gpu", llama_backend="vulkan")

        self.assertEqual(plan.get("llama").backend_preference, "vulkan")
        self.assertIsNone(plan.get("onnx").backend_preference)
        self.assertEqual(plan.get("llama").device_index, 0)


class TTSEngineLlamaBackendForwardingTests(unittest.TestCase):
    """验证 TTSEngine 是否把 llama_backend 正确下发给 adapter。"""

    def setUp(self) -> None:
        self.resolve_calls: list[tuple[str, str | None]] = []
        self.fake_adapter = self._build_fake_adapter()
        self.registry = ModelRegistry({"fake-model": self.fake_adapter})

    def _build_fake_adapter(self) -> types.SimpleNamespace:
        def resolve_execution_plan(device: str, llama_backend: str | None = None):
            self.resolve_calls.append((device, llama_backend))
            return ExecutionPlan(raw_device=device, targets={})

        def load_runtime(_request):
            return types.SimpleNamespace(
                sample_rate=24000,
                runtime_info=types.SimpleNamespace(model="fake-model"),
                shutdown=lambda: None,
            )

        return types.SimpleNamespace(
            model_name="fake-model",
            resolve_execution_plan=resolve_execution_plan,
            load_runtime=load_runtime,
            get_text_preprocess_config=lambda: None,
            get_hardware_profile=lambda: None,
        )

    def _build_fake_prompt_cache(self, *_args, **_kwargs):
        return types.SimpleNamespace(roles=set())

    def test_engine_defaults_llama_backend_to_none(self) -> None:
        with (
            patch.object(TTSEngine, "_resolve_model_init", return_value=("fake-model", Path.cwd())),
            patch("streamvox.engine.PromptCache", side_effect=self._build_fake_prompt_cache),
            patch("streamvox.engine.assess_and_log_hardware", return_value=None),
        ):
            TTSEngine(
                model="fake-model",
                device="gpu:1",
                verify_model_sha256=False,
                registry=self.registry,
            )

        self.assertEqual(self.resolve_calls, [("gpu:1", None)])

    def test_engine_forwards_explicit_llama_backend(self) -> None:
        with (
            patch.object(TTSEngine, "_resolve_model_init", return_value=("fake-model", Path.cwd())),
            patch("streamvox.engine.PromptCache", side_effect=self._build_fake_prompt_cache),
            patch("streamvox.engine.assess_and_log_hardware", return_value=None),
        ):
            TTSEngine(
                model="fake-model",
                device="gpu:0",
                llama_backend="vulkan",
                verify_model_sha256=False,
                registry=self.registry,
            )

        self.assertEqual(self.resolve_calls, [("gpu:0", "vulkan")])


if __name__ == "__main__":
    unittest.main()
