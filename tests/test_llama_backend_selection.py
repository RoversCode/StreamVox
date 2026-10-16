from __future__ import annotations

import unittest

from streamvox.models.llama_backend import select_llama_backend


class SelectLlamaBackendTests(unittest.TestCase):
    """验证 GGUF backend 显式选择与回退规则。"""

    def test_auto_selects_first_available_packaged_backend(self) -> None:
        resolution = select_llama_backend(
            mode="auto",
            device_index=1,
            model_label="test-model",
            packaged_backends=("cuda", "vulkan"),
            availability_checker=lambda backend: backend == "vulkan",
        )

        self.assertEqual(resolution.backend, "vulkan")
        self.assertEqual(resolution.gpu_index, 1)

    def test_explicit_vulkan_selects_vulkan_when_available(self) -> None:
        resolution = select_llama_backend(
            mode="gpu",
            device_index=0,
            model_label="test-model",
            packaged_backends=("cuda", "vulkan"),
            availability_checker=lambda backend: backend == "vulkan",
            explicit_backend="vulkan",
        )

        self.assertEqual(resolution.backend, "vulkan")
        self.assertEqual(resolution.gpu_index, 0)

    def test_explicit_vulkan_raises_when_not_packaged(self) -> None:
        with self.assertRaisesRegex(ValueError, "not packaged"):
            select_llama_backend(
                mode="gpu",
                device_index=0,
                model_label="test-model",
                packaged_backends=("cuda",),
                availability_checker=lambda _backend: True,
                explicit_backend="vulkan",
            )

    def test_device_cpu_ignores_explicit_backend_and_warns(self) -> None:
        with self.assertLogs("AppLogger", level="WARNING") as captured:
            resolution = select_llama_backend(
                mode="cpu",
                device_index=None,
                model_label="test-model",
                packaged_backends=("cuda", "vulkan"),
                availability_checker=lambda backend: backend == "vulkan",
                explicit_backend="vulkan",
            )

        self.assertEqual(resolution.backend, "cpu")
        self.assertIn("ignored explicit llama_backend=vulkan", "\n".join(captured.output))

    def test_explicit_cuda_unavailable_warns_and_falls_back_to_vulkan(self) -> None:
        with self.assertLogs("AppLogger", level="WARNING") as captured:
            resolution = select_llama_backend(
                mode="gpu",
                device_index=0,
                model_label="test-model",
                packaged_backends=("cuda", "vulkan"),
                availability_checker=lambda backend: backend == "vulkan",
                explicit_backend="cuda",
            )

        self.assertEqual(resolution.backend, "vulkan")
        joined_logs = "\n".join(captured.output).lower()
        self.assertIn("ignored explicit cuda request", joined_logs)
        self.assertIn("defaulted to vulkan", joined_logs)


if __name__ == "__main__":
    unittest.main()
