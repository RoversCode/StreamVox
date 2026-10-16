from __future__ import annotations

from dataclasses import dataclass, field

from .base import ModelAdapter
from ..hardware import ModelFootprintProfile


@dataclass
class ModelRegistry:
    _adapters: dict[str, ModelAdapter] = field(default_factory=dict)

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.model_name] = adapter

    def get(self, model_name: str) -> ModelAdapter:
        try:
            return self._adapters[model_name]
        except KeyError as exc:
            raise KeyError(f"unsupported model: {model_name}") from exc

    def list_models(self) -> list[str]:
        return sorted(self._adapters)

    def get_hardware_profile(self, model_name: str) -> ModelFootprintProfile | None:
        adapter = self.get(model_name)
        getter = getattr(adapter, "get_hardware_profile", None)
        if getter is None:
            return None
        return getter()


default_registry = ModelRegistry()
