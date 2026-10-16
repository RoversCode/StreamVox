from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .paths import get_streamvox_home
from .models.base import BasePromptAsset, ModelAdapter, PromptCacheRecord, PromptReference, SerializedPromptAsset
from .utils.log import logger


class PromptCache:
    def __init__(self, model: str = "qwen3-tts-clone-1.7b-gguf") -> None:
        '''
            初始化Prompt data
        '''
        self.model = model
        self.base_dir = get_streamvox_home() / "prompts"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # 不同模型的 prompt 资产按模型分目录，避免混放后字段语义冲突。
        self.model_dir = self._model_dir(model)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # 初始化时预读当前模型已有的 role，后续查重不再反复扫描目录。
        self.roles: set[str] = self._load_existing_roles()

    def store(
        self,
        adapter: ModelAdapter,
        prompt: BasePromptAsset,
        record: PromptCacheRecord | None = None,
    ) -> Path:
        """
        保存模型 prompt 资产和公开 prompt 资料。

        参数:
            adapter: 当前模型 adapter，用于序列化模型私有资产。
            prompt: 当前模型的推理资产。
            record: 与模型无关的公开 prompt 资料。

        返回:
            返回写入的 npz 文件路径。
        """
        serialized = adapter.serialize_prompt_asset(prompt)
        prompt_record = self._normalize_prompt_record(record, serialized.sample_rate)
        target = self._path_for(serialized.model, serialized.role_name)
        arrays = {key: np.asarray(value) for key, value in serialized.arrays.items()}
        payload_index = sorted(arrays) # 返回的是排序好的key name
        prompt_text_index = list(prompt_record.prompt_text)
        reference_index = [
            {
                "index": int(reference.index),
                "audio_sha256": reference.audio_sha256,
            }
            for reference in prompt_record.references
        ]
        payload = {
            "role_name": serialized.role_name,
            "model": serialized.model,
            "sample_rate": serialized.sample_rate,
            "metadata": json.dumps(serialized.metadata),
            "json_data": json.dumps(serialized.json_data, ensure_ascii=False),
            "payload_index": json.dumps(payload_index),
            "prompt_text_index": json.dumps(prompt_text_index, ensure_ascii=False),
            "reference_index": json.dumps(reference_index, ensure_ascii=False),
        }
        flat_archive: dict[str, Any] = {
            **payload,
        }
        for key, value in arrays.items():
            flat_archive[f"payload__{key}"] = value
        for reference in prompt_record.references:
            flat_archive[f"reference_audio__{int(reference.index)}"] = np.asarray(reference.audio, dtype=np.float32)
        # prompt data 体积不大，这里优先读写速度，使用非压缩 npz。
        np.savez(target, **flat_archive)
        if serialized.model == self.model:
            self.roles.add(serialized.role_name)
        logger.info(f"存储角色: {serialized.role_name} Prompt Data到: {target}")
        return target

    def load(
        self,
        adapter: ModelAdapter,
        model: str,
        role_name: str,
    ) -> BasePromptAsset | None:
        serialized = self.load_serialized(model, role_name)
        if serialized is None:
            return None
        return adapter.deserialize_prompt_asset(serialized)

    def load_serialized(
        self,
        model: str,
        role_name: str,
    ) -> SerializedPromptAsset | None:
        """
        读取缓存中的通用序列化 prompt 数据。

        参数:
            model: prompt 所属模型名。
            role_name: prompt 角色名。

        返回:
            命中时返回 SerializedPromptAsset；缓存不存在时返回 None。
        """
        target = self._path_for(model, role_name)
        if not target.exists():
            return None

        with np.load(target, allow_pickle=False) as bundle:
            metadata = json.loads(bundle["metadata"].item())
            json_data = self._decode_json_item(bundle, "json_data", default={})
            payload_index = self._decode_json_item(bundle, "payload_index", default=[])
            arrays: dict[str, np.ndarray] = {}
            for key in payload_index:
                bundle_key = f"payload__{key}"
                if bundle_key in bundle.files:
                    arrays[str(key)] = np.asarray(bundle[bundle_key])
            return SerializedPromptAsset(
                role_name=bundle["role_name"].item(),
                model=bundle["model"].item(),
                sample_rate=int(bundle["sample_rate"].item()),
                metadata=metadata,
                arrays=arrays,
                json_data=json_data,
            )

    def load_prompt_data_record(
        self,
        model: str,
        role_name: str,
    ) -> PromptCacheRecord | None:
        """
        读取缓存中的公开 prompt 资料。

        参数:
            model: prompt 所属模型名。
            role_name: prompt 角色名。

        返回:
            命中时返回 PromptCacheRecord；缓存不存在时返回 None。
        """
        target = self._path_for(model, role_name)
        if not target.exists():
            return None

        with np.load(target, allow_pickle=False) as bundle:
            prompt_text = self._decode_json_item(bundle, "prompt_text_index", default=[])
            if not isinstance(prompt_text, list):
                prompt_text = []
            references = self._load_prompt_references(bundle)
            if len(prompt_text) != len(references):
                logger.warning(
                    "PromptData 公共资料长度不一致，已忽略 role=%s model=%s 的公开 prompt_text/references",
                    role_name,
                    model,
                )
                return PromptCacheRecord()
            return PromptCacheRecord(
                prompt_text=[None if item is None else str(item) for item in prompt_text],
                references=references,
            )

    def delete(self, model: str, role_name: str) -> bool:
        '''
            删除单个role对应的Prompt data
        '''
        target = self._path_for(model, role_name)
        if not target.exists():
            return False
        target.unlink()
        if model == self.model:
            self.roles.discard(role_name)
        logger.info(f"删除角色: {role_name} Prompt Data: {target}")
        return True

    def _path_for(self, model: str, role_name: str) -> Path:
        return self._model_dir(model) / f"{self._sanitize(role_name)}.npz"

    def _model_dir(self, model: str) -> Path:
        return self.base_dir / self._sanitize(model)

    def _load_existing_roles(self) -> set[str]:
        if not self.model_dir.exists():
            return set()
        roles: set[str] = set()
        # 现在文件名只保存 role_name，模型信息已经体现在目录层级里。
        for file in self.model_dir.glob("*.npz"):
            roles.add(file.stem)
        return roles

    def _load_prompt_references(
        self,
        bundle: np.lib.npyio.NpzFile,
    ) -> tuple[PromptReference, ...]:
        """
        从 npz 包中恢复原始参考音频信息。

        参数:
            bundle: 已打开的 npz 文件对象。

        返回:
            返回缓存中的参考音频列表；旧缓存没有该字段时返回空元组。
        """
        reference_index = self._decode_json_item(bundle, "reference_index", default=[])
        if not isinstance(reference_index, list):
            return ()
        references: list[PromptReference] = []
        for raw_item in reference_index:
            if not isinstance(raw_item, dict):
                continue
            index = int(raw_item.get("index", len(references)))
            bundle_key = f"reference_audio__{index}"
            if bundle_key not in bundle.files:
                continue
            references.append(
                PromptReference(
                    audio=np.asarray(bundle[bundle_key], dtype=np.float32).reshape(-1),
                    audio_sha256=str(raw_item.get("audio_sha256", "")),
                    index=index,
                )
            )
        return tuple(sorted(references, key=lambda item: item.index))

    def _normalize_prompt_record(
        self,
        record: PromptCacheRecord | None,
        sample_rate: int,
    ) -> PromptCacheRecord:
        """
        校验并规整公开 prompt 资料。

        参数:
            record: 调用方提供的公开 prompt 资料。
            sample_rate: 模型 prompt 采样率，仅用于错误上下文。

        返回:
            返回可安全持久化的 PromptCacheRecord。

        异常:
            当 prompt_text 与 references 长度不一致时抛出 ValueError。
        """
        if record is None:
            return PromptCacheRecord()
        if len(record.prompt_text) != len(record.references):
            raise ValueError(
                f"prompt_text length must match references length for sample_rate={sample_rate}"
            )
        return PromptCacheRecord(
            prompt_text=[None if item is None else str(item) for item in record.prompt_text],
            references=tuple(record.references),
        )

    @staticmethod
    def _sanitize(value: str) -> str:
        return value.replace("/", "_").replace(" ", "_")

    @staticmethod
    def _decode_json_item(
        bundle: np.lib.npyio.NpzFile,
        key: str,
        default: dict[str, object] | list[object],
    ) -> dict[str, object] | list[object]:
        if key not in bundle.files:
            return default
        raw = bundle[key].item()
        if raw == "None":
            return default
        return json.loads(raw)
