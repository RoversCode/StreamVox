from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from .text_processing import TextPreprocessConfig
from ..hardware import ModelFootprintProfile


@dataclass(slots=True)
class PromptReference:
    """
    保存公开 PromptData 中的参考音频。

    参数:
        audio: 已规整到当前模型目标采样率的单声道参考音频数组。
        audio_sha256: 根据音频内容和模型采样率计算出的稳定哈希。
        index: 多参考音频中的位置序号。

    返回:
        该对象仅作为 PromptData 和 PromptCache 通用资料的数据载体。
    """

    audio: np.ndarray
    audio_sha256: str
    index: int = 0


@dataclass(slots=True)
class BasePromptAsset:
    role_name: str
    model: str
    sample_rate: int
    metadata: dict[str, str] = field(default_factory=dict)
    # 内部缓存资料由 adapter 在 make_prompt 阶段填充；模型 runtime 不消费该字段。
    prompt_cache_record: PromptCacheRecord | None = None


ModelPrompt = BasePromptAsset


@dataclass(slots=True)
class SerializedPromptAsset:
    role_name: str
    model: str
    sample_rate: int
    metadata: dict[str, str] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    json_data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptCacheRecord:
    """
    保存 PromptCache 中和模型无关的公开 prompt 资料。

    参数:
        prompt_text: 与参考音频逐条对齐的文本列表；不需要文本的模型用 None 占位。
        references: 与 prompt_text 等长的参考音频列表。

    返回:
        该对象用于 PromptCache 持久化和 TTSEngine.get_prompt_data 组装公开结果。
    """

    prompt_text: list[str | None] = field(default_factory=list)
    references: tuple[PromptReference, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class PromptData:
    """
    返回给调用方查看的完整 prompt 数据。

    参数:
        role_name: prompt 角色名。
        model: prompt 所属模型名。
        sample_rate: 模型侧 prompt 采样率。
        prompt_text: 与 references 一一对应的参考文本列表；不需要文本的模型用 None 占位。
        references: 当前模型目标采样率下的参考音频列表。
        prompt: 当前模型 adapter 反序列化后的推理资产。

    返回:
        该对象用于 TTSEngine.get_prompt_data 的只读查询结果。
    """

    role_name: str
    model: str
    sample_rate: int
    prompt_text: list[str | None]
    references: tuple[PromptReference, ...]
    prompt: BasePromptAsset

@dataclass(slots=True)
class RuntimeInfo:
    model: str
    adapter: str
    bundle_loaded: bool
    asset_count: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionTarget:
    '''
        引擎层只表达“用户意图”，不直接绑到底层框架名字。
        mode 固定为 auto / cpu / gpu，具体是 cuda、dml、vulkan 还是 cpu，
        由模型 runtime 再结合平台和能力解析。
    '''
    family: str
    mode: str
    device_index: int | None = None
    backend_preference: str | None = None


@dataclass(slots=True)
class ExecutionPlan:
    '''
        统一的执行计划。不同模型只需要声明自己有哪些 runtime family，
        比如 llama / onnx / torch，再把用户给的 device 翻译到这里。
    '''
    raw_device: str
    targets: dict[str, ExecutionTarget] = field(default_factory=dict)

    def get(self, family: str) -> ExecutionTarget | None:
        return self.targets.get(family)


@dataclass(slots=True)
class RuntimeRequest:
    model_dir: Path
    device: str
    # n_ctx 表示用户期望的 GGUF 上下文窗口大小；为空时由各模型配置文件决定。
    n_ctx: int | None = None
    # verify_model_sha256 表示 runtime 加载模型前是否执行 manifest 中声明的 sha256 完整性校验。
    # 默认保持开启，保证旧调用方仍然获得和历史版本一致的模型文件篡改检测能力。
    verify_model_sha256: bool = True
    execution_plan: ExecutionPlan | None = None
    text_preprocess_config: TextPreprocessConfig | None = None


class ModelRuntime(Protocol):
    model_name: str
    sample_rate: int
    runtime_info: RuntimeInfo

    def shutdown(self) -> None:
        ...


class ModelAdapter(Protocol):
    '''
        它主要是“接口契约”，告诉 engine / registry / 类型检查器：接入新模型时，适配器应该长成什么样。
    '''
    
    model_name: str

    def validate_make_prompt_args(
        self,
        *,
        role_name: str,
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        sample_rate: int | list[int] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> None:
        ...

    def load_runtime(self, request: RuntimeRequest) -> ModelRuntime:
        ...

    def get_text_preprocess_config(self) -> TextPreprocessConfig | None:
        ...

    def get_hardware_profile(self) -> ModelFootprintProfile | None:
        ...

    def resolve_execution_plan(
        self,
        device: str,
        llama_backend: str | None = None,
    ) -> ExecutionPlan:
        ...

    def make_prompt(
        self,
        runtime: ModelRuntime,
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BasePromptAsset:
        ...

    def serialize_prompt_asset(
        self,
        asset: BasePromptAsset,
    ) -> SerializedPromptAsset:
        ...

    def deserialize_prompt_asset(
        self,
        payload: SerializedPromptAsset,
    ) -> BasePromptAsset:
        ...

    def normalize_synthesize_kwargs(
        self,
        runtime: ModelRuntime,
        **model_kwargs: object,
    ) -> dict[str, object]:
        ...

    def synthesize(
        self,
        runtime: ModelRuntime,
        text: str,
        prompt: BasePromptAsset,
        stream: bool = True,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        ...
