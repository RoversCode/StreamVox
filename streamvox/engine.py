from __future__ import annotations

from pathlib import Path
import uuid
from typing import Iterable, Literal

import numpy as np

from .hardware import assess_and_log_hardware
from .model_bundle import ModelBundleError, load_local_manifest, manifest_path_for
from .models import default_registry
from .models.base import BasePromptAsset, PromptCacheRecord, PromptData, RuntimeRequest
from .models.registry import ModelRegistry
from .paths import get_default_model_root
from .prompt_cache import PromptCache
from .runtime_compat import ensure_openmp_runtime_compat
from .utils.log import logger


# 中文说明：
# 有些调用方会直接 `from streamvox.engine import TTSEngine`，不会经过包级 `__init__` 的兼容注入；
# 因此这里再做一次幂等调用，确保主进程在加载原生模型前已经具备 OpenMP 兼容环境。
ensure_openmp_runtime_compat(source="streamvox.engine")


class TTSEngine:
    """统一管理模型运行时、参考音频缓存和流式语音合成。"""

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str = "auto",
        llama_backend: Literal["cpu", "cuda", "vulkan", "metal"] | None = None,
        n_ctx: int | None = None,
        verify_model_sha256: bool = True,
        registry: ModelRegistry | None = None,
        onnx_provider: Literal["directml", "cpu", "cuda"] | None = None,
    ) -> None:
        if onnx_provider not in {None, "directml", "cpu", "cuda"}:
            raise ValueError("onnx_provider must be directml, cpu, cuda or None")
        self.onnx_provider = onnx_provider
        self.device = device
        self.llama_backend = self._normalize_llama_backend(llama_backend)
        # 未传 n_ctx 时不覆盖模型配置，让不同模型按照自己的 yaml 默认上下文长度初始化。
        if n_ctx is None:
            self.n_ctx = None
        else:
            # bool 是 int 的子类，但把 True/False 当上下文长度没有业务意义，因此显式拒绝。
            if not isinstance(n_ctx, int) or isinstance(n_ctx, bool):
                raise TypeError("n_ctx must be an integer or None")
            # 引擎层不能假设所有模型共享同一套上下文窗口边界，因此只透传用户显式意图给 adapter/runtime。
            self.n_ctx = n_ctx
        # sha256 校验开关会影响安全边界，必须要求用户显式传入 bool，避免 "False" 这类字符串被 Python 当成真值。
        if not isinstance(verify_model_sha256, bool):
            raise TypeError("verify_model_sha256 must be a boolean")
        # 返回用户的显式选择；默认值在 __init__ 签名中保持 True，确保向后兼容。
        self.verify_model_sha256 = verify_model_sha256
        self.registry = registry or default_registry
        resolved_model_name, resolved_model_dir = self._resolve_model_init(
            model=model,
        )
        self.model_dir = resolved_model_dir
        self.model = resolved_model_name
        self.adapter = self.registry.get(resolved_model_name)
        self.prompt_cache = PromptCache(model=resolved_model_name)
        # 初始化阶段先做一次本机资源体检，只打印建议，不阻断后续 runtime 加载。
        hardware_profile_getter = getattr(self.adapter, "get_hardware_profile", None)
        self.hardware_requirement = None if hardware_profile_getter is None else hardware_profile_getter()
        self.hardware_recommendation = None
        if self.hardware_requirement is not None:
            self.hardware_recommendation = assess_and_log_hardware(
                active_model_name=resolved_model_name,
                requirement=self.hardware_requirement,
            )

        # 引擎层只保存通用执行意图，具体 family 到 backend 的映射交给 adapter。
        self.execution_plan = self.adapter.resolve_execution_plan(
            device,
            llama_backend=self.llama_backend,
        )

        onnx_target = self.execution_plan.get("onnx")
        if onnx_target is not None and onnx_provider is not None:
            onnx_target.backend_preference = onnx_provider
            onnx_target.mode = "cpu" if onnx_provider == "cpu" else "gpu"
        llama_target = self.execution_plan.get("llama")
        if llama_target is not None and self.llama_backend is not None:
            llama_target.mode = "cpu" if self.llama_backend == "cpu" else "gpu"

        self.runtime = self.adapter.load_runtime(
            RuntimeRequest(
                model_dir=self.model_dir,
                device=device,
                n_ctx=self.n_ctx,
                verify_model_sha256=self.verify_model_sha256,
                execution_plan=self.execution_plan,
                text_preprocess_config=(
                    None
                    if getattr(self.adapter, "get_text_preprocess_config", None) is None
                    else self.adapter.get_text_preprocess_config()
                ),
            )
        )
        # 中文说明：
        # runtime 级初始化预热是可选内部约定，不修改 adapter 公共契约；
        # 但一旦 runtime 声明支持该钩子，就必须在引擎初始化阶段执行并把失败直接上抛，
        # 避免“底层模型只加载成功一半”却仍被误记成初始化成功。
        runtime_warmup_for_init = getattr(self.runtime, "warmup_for_init", None)
        if callable(runtime_warmup_for_init):
            runtime_warmup_for_init()
        logger.info("✅ [StreamVox] Initialization successful")
        self.runtime_info = self.runtime.runtime_info
        self._verify_requested_backends()

    @property
    def actual_backends(self) -> dict[str, object]:
        """读取运行时已经解析并完成初始化的后端，避免使用构造时的旧日志快照。

        Returns:
            当前模型实际使用的 llama 与 ONNX 后端；纯 ONNX 模型没有 llama 键。
        """
        resolved = getattr(self.runtime, "resolved_device", self.runtime)
        llama = getattr(resolved, "gguf_backend", None)
        onnx = getattr(resolved, "onnx_backend", getattr(resolved, "decoder_backend", None))
        result = {}
        if llama is not None:
            result["llama"] = str(llama)
        if onnx is not None:
            result["onnx"] = "directml" if onnx == "dml" else str(onnx)
        # 编解码器有意使用的 CPU 会话单独公开，不能把混合执行伪装成全 GPU。
        sessions: dict[str, list[str]] = {}
        visited: set[int] = set()
        pending = [("runtime", self.runtime, 0)]
        while pending:
            path, value, depth = pending.pop()
            if id(value) in visited:
                continue
            visited.add(id(value))
            if callable(getattr(value, "get_providers", None)):
                sessions[path] = list(value.get_providers())
                continue
            if depth >= 4 or not type(value).__module__.startswith("streamvox"):
                continue
            for name, child in vars(value).items() if hasattr(value, "__dict__") else ():
                pending.append((f"{path}.{name}", child, depth + 1))
        result["onnx_sessions"] = sessions
        return result

    def _verify_requested_backends(self) -> None:
        """调用方明确选中的后端不能在模型加载时被静默换成 CPU。

        Returns:
            后端符合请求时返回，否则释放运行时并抛出 RuntimeError。
        """
        requested = {"llama": self.llama_backend, "onnx": self.onnx_provider}
        actual = self.actual_backends
        for family, backend in requested.items():
            if backend is None or family not in actual or actual[family] == backend:
                continue
            self.runtime.shutdown()
            raise RuntimeError(f"{family} 后端发生回退：请求 {backend}，实际 {actual[family]}")

    def make_prompt(
        self,
        role_name: str,
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        sample_rate: int | list[int] | None = None,
        prompt_text: str | list[str] | None = None,
        persist: bool = True,
    ) -> BasePromptAsset:
        """
        构建并可选持久化一个角色 prompt。

        参数:
            role_name: 角色名称，作为 prompt 缓存中的唯一键。
            audio_path: 单个音频路径或多个音频路径；多音频能力由具体模型 adapter 决定。
            audio_data: 单个内存音频数组或多个内存音频数组；与 audio_path 二选一。
            sample_rate: 内存音频采样率；传入多个 audio_data 时可传同一个采样率或等长采样率列表。
            prompt_text: 单条参考文本或多条参考文本；多文本对齐规则由具体模型 adapter 决定。
            persist: 是否写入本地 prompt 缓存。

        返回:
            返回具体模型 adapter 构建的 prompt 资产对象。

        异常:
            当角色名重复、输入类型非法、音频来源缺失或模型不支持多 prompt 时抛出异常。
        """
        # 持久化角色必须先查重，避免后续 adapter 做了昂贵的 codec / prefill 计算后才发现缓存键冲突。
        if persist and role_name in self.prompt_cache.roles:
            raise ValueError(f"role name {role_name} already exists in {self.model}")
        # 引擎层只校验通用形态：音频来源是否存在、列表长度是否自洽、基础类型是否正确。
        # 具体模型是否允许多参考音频，交给 adapter 在 make_prompt 内用模型语义判断。
        self._validate_make_prompt_args(
            role_name=role_name,
            audio_path=audio_path,
            audio_data=audio_data,
            sample_rate=sample_rate,
            prompt_text=prompt_text,
        )

        # sample_rate 已显式传入时原样交给 adapter；缺省时用 runtime 采样率作为路径音频和单音频场景的默认值。
        adapter_sample_rate = sample_rate if sample_rate is not None else self.runtime.sample_rate
        prompt = self.adapter.make_prompt(
            self.runtime,
            role_name=role_name,
            sample_rate=adapter_sample_rate,
            audio_path=audio_path,
            audio_data=audio_data,
            prompt_text=prompt_text,
        )
        self._verify_requested_backends()
        # persist=False 的临时 prompt 直接返回，避免污染用户显式管理的角色缓存。
        if persist:
            self.prompt_cache.store(
                self.adapter,
                prompt,
                self._prompt_cache_record_from_asset(prompt),
            )
        return prompt

    def stream(
        self,
        text: str,
        role_name: str | BasePromptAsset | None = None,
        prompt_audio_path: str | None = None,
        prompt_audio_data: np.ndarray | None = None,
        prompt_audio_sample_rate: int | None = None,
        prompt_text: str | None = None,
        track_performance: bool = False,
        max_length: int | None = None,
        min_length: int | None = None,
        remove_meaningless_chars: bool = True,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        """按块流式合成语音（生成器）。

        若 ``role_name`` 直接传入 prompt 对象，则优先使用该 prompt；若 ``role_name`` 已在缓存中，
        则直接使用对应 prompt；否则须通过 ``prompt_audio_*`` /
        ``prompt_text`` 提供参考音频与文本以即时构建 prompt（或由适配器决定的具体策略）。

        Args:
            text: 待合成的目标文本。
            role_name: 既可以是角色名字符串，也可以直接传 ``make_prompt`` 返回的 prompt 对象。
                传 prompt 对象时会直接复用该内存态资产；传字符串且命中缓存时复用缓存角色；
                为 ``None`` 时使用不落盘的临时角色，必须提供参考音频；非空但未命中缓存时同样必须提供参考音频。
            prompt_audio_path: 参考音频文件路径；与 ``prompt_audio_data`` 二选一，不可同时给。
            prompt_audio_data: 内存中的参考音频波形；若提供则须同时给出 ``prompt_audio_sample_rate``。
            prompt_audio_sample_rate: ``prompt_audio_data`` 的采样率（Hz）；只传文件路径时可省略。
            prompt_text: 与参考音频对应的朗读/转写文本，用于音色克隆对齐。
            track_performance: 是否让底层记录性能数据（具体字段由 ``adapter.synthesize`` 实现决定）。
            max_length: 长文本预处理分段时单段上限（字符或 token 等含义由 ``adapter.normalize_synthesize_kwargs`` 解释）。
            min_length: 分段后合并过短片段的下限，减少碎句。
            remove_meaningless_chars: 是否让具体模型按自己的允许字符正则清除无语义字符。
            **model_kwargs: 透传给 ``adapter.normalize_synthesize_kwargs`` 的模型私有合成参数。

        Yields:
            一段 ``float32`` PCM 音频数组；布局与采样率与当前 ``runtime`` 一致。
        """
        # stream 允许直接传入 make_prompt 返回的 prompt，避免调用方必须先落盘再按角色名二次查询。
        if role_name is None:
            prompt = None
            transient_role_name = None
        elif isinstance(role_name, BasePromptAsset):
            # prompt 资产必须和当前引擎模型一致，避免把别的模型的 prompt 误喂给当前 runtime。
            if role_name.model != self.model:
                raise ValueError(f"prompt model mismatch: expected {self.model!r}, got {role_name.model!r}")
            prompt = role_name
            transient_role_name = None
        elif isinstance(role_name, str):
            prompt = self.prompt_cache.load(self.adapter, self.model, role_name)
            transient_role_name = role_name
        else:
            raise TypeError("role_name must be a string, BasePromptAsset, or None")

        if prompt is None:
            if prompt_audio_path is None and prompt_audio_data is None:
                # 少数模型支持纯文本无 prompt 推理；只有 adapter 显式提供 make_default_prompt 时才走该分支，
                # 避免改变 qwen3/s2-pro 这类必须参考音频的既有模型语义。
                default_prompt_factory = getattr(self.adapter, "make_default_prompt", None)
                if transient_role_name is None and callable(default_prompt_factory):
                    prompt = default_prompt_factory(self.runtime)
                elif transient_role_name is None:
                    raise ValueError("role_name or prompt_audio input is required")
                else:
                    raise ValueError(
                        f"role_name {transient_role_name!r} does not exist for model {self.model}, and no prompt_audio was provided"
                    )
            else:
                transient_role_name = transient_role_name or f"__transient__{uuid.uuid4().hex}"
                self._validate_make_prompt_args(
                    role_name=transient_role_name,
                    audio_path=prompt_audio_path,
                    audio_data=prompt_audio_data,
                    sample_rate=prompt_audio_sample_rate,
                    prompt_text=prompt_text,
                )
                prompt = self.adapter.make_prompt(
                    self.runtime,
                    role_name=transient_role_name,
                    sample_rate=prompt_audio_sample_rate or self.runtime.sample_rate,
                    audio_path=prompt_audio_path,
                    audio_data=prompt_audio_data,
                    prompt_text=prompt_text,
                )

        # 给“模型私有参数”一个正式的下沉入口
        # 分段阈值在引擎层显式暴露，避免用户只能深入模型私有参数才能控制长文本切分。
        normalized_model_kwargs = self.adapter.normalize_synthesize_kwargs(
            max_length=max_length,
            min_length=min_length,
            remove_meaningless_chars=remove_meaningless_chars,
            **model_kwargs,
        )
        # 流式生成
        chunks = self.adapter.synthesize(
            self.runtime,
            text,
            prompt,
            track_performance=track_performance,
            **normalized_model_kwargs,
        )
        try:
            for chunk in chunks:
                self._verify_requested_backends()
                audio = np.asarray(chunk, dtype=np.float32)
                yield audio
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

    def list_roles(self) -> list[str]:
        return sorted(self.prompt_cache.roles)

    def get_prompt_data(self, role_name: str) -> PromptData | None:
        """
        读取指定角色的完整 prompt 数据。

        参数:
            role_name: prompt 角色名。

        返回:
            命中缓存时返回 PromptData；角色不存在时返回 None。

        异常:
            当 role_name 不是非空字符串时抛出 TypeError。
        """
        if not isinstance(role_name, str) or not role_name.strip():
            raise TypeError("role_name must be a non-empty string")
        serialized = self.prompt_cache.load_serialized(self.model, role_name)
        if serialized is None:
            return None
        record = self.prompt_cache.load_prompt_data_record(self.model, role_name) or PromptCacheRecord()
        asset = self.adapter.deserialize_prompt_asset(serialized)
        return PromptData(
            role_name=serialized.role_name,
            model=serialized.model,
            sample_rate=serialized.sample_rate,
            prompt_text=list(record.prompt_text),
            references=record.references,
            prompt=asset,
        )

    def del_roles(self, role_names: str | list[str]) -> list[str]:
        '''删除角色'''
        if isinstance(role_names, str):
            normalized_role_names = [role_names]
        elif isinstance(role_names, list):
            normalized_role_names = role_names
        else:
            raise TypeError("role_names must be a string or list[str]")

        deleted_roles: list[str] = []
        # 删除动作只落在 PromptCache，不碰模型运行时，避免影响正在持有的内存态 prompt。
        for role_name in normalized_role_names:
            if not isinstance(role_name, str) or not role_name.strip():
                raise TypeError("each role name must be a non-empty string")
            if self.prompt_cache.delete(self.model, role_name):
                deleted_roles.append(role_name)
        return deleted_roles

    def shutdown(self) -> None:
        self.runtime.shutdown()

    def _prompt_cache_record_from_asset(self, prompt: BasePromptAsset) -> PromptCacheRecord:
        """
        从模型资产中读取公开 prompt 资料。

        参数:
            prompt: adapter.make_prompt 返回的模型资产。

        返回:
            返回 PromptCache 可持久化的公开 prompt 资料。
        """
        record = getattr(prompt, "prompt_cache_record", None)
        if isinstance(record, PromptCacheRecord):
            return record
        return PromptCacheRecord()

    def _resolve_model_init(
        self,
        *,
        model: str | None,
    ) -> tuple[str, Path]:
        '''
            解析模型初始化入口：
            1. 传本地目录时，从本地 manifest 读取 model 名称并校验；
            2. 传模型名时，先校验支持范围，再自动下载到默认目录。
        '''
        if model is None:
            raise ValueError("model is required")

        # 单参数入口既支持本地 bundle 目录，也支持模型名。
        # 这里优先按“像路径”的输入处理，避免本地 bundle 使用者还要额外传标志位。
        if self._looks_like_model_dir(model):
            resolved_model_dir = Path(model).expanduser().resolve()
            manifest_model_name = self._resolve_model_name_from_manifest(resolved_model_dir)
            self._ensure_supported_model(manifest_model_name)
            return manifest_model_name, resolved_model_dir

        self._ensure_supported_model(model)
        resolved_model_dir = self._ensure_downloaded_model(model)
        return model, resolved_model_dir

    def _looks_like_model_dir(self, model: str) -> bool:
        '''
            判断单参数 model 更像“本地目录”还是“模型名”。
        '''
        candidate = Path(model).expanduser()
        if candidate.exists():
            return True
        if "/" in model or "\\" in model:
            return True
        if model.startswith("."):
            return True
        return False

    def _resolve_model_name_from_manifest(self, model_dir: Path) -> str:
        '''
            从本地 manifest 读取模型名，避免用户初始化时重复传递 model_name。
        '''
        try:
            manifest = load_local_manifest(model_dir)
        except ModelBundleError as exc:
            raise ValueError(
                f"model_dir {model_dir} does not contain a valid {manifest_path_for(model_dir).name}"
            ) from exc

        model_name = manifest.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"local manifest in {model_dir} is missing a valid model field")
        return model_name

    def _ensure_supported_model(self, model_name: str) -> None:
        '''
            统一在入口检查模型是否已经接入，避免后面初始化 runtime 时才报不支持。
        '''
        if model_name not in self.registry.list_models():
            raise ValueError(f"unsupported model: {model_name}")

    def _ensure_downloaded_model(self, model_name: str) -> Path:
        '''
            只给 model_name 时，优先复用默认目录已有 bundle；不存在时再自动下载。
        '''
        default_model_root = get_default_model_root()
        candidate_dir = default_model_root / model_name
        manifest_path = manifest_path_for(candidate_dir)
        if manifest_path.exists():
            manifest_model_name = self._resolve_model_name_from_manifest(candidate_dir)
            if manifest_model_name != model_name:
                raise ValueError(
                    f"default model directory {candidate_dir} belongs to {manifest_model_name!r}, expected {model_name!r}"
                )
            return candidate_dir

        # 延迟导入 download：避免 `import streamvox` 时加载 streamvox.download，否则
        # `python -m streamvox.download` 会在 runpy 里触发「模块已先于 __main__ 加载」的 RuntimeWarning。
        from .download import download_model

        # 这里固定走 modelscope，符合当前产品预期的默认下载来源。
        result = download_model(
            model=model_name,
            source="modelscope",
            target=str(default_model_root),
        )
        return result.target_dir.resolve()

    def _normalize_llama_backend(
        self,
        llama_backend: Literal["cpu", "cuda", "vulkan", "metal"] | None,
    ) -> Literal["cpu", "cuda", "vulkan", "metal"] | None:
        """
        规范化 GGUF llama.cpp backend 显式选择参数。

        参数:
            llama_backend: 用户传入的 GGUF backend 名称；为空时表示保持自动选择。

        返回:
            返回规整后的 backend 名称，或 None。

        异常:
            当传入值不是字符串、为空白字符串或不在允许集合中时抛出异常。
        """
        # llama_backend 是公开 API；这里统一在引擎入口完成校验，避免不同模型 adapter 各自散落一套规则。
        if llama_backend is None:
            return None
        if not isinstance(llama_backend, str):
            raise TypeError("llama_backend must be a string or None")
        normalized = llama_backend.strip().lower()
        if not normalized:
            raise ValueError("llama_backend must not be empty")
        if normalized not in {"cpu", "cuda", "vulkan", "metal"}:
            raise ValueError("llama_backend must be one of: cpu, cuda, vulkan, metal")
        return normalized  # type: ignore[return-value]

    def _validate_make_prompt_args(
        self,
        *,
        role_name: str,
        audio_path: str | list[str] | None,
        audio_data: np.ndarray | list[np.ndarray] | None,
        sample_rate: int | list[int] | None,
        prompt_text: str | list[str] | None,
    ) -> None:
        """
        校验 make_prompt 的通用参数形态。

        参数:
            role_name: prompt 角色名。
            audio_path: 单个或多个音频路径。
            audio_data: 单个或多个内存音频数组。
            sample_rate: 单个或多个采样率。
            prompt_text: 单个或多个参考文本。

        返回:
            本函数无返回值，校验通过即静默结束。

        异常:
            当角色名、音频来源、列表长度或元素类型不合法时抛出 TypeError 或 ValueError。
        """
        # 角色名是 prompt 缓存、日志任务名和 adapter 下游状态键的公共标识，必须是非空字符串。
        if not isinstance(role_name, str) or not role_name.strip():
            raise TypeError("role_name must be a non-empty string")
        # 任意 prompt 字段传入列表时，进入多参考输入校验分支，保证列表内部元素先满足通用契约。
        if self._has_prompt_list_input(
            audio_path=audio_path,
            audio_data=audio_data,
            sample_rate=sample_rate,
            prompt_text=prompt_text,
        ):
            self._validate_make_prompt_list_args(
                audio_path=audio_path,
                audio_data=audio_data,
                sample_rate=sample_rate,
                prompt_text=prompt_text,
            )
            return
        # 单参考音频必须提供路径，或者同时提供内存音频和对应采样率。
        if audio_path is None and (audio_data is None or sample_rate is None):
            raise ValueError("audio_path or audio_data and sample_rate are required")
        # 路径输入与内存输入代表两条互斥的数据来源，不能混用，避免采样率和音频主体错配。
        if audio_path is not None and audio_data is not None:
            raise ValueError("provide either audio_path or audio_data, not both")
        # 内存音频没有文件头信息，必须显式给采样率，adapter 才能按 runtime 采样率重采样。
        if audio_path is None and audio_data is not None and sample_rate is None:
            raise ValueError("sample_rate is required when audio_data is provided")
        # 文件路径只允许字符串，避免 Path 等对象在不同 adapter / librosa 版本里出现隐式行为差异。
        if audio_path is not None and not isinstance(audio_path, str):
            raise TypeError("audio_path must be a string or None")
        # 内存音频只允许 numpy 数组，后续 adapter 会统一转 float32 并展平。
        if audio_data is not None and not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be a numpy.ndarray or None")
        # 采样率必须是整数，避免浮点采样率在重采样库中被隐式截断。
        if sample_rate is not None and not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an int or None")
        # 参考文本允许缺省；一旦提供，必须是字符串，模型私有的必填规则由 adapter 再判断。
        if prompt_text is not None and not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string or None")

    def _has_prompt_list_input(
        self,
        *,
        audio_path: str | list[str] | None,
        audio_data: np.ndarray | list[np.ndarray] | None,
        sample_rate: int | list[int] | None,
        prompt_text: str | list[str] | None,
    ) -> bool:
        """
        判断 make_prompt 是否使用了列表形态输入。

        参数:
            audio_path: 单个或多个音频路径。
            audio_data: 单个或多个内存音频数组。
            sample_rate: 单个或多个采样率。
            prompt_text: 单个或多个参考文本。

        返回:
            任一字段为 list 时返回 True，否则返回 False。

        异常:
            本函数不主动抛出异常。
        """
        # 只把用户明确传入的 list 视为多参考输入；numpy.ndarray 自身也可迭代，不能按序列误判。
        return any(isinstance(value, list) for value in (audio_path, audio_data, sample_rate, prompt_text))

    def _validate_make_prompt_list_args(
        self,
        *,
        audio_path: str | list[str] | None,
        audio_data: np.ndarray | list[np.ndarray] | None,
        sample_rate: int | list[int] | None,
        prompt_text: str | list[str] | None,
    ) -> None:
        """
        校验 make_prompt 的多参考列表参数。

        参数:
            audio_path: 多个音频路径，或在内存音频模式下为 None。
            audio_data: 多个内存音频数组，或在路径模式下为 None。
            sample_rate: 多个采样率，或单个采样率广播到所有内存音频。
            prompt_text: 多条参考文本，或单条由 adapter 自行解释的参考文本。

        返回:
            本函数无返回值，校验通过即静默结束。

        异常:
            当列表为空、路径和内存音频混用、长度错配或元素类型不合法时抛出异常。
        """
        # 列表模式下仍然要求音频来源二选一，避免第 N 条路径和第 M 条数组产生不可解释的混合输入。
        if audio_path is not None and audio_data is not None:
            raise ValueError("provide either audio_path or audio_data, not both")
        # 完全缺少音频来源时，adapter 没有任何参考音频可处理，因此在通用层直接拒绝。
        if audio_path is None and audio_data is None:
            raise ValueError("audio_path or audio_data is required")
        # 路径列表模式用于从磁盘加载多个参考音频，列表不能为空且每一项都必须是非空字符串。
        if isinstance(audio_path, list):
            if not audio_path:
                raise ValueError("audio_path list must not be empty")
            for item in audio_path:
                if not isinstance(item, str) or not item:
                    raise TypeError("each audio_path item must be a non-empty string")
            expected_count = len(audio_path)
        # 单路径仍允许和其它字段的列表形态一起传入；是否有模型意义由具体 adapter 决定。
        elif audio_path is not None:
            if not isinstance(audio_path, str):
                raise TypeError("audio_path must be a string or list[str]")
            expected_count = None
        # 内存音频列表模式用于直接传入多个波形数组，列表不能为空且每一项都必须是 numpy.ndarray。
        elif isinstance(audio_data, list):
            if not audio_data:
                raise ValueError("audio_data list must not be empty")
            for item in audio_data:
                if not isinstance(item, np.ndarray):
                    raise TypeError("each audio_data item must be a numpy.ndarray")
            expected_count = len(audio_data)
        # 单条内存音频仍允许和其它字段的列表形态一起传入，让 adapter 给出模型专属错误或解释。
        else:
            if not isinstance(audio_data, np.ndarray):
                raise TypeError("audio_data must be a numpy.ndarray or list[numpy.ndarray]")
            expected_count = None
        # 内存音频列表没有文件头采样率，必须提供单个采样率或等长采样率列表供 adapter 重采样。
        if isinstance(audio_data, list) and sample_rate is None:
            raise ValueError("sample_rate is required when audio_data list is provided")
        # 单条内存音频同样没有文件头采样率，列表模式下也必须提供采样率，避免 adapter 收到无法重采样的裸波形。
        if isinstance(audio_data, np.ndarray) and sample_rate is None:
            raise ValueError("sample_rate is required when audio_data is provided")
        # 等长采样率列表用于逐条音频重采样，长度必须和音频列表完全一致。
        if isinstance(sample_rate, list):
            if expected_count is not None and len(sample_rate) != expected_count:
                raise ValueError("sample_rate list length must match prompt audio list length")
            for item in sample_rate:
                if not isinstance(item, int):
                    raise TypeError("each sample_rate item must be an int")
        # 单个采样率可以广播到全部内存音频，路径音频则允许省略，由 loader 统一转到 runtime 采样率。
        elif sample_rate is not None and not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an int, list[int], or None")
        # 多条参考文本按位置对齐多条参考音频；长度不一致时下游无法可靠建立 speaker 与 speech token 的对应关系。
        if isinstance(prompt_text, list):
            if expected_count is not None and len(prompt_text) != expected_count:
                raise ValueError("prompt_text list length must match prompt audio list length")
            for item in prompt_text:
                if not isinstance(item, str):
                    raise TypeError("each prompt_text item must be a string")
        # 单条参考文本交给 adapter 自行解释；比如它可以包含显式 speaker tag 或由模型决定是否允许广播。
        elif prompt_text is not None and not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string, list[str], or None")
