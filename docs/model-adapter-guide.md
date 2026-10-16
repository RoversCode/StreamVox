# StreamVox 新模型适配指南

[返回 README](../README.md) · [使用文档](usage.md) · [构建与验证](development.md) · [模型许可](model-licenses.md)

本文面向准备把一个新的 TTS 模型接入 StreamVox 的开发者。目标是让新模型完成适配后，能像现有 `qwen3-tts-clone-0.6b-gguf`、`qwen3-tts-clone-1.7b-gguf`、`s2-pro-4b-gguf`、`voxcpm2-gguf` 一样，通过统一的 `TTSEngine` 初始化、构建 prompt、流式合成、缓存角色、下载模型和发布 wheel。

## 职责拆分设计

新模型适配只需要新增或修改少量明确职责的文件，不要把模型加载、prompt 处理、文本预处理、设备解析和注册逻辑堆到一个文件里。推荐拆分如下：

```text
streamvox/models/<model_key>/
  __init__.py            # 只导出 adapter 类，供默认注册入口导入。
  spec.py                # 只描述模型事实：模型名、尺寸、硬件档位、权重子目录。
  adapter.py             # 连接 TTSEngine 与模型 runtime，负责参数归一化和 prompt 入口。
  runtime.py             # 负责加载权重、持有后端对象、创建 session、关闭资源。
  assets.py              # 可选；复杂模型用它集中解析 ckpt 文件、资产 manifest 或后端配置。
  text/
    __init__.py          # 可选；导出文本预处理配置和 preprocess_text。
    preprocess_text.py   # 可选；模型专属文本清洗、分段和特殊 token 保护。
  schema/
    __init__.py          # 可选；导出模型内部结构。
    *.py                 # 可选；decoder 协议、record、item 等结构体。
  worker/
    __init__.py          # 可选；子进程 decoder 或后台 worker。
    *.py                 # 可选；跨进程运行的代码必须独立成模块，避免 pickle/import 问题。

streamvox/models/config/<model_key>.yaml  # 模型 runtime 默认参数和文件名映射。
streamvox/models/builtin_models.py        # 默认模型注册实现。
streamvox/manifests.py                    # 自动下载时的 ModelScope 仓库映射。
streamvox/release_audit.py                # 检查源码发行包的模块、资源与许可文件。
pyproject.toml                            # 声明依赖、源码包发现和运行资源规则。
tests/test_<model_key>_*.py               # adapter、runtime 初始化、prompt 缓存和 stream 契约测试。
```

这种拆分足够表达 StreamVox 的核心边界：

- `TTSEngine` 是统一产品入口，负责下载、本地 bundle 解析、prompt 缓存、设备选择和统一流式 API。
- `ModelRegistry` 只保存 `model_name -> adapter` 的映射，不关心模型实现细节。
- `ModelAdapter` 是模型接入层，负责把通用 API 翻译成模型私有语义。
- `ModelRuntime` 是模型运行层，负责加载本地权重、持有后端对象和创建合成 session。
- `PromptCache` 只认识 `SerializedPromptAsset`，不理解任何模型内部结构。

如果一个新模型很简单，可以没有 `assets.py`、`schema/`、`worker/`、`text/`。但只要一个文件承担了两个以上职责，就应该拆开；反过来，如果某个函数只是校验参数后完整透传给另一个私有函数，应合并，避免为了拆分而制造空壳调用链。

## 总体接入流程

1. 选定稳定的 `model_name`，例如 `my-tts-1b-gguf`。这个名字会同时出现在 registry、默认模型目录、manifest、prompt cache 子目录和下载映射里。
2. 新建 `streamvox/models/<model_key>/spec.py`，声明不可变模型规格和硬件档位。
3. 新建 `streamvox/models/config/<model_key>.yaml`，声明 sample rate、默认上下文、权重文件名、采样参数和后端默认值。
4. 新建 `runtime.py`，实现模型加载、bundle 校验、`runtime_info`、`sample_rate`、`create_session()` 和 `shutdown()`。
5. 新建 `adapter.py`，实现 `ModelAdapter` 约定：加载 runtime、解析 device、构建 prompt、序列化 prompt、归一化合成参数、调用 session 合成。
6. 在模型包 `__init__.py` 导出 adapter，在 `streamvox/models/builtin_models.py` 注册 adapter。
7. 准备模型 bundle 的 `manifest.json`，并把运行期资产放到 `ckpt/` 下。
8. 如果要支持 `TTSEngine(model="<model_name>")` 自动下载，在 `streamvox/manifests.py` 增加 ModelScope 仓库映射。
9. 确认新模型源码、配置与必要运行资产被 wheel 和 sdist 收集，保留适用的第三方声明。
10. 写测试覆盖本地 bundle 初始化、prompt 缓存、参数归一化、stream 输出、无 prompt 模式等关键路径。

## 统一入口如何调用新模型

`TTSEngine` 的初始化链路是：

```text
TTSEngine(model=...)
  -> 判断 model 是本地目录还是模型名
  -> 本地目录读取 manifest.json 的 model 字段
  -> 模型名检查 default_registry 是否支持
  -> 模型名缺少本地缓存时调用 streamvox.download.download_model
  -> registry.get(model_name) 拿到 adapter
  -> adapter.resolve_execution_plan(device)
  -> adapter.load_runtime(RuntimeRequest(...))
  -> 如 runtime 实现了 warmup_for_init()，则立即执行初始化预热
```

合成链路是：

```text
TTSEngine.stream(...)
  -> 解析 role_name 或 BasePromptAsset
  -> 如未命中缓存，根据 prompt_audio_* 调 adapter.make_prompt(...)
  -> adapter.normalize_synthesize_kwargs(...)
  -> adapter.synthesize(runtime, text, prompt, **normalized_kwargs)
  -> 逐块产出 float32 PCM ndarray
```

因此，新模型接入时不要修改 `TTSEngine`。如果发现必须改引擎层才能支持某个模型，优先判断是不是 adapter 没有正确表达模型能力。只有当所有模型都需要的新公共能力出现时，才应该扩展 `TTSEngine`。

## 1. 定义模型规格

`spec.py` 只保存稳定的模型事实，不加载任何权重，也不 import 重型后端。现有 `qwen3` 用一个 runtime 支持两个尺寸，所以把尺寸、权重子目录和硬件档位放进 spec；`s2_pro` 和 `voxcpm2` 只有一个规格，也仍然使用 spec，避免硬编码散落在 adapter/runtime 中。

```python
# filepath: streamvox/models/my_tts/spec.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MyTTSModelSpec:
    """
    MyTTS 模型规格。

    参数:
        model_name: StreamVox 对外暴露的模型名，也对应默认模型缓存目录名。
        adapter_name: 当前 adapter 类名，用于 runtime_info 和 manifest 追踪。
        parameter_billion: 参数量估算值，用于硬件提示。
        hardware_profile_id: 硬件档位标识，建议包含尺寸、量化和后端关键信息。
        min_vram_gb: 最低显存建议。
        recommended_vram_gb: 推荐显存建议。
        min_ram_gb: 最低内存建议。
        recommended_ram_gb: 推荐内存建议。
        notes: 展示给用户和日志的资源说明。

    返回:
        该类只作为不可变配置容器，不提供运行时返回值。

    边界异常:
        dataclass 本身不做额外校验；非法字段会在 adapter/runtime 使用时暴露。
    """

    model_name: str
    adapter_name: str
    parameter_billion: float
    hardware_profile_id: str
    min_vram_gb: float
    recommended_vram_gb: float
    min_ram_gb: float
    recommended_ram_gb: float
    notes: str


# 关键常量只描述模型事实，避免 runtime 和 adapter 重复写同一份模型名。
MY_TTS_1B_SPEC = MyTTSModelSpec(
    model_name="my-tts-1b-gguf",
    adapter_name="MyTTS1BGGUFAdapter",
    parameter_billion=1.0,
    hardware_profile_id="my-tts-1b-q8-fp16",
    min_vram_gb=3.0,
    recommended_vram_gb=4.0,
    min_ram_gb=4.0,
    recommended_ram_gb=6.0,
    notes="MyTTS 1B GGUF + ONNX FP16 的经验档位。",
)
```

## 2. 编写模型配置

配置文件放在 `streamvox/models/config/`，由 runtime 读取。推荐把“用户可调默认值”和“模型资产文件名”都放进 yaml，避免文件名硬编码在多个类里。

```yaml
# filepath: streamvox/models/config/my_tts.yaml
runtime:
  sample_rate: 24000
  default_n_ctx: 4096
  min_n_ctx: 2048
  max_n_ctx: 32768
  max_length: 300
  min_length: 30
  llama_n_gpu_layers: -1
  decoder_start_method: spawn

sample_config:
  temperature: 0.8
  top_p: 0.9
  top_k: 30
  seed: 42

paths:
  model_filename: my_tts.q8_0.gguf
  decoder_filename: my_tts_decoder.fp16.onnx
  tokenizer_filename: tokenizer.json
```

配置读取建议遵守三个规则：

- `default_n_ctx` 是未传 `TTSEngine(..., n_ctx=...)` 时的默认上下文。
- `min_n_ctx` 和 `max_n_ctx` 是模型能力边界，用户显式传入的 `n_ctx` 应在 runtime 里裁剪或拒绝，并打印清晰日志。
- `paths` 只描述 bundle 内的相对文件名；完整路径由 `model_dir / "ckpt" / filename` 组合。

## 3. 准备模型 bundle 和 manifest

StreamVox 的本地模型目录必须包含 `manifest.json`。标准目录如下：

```text
ckpts/my-tts-1b-gguf/
  manifest.json
  ckpt/
    my_tts.q8_0.gguf
    my_tts_decoder.fp16.onnx
    tokenizer.json
```

`manifest.json` 至少需要包含：

```json
{
  "model": "my-tts-1b-gguf",
  "sources": ["local"],
  "adapter": "MyTTS1BGGUFAdapter",
  "assets": [
    {
      "path": "ckpt/my_tts.q8_0.gguf",
      "sha256": "<64位sha256>"
    },
    {
      "path": "ckpt/my_tts_decoder.fp16.onnx",
      "sha256": "<64位sha256>"
    },
    {
      "path": "ckpt/tokenizer.json",
      "sha256": "<64位sha256>"
    }
  ]
}
```

重要约束：

- `manifest.model` 必须和 adapter 的 `model_name` 完全一致。
- 运行期资产应该放在 `ckpt/` 下；当前 `validate_local_bundle` 只对 `ckpt/` 下的资产做启动前存在性和 sha256 校验。
- `README.md`、`LICENSE`、`NOTICE`、`configuration.json` 这类仓库元信息不会阻断 runtime 启动。
- 自动下载链路会先校验远端 manifest，再把运行期资产复制到默认缓存目录，最后重写本地 manifest。

生成本地 manifest 可以使用项目脚本：

```powershell
uv run python -m streamvox.model_bundle --model my-tts-1b-gguf --root-dir ckpts/my-tts-1b-gguf --asset ckpts/my-tts-1b-gguf/ckpt/my_tts.q8_0.gguf --asset ckpts/my-tts-1b-gguf/ckpt/my_tts_decoder.fp16.onnx --asset ckpts/my-tts-1b-gguf/ckpt/tokenizer.json --source local --adapter MyTTS1BGGUFAdapter
```

## 4. 实现 Runtime

Runtime 是真正持有模型后端的对象。它应该满足 `streamvox/models/base.py` 中 `ModelRuntime` 的协议：

- `model_name: str`
- `sample_rate: int`
- `runtime_info: RuntimeInfo`
- `shutdown() -> None`

此外，runtime 现在还可以选择实现一个约定式初始化预热钩子：

- `warmup_for_init() -> None`

这个钩子不是 `ModelAdapter` 的公开契约，也不要求所有模型都实现。`TTSEngine` 会在 `adapter.load_runtime(...)` 成功返回后，通过 `getattr(runtime, "warmup_for_init", None)` 检查并调用它。设计约束如下：

- 预热应该是 runtime 内部行为，不新增用户开关，不暴露到公开 API。
- 预热失败必须直接抛出异常，终止 `TTSEngine` 初始化；不要把“权重加载成功但关键链路未热起”误记成初始化成功。
- runtime 内部应自行保证幂等，例如用 `_init_warmed` 标记避免重复调用。
- 预热只负责把 ONNX、GGUF、decoder worker 等关键路径各走一遍最小合法输入，不应污染真实请求状态；如果预热会写入 KV cache、decoder state 或上下文缓存，必须在结束前显式清理。

推荐 runtime 做这些事情：

- 读取 yaml 配置。
- 解析 `RuntimeRequest.execution_plan` 中的 backend 意图。
- 调用 `validate_local_bundle(model_dir, model_name, verify_model_sha256=...)` 校验本地 bundle。
- 解析 `ckpt/` 下的模型文件路径。
- 加载 GGUF、ONNX、Torch 或其它后端对象。
- 如模型存在明显的首轮初始化抖动，实现 `warmup_for_init()`，在 runtime 内预热关键推理链路。
- 暴露 `create_session()`，让每次合成有独立的短生命周期状态。
- `shutdown()` 幂等关闭 native 资源、子进程和 worker。

```python
# filepath: streamvox/models/my_tts/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from ...model_bundle import validate_local_bundle
from ..base import BasePromptAsset, ExecutionPlan, RuntimeInfo, SerializedPromptAsset
from .spec import MY_TTS_1B_SPEC, MyTTSModelSpec


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "my_tts.yaml"


@dataclass(slots=True)
class MyTTSConfig:
    """
    MyTTS runtime 配置。

    参数:
        sample_rate: 输出音频采样率。
        n_ctx: 当前 runtime 使用的上下文窗口。
        min_n_ctx: 允许的最小上下文窗口。
        max_n_ctx: 允许的最大上下文窗口。
        max_length: 文本预处理默认最大分段长度。
        min_length: 文本预处理默认最小合并长度。
        model_filename: 主模型权重文件名。
        decoder_filename: decoder 权重文件名。
        tokenizer_filename: tokenizer 文件名。

    返回:
        配置对象只承载 runtime 初始化需要的默认值。

    边界异常:
        字段类型转换失败会在配置加载阶段抛出异常，避免后端加载到一半才失败。
    """

    sample_rate: int = 24000
    n_ctx: int = 4096
    min_n_ctx: int = 2048
    max_n_ctx: int = 32768
    max_length: int = 300
    min_length: int = 30
    model_filename: str = "my_tts.q8_0.gguf"
    decoder_filename: str = "my_tts_decoder.fp16.onnx"
    tokenizer_filename: str = "tokenizer.json"


@dataclass(slots=True)
class MyTTSPromptAsset(BasePromptAsset):
    """
    MyTTS prompt 资产。

    参数:
        role_name: prompt 角色名。
        model: prompt 所属 StreamVox 模型名。
        sample_rate: prompt 音频采样率。
        prompt_text: 参考音频对应文本；无参考文本模型可以为空。
        speaker_embedding: 说话人向量，供后续合成复用。

    返回:
        该对象由 adapter.make_prompt 构建，并可通过 PromptCache 序列化到本地。

    边界异常:
        prompt 与 runtime 的模型名不一致时，应在 adapter.synthesize 中拒绝。
    """

    prompt_text: str | None = None
    speaker_embedding: np.ndarray | None = None


def load_my_tts_config() -> MyTTSConfig:
    """
    加载 MyTTS yaml 配置。

    参数:
        本函数无显式参数；配置路径固定为当前包的 config/my_tts.yaml。

    返回:
        返回完成默认值合并后的 MyTTSConfig。

    边界异常:
        yaml 结构非法或字段无法转换为目标类型时抛出异常。
    """

    # 配置文件缺失时使用保守默认值，便于单元测试构造最小 runtime。
    if not CONFIG_PATH.exists():
        return MyTTSConfig()

    # yaml 分成 runtime 和 paths 两块，避免采样参数、上下文参数和文件名互相污染。
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime", {})
    paths = payload.get("paths", {})
    default_n_ctx = int(runtime.get("default_n_ctx", runtime.get("n_ctx", 4096)))
    return MyTTSConfig(
        sample_rate=int(runtime.get("sample_rate", 24000)),
        n_ctx=default_n_ctx,
        min_n_ctx=int(runtime.get("min_n_ctx", 2048)),
        max_n_ctx=int(runtime.get("max_n_ctx", 32768)),
        max_length=int(runtime.get("max_length", 300)),
        min_length=int(runtime.get("min_length", 30)),
        model_filename=str(paths.get("model_filename", "my_tts.q8_0.gguf")),
        decoder_filename=str(paths.get("decoder_filename", "my_tts_decoder.fp16.onnx")),
        tokenizer_filename=str(paths.get("tokenizer_filename", "tokenizer.json")),
    )


def _clamp_n_ctx(config: MyTTSConfig, requested_n_ctx: int) -> int:
    """
    将上下文窗口限制在 MyTTS 声明的能力范围内。

    参数:
        config: 已加载的 MyTTS 配置。
        requested_n_ctx: 用户显式传入或 yaml 默认的上下文窗口。

    返回:
        返回位于 [min_n_ctx, max_n_ctx] 内的上下文窗口。

    边界异常:
        本函数不主动抛出异常；调用方负责传入整数。
    """

    # 上下文窗口超界时按模型能力裁剪，避免后端因为不可恢复的 native 错误崩溃。
    return max(config.min_n_ctx, min(config.max_n_ctx, int(requested_n_ctx)))


class MyTTSRuntime:
    """
    MyTTS 模型运行时。

    参数:
        model_dir: 本地模型 bundle 根目录。
        model_spec: 当前模型规格。
        device: 用户传给 TTSEngine 的原始设备字符串。
        n_ctx: 用户显式传入的上下文窗口；为空时使用 yaml 默认值。
        verify_model_sha256: 是否校验 manifest 中声明的运行期资产 sha256。
        execution_plan: adapter 根据 device 解析出的统一执行计划。

    返回:
        初始化后得到可创建 session 的 runtime 对象。

    边界异常:
        bundle 缺失、manifest 模型名不匹配、资产缺失、后端加载失败时抛出异常。
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        model_spec: MyTTSModelSpec = MY_TTS_1B_SPEC,
        device: str = "auto",
        n_ctx: int | None = None,
        verify_model_sha256: bool = True,
        execution_plan: ExecutionPlan | None = None,
    ) -> None:
        self.model_name = model_spec.model_name
        self.model_spec = model_spec
        self.bundle_root = Path(model_dir)
        self.ckpt_root = self.bundle_root / "ckpt"
        self.device = device
        self.execution_plan = execution_plan

        # 先加载轻量配置，再用用户显式 n_ctx 覆盖默认值，保持 TTSEngine 的入口语义。
        self.config = load_my_tts_config()
        self.config.n_ctx = _clamp_n_ctx(self.config, n_ctx if n_ctx is not None else self.config.n_ctx)

        # 本地 bundle 校验放在后端加载前，失败时给用户清晰的文件级错误。
        bundle = validate_local_bundle(
            self.bundle_root,
            self.model_name,
            verify_model_sha256=verify_model_sha256,
        )

        # 运行期路径统一从 ckpt_root 和 yaml 文件名组合，避免散落硬编码。
        self.paths = {
            "model": self.ckpt_root / self.config.model_filename,
            "decoder": self.ckpt_root / self.config.decoder_filename,
            "tokenizer": self.ckpt_root / self.config.tokenizer_filename,
        }

        # 这里加载真实后端对象；示例中保留占位，实际模型应替换成 GGUF/ONNX/Torch 初始化。
        self.model_backend = None
        self.decoder_backend = None

        # runtime_info 是用户排查环境和 bundle 状态的第一入口，metadata 应保留关键路径和后端信息。
        self.runtime_info = RuntimeInfo(
            model=self.model_name,
            adapter=model_spec.adapter_name,
            bundle_loaded=True,
            asset_count=len(bundle.asset_paths),
            metadata={
                "model_dir": str(self.bundle_root),
                "device": str(device),
                "n_ctx": str(self.config.n_ctx),
            },
        )
        # 初始化预热是可选约定，不属于公共协议强制项；
        # 如果你的模型首轮 synthesize 会明显卡在 ONNX session、decoder worker 或 GGUF context 冷启动，
        # 应实现 warmup_for_init() 并在里面做最小合法输入预热。
        self._init_warmed = False

    @property
    def sample_rate(self) -> int:
        """
        返回当前 runtime 输出采样率。

        参数:
            本属性无参数。

        返回:
            返回整数 Hz 采样率。

        边界异常:
            配置加载阶段已保证 sample_rate 可转成整数。
        """

        return int(self.config.sample_rate)

    def create_session(self) -> "MyTTSSession":
        """
        创建一次合成会话。

        参数:
            本函数无显式参数；会话复用当前 runtime 的后端对象和配置。

        返回:
            返回 MyTTSSession，用于持有单次合成的短生命周期状态。

        边界异常:
            runtime 已 shutdown 后继续创建 session 时，应由真实实现抛出清晰异常。
        """

        return MyTTSSession(self)

    def warmup_for_init(self) -> None:
        """
        可选的初始化预热钩子。

        参数:
            本函数无显式参数；依赖当前 runtime 已完成后端对象加载。

        返回:
            无返回值；成功后应把 runtime 标记成“已完成初始化预热”。

        边界异常:
            预热失败时直接抛出异常，阻止 TTSEngine 继续初始化。
        """

        # 用幂等标记保护重复调用；测试和异常恢复路径都可能再次触发这个钩子。
        if self._init_warmed:
            return

        # 真实实现应该在这里构造最小合法输入，分别预热模型主干、decoder 和必要的辅助编码器。
        # 如果预热过程中写入了 KV cache、decoder state 或上下文缓存，必须在成功/失败两条路径都显式清理。

        self._init_warmed = True

    def shutdown(self) -> None:
        """
        关闭 runtime 持有的后端资源。

        参数:
            本函数无显式参数。

        返回:
            无返回值；重复调用应安全返回。

        边界异常:
            关闭底层 native 资源失败时应记录日志，避免解释器退出阶段产生噪音。
        """

        # shutdown 必须幂等，TTSEngine 用户可能在异常清理路径中重复调用。
        if getattr(self, "_shutdown_started", False):
            return
        self._shutdown_started = True

        # 真实实现应在这里关闭 decoder worker、ONNX session、llama context 等后端资源。
        self.model_backend = None
        self.decoder_backend = None


class MyTTSSession:
    """
    MyTTS 单次合成会话。

    参数:
        runtime: 已初始化的 MyTTSRuntime。

    返回:
        session 本身不直接返回音频；音频由 synthesize 生成器逐块产出。

    边界异常:
        文本为空、prompt 类型不匹配或后端推理失败时抛出异常。
    """

    def __init__(self, runtime: MyTTSRuntime) -> None:
        self.runtime = runtime

    def synthesize(
        self,
        *,
        text: str,
        prompt: MyTTSPromptAsset,
        max_length: int | None = None,
        min_length: int | None = None,
        **generation_options: object,
    ) -> Iterable[np.ndarray]:
        """
        执行一次 MyTTS 流式合成。

        参数:
            text: 待合成文本。
            prompt: 已构建或缓存恢复的 MyTTS prompt。
            max_length: 文本分段最大长度；为空时使用 runtime 配置。
            min_length: 文本分段最小合并长度；为空时使用 runtime 配置。
            generation_options: temperature/top_p 等模型私有采样参数。

        返回:
            逐块产出 float32 PCM ndarray，采样率等于 runtime.sample_rate。

        边界异常:
            文本为空、prompt 模型不匹配或后端返回非法音频时抛出异常。
        """

        # 入口先做业务级校验，避免后端收到空文本后报出难读的 native 错误。
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if prompt.model != self.runtime.model_name:
            raise ValueError("prompt model does not match current runtime")

        # 真实实现应在这里调用文本预处理、语义模型、codec decoder，并按流式块 yield。
        yield np.zeros(self.runtime.sample_rate // 10, dtype=np.float32)


def serialize_prompt_asset(asset: MyTTSPromptAsset) -> SerializedPromptAsset:
    """
    序列化 MyTTS prompt 资产。

    参数:
        asset: adapter.make_prompt 返回的 MyTTSPromptAsset。

    返回:
        返回 PromptCache 可保存的 SerializedPromptAsset。

    边界异常:
        prompt 类型错误或数组字段无法转成 numpy.ndarray 时抛出异常。
    """

    # 数组字段放 arrays，字符串和轻量结构放 json_data，便于 PromptCache 稳定落盘。
    arrays = {}
    if asset.speaker_embedding is not None:
        arrays["speaker_embedding"] = np.asarray(asset.speaker_embedding, dtype=np.float32)
    return SerializedPromptAsset(
        role_name=asset.role_name,
        model=asset.model,
        sample_rate=asset.sample_rate,
        metadata=dict(asset.metadata),
        arrays=arrays,
        json_data={"prompt_text": asset.prompt_text},
    )


def deserialize_prompt_asset(payload: SerializedPromptAsset) -> MyTTSPromptAsset:
    """
    从 PromptCache 载荷恢复 MyTTS prompt。

    参数:
        payload: PromptCache 从 npz 文件读取出的序列化载荷。

    返回:
        返回可直接传给 adapter.synthesize 的 MyTTSPromptAsset。

    边界异常:
        必要数组缺失或 json_data 类型非法时抛出异常。
    """

    return MyTTSPromptAsset(
        role_name=payload.role_name,
        model=payload.model,
        sample_rate=payload.sample_rate,
        metadata=dict(payload.metadata),
        prompt_text=payload.json_data.get("prompt_text"),
        speaker_embedding=payload.arrays.get("speaker_embedding"),
    )
```

## 5. 实现 Adapter

Adapter 是新模型最关键的适配层。它不应该直接承担重型推理，只负责把 StreamVox 的通用调用转成模型私有调用。

必须覆盖的能力：

- `model_name`
- `load_runtime(request)`
- `get_text_preprocess_config()`
- `get_hardware_profile()`
- `resolve_execution_plan(device)`
- `make_prompt(...)`
- `serialize_prompt_asset(asset)`
- `deserialize_prompt_asset(payload)`
- `normalize_synthesize_kwargs(**model_kwargs)`
- `synthesize(runtime, text, prompt, **kwargs)`

`ModelAdapter` 协议里还有 `validate_make_prompt_args`，但当前 `TTSEngine` 已经在 `_validate_make_prompt_args` 中做通用参数校验，现有正式 adapter 也把模型语义校验放在 `make_prompt` 内。因此新增模型通常不需要额外实现一个只做透传的 `validate_make_prompt_args`；如果你的类型检查或自定义 registry 强制要求完整协议，可以实现它，但不要再拆出 `_validate_xxx` 空壳函数。

```python
# filepath: streamvox/models/my_tts/adapter.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import librosa
import numpy as np

from ...hardware import ModelFootprintProfile
from ...utils.log import logger
from ..base import BasePromptAsset, ExecutionPlan, ExecutionTarget, RuntimeRequest, SerializedPromptAsset
from ..text_processing import TextPreprocessConfig
from .spec import MY_TTS_1B_SPEC, MyTTSModelSpec

if TYPE_CHECKING:
    from .runtime import MyTTSRuntime


class MyTTS1BGGUFAdapter:
    """
    MyTTS 1B GGUF 的 StreamVox 适配器。

    参数:
        spec: 模型规格，默认使用 MY_TTS_1B_SPEC。

    返回:
        adapter 本身不直接返回音频；音频由 synthesize 生成器逐块产出。

    边界异常:
        device、prompt、采样参数和模型私有参数非法时在 adapter 层提前抛出清晰异常。
    """

    spec: MyTTSModelSpec = MY_TTS_1B_SPEC

    @property
    def model_name(self) -> str:
        """
        返回 StreamVox 对外模型名。

        参数:
            本属性无参数。

        返回:
            返回 registry 和 manifest 使用的模型名称。

        边界异常:
            本属性不主动抛出异常。
        """

        return self.spec.model_name

    def load_runtime(self, request: RuntimeRequest) -> "MyTTSRuntime":
        """
        按 RuntimeRequest 加载 MyTTS runtime。

        参数:
            request: TTSEngine 构造的运行时请求，包含模型目录、设备、n_ctx 和校验开关。

        返回:
            返回已加载权重的 MyTTSRuntime。

        边界异常:
            模型 bundle 或后端加载失败时由 runtime 抛出异常。
        """

        from .runtime import MyTTSRuntime

        logger.info("[StreamVox] loading my-tts runtime")
        return MyTTSRuntime(
            model_dir=Path(request.model_dir),
            model_spec=self.spec,
            device=request.device,
            n_ctx=request.n_ctx,
            verify_model_sha256=request.verify_model_sha256,
            execution_plan=request.execution_plan,
        )

    def get_text_preprocess_config(self) -> TextPreprocessConfig | None:
        """
        返回 MyTTS 文本预处理配置。

        参数:
            本函数无参数。

        返回:
            如果模型需要保护特殊 token，返回 TextPreprocessConfig；否则返回 None。

        边界异常:
            本函数不主动抛出异常。
        """

        # 无模型专属特殊 token 时返回 None，让 runtime/session 自行决定是否调用通用预处理。
        return None

    def get_hardware_profile(self) -> ModelFootprintProfile:
        """
        返回 MyTTS 硬件资源建议。

        参数:
            本函数无参数。

        返回:
            返回 TTSEngine 初始化阶段用于资源体检和日志提示的硬件档位。

        边界异常:
            本函数不主动抛出异常。
        """

        return ModelFootprintProfile(
            profile_id=self.spec.hardware_profile_id,
            parameter_billion=self.spec.parameter_billion,
            quantization="gguf(q8_0)+onnx(fp16)",
            min_vram_gb=self.spec.min_vram_gb,
            recommended_vram_gb=self.spec.recommended_vram_gb,
            min_ram_gb=self.spec.min_ram_gb,
            recommended_ram_gb=self.spec.recommended_ram_gb,
            notes=self.spec.notes,
        )

    def resolve_execution_plan(self, device: str) -> ExecutionPlan:
        """
        把用户 device 字符串解析成 runtime family 执行计划。

        参数:
            device: 用户传入的 auto/cpu/gpu/gpu:N。

        返回:
            返回包含 llama 和 onnx 两类 family 的 ExecutionPlan。

        边界异常:
            device 类型错误或格式不支持时抛出异常。
        """

        mode, index = self._parse_device_spec(device)
        return ExecutionPlan(
            raw_device=device,
            targets={
                "llama": ExecutionTarget(family="llama", mode=mode, device_index=index),
                "onnx": ExecutionTarget(family="onnx", mode=mode, device_index=index),
            },
        )

    def _parse_device_spec(self, device: str) -> tuple[str, int | None]:
        """
        解析 auto/cpu/gpu/gpu:N 设备字符串。

        参数:
            device: 用户传入的设备描述。

        返回:
            返回标准化模式和可选 GPU 序号。

        边界异常:
            非字符串、非法 GPU 序号或未知模式会抛出异常。
        """

        if not isinstance(device, str):
            raise TypeError("device must be a string: auto, cpu, gpu or gpu:<index>")
        raw = device.strip().lower()
        if raw in {"auto", "cpu"}:
            return raw, None
        if raw == "gpu":
            return "gpu", 0
        if raw.startswith("gpu:"):
            index_text = raw.split(":", 1)[1]
            if not index_text.isdigit():
                raise ValueError(f"device index must be a non-negative integer: {device}")
            return "gpu", int(index_text)
        raise ValueError(f"unsupported device value: {device}")

    def normalize_synthesize_kwargs(self, **model_kwargs: object) -> dict[str, object]:
        """
        校验并归一化 MyTTS 私有合成参数。

        参数:
            model_kwargs: TTSEngine.stream 透传的模型私有参数。

        返回:
            返回只包含受支持字段的参数字典。

        边界异常:
            参数类型非法或数值越界时抛出异常。
        """

        valid_params = {"temperature", "top_p", "top_k", "seed", "max_length", "min_length", "remove_meaningless_chars"}
        normalized = {key: value for key, value in model_kwargs.items() if key in valid_params and value is not None}

        # 数值采样参数在 adapter 层提前校验，避免推理跑到一半才因后端类型错误中断。
        for name in ("temperature", "top_p"):
            if name in normalized and not isinstance(normalized[name], (int, float)):
                raise TypeError(f"{name} must be a number")
        for name in ("top_k", "seed", "max_length", "min_length"):
            if name in normalized and (not isinstance(normalized[name], int) or isinstance(normalized[name], bool)):
                raise TypeError(f"{name} must be an integer")
        if "remove_meaningless_chars" in normalized and not isinstance(normalized["remove_meaningless_chars"], bool):
            raise TypeError("remove_meaningless_chars must be a boolean")
        if "max_length" in normalized and int(normalized["max_length"]) <= 0:
            raise ValueError("max_length must be greater than 0")
        if "min_length" in normalized and int(normalized["min_length"]) < 0:
            raise ValueError("min_length must be greater than or equal to 0")
        if (
            "max_length" in normalized
            and "min_length" in normalized
            and int(normalized["min_length"]) > int(normalized["max_length"])
        ):
            raise ValueError("min_length must be less than or equal to max_length")
        return normalized

    def make_prompt(
        self,
        runtime: "MyTTSRuntime",
        role_name: str,
        sample_rate: int | list[int],
        audio_path: str | list[str] | None = None,
        audio_data: np.ndarray | list[np.ndarray] | None = None,
        prompt_text: str | list[str] | None = None,
    ) -> BasePromptAsset:
        """
        构建 MyTTS 单参考音频 prompt。

        参数:
            runtime: 已加载的 MyTTS runtime。
            role_name: prompt 角色名。
            sample_rate: 内存音频采样率；路径模式可由音频文件自身决定。
            audio_path: 单条参考音频路径。
            audio_data: 单条参考音频数组。
            prompt_text: 参考音频对应文本，可按模型需要设为必填或可选。

        返回:
            返回 MyTTSPromptAsset。

        边界异常:
            多参考输入、音频缺失、采样率非法或 prompt 编码失败时抛出异常。
        """

        from .runtime import MyTTSPromptAsset

        # 当前示例模型只支持单参考音频；列表输入不能静默降级，否则 prompt 与文本对齐会变得不可解释。
        if any(isinstance(value, list) for value in (sample_rate, audio_path, audio_data, prompt_text)):
            raise ValueError("my-tts make_prompt supports exactly one prompt audio")
        if prompt_text is not None and not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string or None")
        if not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an int")

        # 未传内存音频时从路径读取；路径也缺失说明调用方没有给模型可用的音色参考。
        if audio_data is None:
            if audio_path is None:
                raise ValueError("audio_path or audio_data is required to build a prompt")
            audio_data, input_rate = librosa.load(audio_path, sr=runtime.sample_rate, mono=True)
        else:
            input_rate = sample_rate

        # 后端 encoder 通常只接受固定采样率，输入不一致时提前重采样，保证 prompt token 时间尺度稳定。
        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        if input_rate != runtime.sample_rate:
            audio = np.asarray(
                librosa.resample(audio, orig_sr=input_rate, target_sr=runtime.sample_rate),
                dtype=np.float32,
            )

        # 真实模型应在这里调用 speaker encoder、codec encoder 或 prompt prefill。
        speaker_embedding = np.zeros((256,), dtype=np.float32)

        return MyTTSPromptAsset(
            role_name=role_name,
            model=self.model_name,
            sample_rate=runtime.sample_rate,
            prompt_text=prompt_text,
            speaker_embedding=speaker_embedding,
            metadata={},
        )

    def serialize_prompt_asset(self, asset: BasePromptAsset) -> SerializedPromptAsset:
        """
        序列化 MyTTS prompt 到 PromptCache。

        参数:
            asset: make_prompt 返回的 prompt 资产。

        返回:
            返回 PromptCache 可保存的 SerializedPromptAsset。

        边界异常:
            prompt 类型不是 MyTTSPromptAsset 时抛出 TypeError。
        """

        from .runtime import MyTTSPromptAsset, serialize_prompt_asset

        if not isinstance(asset, MyTTSPromptAsset):
            raise TypeError("my-tts adapter can only serialize MyTTSPromptAsset")
        return serialize_prompt_asset(asset)

    def deserialize_prompt_asset(self, payload: SerializedPromptAsset) -> BasePromptAsset:
        """
        从 PromptCache 载荷恢复 MyTTS prompt。

        参数:
            payload: PromptCache 读取出的序列化载荷。

        返回:
            返回 MyTTSPromptAsset。

        边界异常:
            payload 缺少必要字段时由 runtime.deserialize_prompt_asset 抛出异常。
        """

        from .runtime import deserialize_prompt_asset

        return deserialize_prompt_asset(payload)

    def synthesize(
        self,
        runtime: "MyTTSRuntime",
        text: str,
        prompt: BasePromptAsset,
        stream: bool = True,
        track_performance: bool = False,
        max_length: int | None = None,
        min_length: int | None = None,
        remove_meaningless_chars: bool = True,
        **model_kwargs: object,
    ) -> Iterable[np.ndarray]:
        """
        创建 session 并执行 MyTTS 合成。

        参数:
            runtime: 已加载的 MyTTS runtime。
            text: 待合成文本。
            prompt: 已构建或缓存恢复的 prompt。
            stream: 是否使用流式路径；如果模型没有非流式路径，可以接受但忽略。
            track_performance: 是否记录模型内部性能指标。
            max_length: 文本分段最大长度。
            min_length: 文本分段最小合并长度。
            remove_meaningless_chars: 是否清除无语义字符。
            model_kwargs: normalize_synthesize_kwargs 允许的模型私有参数。

        返回:
            逐块产出 float32 PCM ndarray。

        边界异常:
            prompt 类型错误、未支持参数绕过归一化或后端推理失败时抛出异常。
        """

        from .runtime import MyTTSPromptAsset

        if not isinstance(prompt, MyTTSPromptAsset):
            raise TypeError("my-tts synthesize requires MyTTSPromptAsset")

        # adapter.synthesize 只应该收到 normalize_synthesize_kwargs 的结果；兜底拒绝未知参数，方便调用方定位拼写错误。
        allowed_generation_keys = {"temperature", "top_p", "top_k", "seed"}
        unknown_keys = sorted(set(model_kwargs) - allowed_generation_keys)
        if unknown_keys:
            raise ValueError(f"unsupported my-tts synthesize parameters: {', '.join(unknown_keys)}")

        session = runtime.create_session()
        yield from session.synthesize(
            text=text,
            prompt=prompt,
            max_length=max_length,
            min_length=min_length,
            remove_meaningless_chars=remove_meaningless_chars,
            track_performance=track_performance,
            stream=stream,
            **model_kwargs,
        )
```

## 6. 支持无 Prompt 纯文本模式

默认情况下，`TTSEngine.stream(...)` 在没有 `role_name`、没有缓存 prompt、也没有 `prompt_audio_*` 时会拒绝请求。像 `voxcpm2-gguf` 这种支持纯文本直接生成的模型，需要在 adapter 上额外提供：

```python
def make_default_prompt(self, runtime: MyTTSRuntime) -> BasePromptAsset:
    """
    构建纯文本模式使用的默认 prompt。

    参数:
        runtime: 已加载的模型 runtime。

    返回:
        返回可传给 synthesize 的空 prompt 或默认 prompt。

    边界异常:
        如果模型不支持纯文本模式，不要实现该函数。
    """
```

实现原则：

- 支持纯文本就实现 `make_default_prompt`，并确保 `adapter.synthesize` 能根据 prompt 形态或显式参数进入 text 模式。
- 不支持纯文本就不要实现该函数，让 `TTSEngine` 继续要求调用方提供参考音频。
- 不要在 `TTSEngine` 里为某个模型写特判。

## 7. Prompt 资产序列化规则

`PromptCache` 会调用：

```text
adapter.serialize_prompt_asset(prompt)
adapter.deserialize_prompt_asset(serialized_payload)
```

`SerializedPromptAsset` 有三类载荷：

- `metadata`: 简单字符串字典，适合版本、模式、说明类字段。
- `arrays`: numpy 数组字典，适合 speaker embedding、codec token、decoder tail、hidden state。
- `json_data`: 可 JSON 序列化的结构，适合 prompt_text、位置、轻量配置、列表。

设计 prompt 资产时要注意：

- prompt asset 必须继承 `BasePromptAsset`，至少包含 `role_name`、`model`、`sample_rate`。
- `asset.model` 必须写 adapter 的 `model_name`，否则跨模型复用 prompt 时无法防错。
- 大数组放 `arrays`，不要塞进 `metadata` 或 JSON。
- 恢复后的 prompt 应能直接用于 `adapter.synthesize`，不要依赖 make_prompt 阶段的临时对象。
- 如果 prompt 里保存了后端状态快照，必须确认它可以跨进程、跨解释器重启恢复；否则只保存可重建的中间结果。

## 8. 文本预处理接入

如果模型需要特殊 token、语言标记或长文本分段规则，应提供文本预处理配置。现有模型做法有两种：

- `qwen3` 在 adapter 中直接返回 `TextPreprocessConfig(special_tokens=(...))`。
- `s2_pro` 和 `voxcpm2` 在 `text/preprocess_text.py` 中维护模型专属配置和预处理函数。

建议：

- 简单模型只在 adapter 返回 `TextPreprocessConfig`。
- 文本规则复杂、需要复用或测试时，新增 `streamvox/models/<model_key>/text/preprocess_text.py`。
- `language`、`mode`、`control_text` 等模型私有语义应通过 `normalize_synthesize_kwargs` 校验，再传给 runtime/session。
- 文本预处理默认阈值优先来自 yaml；用户通过 `TTSEngine.stream(max_length=..., min_length=...)` 显式传入时覆盖默认值。

## 9. 注册默认模型

模型包导出 adapter：

```python
# filepath: streamvox/models/my_tts/__init__.py
from __future__ import annotations

from .adapter import MyTTS1BGGUFAdapter

__all__ = ["MyTTS1BGGUFAdapter"]
```

默认注册入口增加导入和注册：

```python
# filepath: streamvox/models/builtin_models.py
from .my_tts import MyTTS1BGGUFAdapter


def register_default_models() -> None:
    """
    统一维护当前包默认支持的模型清单。

    参数:
        本函数无参数。

    返回:
        无返回值；重复调用保持幂等。

    边界异常:
        adapter 构造失败时会在导入阶段暴露，避免 registry 处于半注册状态。
    """

    # 新模型只需要加入 adapter 元组；registry 自身只负责存取，不承载模型清单。
    for adapter in (..., MyTTS1BGGUFAdapter()):
        if adapter.model_name not in default_registry.list_models():
            default_registry.register(adapter)
```

注册后应满足：

```python
from streamvox.models import default_registry

assert "my-tts-1b-gguf" in default_registry.list_models()
```

## 10. 自动下载接入

如果希望用户只传模型名即可自动下载：

```python
# filepath: streamvox/manifests.py
DEFAULT_MODELSCOPE_REPOS: dict[str, str] = {
    "my-tts-1b-gguf": "YourOrg/my-tts-1b-gguf",
}
```

ModelScope 仓库必须包含标准 `manifest.json`。下载链路会：

1. 根据 `DEFAULT_MODELSCOPE_REPOS[model_name]` 找到仓库。
2. 调用 ModelScope SDK 的 `snapshot_download`。
3. 复制仓库内容到默认模型目录下的 `<model_name>`。
4. 校验 manifest 中 `ckpt/` 运行期资产的 sha256。
5. 写入本地缓存 manifest，补充 `sources=["local", "modelscope", ...]`。

如果你的模型还没有发布远端仓库，可以先只支持本地 bundle：

```python
engine = TTSEngine(model="ckpts/my-tts-1b-gguf")
```

## 11. 打包和发布接入

2.0.0 在 Windows x64 / CPython 3.12 上分发可读 Python 源码，wheel tag 为 `cp312-none-win_amd64`。新增 adapter、runtime、schema、worker 和文本处理模块应直接作为源码包含在包中。

`pyproject.toml` 负责包发现和资源收集。新增其他类型的运行资产时，补充包数据规则；不要将模型权重塞进 SDK 安装包。模型 bundle 应单独分发并保留其 LICENSE/NOTICE。新模型的第三方代码或派生实现也应记录实际来源提交，保留原始版权与许可。

公开 Git 仓库不保存原生 DLL。使用 `python -m streamvox.native_assets --project-root .` 恢复固定官方归档中的运行库，再使用发行工具构建 wheel 与含 DLL 的 sdist。完整命令、来源记录与安装验收见 [构建与验证](development.md)。

构建后应检查最终归档中的源码、配置、原生库和法律文件，并在干净 CPython 3.12 环境中导入、初始化和运行模型。
## 12. 测试清单

新增模型至少覆盖下面几类测试。

### Registry 和初始化

- `default_registry.list_models()` 包含新 `model_name`。
- `TTSEngine(model="<本地bundle路径>")` 能从 manifest 解析模型名并初始化。
- `TTSEngine(model="<model_name>")` 在默认缓存存在时复用本地目录，不触发下载。
- `verify_model_sha256=False` 能透传到 runtime。

### Adapter 参数契约

- `resolve_execution_plan("auto")`、`"cpu"`、`"gpu"`、`"gpu:1"` 行为正确。
- 非法 device、非法 mode、非法采样参数能在 adapter 层抛出清晰异常。
- `normalize_synthesize_kwargs` 只保留支持字段，并拒绝类型错误。
- 未知合成参数不会静默进入后端。

### Prompt 和缓存

- `make_prompt(audio_path=...)` 能构建 prompt。
- `make_prompt(audio_data=..., sample_rate=...)` 能构建 prompt。
- 模型不支持多参考音频时，列表输入必须明确报错。
- `serialize_prompt_asset` 和 `deserialize_prompt_asset` 往返后仍能合成。
- prompt 的 `model` 不匹配当前 runtime 时必须拒绝。

### Stream 输出

- `adapter.synthesize` 产出 `np.ndarray`，dtype 应可转为 `float32`。
- `TTSEngine.stream` 原样返回模型生成的 `float32` 音频片段，返回类型为 `numpy.ndarray`。
- 支持纯文本模式的模型应覆盖 `make_default_prompt` 和 `--skip-make-prompt` benchmark 路径。

### Bundle 和下载

- manifest 缺失时报错。
- manifest 的 `model` 与请求模型不一致时报错。
- `ckpt/` 资产缺失时报错。
- sha256 不匹配时默认报错，关闭校验时仍检查文件存在。
- ModelScope 仓库映射缺失时，`download_model` 给出清晰错误。

## 13. 推荐验证命令

先运行项目实际维护的测试：

```powershell
uv run python -m pytest tests -q
```

新模型本地 bundle 冒烟检查应使用自己的参考音频，模型私有参数按适配器能力提供：

```powershell
uv run python example/synthesize.py --model ckpts/my-tts-1b-gguf --reference-audio C:/audio/reference.wav --reference-text "参考音频的实际内容" --text "这是一段新模型适配检查。" --output smoke.wav
```

该示例逐块接收 `float32` 的 `numpy.ndarray` 并写成 WAV。直接连接字节只适用于调用方自行约定的 PCM 传输协议，不是 `stream()` 的返回类型。

资源测试示例：

```powershell
uv run python tools/benchmark_streamvox_models.py --model ckpts/my-tts-1b-gguf --model-key my-tts --display-name "MyTTS 1B GGUF" --prompt-audio C:/audio/reference.wav --prompt-text "参考音频的实际内容" --stream-kwargs-json '{"temperature":0.8,"top_p":0.9}'
```

如果模型支持纯文本默认 Prompt：

```powershell
uv run python tools/benchmark_streamvox_models.py --model ckpts/my-tts-1b-gguf --model-key my-tts-text --skip-make-prompt --stream-kwargs-json '{"mode":"text"}'
```

测试和示例运行不能代替数值对照和人工试听验收，具体边界见 [构建与验证](development.md#当前验收边界)。
## 14. 适配完成标准

一个新模型可以认为已经完整成为 StreamVox 的一部分，需要同时满足：

- 可以通过 `TTSEngine(model="<model_name>")` 或 `TTSEngine(model="<本地bundle路径>")` 初始化。
- `engine.runtime_info` 能准确展示模型名、adapter、资产数量和关键 metadata。
- `engine.make_prompt(...)` 能构建并持久化模型 prompt，或模型显式支持 `make_default_prompt`。
- `engine.stream(...)` 能用同一套公开 API 产出 `float32` PCM chunk。
- `engine.list_roles()` 和 `engine.del_roles(...)` 对该模型的 prompt cache 正常工作。
- `streamvox.download.download_model(model="<model_name>")` 能下载并校验 bundle，除非该模型明确只支持本地 bundle。
- 新模型源码和必要资源进入 wheel 与 sdist，安装后仍可导入和运行。
- 单测覆盖 adapter 契约、初始化、prompt 缓存和 stream 输出。
- benchmark 脚本无需改源码即可通过 `--model`、`--stream-kwargs-json`、`--skip-make-prompt` 覆盖新模型。

## 15. 常见设计错误

- 不要把新模型特判写进 `TTSEngine`；模型差异应在 adapter/runtime 中表达。
- 不要让 registry import 重型后端；adapter 顶层 import 应保持轻量，重型依赖延迟到 `load_runtime`。
- 不要把 prompt 的大数组写进 JSON；使用 `SerializedPromptAsset.arrays`。
- 不要默默吞掉未知模型参数；`normalize_synthesize_kwargs` 应白名单过滤并在 `synthesize` 兜底拒绝绕过归一化的参数。
- 不要把 `model_name` 写成多个不同常量；以 `spec.py` 为单一来源。
- 不要让 `shutdown()` 只能调用一次；用户异常路径和解释器退出阶段都可能重复关闭资源。
- 不要让路径音频和内存音频混用；两种来源必须二选一。
- 不要把支持多参考音频的行为做成隐式拼接；如果支持多参考，必须定义文本对齐规则。
- 不要为了“看起来模块化”把一个简单校验函数拆成 `validate_xxx -> _validate_xxx` 的空壳链路。

## 16. 最小改动清单

只支持本地 bundle、单参考音频、普通流式合成的最小接入清单：

- 新增 `streamvox/models/<model_key>/spec.py`
- 新增 `streamvox/models/<model_key>/runtime.py`
- 新增 `streamvox/models/<model_key>/adapter.py`
- 新增 `streamvox/models/<model_key>/__init__.py`
- 新增 `streamvox/models/config/<model_key>.yaml`
- 修改 `streamvox/models/builtin_models.py`
- 新增测试文件

如果还要支持自动下载：

- 修改 `streamvox/manifests.py`
- 确保远端仓库包含标准 `manifest.json`

如果还要进入正式发布：

- 确认新增源码、配置和运行资源被 wheel、sdist 收集
- 同步新增第三方材料的来源与许可声明
- 在干净 CPython 3.12 环境安装最终产物并执行运行契约检查