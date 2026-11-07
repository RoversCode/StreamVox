# StreamVox 新模型适配指南

[返回 README](../README.md) · [使用文档](usage.md) · [构建与验证](development.md) · [模型许可](model-licenses.md)

把新的文本转语音模型接入 StreamVox，需要让引擎能够找到它、加载权重、准备参考音色并返回音频。本指南从 `TTSEngine` 的真实调用顺序出发，说明每个接口的输入、输出和实现位置。

模型目录（bundle）是一套可加载的模型文件，根目录的 `manifest.json` 标明模型名和文件摘要，`ckpt/` 保存权重、分词器等运行资产。参考音色资产（Prompt）是模型从参考音频提取出的可复用资料，例如说话人向量或音频编码。它通过模型自己的格式保存到角色缓存。安装包（wheel）提供 SDK 源码、配置和原生运行库，模型目录单独分发。

下文用假设的 `my-tts-1b-gguf` 演示命名和接口。示例中的文件名、采样率与参数范围需要按目标模型确定。实际模型的编码器、生成器和音频解码器需要开发者接入，本文的接口片段分别展示通用连接部分。

GGUF 是 llama.cpp 读取的模型权重格式，ONNX 文件保存 ONNX Runtime 可执行的模型计算图。同一 TTS 模型可以把生成组件交给 llama.cpp，把参考编码或波形解码组件交给 ONNX Runtime。适配器需要为这些组件分别保存设备意图，运行时再选择实际后端。

## 职责拆分设计

先区分描述模型的代码和持有模型的对象。`ModelAdapter` 与 `ModelRuntime` 都定义在 `models/base.py`，两者是 Python `Protocol`，用来表达需要有哪些属性和方法。开发者编写的适配器类实现前者，真正持有权重和后端资源的运行时对象满足后者。现有模型通过普通类实现这些接口，无需继承协议类。[接口定义](../streamvox/models/base.py)

建议按下列职责建立文件。可选目录仅在模型确有相应逻辑时建立。

```text
streamvox/models/<model_key>/
  __init__.py            # 导出适配器，供注册入口使用
  spec.py                # 保存模型名、规格和硬件建议等稳定事实
  adapter.py             # 处理公开参数、设备意图、Prompt 和引擎调用
  runtime.py             # 加载权重、执行推理、管理会话与资源释放
  assets.py              # 可选，解析复杂的模型文件组合和资产元数据
  prompt.py              # 可选，定义 Prompt 数据与序列化规则
  text/                  # 可选，维护模型专属的文本清理与分段规则
  schema/                # 可选，维护多个模块共用的结构和消息协议
  worker/                # 可选，放置能被子进程独立导入的工作代码

streamvox/models/config/<model_key>.yaml  # 配置默认值与文件名
streamvox/models/builtin_models.py        # 登记内置模型
streamvox/manifests.py                    # 映射自动下载仓库
pyproject.toml                            # 声明依赖和安装包资源
MANIFEST.in                               # 声明源码归档内容
tests/test_<model_key>_*.py               # 检查模型接入行为
```

`TTSEngine` 负责选择模型、取得模型目录、管理角色缓存和返回音频。`ModelRegistry` 保存模型名到适配器实例的映射。适配器把通用输入转成目标模型的语义，运行时负责具体后端计算。`PromptCache` 通过适配器把私有 Prompt 转成统一的 `SerializedPromptAsset` 后存盘，所以缓存层无需了解说话人向量或模型缓存的内部结构。

同一系列多个尺寸可以共享适配器基类和运行时，由 `spec.py` 区分规格。复杂的资产解析、文本规则和子进程协议拆到各自模块，简单的参数校验与执行代码可放在同一方法中。保持注册入口可轻量导入，重型后端延迟到 `load_runtime()` 加载。

## 总体接入流程

1. 确定模型实际需要的输入和资产。列明参考音频数量、参考文本要求、输出采样率、支持的后端，以及可否在没有参考音频时生成。
2. 选定稳定的 `model_name`，例如 `my-tts-1b-gguf`。注册表、模型清单、下载映射、默认模型目录与角色缓存目录都使用这个名字。
3. 编写规格和配置，再准备含完整运行资产的模型目录。离线转换工具与 SDK 运行环境分别准备依赖。
4. 实现运行时，先校验模型目录和配置，再加载后端，提供运行信息并管理资源释放。
5. 实现适配器，把引擎请求传给运行时，完成设备计划、Prompt 编码、序列化和合成参数校验。
6. 先用独立 `ModelRegistry` 和本地模型目录，从 `TTSEngine` 入口验证接入。通过后再加入内置注册入口。
7. 需要按模型名下载时，登记 ModelScope 映射并验证远端清单。需要发行时，检查源码、配置、依赖和许可进入相应归档。
8. 分别完成接口测试、真实合成、数值对照、试听和资源测量，保存各自的输入与结论范围。

## 统一入口如何调用新模型

初始化先解决“加载哪个模型”和“文件在哪里”，再确定各后端的执行位置。

```text
TTSEngine(model=..., device=..., llama_backend=..., onnx_provider=...)
  -> model 看起来像路径时，从目录 manifest.json 读取 model 字段
  -> 按模型名初始化时，确认注册表支持，再查默认模型缓存
  -> 默认模型目录没有 manifest.json 时，调用 download_model()
  -> 从注册表取得对应适配器，并创建该模型的 PromptCache
  -> 读取适配器的硬件建议，输出资源提示
  -> adapter.resolve_execution_plan(device, llama_backend=...)
  -> 将显式 onnx_provider 与 llama_backend 写入相应执行目标
  -> adapter.load_runtime(RuntimeRequest(...))
  -> 运行时提供 warmup_for_init() 时执行初始化预热
  -> 保存 runtime_info，并核对显式请求的实际后端
```

`model` 公开参数使用字符串。本地路径中的模型名也必须已经注册。默认缓存已有清单时，引擎先核对清单模型名，后续资产存在性与摘要校验由运行时完成。默认数据根目录在 Windows 为 `%APPDATA%/StreamVox`，其他系统为 `~/.streamvox`，`STREAMVOX_HOME` 可覆盖它。模型位于根目录下的 `ckpts/<model_name>/`，角色缓存位于 `prompts/<model_name>/`。[引擎入口](../streamvox/engine.py) · [目录解析](../streamvox/paths.py)

合成时，引擎先取得 Prompt，然后才进入参数归一化和推理。

```text
TTSEngine.stream(text, role_name=..., **model_kwargs)
  -> 直接传 BasePromptAsset 时，检查其模型名并复用
  -> 传角色名时，从该模型的角色缓存恢复 Prompt
  -> 尚无 Prompt 且提供参考音频时，调用 adapter.make_prompt()
  -> role_name=None 且未提供参考音频时，尝试 adapter.make_default_prompt()
  -> adapter.normalize_synthesize_kwargs(...) 校验模型参数
  -> adapter.synthesize(runtime, text, prompt, track_performance=..., **参数)
  -> 将每块音频转成 numpy.float32，按原顺序返回
  -> 消费结束、异常或关闭生成器时，关闭底层合成生成器
```

一块音频（chunk）是生成器一次返回的波形数组。它的大小由模型生成和解码策略决定。引擎将采样值转换为 `float32`，即 32 位浮点数，保留块的内容、形状和顺序，采样率来自 `runtime.sample_rate`。适配器与运行时须共同保证自己的音频布局和连续性。当前内置 TTS 模型使用单声道波形，调用方可顺序播放，或连接各块后写 WAV。[流式契约测试](../tests/test_open_source_runtime.py)

新增模型的差异应先在适配器和运行时表达。涉及所有模型共同需要的能力时，再评估是否扩展引擎公开接口。

## 1. 定义模型规格

`spec.py` 保存稳定事实，供适配器、运行时和日志引用。下面示例集中保存模型名和适配器名，避免同一名字在多个文件中逐渐分叉。

```python
# filepath: streamvox/models/my_tts/spec.py
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MyTTSModelSpec:
    """保存模型接入时共享的固定标识。

    参数:
        model_name: 注册表、清单与缓存共同使用的模型名。
        adapter_name: 运行信息和清单使用的适配器名。
    """

    model_name: str
    adapter_name: str


MY_TTS_1B_SPEC = MyTTSModelSpec(
    model_name="my-tts-1b-gguf",
    adapter_name="MyTTS1BGGUFAdapter",
)
```

如果多个规格共享运行时，可在规格中增加权重子目录等差异。硬件建议通过 `get_hardware_profile()` 返回 `ModelFootprintProfile`，包括档位标识、参数量、量化描述、最低和推荐内存与显存、测量说明。缺少可靠资源数据时可返回 `None`。引擎资源体检只输出建议，后端能否运行仍由实际加载决定。[Qwen3 规格示例](../streamvox/models/qwen3/spec.py) · [硬件结构](../streamvox/hardware.py)

## 2. 编写模型配置

配置文件保存默认值，运行时负责读取并传给真正消费它们的后端。仅增加 YAML 字段还不会改变推理。以下数值为假设模型的示例配置。

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

`sample_rate` 表示每秒输出多少个音频采样点。`n_ctx` 表示模型单次推理可容纳的上下文位置数，包含哪些参考、文本或生成位置由模型实现决定。上下文越长，通常需要更大的模型缓存。`TTSEngine` 对显式 `n_ctx` 只要求为整数并排除布尔值，边界由各模型运行时判断。未传时，`RuntimeRequest.n_ctx` 为 `None`，由模型使用自己的默认值。

模型应明确超界时的行为。Qwen3 按配置最小值和最大值裁剪并记录实际值，Breeze 对非法值和请求超出容量报错，VoxCPM2 使用自己的配置与加载路径。新模型需要根据后端能力选择并测试规则，避免把一种模型的上下文约束当成全局约束。[Qwen3 上下文处理](../streamvox/models/qwen3/runtime.py) · [Breeze 上下文处理](../streamvox/models/breeze/runtime.py)

`max_length` 和 `min_length` 控制文本分段与短段合并，它们与模型缓存容量分别处理。`llama_n_gpu_layers` 描述模型给 llama.cpp 的层卸载配置，只有运行时读取后才生效。`decoder_start_method` 在使用独立解码进程时决定进程启动方式，Windows 工作进程应使用可独立导入的模块。`paths` 保存模型目录内的相对文件名，由运行时与 `model_dir / "ckpt"` 组合。

## 3. 准备模型 bundle 和 manifest

一个标准模型目录可以包含如下文件。

```text
ckpts/my-tts-1b-gguf/
  manifest.json
  LICENSE
  NOTICE
  ckpt/
    my_tts.q8_0.gguf
    my_tts_decoder.fp16.onnx
    tokenizer.json
```

清单里的 `model` 选择适配器，`assets` 列出模型文件与各自的 SHA256。SHA256 用于核对文件内容是否与清单一致。下面的摘要是占位内容，实际应由工具读取完整文件生成。

```json
{
  "model": "my-tts-1b-gguf",
  "sources": ["local"],
  "adapter": "MyTTS1BGGUFAdapter",
  "assets": [
    {"path": "ckpt/my_tts.q8_0.gguf", "sha256": "<64位sha256>"},
    {"path": "ckpt/my_tts_decoder.fp16.onnx", "sha256": "<64位sha256>"},
    {"path": "ckpt/tokenizer.json", "sha256": "<64位sha256>"}
  ]
}
```

准备好文件后，在仓库根目录生成本地清单。

```powershell
uv run --no-sync python -m streamvox.model_bundle --model my-tts-1b-gguf --root-dir ckpts/my-tts-1b-gguf --asset ckpts/my-tts-1b-gguf/ckpt/my_tts.q8_0.gguf --asset ckpts/my-tts-1b-gguf/ckpt/my_tts_decoder.fp16.onnx --asset ckpts/my-tts-1b-gguf/ckpt/tokenizer.json --source local --adapter MyTTS1BGGUFAdapter
```

运行时应在加载后端前调用 `validate_local_bundle(model_dir, model_name, verify_model_sha256=...)`。该函数核对清单中的模型名，并检查清单列出的 `ckpt/` 文件。默认还计算文件 SHA256，关闭摘要校验后仍检查文件存在。

当前校验只处理清单中的 `ckpt/` 路径，运行时还须检查自己必需的文件是否完整列出、格式和维度是否可用。ONNX 外部参数、嵌入表和分词器等实际读取的文件也应列入清单。根目录的 `README.md`、`LICENSE`、`NOTICE` 和 `configuration.json` 属于仓库资料，启动校验会跳过这些文件，模型分发仍须保留适用许可。[模型目录实现](../streamvox/model_bundle.py)

清单的 `adapter` 字段用于描述来源，当前引擎以 `model` 查注册表。`sources=["local"]` 可以被 ModelScope 下载入口接受。最新快照下载会生成本地清单并补充下载来源，固定版本下载保留原始清单，具体流程见第 10 节。

## 4. 实现 Runtime

运行时在引擎初始化期间创建，持有模型权重和长生命周期资源。`ModelRuntime` 协议要求以下成员。

| 成员 | 运行时需要提供的内容 |
| --- | --- |
| `model_name: str` | 当前加载的注册模型名 |
| `sample_rate: int` | 合成输出采样率 |
| `runtime_info: RuntimeInfo` | 模型、适配器、模型目录加载状态、资产数量与字符串元数据 |
| `shutdown() -> None` | 释放原生对象、会话、线程和工作进程 |

`RuntimeInfo` 的构造字段为 `model`、`adapter`、`bundle_loaded`、`asset_count` 和 `metadata`。分别填写模型名、适配器名、模型目录加载状态、已校验运行资产数量，以及需要公开的字符串配置资料。

`RuntimeRequest` 是适配器收到的初始化请求，字段需要有明确去向。

| 字段 | 实现时的处理 |
| --- | --- |
| `model_dir: Path` | 作为模型目录根路径，读取清单与 `ckpt/` 资产 |
| `device: str` | 保留原始设备意图，用于诊断或模型自己的解析 |
| `n_ctx: int \| None` | 用显式值覆盖默认值，并按本模型规则校验 |
| `verify_model_sha256: bool` | 传给 `validate_local_bundle` |
| `execution_plan: ExecutionPlan \| None` | 分别解析本模型实际使用的后端类别 |
| `text_preprocess_config: TextPreprocessConfig \| None` | 传给本模型的文本处理路径 |

适配器的 `load_runtime()` 应把这些输入传给对应运行时。下面以已有 Qwen3 实现展示完整的请求传递，方法位于适配器类中。

```python
# filepath: streamvox/models/qwen3/adapter.py
from pathlib import Path

from ..base import RuntimeRequest


def load_runtime(self, request: RuntimeRequest) -> "Qwen3Runtime":
    """按引擎请求创建当前规格的运行时。

    参数:
        request: 模型目录、设备计划、上下文与文本配置。
    返回:
        已加载的 Qwen3 运行时。
    """
    from .runtime import Qwen3Runtime

    return Qwen3Runtime(
        model_dir=Path(request.model_dir),
        model_spec=self.spec,
        device=request.device,
        n_ctx=request.n_ctx,
        verify_model_sha256=request.verify_model_sha256,
        execution_plan=request.execution_plan,
        text_preprocess_config=request.text_preprocess_config,
    )
```

运行时初始化按轻量输入到昂贵资源的顺序进行。先读配置、检查模型目录和执行计划，再加载权重、创建后端会话，最后填充 `RuntimeInfo`。初始化失败时，需要释放此前已创建的资源。`RuntimeInfo.metadata` 的键和值均使用字符串，适合记录模型目录、有效 `n_ctx`、后端和关键资产版本。

`create_session()` 是多种内置模型分离单次请求状态的方法，协议只要求上表的公共成员。Breeze 直接在运行时中串行执行请求。选择会话结构时，权重可长期复用，采样状态、生成缓存、解码历史和取消状态应按请求隔离。多个会话共享可变后端时，还需确定串行或加锁规则。[Qwen3 运行时](../streamvox/models/qwen3/runtime.py) · [Breeze 运行时](../streamvox/models/breeze/runtime.py)

### 初始化预热与关闭

运行时可以实现内部约定 `warmup_for_init() -> None`。引擎会在 `load_runtime()` 成功后检查它是否可调用，存在时立即执行。预热把常用的会话创建、工作进程启动或最小推理移到初始化阶段。失败会传播给调用方，初始化随之失败。

预热应使用最小合法输入，并在完成或失败后清理请求缓存。实现需要保证重复调用不会重复污染状态。Qwen3、VoxCPM2、MOSS-TTS-Nano 和 Breeze 的具体预热范围不同，可从最接近目标模型的运行时查看。

预热异常发生在运行时已经加载之后，此时引擎构造尚未返回，调用方还拿不到引擎对象。预热实现需要在失败路径调用自身的 `shutdown()`，释放已经加载的权重、会话、线程和工作进程，再继续抛出原始异常。当前引擎将预热异常直接向上传播，这条清理路径由运行时负责。

`shutdown()` 应允许重复调用，并清理部分初始化状态。用户可能在失败处理或提前结束请求后调用它。合成生成器自身也需要在 `finally` 中等待在途任务、关闭请求资源和清理缓存，让引擎的生成器关闭操作能到达实际后端。

### 实际后端的可见信息

`ExecutionPlan` 保存请求，实际后端由运行时加载结果确定。引擎的 `actual_backends` 从运行时或其 `resolved_device` 读取 `gguf_backend`，再读取 `onnx_backend` 或 `decoder_backend`，并遍历模型对象中可发现的 ONNX 会话以列出提供程序。DirectML 的内部 `dml` 名称会转换为公开 `directml`。

新运行时应按现有命名公开实际后端，混合执行的 CPU 辅助会话也应准确显示。显式选择的后端一旦与已报告的实际后端不符，引擎会释放运行时并报错。目标机器能否加载该后端仍须在真实初始化与推理中验证。[后端核对实现](../streamvox/engine.py) · [显式后端测试](../tests/test_explicit_backend.py)

## 5. 实现 Adapter

适配器是引擎认识模型的入口。需要提供下列属性和方法，名称及参数应与当前引擎一致。

| 接口 | 引擎如何使用 |
| --- | --- |
| `model_name: str` | 注册、校验模型归属和隔离缓存 |
| `load_runtime(request)` | 加载模型，返回满足 `ModelRuntime` 的对象 |
| `get_text_preprocess_config()` | 返回 `TextPreprocessConfig` 或 `None` |
| `get_hardware_profile()` | 返回 `ModelFootprintProfile` 或 `None` |
| `resolve_execution_plan(device, llama_backend=None)` | 返回本模型各后端类别的设备意图 |
| `make_prompt(runtime, role_name, sample_rate, audio_path=None, audio_data=None, prompt_text=None)` | 从单条或多条参考输入构建模型自己的 `BasePromptAsset` 子类 |
| `serialize_prompt_asset(asset)` | 返回缓存可保存的 `SerializedPromptAsset` |
| `deserialize_prompt_asset(payload)` | 恢复可直接合成的私有 Prompt |
| `normalize_synthesize_kwargs(**model_kwargs)` | 校验并返回本模型接受的合成参数字典 |
| `synthesize(runtime, text, prompt, stream=True, **model_kwargs)` | 返回逐块生成音频的可迭代对象 |

协议还声明 `validate_make_prompt_args(...)`，其关键字参数为 `role_name`、`audio_path`、`audio_data`、`sample_rate` 和 `prompt_text`。引擎会执行自己的通用校验，当前没有单独调用这个协议方法。部分适配器提供该方法并从 `make_prompt()` 调用，其他适配器直接在 `make_prompt()` 内校验。新实现若需完整满足静态协议，可提供该方法并让模型入口复用它，确保调用路径实际执行了校验。

### 把设备意图交给各后端

`device` 提供默认位置和设备编号。`auto` 允许运行时选择，`cpu` 请求处理器，`gpu` 等价于首块显卡，`gpu:N` 指定显卡编号。执行计划用 `llama`、`onnx` 等类别区分模型中的后端组件，`ExecutionTarget` 保存 `mode`、`device_index` 和可选 `backend_preference`。

下面是同时使用 llama.cpp 和 ONNX 的模型可采用的解析片段，放入适配器类。纯 ONNX 模型只声明 `onnx` 目标。

```python
# filepath: streamvox/models/my_tts/adapter.py
from ..base import ExecutionPlan, ExecutionTarget
from ..llama_backend import normalize_explicit_llama_backend


def resolve_execution_plan(
    self,
    device: str,
    llama_backend: str | None = None,
) -> ExecutionPlan:
    """为生成模型与音频组件保存设备请求。

    参数:
        device: 自动、处理器或显卡设备字符串。
        llama_backend: 显式 llama.cpp 后端，空值保留自动选择。
    返回:
        分别包含 llama 与 ONNX 目标的执行计划。
    """
    backend = normalize_explicit_llama_backend(llama_backend)
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    normalized = device.strip().lower()
    index = None
    if normalized in {"auto", "cpu"}:
        mode = normalized
    elif normalized == "gpu":
        mode, index = "gpu", 0
    elif normalized.startswith("gpu:") and normalized[4:].isascii() and normalized[4:].isdigit():
        mode, index = "gpu", int(normalized[4:])
    else:
        raise ValueError(f"unsupported device value: {device}")
    return ExecutionPlan(
        raw_device=device,
        targets={
            "llama": ExecutionTarget(
                family="llama", mode=mode, device_index=index,
                backend_preference=backend,
            ),
            "onnx": ExecutionTarget(
                family="onnx", mode=mode, device_index=index,
            ),
        },
    )
```

引擎随后将显式 `onnx_provider` 写入 ONNX 目标，将显式 `llama_backend` 对应的 CPU/GPU 模式写入 llama 目标。运行时要分别尊重这两条链路的选择，例如 `device="gpu:0", llama_backend="cpu", onnx_provider="cuda"` 表示 CPU 生成与 CUDA ONNX 的组合。`llama_backend` 的公开取值为 `cpu`、`cuda`、`vulkan`、`metal`，模型和平台的实际支持范围还需由后端加载路径判断。ONNX 的公开取值为 `cpu`、`cuda`、`directml`，DirectML 用于 Windows。[执行目标定义](../streamvox/models/base.py) · [后端选择工具](../streamvox/models/llama_backend.py)

### 先校验模型参数，再进入推理

引擎调用 `normalize_synthesize_kwargs()` 时，会传入公开的 `max_length`、`min_length`、`remove_meaningless_chars` 及用户的模型私有参数。该方法只接收这些关键字参数。`track_performance` 随后直接传给 `synthesize()`，因此合成方法应接受它。用户经私有参数传入的 `stream` 也需要由适配器解释。

为新模型明确参数白名单、类型、数值范围和未知参数行为。现有适配器的未知参数策略存在差异，新实现可以在归一化阶段直接拒绝未知字段，帮助调用方定位拼写错误。布尔值在 Python 中属于整数子类，整数参数应显式排除它。浮点采样参数还应检查有限值和模型允许的范围。参数默认值由配置或运行时统一提供，用户显式值在校验后覆盖默认值。

文本分段参数至少需要确定 `max_length > 0`、`min_length >= 0`，同时给出两者时核对 `min_length <= max_length`。对不做分段的模型，应明确接受或拒绝这些字段的规则，避免参数看似生效却改变不了文本路径。[Qwen3 参数示例](../streamvox/models/qwen3/adapter.py) · [Breeze 参数示例](../streamvox/models/breeze/adapter.py)

### 构建单参考 Prompt

引擎先检查音频来源二选一、基础类型和列表长度。文件路径的采样率可省略，传给适配器时会使用运行时采样率作为默认值。内存波形必须同时提供采样率。模型还须确定参考文本是否必填、允许的音频长度以及多参考规则。

下面演示单参考模型的音频读取与公共资料保存。假设新运行时提供 `encode_reference(audio)` 返回说话人向量，开发者须把这一行接到目标模型的真实编码器。`MyTTSPromptAsset` 的结构在第 7 节定义。

```python
# filepath: streamvox/models/my_tts/adapter.py
import librosa
import numpy as np

from ...prompt_audio import build_prompt_reference, normalize_prompt_audio
from ..base import PromptCacheRecord
from .prompt import MyTTSPromptAsset


def make_prompt(
    self,
    runtime,
    role_name: str,
    sample_rate: int | list[int],
    audio_path: str | list[str] | None = None,
    audio_data: np.ndarray | list[np.ndarray] | None = None,
    prompt_text: str | list[str] | None = None,
) -> MyTTSPromptAsset:
    """编码单条参考音频并保存可查询的参考资料。

    参数:
        runtime: 已加载真实编码器的模型运行时。
        role_name: 当前参考音色的角色名。
        sample_rate: 内存音频采样率，路径输入使用文件信息。
        audio_path: 单条参考音频路径，与内存音频二选一。
        audio_data: 单条内存波形，与路径二选一。
        prompt_text: 参考转录，本示例允许缺省。
    返回:
        含模型私有向量和公共参考资料的 Prompt。
    """
    if any(isinstance(value, list) for value in (sample_rate, audio_path, audio_data, prompt_text)):
        raise ValueError("my-tts requires exactly one reference audio")
    if prompt_text is not None and not isinstance(prompt_text, str):
        raise TypeError("prompt_text must be a string or None")
    if (audio_path is None) == (audio_data is None):
        raise ValueError("provide exactly one reference audio source")

    if audio_path is not None:
        audio, input_rate = librosa.load(audio_path, sr=None, mono=True)
    else:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        audio, input_rate = normalize_prompt_audio(audio_data), sample_rate
    if input_rate != runtime.sample_rate:
        audio = librosa.resample(audio, orig_sr=input_rate, target_sr=runtime.sample_rate)
    audio = normalize_prompt_audio(audio)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("reference audio must be non-empty and finite")

    # 编码器和公共资料使用同一份波形，查询结果才能对应实际输入。
    embedding = runtime.encode_reference(audio)
    prompt = MyTTSPromptAsset(
        role_name=role_name,
        model=self.model_name,
        sample_rate=runtime.sample_rate,
        prompt_text=prompt_text,
        speaker_embedding=np.asarray(embedding, dtype=np.float32),
    )
    prompt.prompt_cache_record = PromptCacheRecord(
        prompt_text=[prompt_text],
        references=(build_prompt_reference(
            audio=audio, sample_rate=runtime.sample_rate, index=0,
        ),),
    )
    return prompt
```

目标模型若分别使用参考编码采样率和输出采样率，应按自己的编码入口处理，公共参考资料记录实际消费的波形与相应哈希语义。多参考模型需要逐条定义音频、采样率、文本与说话人的对应关系。引擎允许采样率标量及列表等通用形态，模型是否接受由适配器判断。可参考 S2-Pro 的多参考路径，Qwen3、VoxCPM2、MOSS 与 Breeze 的参考能力按各模型实现使用。[S2-Pro 适配器](../streamvox/models/s2_pro/adapter.py) · [公共参考音频工具](../streamvox/prompt_audio.py)

### 合成方法连接运行时

适配器在 `synthesize()` 中检查 Prompt 类型与模型归属，再进入模型运行时。使用会话的模型通常先 `runtime.create_session()`，再把文本、Prompt、分段选项和已归一化的私有参数交给 `session.synthesize()`。Breeze 则调用运行时自己的生成方法。对引擎暴露的结果始终是逐块音频迭代器。

后端失败应传播原始原因，请求资源在生成器清理路径释放。引擎会关闭底层生成器，`yield from` 可以把关闭传递给子生成器。调用方提前停止消费时需关闭 `engine.stream()` 返回的生成器，再关闭引擎。资源清理与音频输出应同时覆盖正常完成、取消和异常三条路径。

## 6. 支持无 Prompt 纯文本模式

模型能在没有参考音频时生成，可额外实现 `make_default_prompt(runtime) -> BasePromptAsset`。这属于引擎通过属性检查发现的可选能力。VoxCPM2 和 Breeze 提供了此类默认 Prompt，默认资产描述该模型的文本生成初始状态。

只有 `role_name=None`、尚无 Prompt 且未提供参考音频时，引擎才会调用该方法。传入未命中的角色名仍会报角色不存在。默认 Prompt 与流内即时创建的 Prompt 都不会自动写入角色缓存。模型的 `synthesize()` 应根据该资产或已校验的 `mode` 等参数选择合成路径。[VoxCPM2 默认 Prompt](../streamvox/models/voxcpm2/adapter.py) · [Breeze 默认 Prompt](../streamvox/models/breeze/adapter.py)

## 7. Prompt 资产序列化规则

序列化把内存中的模型私有 Prompt 拆成普通数据，使它可以在退出 Python 后保存、下次启动后恢复。`engine.make_prompt(persist=True)` 调用 `PromptCache.store()`，缓存层再调用适配器的序列化方法。读取角色时由缓存层恢复 `SerializedPromptAsset`，再让适配器恢复私有 Prompt。

`SerializedPromptAsset` 共有四个公共身份字段和两组载荷。`role_name`、`model`、`sample_rate` 和 `metadata` 描述角色归属与版本等信息，`metadata` 使用字符串字典。`arrays` 保存 NumPy 数组，例如说话人向量、音频编码和可恢复的解码状态。`json_data` 保存可 JSON 编码的文本、位置和轻量配置。

下面为假设模型定义完整的最小序列化往返。说话人向量是本示例唯一必需的模型私有数组。

```python
# filepath: streamvox/models/my_tts/prompt.py
from dataclasses import dataclass

import numpy as np

from ..base import BasePromptAsset, SerializedPromptAsset
from .spec import MY_TTS_1B_SPEC


@dataclass(slots=True)
class MyTTSPromptAsset(BasePromptAsset):
    """保存单参考模型后续合成需要的资料。

    参数:
        prompt_text: 参考音频的转录，可以为空。
        speaker_embedding: 编码器生成的说话人向量。
    """

    prompt_text: str | None = None
    speaker_embedding: np.ndarray | None = None


def serialize_prompt_asset(asset: BasePromptAsset) -> SerializedPromptAsset:
    """把模型 Prompt 转成缓存可保存的数据。

    参数:
        asset: 当前模型构建的 Prompt。
    返回:
        含说话人向量和参考转录的序列化载荷。
    """
    if not isinstance(asset, MyTTSPromptAsset) or asset.model != MY_TTS_1B_SPEC.model_name:
        raise TypeError("my-tts requires its own prompt asset")
    if asset.speaker_embedding is None:
        raise ValueError("speaker_embedding is required")
    return SerializedPromptAsset(
        role_name=asset.role_name,
        model=asset.model,
        sample_rate=asset.sample_rate,
        metadata=dict(asset.metadata),
        arrays={"speaker_embedding": np.asarray(asset.speaker_embedding, dtype=np.float32)},
        json_data={"prompt_text": asset.prompt_text},
    )


def deserialize_prompt_asset(payload: SerializedPromptAsset) -> MyTTSPromptAsset:
    """从保存的数据恢复可合成的模型 Prompt。

    参数:
        payload: 缓存读取的统一载荷。
    返回:
        当前模型可直接使用的 Prompt。
    """
    if payload.model != MY_TTS_1B_SPEC.model_name:
        raise ValueError("prompt model mismatch")
    if "speaker_embedding" not in payload.arrays:
        raise ValueError("cached prompt is missing speaker_embedding")
    prompt_text = payload.json_data.get("prompt_text")
    if prompt_text is not None and not isinstance(prompt_text, str):
        raise TypeError("cached prompt_text must be a string or None")
    return MyTTSPromptAsset(
        role_name=payload.role_name,
        model=payload.model,
        sample_rate=payload.sample_rate,
        metadata=dict(payload.metadata),
        prompt_text=prompt_text,
        speaker_embedding=np.asarray(payload.arrays["speaker_embedding"], dtype=np.float32),
    )
```

适配器应把这两个函数作为自己的 `serialize_prompt_asset()` 和 `deserialize_prompt_asset()` 实现。可以像 Qwen3 一样在方法内延迟导入，也可以直接在适配器类中实现，按模块依赖与逻辑规模选择。实际模型还需要校验向量形状、数据类型和版本，恢复后的资产应拥有合成所需的全部资料。原生句柄、会话对象和进程内指针需要在恢复时重建。

缓存保存为 `prompts/<model_name>/<role_name>.npz`，路径中的模型名和角色名会把 `/` 与空格替换为 `_`。模型私有数组使用 `payload__` 前缀，读取时 `allow_pickle=False`。应使用数值数组，较大的数组通过 `arrays` 保存。`PromptCacheRecord` 则单独保存公开参考资料，`prompt_text` 列表与 `references` 数量一致，文本不适用时用 `None` 占位。`PromptReference` 包含音频数组、内容哈希和序号，哈希还包含采样率。

`engine.get_prompt_data(role_name)` 同时返回模型私有 Prompt 和公开参考资料。旧缓存缺少公开资料时，查询结果的参考列表可以为空，私有资产能否继续合成由其自身数据决定。第 5 节在编码后写入 `prompt_cache_record`，使新缓存能够完整支持这一查询。[缓存实现](../streamvox/prompt_cache.py) · [缓存测试](../tests/test_prompt_data_cache.py)

## 8. 文本预处理接入

文本预处理负责把输入变成模型将要编码的文本片段。通用 `TextPreprocessConfig` 可声明需要原样保护的特殊 token、包裹规则以及句级和子句级分隔符。特殊 token 是模型用来表达边界或控制信息的文本标记，清理时误删会改变模型输入。配置本身只保存规则，运行时或会话须实际调用相应处理函数。

Qwen3 在适配器中返回轻量配置，S2-Pro 和 VoxCPM2 把专属规则放在 `text/` 中。Breeze 保留上游文本与事件标记，不额外清理或自动分段。选择实现时先检查上游模型对原始文本的要求，再决定是否复用通用工具。[通用文本工具](../streamvox/models/text_processing.py) · [模型专属文本示例](../streamvox/models/voxcpm2/text/preprocess_text.py)

`language`、`mode`、`control_text`、`instruction` 等字段由各模型定义，应在适配器校验后传给文本或推理入口。分段默认值来自模型配置，`engine.stream(max_length=..., min_length=...)` 的显式值按模型规则覆盖。`remove_meaningless_chars` 默认为 `True`，模型应说明其允许字符规则或保留原始文本的处理方式。

## 9. 注册默认模型

开发阶段可先使用独立注册表，让本地模型从完整引擎入口运行。下面代码的前提是模型包已经导出真实实现的适配器，且本地模型目录已准备完成。建议将脚本保存为 `example/my_tts_smoke.py`。入口保护让 Windows `spawn` 子进程导入脚本时只取得定义，避免再次初始化引擎和发起合成。

```python
# filepath: example/my_tts_smoke.py
from streamvox import TTSEngine
from streamvox.models.my_tts import MyTTS1BGGUFAdapter
from streamvox.models.registry import ModelRegistry


def main() -> None:
    """从引擎入口检查新模型，并在请求结束后释放资源。

    Returns:
        无返回值，逐块打印音频形状和数据类型。
    """
    registry = ModelRegistry()
    registry.register(MyTTS1BGGUFAdapter())
    engine = TTSEngine(model="ckpts/my-tts-1b-gguf", device="cpu", registry=registry)
    try:
        prompt = engine.make_prompt(
            role_name="my_tts_smoke",
            audio_path="reference.wav",
            prompt_text="参考音频的实际内容",
            persist=False,
        )
        for chunk in engine.stream("这是一段适配检查。", role_name=prompt):
            print(chunk.shape, chunk.dtype)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
```

`ModelRegistry.register()` 以 `model_name` 为键保存适配器，同名注册会替换原实例。内置入口 `builtin_models.py` 自行检查模型名，避免重复登记。模型包先导出适配器。

```python
# filepath: streamvox/models/my_tts/__init__.py
from .adapter import MyTTS1BGGUFAdapter

__all__ = ["MyTTS1BGGUFAdapter"]
```

再在 `streamvox/models/builtin_models.py` 导入 `MyTTS1BGGUFAdapter`，将 `MyTTS1BGGUFAdapter()` 加入现有适配器元组。保留其他模型与既有幂等判断。`streamvox/models/__init__.py` 导入时会调用统一注册函数，随后应能验证新模型名。

```python
from streamvox.models import default_registry

assert "my-tts-1b-gguf" in default_registry.list_models()
```

## 10. 自动下载接入

本地模型运行通过后，可以在 `DEFAULT_MODELSCOPE_REPOS` 现有字典中加入模型名到仓库的映射。下面演示新增项，`YourOrg` 需要替换为实际仓库所有者。

```python
# filepath: streamvox/manifests.py
DEFAULT_MODELSCOPE_REPOS["my-tts-1b-gguf"] = "YourOrg/my-tts-1b-gguf"
```

远端仓库根目录须包含 `manifest.json`，并保留模型许可与归属材料。清单中的 `model` 与注册名一致，运行资产和摘要完整。`sources` 可声明 `local`，下载器接受其本地模型目录语义。

`download_model()` 当前有最新快照和固定版本两条路径。

1. 仅给模型名时，下载器按映射调用 ModelScope `snapshot_download`，默认缓存位于目标根目录下的 `.streamvox-download-cache`。引擎自动下载走这条路径。
2. 同时提供 `revision` 与 `expected_manifest_sha256` 时，下载器按完整远端 Git 提交和原始清单摘要取得固定快照。两参数必须成对出现，适合需要固定模型版本的产品调用方。
3. 两条路径都先写入目标根目录下的独立隐藏目录，核对清单模型名、来源与运行资产 SHA256。最新快照路径重写本地清单并补充 `local` 和 `modelscope` 来源，固定版本路径保留原始清单。
4. 全部校验通过后才切换到 `<target>/<model_name>`。取消或失败会清理待发布目录，下载缓存保留供重试。已有可用目标在切换前保持可用，切换失败会尝试恢复原目录。

下载 API 还支持 `progress_callback`、`cancel_event`、`cache_dir`。初始化引擎时当前仅传模型名、下载源与默认目标目录，产品若需要版本固定、进度或取消控制，应先调用下载 API，再把完成的本地目录交给引擎。[下载实现](../streamvox/download.py) · [固定版本下载](../streamvox/pinned_download.py)

固定版本路径还会下载该提交中名称被识别的法律文件，包括 `LICENSE`、`MODEL_LICENSE`、`NOTICE`、`COPYING` 及代码列出的扩展名形式，并核对远端提交记录的摘要。模型采用其他法律文件名时，发布前应检查它是否包含在下载结果中，并同步所需的识别规则与测试。

开发者可以先保持只支持本地目录，在准备远端仓库后再登记下载映射。SDK 注册模型和配置远端下载是两个独立步骤。

## 11. 打包和发布接入

2.1.2 的 wheel 目标为 Windows x64 / CPython 3.12。`pyproject.toml` 自动发现 `streamvox` 下的 Python 模块，并收集 `models/config/*.yaml`、`*.yml` 与指定原生资源。`MANIFEST.in` 还控制 sdist 中的源码、测试、工具和文档。新增文件类型时，分别检查两个归档实际收集的结果。

新模型的运行依赖进入 SDK 依赖声明，导出与参考对照依赖放在独立工具环境。SDK 运行时维持无需 Torch 的依赖边界，模型目录与权重独立分发。新增第三方代码或派生实现时，保存实际来源提交、原始版权和许可，检查模型权重的使用限制。[项目依赖与资源](../pyproject.toml) · [源码归档规则](../MANIFEST.in) · [模型许可](model-licenses.md)

公开 Git 提交省略 Windows DLL，发行工具从固定公开归档恢复三种后端。源码提交、SDK 版本、原生库来源和最终附件通过发行清单连接。构建要求当前 `HEAD` 与声明的完整提交一致，已跟踪文件无未提交修改，安装包从该提交的独立暂存目录构建。完整构建、汇总、重建和干净环境验收步骤见开发文档。[发行构建](development.md#发行构建)

## 12. 测试清单

先用轻量后端替身检查引擎与适配器，再用真实模型检查输入输出。替身用于确定失败发生在哪一层，真实合成用于确认权重和后端实际可用。

### Registry 和初始化

- 注册表能查到新模型，本地目录的清单能选到对应适配器。
- 默认模型目录有清单时复用本地目录，缺失时进入下载路径，清单模型名不符时提前报错。
- `RuntimeRequest` 的 `n_ctx`、校验开关、执行计划与文本配置传递完整。
- 初始化中途失败能释放已创建资源，预热失败传播给调用方，`shutdown()` 可以重复执行。

### Adapter 参数契约

- `auto`、`cpu`、`gpu`、`gpu:1` 和非法设备字符串有确定行为，`llama_backend` 关键字可被引擎调用。
- 显式 llama 与 ONNX 选择相互独立，实际后端信息准确，可用性与回退规则覆盖目标平台。
- 私有参数、分段参数、布尔值、非有限浮点值与范围错误按模型规则处理，未知字段行为有测试。

### Prompt 和缓存

- 路径音频与内存音频均能编码，来源冲突、缺少采样率和参考文本错误有明确结果。
- 多参考能力与列表对齐规则一致，单参考模型拒绝列表输入。
- 私有 Prompt 序列化往返后仍可合成，模型名、版本和必要数组缺失时能定位问题。
- 新缓存可通过 `get_prompt_data()` 取得真实参考音频和文本，旧缓存缺少公开资料时保持其声明的读取行为。
- `persist=False` 与即时合成的 Prompt 保持临时状态，角色列举和删除只影响相应模型缓存。

### Stream 输出

- 音频块采样率、布局、内容顺序与 `float32` 类型符合模型声明，最后一块能正确收尾。
- 取消、提前关闭和异常均会清理会话、在途任务与缓存，后续请求可按模型并发规则继续运行。
- 支持无参考生成的模型覆盖默认 Prompt，以及角色名未命中时的错误分支。

### Bundle 和下载

- 清单缺失、模型名不符、必需资产缺失和 SHA256 不符时能在后端加载前报错。
- 关闭摘要校验仍检查文件存在，模型自己检查必需文件集合和格式。
- 自动下载映射、固定版本参数对、进度、取消、校验失败与最终切换有对应测试。
- 发行归档保留新模型源码、配置、依赖及许可，干净环境安装后仍可导入和运行。

可参考引擎流式测试、缓存测试、执行计划测试与具体模型测试。[引擎流式测试](../tests/test_open_source_runtime.py) · [Prompt 测试](../tests/test_prompt_data_cache.py) · [执行计划测试](../tests/test_execution_plan_llama_backend.py)

## 13. 推荐验证命令

先在 Windows 开发环境运行项目维护的测试。Linux 从已激活环境执行与修改及平台相关的测试，Windows 发行约束按开发文档在 Windows 验证。

```powershell
uv run --no-sync python -m pytest tests -q
uv run --no-sync python -m example.synthesize --help
uv run --no-sync python tools/benchmark_streamvox_models.py --help
```

新模型加入内置注册后，使用准备好的本地目录和自己的参考音频执行冒烟合成。下面路径与文本需要按素材替换。

```powershell
uv run --no-sync python -m example.synthesize --model ckpts/my-tts-1b-gguf --reference-audio C:/audio/reference.wav --reference-text "参考音频的实际内容" --text "这是一段新模型适配检查。" --output smoke.wav
```

示例接收 `float32` NumPy 音频块，连接后写成 WAV。资源测量可通过 JSON 传入适配器支持的私有参数。下列命令显式选择物理 GPU 0，工具会据此使用 `device="gpu:0"`，多卡环境按实际编号调整。先准备已核验的模型目录，基准工具加载时关闭模型 SHA256 复核，其初始化耗时需与包含校验的完整初始化分开记录。

```powershell
uv run --no-sync python tools/benchmark_streamvox_models.py --model ckpts/my-tts-1b-gguf --model-key my-tts --display-name "MyTTS 1B GGUF" --physical-gpu-index 0 --prompt-audio C:/audio/reference.wav --prompt-text "参考音频的实际内容" --stream-kwargs-json '{"temperature":0.8,"top_p":0.9}'
```

若该假设模型还实现了 `make_default_prompt()` 和 `mode="text"`，可跳过参考准备。

```powershell
uv run --no-sync python tools/benchmark_streamvox_models.py --model ckpts/my-tts-1b-gguf --model-key my-tts-text --physical-gpu-index 0 --skip-make-prompt --stream-kwargs-json '{"mode":"text"}'
```

保存模型版本、SDK 提交、硬件、后端、输入、参考长度、采样参数和预热方法。数值对照还需固定上游源码与检查点，并保存误差门限、失败项及比较范围。Breeze Q8 已有有损任务验收和用户试听反馈，同时保留严格数值超限记录，其他模型和配置按各自证据判断。[当前验收边界](development.md#当前验收边界)

## 14. 适配完成标准

适配完成时，需要从用户实际入口确认下列结果。

- `TTSEngine` 能按已注册的本地模型目录初始化，`runtime_info` 准确显示模型、适配器、资产和关键配置。
- `make_prompt()` 能构建模型自己的 Prompt，角色缓存往返后仍可合成，公开参考资料可查询，或模型已明确提供无参考生成能力。
- `stream()` 按声明的采样率和布局返回音频，正常完成、取消与失败的资源清理均已验证。
- `list_roles()`、`get_prompt_data()` 和 `del_roles()` 对该模型工作，Prompt 的模型归属有校验。
- 声明支持按模型名下载时，仓库映射、清单校验和完成后切换均已验证。
- 最终 wheel 与 sdist 包含源码、配置、必要运行资源和许可，安装及 sdist 重建检查通过。
- benchmark 能通过模型名或本地路径与参数 JSON 运行，无参考模式可通过 `--skip-make-prompt` 覆盖。
- 对外声明的数值、效果和性能结论均有对应记录，范围准确对应模型版本、精度、后端和输入。

## 15. 常见设计错误

重复实现公共流程会让同一个参数在不同入口拥有不同含义。模型加载和推理差异放在适配器与运行时，注册表只维护映射，缓存层只处理统一载荷。权重加载延迟到运行时，才能让列举模型和查看帮助保持轻量。

Prompt 恢复后需要完整可用。将大数组转成 JSON、保存进程内指针，或依赖创建 Prompt 时的临时会话，都会使跨次启动恢复失效。应保存可重新创建推理状态的数据，并验证跨解释器重启后的合成。

设备请求、配置与实际计算之间需要逐项对应。读取了参数却未传给后端、混用不同构建的原生库、把多参考音频隐式连接，都会改变调用方对执行位置或参考资料的理解。用真实入口测试输入到后端的每次转换，并记录实际使用值。

文件拆分围绕独立职责。复杂文本规则、资产解析和进程协议可分别维护，简单的校验与执行保持在同一方法中。方法仅转发全部参数给另一个同义私有方法时，可合并以缩短调用路径。

## 16. 最小改动清单

只支持本地模型目录、单参考音频和流式合成时，通常需要模型包中的 `spec.py`、`runtime.py`、`adapter.py`、`__init__.py`，以及配置和测试。Prompt 结构较复杂时独立为 `prompt.py`。开发阶段用独立注册表验证，通过后把适配器加入 `builtin_models.py`。

支持自动下载还需修改 `manifests.py` 并准备远端标准清单。进入正式发行还需核对 `pyproject.toml`、`MANIFEST.in` 和实际归档，补齐新增依赖及第三方许可材料，执行目标环境安装、真实运行和 sdist 重建检查。每项工作对应一个实际入口或交付文件，按模型的能力完成相应路径即可。
