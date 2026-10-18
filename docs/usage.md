# StreamVox 使用文档

[返回 README](../README.md) · [模型许可](model-licenses.md) · [构建与开发](development.md)

本文描述 2.0.0 的公开 API。发行支持 Windows x64 / CPython 3.12，安装方式见 README。示例中的参考音频由你自行准备，参考文本应与音频内容一致。

## 初始化

```python
from streamvox import TTSEngine


engine = TTSEngine(
    model="qwen3-tts-clone-0.6b-gguf",
    device="cpu",
    llama_backend="cpu",
    onnx_provider="cpu",
    verify_model_sha256=True,
)
```

除 `model` 外的初始化参数均须按名称传入。

| 参数 | 说明 |
| --- | --- |
| `model` | 已注册模型名或本地模型 bundle 路径。 |
| `device` | `auto`、`cpu`、`gpu` 或 `gpu:<index>`，表达设备意图。 |
| `llama_backend` | `cpu`、`cuda`、`vulkan`，或 `None` 自动选择；控制 GGUF 链路。 |
| `onnx_provider` | `cpu`、`directml`、`cuda`，或 `None` 自动选择；控制 ONNX 链路。 |
| `n_ctx` | 可选的上下文窗口；默认沿用各模型配置，边界由模型检查。 |
| `verify_model_sha256` | 默认 `True`；校验模型运行资产的 SHA256。 |
| `registry` | 可选的自定义 `ModelRegistry`，供新模型适配使用。 |

使用完成后调用 `engine.shutdown()`。含模型初始化的独立脚本必须使用 `if __name__ == "__main__":` 入口，避免 Windows 子进程重复执行主程序。完整示例见 [synthesize.py](../example/synthesize.py)。

## 模型与本地目录

模型能力说明：

- [Qwen3 TTS Clone](models/qwen3-tts-clone.md)
- [S2-Pro](models/s2-pro.md)
- [VoxCPM2](models/voxcpm2.md)
- [MOSS-TTS-Nano](models/moss-tts-nano.md)

传入模型名时，SDK 检查默认模型目录，缺失时尝试从 ModelScope 下载。传入目录时，该目录必须包含 `manifest.json`，其中的 `model` 必须是当前 registry 已注册的模型名：

```python
engine = TTSEngine(
    model="./ckpts/qwen3-tts-clone-0.6b-gguf",
    device="cpu",
    llama_backend="cpu",
    onnx_provider="cpu",
)
```

生产和离线环境建议使用已核验的本地目录。模型权重遵循独立许可，详见 [模型许可表](model-licenses.md)。

Windows 默认数据目录为 `%APPDATA%/StreamVox`，模型位于其中的 `ckpts/`，参考音色缓存位于 `prompts/`。可在启动 Python 前设置 `STREAMVOX_HOME`，将这些数据放到你指定的目录。

## Prompt 与缓存

`make_prompt()` 将参考音频编码为模型可复用的音色资产。提前构建 Prompt 可以避免在首个合成片段前重复计算参考音频。

临时 Prompt 不写入缓存，直接把返回对象传给 `role_name`：

```python
prompt = engine.make_prompt(
    role_name="temporary_voice",
    audio_path="your-reference.wav",
    prompt_text="请填写参考音频中实际说出的内容。",
    persist=False,
)
chunks = engine.stream(
    text="你好，这是一段新的文本。",
    role_name=prompt,
    language="chinese",
)
```

重复使用的角色可传 `persist=True`，后续用 `role_name="角色名称"` 合成。持久化角色名不能重复；需要替换时先删除已有角色。缓存按模型隔离，不同模型的 Prompt 不能混用。

内存音频通过 `audio_data` 传入，并显式提供 `sample_rate`；它与 `audio_path` 二选一：

```python
prompt = engine.make_prompt(
    role_name="memory_voice",
    audio_data=audio_array,
    sample_rate=24000,
    prompt_text="请填写音频转写。",
    persist=False,
)
```

`stream()` 也能即时接收 `prompt_audio_path` 或 `prompt_audio_data` / `prompt_audio_sample_rate`，但这会将 Prompt 构建计入请求耗时。低延迟接入更适合先调用 `make_prompt()`。

角色管理：

```python
print(engine.list_roles())
data = engine.get_prompt_data("saved_voice")
if data is not None:
    print(data.role_name, data.sample_rate, data.prompt_text)
    print([reference.audio.shape for reference in data.references])
deleted = engine.del_roles("saved_voice")
```

`get_prompt_data()` 返回 `PromptData`，包含参考文本、参考音频和模型 Prompt；找不到角色时返回 `None`。`del_roles()` 接受单个角色名或角色名列表，并返回实际删除的名称列表。

## 流式输出与保存

`engine.stream()` 返回音频片段迭代器。每个片段是 `numpy.ndarray`，类型为 `float32`，表示单声道 PCM 采样值；采样率由 `engine.runtime.sample_rate` 提供。返回对象不是编码后的 WAV，也不是 `bytes`。

```python
import numpy as np
import soundfile as sf


chunks = list(engine.stream(
    text="这是一段保存音频的示例。",
    role_name=prompt,
    language="chinese",
))
if not chunks:
    raise RuntimeError("未生成音频。")
audio = np.concatenate(chunks, axis=-1)
sf.write("result.wav", audio, engine.runtime.sample_rate)
```

较长音频可逐块写入或播放，见 [命令行示例](../example/synthesize.py)。若需要传输原始浮点 PCM，可自行执行 `chunk.tobytes()`，接收端必须另外约定采样率、通道数、字节序和数据类型。该转换不属于 `stream()` 的返回契约。

迭代时发生错误应关闭生成器，并在 `finally` 中调用 `engine.shutdown()`。合成参数 `track_performance=True` 可记录性能分解，日志不能代替实际试听。

## 设备与实际后端

| `device` | 意图 |
| --- | --- |
| `auto` | 自动选择可用设备。 |
| `cpu` | CPU。 |
| `gpu` | 默认 GPU，等价于 `gpu:0`。 |
| `gpu:<index>` | 指定设备编号，例如 `gpu:1`。 |

`llama_backend` 和 `onnx_provider` 分别作用于两条推理链路。显式设置优先表达相应链路的后端要求；为了便于理解，建议让 `device` 与它们保持一致。MOSS 是纯 ONNX 模型，没有 llama 链路。

```python
engine = TTSEngine(
    model="voxcpm2-gguf",
    device="gpu:0",
    llama_backend="cuda",
    onnx_provider="directml",
)
try:
    print(engine.actual_backends)
finally:
    engine.shutdown()
```

显式要求的 ONNX provider 缺失、会话初始化发生不允许的回退，或实际后端不符合显式选择时，SDK 抛出异常。未显式选择时，自动选择行为由各模型 runtime 决定。

`actual_backends` 包含实际 `llama` / `onnx` 后端，以及主进程可见的 `onnx_sessions` provider 列表；纯 ONNX 模型没有 `llama` 键。子进程会话不会出现在主进程对象树中，但初始化时同样核对 provider。MOSS 的 DirectML 模式保留 codec decode step 使用 CPU 的已知兼容策略，该主进程会话单独列出。

Vulkan 设备编号不一定遵循 `CUDA_VISIBLE_DEVICES` 的重映射。多卡环境应结合日志、`actual_backends` 和系统监测确认实际设备，不能只凭显存变化判断。

依赖组 `cpu`、`directml`、`onnx-cuda` 互斥；`llama-cuda` 可独立组合。单个 wheel 始终包含 llama 的 CPU、CUDA、Vulkan 三组动态库，安装 extra 不会删减其中的后端。

## 模型参数

`stream(..., **model_kwargs)` 接收模型私有参数。各模型开放的字段和模式不同，应按对应指南选择。

| 模型 | 常用参数 |
| --- | --- |
| Qwen3 | `language`、`icl`、`stream`、`max_length`、`min_length`、`remove_meaningless_chars` |
| S2-Pro | `temperature`、`top_p`、`top_k`、`max_length`、`min_length`、`remove_meaningless_chars` |
| VoxCPM2 | `mode`、`control_text` 及模型指南列出的采样参数 |
| MOSS | `mode`、`seed`、`remove_meaningless_chars` |

通用性能参数为 `track_performance`。文本会经过相应模型的轻量清理和分段，但 SDK 不保证数字、日期、金额、缩写和单位一定按你期待的方式朗读；需要确定读法时，先将其改写为自然语言。

### 多参考音频

Qwen3、VoxCPM2 和 MOSS 当前只支持单参考音频。S2-Pro 支持多参考，音频和转写必须一一对齐，参考文本及目标文本应使用相同的 speaker 标记：

```python
prompt = engine.make_prompt(
    role_name="two_speakers",
    audio_path=["your-speaker-a.wav", "your-speaker-b.wav"],
    prompt_text=[
        "<|speaker:0|>第一位说话人的实际转写。",
        "<|speaker:1|>第二位说话人的实际转写。",
    ],
    persist=False,
)
chunks = engine.stream(
    text="<|speaker:0|>你好。<|speaker:1|>很高兴见到你。",
    role_name=prompt,
)
```

这个片段应在 S2-Pro 引擎上运行；使用其权重前先阅读 [独立许可](model-licenses.md#s2-pro)。

## 模型下载与固定版本

普通下载使用默认 ModelScope 仓库的最新快照：

```python
from streamvox.download import download_model


result = download_model(model="qwen3-tts-clone-0.6b-gguf", target="./ckpts")
print(result.target_dir)
```

需要精确固定版本时，从自己的模型发布记录取得完整 40 位提交和原始 `manifest.json` 的 64 位 SHA256，两项同时传入：

```python
result = download_model(
    model="qwen3-tts-clone-0.6b-gguf",
    target="./ckpts",
    revision=revision_from_your_catalog,
    expected_manifest_sha256=manifest_sha256_from_your_catalog,
)
```

固定版本下载先核对原始清单，再按同一提交下载、校验并原子发布；旧正式目录在验证完成前不会被替换。`progress_callback` 接收阶段与进度，`cancel_event` 可取消任务并保留下载缓存，`cache_dir` 可指定持久缓存位置。生产分发还应保留对应模型 LICENSE/NOTICE，详见 [模型许可](model-licenses.md)。

## 故障排查

| 情况 | 检查内容 |
| --- | --- |
| 安装失败 | Windows 发行包需要 x64、CPython 3.12 和 `cp312-none-win_amd64` wheel；Linux 按 [源码运行](../README.md#linux-源码运行) 使用 `--no-install-project` 准备环境。环境只安装一种 ONNX Runtime。 |
| 找不到 DLL 或共享库 | Windows wheel 应携带 DLL，Windows 源码环境先执行 `python -m streamvox.native_assets --project-root .`；Linux 按 [源码运行](../README.md#linux-源码运行) 下载预编译 `.so`，放到 `streamvox/bin/` 并在启动前设置 `LD_LIBRARY_PATH`，系统不兼容时再选择自编译。 |
| 模型目录无效 | 检查 `manifest.json`、模型名、`ckpt/` 文件和哈希。 |
| 角色不存在 | 缓存角色须先以 `persist=True` 创建；临时 Prompt 直接传返回对象。 |
| 后端不可用 | 核对安装的 extra、GPU 驱动、显式 provider 和实际后端；明确选择失败会报错。 |
| 吞字、异常音或音色漂移 | 检查参考音频、转写、模型模式、文本和采样配置；保留最小复现用例。 |

全模型数值对照和人工试听仍未验收。基础测试通过不表示所有模型的音质、数值和硬件组合都已验收。
