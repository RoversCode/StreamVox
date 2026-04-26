# StreamVox 完整功能使用文档

本文档面向已经完成环境安装的开发者，集中说明 StreamVox SDK 的主要功能入口和常见使用方式。快速安装和最小示例请先阅读仓库根目录的 [README.md](../README.md)。

## 1. 基础入口

StreamVox 的核心入口是 `TTSEngine`：

```python
from streamvox import TTSEngine


engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    verify_model_sha256=False,
)
```

常用初始化参数：

| 参数 | 说明 |
| --- | --- |
| `model` | 模型名或本地模型 bundle 目录。 |
| `license_key` | 在线授权 key。为空或无效时进入 trial 模式。 |
| `license_path` | 本地离线授权文件路径。通常和 `license_key` 二选一。 |
| `device` | 设备选择，支持 `auto`、`cpu`、`gpu`、`gpu:<index>`。 |
| `n_ctx` | 上下文窗口配置。一般不需要手动传入。 |
| `verify_model_sha256` | 是否校验模型文件 sha256。测试阶段可关闭，正式交付按需要开启。 |

使用完成后建议显式关闭：

```python
engine.shutdown()
```

## 2. 模型选择

### 2.1 Qwen3 TTS Clone 0.6B

```python
engine = TTSEngine(
    model="qwen3-tts-clone-0.6b-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    verify_model_sha256=False,
)
```

适合更关注启动速度、低资源占用。

### 2.2 Qwen3 TTS Clone 1.7B

```python
engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    verify_model_sha256=False,
)
```

适合更关注音色克隆质量的单参考音频场景。

### 2.3 S2-Pro 4B

```python
engine = TTSEngine(
    model="s2-pro-4b-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    verify_model_sha256=False,
)
```

S2-Pro 默认采样率为 `44100 Hz`，支持更高保真输出，也支持多参考 prompt，适合有声书、专业配音和复杂情感合成。

## 3. 本地模型目录

`model` 可以传模型名，也可以传本地 bundle 目录。

传模型名：

```python
engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
)
```

传本地目录：

```python
engine = TTSEngine(
    model="./ckpts/qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
)
```

本地目录必须包含 `manifest.json`。引擎会从 manifest 中读取模型名，并检查该模型是否已被当前 SDK 注册。

## 4. Prompt 使用

Prompt 是参考音频经过模型处理后的音色资产。StreamVox 支持可缓存 Prompt、临时 Prompt 和在 `stream()` 中即时创建 Prompt。

### 4.1 创建可缓存角色

```python
prompt = engine.make_prompt(
    role_name="gemi_voice",
    audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    persist=True,
)
```

`persist=True` 会把 prompt 写入本地缓存。后续可以直接使用角色名合成：

```python
chunks = engine.stream(
    text="你好，这是一段使用缓存角色合成的文本。",
    role_name="gemi_voice",
    language="chinese",
)
```

### 4.2 创建临时角色

```python
prompt = engine.make_prompt(
    role_name="tmp_voice",
    audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    persist=False,
)

chunks = engine.stream(
    text="这是一段不落盘的临时克隆示例。",
    role_name=prompt,
    language="chinese",
)
```

`persist=False` 适合试听、用户上传音频或不希望落盘的临时合成场景。

### 4.3 使用内存音频创建 Prompt

```python
prompt = engine.make_prompt(
    role_name="memory_voice",
    audio_data=audio_array,
    sample_rate=24000,
    prompt_text="这是参考音频的转写文本。",
    persist=False,
)
```

传 `audio_data` 时必须显式传入 `sample_rate`。

### 4.4 直接在 `stream()` 中传参考音频

这种方式**可以使用，但不推荐作为正式接入方案**。  
原因是：构建prompt data需要耗时，会拖慢首chunk延迟。。

更推荐的做法是：

1. 先通过 `make_prompt(...)` 预先生成 Prompt。
2. 如果角色需要复用，使用 `persist=True` 落盘缓存，后续正式推理时直接传 `role_name="xxx"`。
3. 如果只是一次性临时合成，使用 `persist=False`，然后在正式推理时直接把返回的 `prompt` 对象传给 `role_name`。

```python
chunks = engine.stream(
    text="这是一段即时克隆的示例。",
    role_name="transient_voice",
    prompt_audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    language="chinese",
)
```

推荐工作流示例一：预先创建可复用角色，再在正式推理时只传 `role_name`

```python
prompt = engine.make_prompt(
    role_name="gemi_voice",
    audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    persist=True,
)

chunks = engine.stream(
    text="这是一段正式推理文本。",
    role_name="gemi_voice",
    language="chinese",
)
```

推荐工作流示例二：预先创建临时 Prompt，再在正式推理时直接传 Prompt 对象

```python
prompt = engine.make_prompt(
    role_name="tmp_voice",
    audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    persist=False,
)

chunks = engine.stream(
    text="这是一段正式推理文本。",
    role_name=prompt,
    language="chinese",
)
```

内存音频入口：

```python
chunks = engine.stream(
    text="这是一段即时克隆的示例。",
    role_name="transient_voice",
    prompt_audio_data=audio_array,
    prompt_audio_sample_rate=24000,
    prompt_text="这是参考音频的转写文本。",
    language="chinese",
)
```

## 5. 保存音频

`stream()` 返回的是 `numpy.float32` 音频分片迭代器。可以边播边处理，也可以合并后保存：

```python
import numpy as np
import soundfile as sf


chunks = engine.stream(
        text="保存 wav 的示例。",
        role_name="gemi_voice",
        language="chinese",
)

audios = []
for chunk in chunks:
    audios.append(chunk)

audio = np.concatenate(audios, axis=-1)
sf.write("result.wav", audio, engine.runtime.sample_rate)
```

## 6. 角色缓存管理

查看当前模型下的缓存角色：

```python
roles = engine.list_roles()
print(roles)
```

删除角色：

```python
deleted = engine.del_roles(["gemi_voice"])
print(deleted)
```

也可以删除单个角色：

```python
deleted = engine.del_roles("gemi_voice")
```

缓存会按模型隔离，不同模型的同名角色不会互相覆盖。

## 7. 设备选择

`device` 当前支持字符串形式：

| 参数 | 含义 |
| --- | --- |
| `auto` | 自动选择可用后端。 |
| `cpu` | 强制 CPU。 |
| `gpu` | 使用默认 GPU，等价于 `gpu:0`。 |
| `gpu:<index>` | 使用指定 GPU，例如 `gpu:1`。 |

示例：

```python
engine = TTSEngine(
    model="s2-pro-4b-gguf",
    license_key="YOUR_SDK_KEY",
    device="gpu:1",
    verify_model_sha256=False,
)
```

注意：当前 GGUF 侧可能使用 llama.cpp Vulkan 后端。Vulkan 的 GPU 编号不一定遵循 `CUDA_VISIBLE_DEVICES` 重映射。多卡机器上建议直接使用 `device="gpu:<物理卡编号>"`。

## 8. 合成参数

### 8.1 Qwen3 TTS Clone

| 参数 | 说明 |
| --- | --- |
| `language` | 目标语言，支持具体语言名或 `auto`。 |
| `stream` | 是否流式生成。 |
| `icl` | 是否启用上下文学习式提示。 |
| `max_length` | 长文本分段上限。 |
| `min_length` | 短段合并下限。 |
| `remove_meaningless_chars` | 是否清理无意义字符。 |
| `track_performance` | 是否输出性能日志。 |

示例：

```python
chunks = engine.stream(
    text="这是一段较长文本，用于测试长文本切分。",
    role_name="gemi_voice",
    language="chinese",
    max_length=150,
    min_length=20,
    icl=False,
    track_performance=True,
)
```

### 8.2 S2-Pro

| 参数 | 说明 |
| --- | --- |
| `max_length` | 长文本分段上限。 |
| `min_length` | 短段合并下限。 |
| `remove_meaningless_chars` | 是否清理无意义字符。 |
| `temperature` | 采样温度。 |
| `top_p` | nucleus sampling 阈值。 |
| `top_k` | top-k 采样数量。 |
| `track_performance` | 是否输出性能日志。 |

示例：

```python
chunks = engine.stream(
    text="<|speaker:0|>你好，这里是 S2-Pro 的示例。",
    role_name="gemi_voice",
    temperature=0.9,
    top_p=0.8,
    top_k=30,
    track_performance=True,
)
```

## 9. 多参考 Prompt

Qwen3 0.6B 和 Qwen3 1.7B 当前只支持单参考音频 prompt。

S2-Pro 支持单参考和多参考 prompt。多参考输入示例：

```python
prompt = engine.make_prompt(
    role_name="multi_voice",
    audio_path=["speaker_a.wav", "speaker_b.wav"],
    prompt_text=[
        "<|speaker:0|>这是第一位说话人的参考文本。",
        "<|speaker:1|>这是第二位说话人的参考文本。",
    ],
    persist=False,
)
```

多参考场景下，请重点注意以下约束：

1. 多参考音频和多参考文本的列表长度必须一致。
2. `audio_path` 中的每一项，都必须与 `prompt_text` 中对应位置的那一项一一匹配，不能错位。
3. **如果给定多参考，文本必须事先分配好 speaker。**  
   也就是说，你传入的每一段参考文本，都必须明确标注它属于哪一个说话人，例如 `<|speaker:0|>`、`<|speaker:1|>`。
4. 正式合成时，输入文本也应继续沿用同一套 speaker 分配规则。否则模型无法稳定理解哪一段内容应该由哪位说话人来表达。

例如，下面这种写法才是推荐的多参考输入方式：

```python
prompt = engine.make_prompt(
    role_name="multi_voice",
    audio_path=["speaker_a.wav", "speaker_b.wav"],
    prompt_text=[
        "<|speaker:0|>这是第一位说话人的参考文本。",
        "<|speaker:1|>这是第二位说话人的参考文本。",
    ],
    persist=True,
)

chunks = engine.stream(
    text="<|speaker:0|>你好，下面由我先开始介绍。<|speaker:1|>接下来这部分由我继续补充说明。",
    role_name="multi_voice",
    language="chinese",
)
```

如果多参考音频已经对应了多个 speaker，但参考文本没有提前分配 speaker，或者正式合成文本没有按相同规则标注 speaker，那么生成结果可能出现说话人混乱、风格漂移或角色切换不稳定的问题。

## 10. 授权说明

| 参数 | 说明 |
| --- | --- |
| `license_key` | 在线授权 key。 |

未授权或授权失败时会进入 trial 模式。trial 模式仍会保持流式输出接口，但输出音频会经过限制处理，不能作为正式结果使用。

## 11. 常见建议

### 11.1 什么时候传模型名

开发和试用阶段可以直接传模型名。StreamVox 会优先检查默认模型目录，缺失时再尝试下载。

### 11.2 什么时候传本地目录

部署和离线环境建议传本地 bundle 目录，确保模型文件版本、manifest 和运行环境完全可控。

### 11.3 什么时候缓存角色

高频复用的角色建议 `persist=True`。一次性试听或用户上传的临时参考音频建议 `persist=False`。

### 11.4 多卡机器如何选卡

建议直接传 `device="gpu:<物理卡编号>"`。如果发现 `nvidia-smi` 中多张卡都有显存变化，优先检查是否设置了 `CUDA_VISIBLE_DEVICES`，以及 GGUF 后端是否走 Vulkan。

## 12. 故障排查

### 12.1 模型目录无效

确认模型目录内存在 `manifest.json`，并且 manifest 中的 `model` 字段是当前 SDK 已支持的模型名。

### 12.2 授权失败

检查 `license_key` 是否有效，授权服务是否可达，或者是否存在本地有效 token。授权失败时会进入 trial 模式。

### 12.3 音频听起来异常

如果日志提示 trial 模式，输出音频会被限制处理。请先确认授权状态。

### 12.4 显存不在预期 GPU 上

多卡机器建议直接使用 `device="gpu:1"` 这类物理卡编号。不要用 `CUDA_VISIBLE_DEVICES=1` 后再传 `device="gpu:0"`，因为 Vulkan 后端可能仍把 `gpu:0` 当成物理 0 卡。

### 12.5 Prompt 角色不存在

如果使用 `role_name="xxx"` 合成，必须先通过 `make_prompt(..., persist=True)` 创建并缓存该角色。临时 Prompt 请直接把 `make_prompt(..., persist=False)` 返回的对象传给 `role_name`。
