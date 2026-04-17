# StreamVox SDK 开发者指南

StreamVox 是一个专为本地化部署和业务线集成打造的流式语音合成（TTS）引擎。它将复杂的底层细节（如 GGUF/ONNX 模型加载、Codec 转换、音色克隆 Prompt 缓存、硬件后端调度及商业授权）完全黑盒化，通过极简的 `TTSEngine` 接口，让开发者能够以最低的接入成本获得企业级的流式语音生成能力。

## 1. 性能基准与资源消耗 (Performance & Resources)

在商业落地中，算力成本和响应延迟是核心考量。以下是 StreamVox 在主流服务器硬件上的实测表现。

**测试环境：**
* **GPU:** NVIDIA GeForce RTX 4090
* **CPU:** Intel(R) Xeon(R) Gold 5320 CPU @ 2.20GHz
* **测试口径:** 每个模型独立子进程运行，Vulkan/CUDA 后端锁定同一物理卡。

### 1.1 推理速度基准 (Speed Benchmarks)

引擎默认采用流式输出，首包延迟决定了用户的等待感，RTF（实时率）决定了持续合成的流畅度。

| 模型 | 平均首包延迟 (First Chunk) | 平均 RTF (实时率) | 测试单句完整推理耗时 |
| :--- | :---: | :---: | :---: |
| **Qwen3 TTS 0.6B** | **~0.289 s** | **0.156** | 1.91 s |
| **Qwen3 TTS 1.7B** | ~0.328 s | 0.185 | 2.06 s |
| **S2-Pro 4B** | ~0.423 s | 0.358 | 3.97 s |

> **提示：** 实际耗时会受输入文本长度、参考音频长度、系统后台负载及具体硬件状态影响，上述数据仅供选型参考。

### 1.2 资源占用评估 (VRAM & Memory)

这是模型 adapter 中声明的最低/推荐资源档位，以及在 4090 上实测的显存峰值占用：

| 模型 | 最低显存 | 推荐显存 | 最低内存 | 实测峰值进程显存 (RTX 4090) |
| :--- | ---: | ---: | ---: | ---: |
| **Qwen3 TTS 0.6B** | 2 GB | 3 GB | 3 GB | **~1892 MB** |
| **Qwen3 TTS 1.7B** | 3 GB | 4 GB | 4 GB | **~2534 MB** |
| **S2-Pro 4B** | 7 GB | 8 GB | 6 GB | **~6580 MB** |

---

## 2. 支持模型矩阵

当前版本 StreamVox 搭载了 3 款业界领先的 TTS 模型，可根据业务场景（高并发 vs. 高保真）灵活切换：

| 模型名 (`model`) | 参数规模 | 默认采样率 | 核心定位与适用场景 |
| :--- | ---: | ---: | :--- |
| `qwen3-tts-clone-0.6b-gguf` | 0.6B | 24000 Hz | **极速轻量**：极低的启动延迟与资源占用，适合高并发对话场景。 |
| `qwen3-tts-clone-1.7b-gguf` | 1.7B | 24000 Hz | **质量均衡**：兼顾速度与生成质量，提供优秀的单参考音色克隆表现。 |
| `s2-pro-4b-gguf` | 4B | 44100 Hz | **高保真演播**：高采样率输出，支持**多参考** prompt，适合有声书、专业配音及复杂情感合成。 |

---

## 3. 核心优势

* **开箱即用：** 单一入口 `TTSEngine` 初始化，自动处理运行环境。
* **灵活部署：** 支持云端自动拉取模型，也支持纯内网物理隔离的本地 Bundle 加载。
* **极致克隆：** 支持零样本（Zero-shot）音色克隆，并内置角色 Prompt 本地缓存机制，大幅提升复用效率。
* **硬件自适应：** 智能探测平台硬件，支持 `auto`、`cpu`、`gpu:<index>` 级调度。
* **商业护航：** 完善的在线 License 校验与离线授权机制，未授权自动降级为 Trial 模式供开发测试。

---

## 4. 安装环境

建议使用 Python `3.10`。当前发布包和测试环境以 Python 3.10 为主。

使用 `uv` 安装：

```bash
uv venv
source .venv/bin/activate
uv sync
```

使用 `pip` 安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

如果需要手动安装 PyTorch CUDA 版本，可以使用：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

## 快速开始

下面示例会完成初始化模型、创建临时音色 prompt、流式合成文本并保存 wav 文件。

```python
import numpy as np
import soundfile as sf

from streamvox import TTSEngine


engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    verify_model_sha256=False,
)

try:
    prompt = engine.make_prompt(
        role_name="demo_role",
        audio_path="example/Condition3.wav",
        prompt_text="所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。",
        persist=False,
    )

    chunks = list(
        engine.stream(
            text="你好，这里是 StreamVox 的快速开始示例。",
            role_name=prompt,
            language="chinese",
            track_performance=False,
        )
    )

    audio = np.concatenate(chunks, axis=-1)
    sf.write("output.wav", audio, engine.runtime.sample_rate)
finally:
    engine.shutdown()
```

说明：

- `license_key` 为空或无效时会进入 trial 模式，输出音频会被限制处理。
- `verify_model_sha256=False` 可以跳过初始化时的模型文件哈希校验，适合本地测试提速；正式分发场景建议按需要开启。
- `device="auto"` 会由 runtime 根据当前平台和已打包后端自动选择。

## 模型选择

### Qwen3 0.6B

```python
engine = TTSEngine(
    model="qwen3-tts-clone-0.6b-gguf",
    license_key="YOUR_SDK_KEY",
    device="gpu",
    verify_model_sha256=False,
)
```

适合更关注启动速度和资源占用的场景。

### Qwen3 1.7B

```python
engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
    device="gpu",
    verify_model_sha256=False,
)
```

适合更关注音色克隆质量的单参考音频场景。

### S2-Pro 4B

```python
engine = TTSEngine(
    model="s2-pro-4b-gguf",
    license_key="YOUR_SDK_KEY",
    device="gpu",
    verify_model_sha256=False,
)
```

S2-Pro 默认采样率为 `44100 Hz`，支持更高采样率输出，也支持多参考 prompt。单说话人文本可以直接传普通文本；多说话人文本建议显式使用 speaker 标签。

## 本地模型目录

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

## Prompt 使用

### 创建可缓存角色

```python
prompt = engine.make_prompt(
    role_name="gemi_voice",
    audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    persist=True,
)
```

`persist=True` 会把 prompt 写入本地缓存。后续可以直接用角色名合成：

```python
chunks = engine.stream(
    text="你好，这是一段使用缓存角色合成的文本。",
    role_name="gemi_voice",
    language="chinese",
)
```

### 使用临时角色

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

### 直接在 stream 中传参考音频

```python
chunks = engine.stream(
    text="这是一段即时克隆的示例。",
    prompt_audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    language="chinese",
)
```

## 保存音频

`stream()` 返回的是 `numpy.float32` 音频分片迭代器。可以边播边处理，也可以合并后保存：

```python
import numpy as np
import soundfile as sf

chunks = list(
    engine.stream(
        text="保存 wav 的示例。",
        role_name="gemi_voice",
        language="chinese",
    )
)

audio = np.concatenate(chunks, axis=-1)
sf.write("result.wav", audio, engine.runtime.sample_rate)
```

## 角色缓存管理

```python
roles = engine.list_roles()
print(roles)

deleted = engine.del_roles(["gemi_voice"])
print(deleted)
```

缓存会按模型隔离，不同模型的同名角色不会互相覆盖。

## 设备选择

`device` 当前支持字符串形式：

| 参数 | 含义 |
| --- | --- |
| `auto` | 自动选择可用后端 |
| `cpu` | 强制 CPU |
| `gpu` | 使用默认 GPU，等价于 `gpu:0` |
| `gpu:<index>` | 使用指定 GPU，例如 `gpu:1` |

示例：

```python
engine = TTSEngine(
    model="s2-pro-4b-gguf",
    license_key="YOUR_SDK_KEY",
    device="gpu:1",
    verify_model_sha256=False,
)
```

注意：当前 GGUF 侧可能使用 llama.cpp Vulkan 后端。Vulkan 的 GPU 编号不一定遵循 `CUDA_VISIBLE_DEVICES` 重映射。多卡机器上建议直接使用 `device="gpu:<物理卡编号>"`，例如 `device="gpu:1"`。

## 合成参数

### Qwen3 TTS Clone

| 参数 | 说明 |
| --- | --- |
| `language` | 目标语言，支持具体语言名或 `auto` |
| `stream` | 是否流式生成 |
| `icl` | 是否启用上下文学习式提示 |
| `max_length` | 长文本分段上限 |
| `min_length` | 短段合并下限 |
| `remove_meaningless_chars` | 是否清理无意义字符 |
| `track_performance` | 是否输出性能日志 |

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

### S2-Pro

| 参数 | 说明 |
| --- | --- |
| `max_length` | 长文本分段上限 |
| `min_length` | 短段合并下限 |
| `remove_meaningless_chars` | 是否清理无意义字符 |
| `temperature` | 采样温度 |
| `top_p` | nucleus sampling 阈值 |
| `top_k` | top-k 采样数量 |
| `track_performance` | 是否输出性能日志 |

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

## 多参考 Prompt

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

## 授权说明

StreamVox 支持两种授权入口：

| 参数 | 说明 |
| --- | --- |
| `license_key` | 在线授权 key |
| `license_path` | 本地 license 文件路径 |

未授权或授权失败时会进入 trial 模式。trial 模式仍会保持流式输出接口，但输出音频会经过限制处理，不能作为正式结果使用。

## 资源需求

### 模型声明档位

这是模型 adapter 中声明的最低和推荐资源档位，用于判断机器是否值得尝试运行：

| 模型 | 最低显存 | 推荐显存 | 最低内存 | 推荐内存 |
| --- | ---: | ---: | ---: | ---: |
| `qwen3-tts-clone-0.6b-gguf` | 2 GB | 3 GB | 3 GB | 5 GB |
| `qwen3-tts-clone-1.7b-gguf` | 3 GB | 4 GB | 4 GB | 6 GB |
| `s2-pro-4b-gguf` | 7 GB | 8 GB | 6 GB | 8 GB |

### 本机实测资源占用

以下结果来自当前测试脚本在一张 NVIDIA GeForce RTX 4090 上运行的资源采样。测试口径为：每个模型独立子进程运行，清理 `CUDA_VISIBLE_DEVICES`，引擎直接使用 `device="gpu:1"`，让 CUDA/ONNX 与 llama.cpp Vulkan 都按物理卡编号命中同一张 GPU。

| 模型 | 峰值内存 MiB | 峰值进程显存 MiB | 峰值整卡显存增量 MiB | 初始化 s | Prompt s | 首包 s | 合成 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 TTS Clone 0.6B GGUF | 2706.74 | 1892.00 | 1912.00 | 3.91 | 2.10 | 0.52 | 41.49 |
| Qwen3 TTS Clone 1.7B GGUF | 3028.66 | 2534.00 | 2548.00 | 5.89 | 1.94 | 0.56 | 45.69 |
| S2-Pro 4B GGUF | 5503.59 | 6580.00 | 6597.00 | 9.19 | 3.63 | 1.03 | 88.10 |

说明：

- `峰值进程显存` 来自 `nvidia-smi --query-compute-apps` 对当前进程树的归因。
- `峰值整卡显存增量` 来自目标 GPU 的整卡已用显存减测试前基线。
- 这些数字与驱动、后端、上下文长度、输入文本长度、参考音频长度和系统后台进程有关，不应当视为所有机器上的固定跑分。

## 常见建议

### 什么时候传模型名

开发和试用阶段可以直接传模型名。StreamVox 会优先检查默认模型目录，缺失时再尝试下载。

### 什么时候传本地目录

部署和离线环境建议传本地 bundle 目录，确保模型文件版本、manifest 和运行环境完全可控。

### 什么时候缓存角色

高频复用的角色建议 `persist=True`。一次性试听或用户上传的临时参考音频建议 `persist=False`。

### 多卡机器如何选卡

建议直接传 `device="gpu:<物理卡编号>"`。如果发现 `nvidia-smi` 中多张卡都有显存变化，优先检查是否设置了 `CUDA_VISIBLE_DEVICES`，以及 GGUF 后端是否走 Vulkan。

## 故障排查

### 模型目录无效

确认模型目录内存在 `manifest.json`，并且 manifest 中的 `model` 字段是当前 SDK 已支持的模型名。

### 授权失败

检查 `license_key` 是否有效，授权服务是否可达，或者是否存在本地有效 token。授权失败时会进入 trial 模式。

### 音频听起来异常

如果日志提示 trial 模式，输出音频会被限制处理。请先确认授权状态。

### 显存不在预期 GPU 上

多卡机器建议直接使用 `device="gpu:1"` 这类物理卡编号。不要用 `CUDA_VISIBLE_DEVICES=1` 后再传 `device="gpu:0"`，因为 Vulkan 后端可能仍把 `gpu:0` 当成物理 0 卡。
