<div align="center">
    <h1>StreamVox SDK</h1>
    <p><strong>工业级流式实时语音合成（TTS）引擎开发者指南</strong></p>
    <p>
        <a href="https://e.tb.cn/h.iABy1AGhcFcpz4G?tk=3ajd5pJJVO1">
            <img alt="SDK Key 购买" src="https://img.shields.io/badge/SDK%20Key-%E8%B4%AD%E4%B9%B0%E9%93%BE%E6%8E%A5-ff6a00?style=for-the-badge">
        </a>
        <img alt="交流群" src="https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-1097818796-12b886?style=for-the-badge">
        <img alt="Trial" src="https://img.shields.io/badge/Trial-%E5%85%88%E6%B5%8B%E5%90%8E%E4%B9%B0-228be6?style=for-the-badge">
    </p>
    <p>
        <a href="doc/usage.md">使用文档</a>&nbsp;&nbsp; | &nbsp;&nbsp;
        <a href="#5-快速开始">快速开始</a>&nbsp;&nbsp; | &nbsp;&nbsp;
        <a href="#1-性能基准与资源消耗-performance--resources">性能基准</a>&nbsp;&nbsp; | &nbsp;&nbsp;
        <a href="#2-支持模型矩阵">模型矩阵</a>&nbsp;&nbsp; | &nbsp;&nbsp;
        <a href="https://modelscope.cn/profile/ChengHee">模型下载</a>
    </p>
    <p>
        <strong>SDK Key 购买：</strong>
        <a href="https://item.taobao.com/item.htm?ft=t&id=1044813462201&spm=a21dvs.23580594.0.0.6781645ez4U7cp">淘宝授权入口</a>
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <strong>QQ交流群：</strong>1097818796
    </p>
</div>

<br>

StreamVox 是一款商用的通用流式实时语音合成（TTS）引擎，全面适配当前业界最先进的 TTS 模型，并对底层推理效率进行了极致优化。

它旨在为开发者提供最低接入成本的企业级实时语音生成能力，具备以下核心特性：

* **极致的推理效率**：性能拉满，Qwen3 1.7B 模型首包延迟低至 **~300ms**，4B 参数级的 S2-Pro 模型首包延迟也仅需 **~400ms**。同时，`MOSS-TTS-Nano` 已完成适配，在 **纯 CPU 场景**下是当前 StreamVox 中最值得优先尝试的实时语音生成模型之一。
* **无视硬件壁垒**：提供丰富尺寸的模型以匹配不同算力。完全不限显卡阵营，无论是 N 卡（NVIDIA）、A 卡（AMD）、I 卡（Intel），还是核显/集显，甚至**纯 CPU 机器**，都能顺畅跑通。
* **先测后买，零试错成本**：无需事先购买 SDK 授权，任何人都可以直接在本地安装并测试真实硬件上的推理速度。确认完全满足业务场景需求后，再按需购买商业授权。
* **极简接入体验**：将复杂的底层细节完全黑盒化。只需一个 `TTSEngine` 接口，几行代码即可完成集成。
* **持续进化的生态**：我们的愿景是**“让高品质的实时语音合成触手可及”**。开发者无需去啃晦涩的 TTS 底层原理，无需耗费心力死磕推理提速，更无需手写繁琐的部署代码。StreamVox 真正做到了开箱即用，让每一位开发者都能以极简的方式，为自己的 AI 助手搭载速度极快、效果惊艳的专属声音引擎。同时，我们将持续集成业界更优秀的模型和微调音色，并不断榨干硬件性能，确保哪怕是纯 CPU 的机器，也能享受到极致的流式语音生成体验。

## 购买前须知！！！注意⚠
1. StreamVox的服务内容是**压低模型的对资源的使用需求，提供更快的推理速度和流畅的流式实时语音合成体验**。不为任何模型的效果负责，购买前请先测试开源模型是否满足需求！！！
2. StreamVox **支持直接本地安装并免费测试推理性能**，你可以先在自己的机器上验证部署方式、资源占用与实时表现。  
   但请注意：**未购买有效 `key` 时，输出音频会处于受限状态，声音不会作为正式效果参考**。如果你需要正常音质、正式交付或线上使用，请购买并配置有效授权。
3. StreamVox **不负责文本预处理**，会按你传入的文本直接执行推理。  
   尤其是**数字、日期、金额、缩写、单位、符号混排**等内容，如果你对最终读法有明确要求，建议**在送入推理前先转换为自然语言文本**，以避免出现不符合预期的播报结果。
4. AI模型都 **存在模型能力边界**，在少量复杂场景或特殊输入下，仍可能出现吞字、异常音、停顿不自然等 bad case，这类问题通常无法通过工程侧完全消除。  如果你遇到类似情况，建议优先**更换 prompt 音频**，或**调整推理文本的句式、断句与表达方式**后重新尝试，很多情况下可以得到更稳定的结果。
5. 如果你的机器使用 **NVIDIA 显卡**，并且驱动环境支持 **CUDA 12.4 及以上**，建议优先选择 **CUDA 后端**版本。  
   在同等硬件条件下，CUDA 后端的推理性能通常会比 Vulkan 后端**快约 30%**。
6. 每个key**最多只能激活3台设备**，如果有工业上的集成需求，请联系淘宝客服。

## News
* **2026.5.8 🎉** **兼容 MOSS-TTS-Nano**，面向低资源与纯 CPU 场景，支持实时流式语音生成、单参考音色克隆与 continuation / voice_clone 两种推理模式。
* **2026.5.4 🎉** **兼容voxcpm2**，最低显存需求6G，首包延迟最低仅180ms！！已完美支持音色设计、副语言控制、方言克隆以及极致的音色克隆能力。
* **2026.4.23 🎉** **StreamVox 正式发布**，现已开放本地试用、性能评估与商业授权接入。

## 目录

- [性能基准与资源消耗](#1-性能基准与资源消耗-performance--resources)
- [支持模型矩阵](#2-支持模型矩阵)
- [核心优势](#3-核心优势)
- [安装教程](#4-安装教程)
- [快速开始](#5-快速开始)
- [完整功能文档](#6-完整功能文档)
- [StreamVox TODO](#7-streamvox-todo)
- [常见问题](#8-常见问题)
- [参考与致谢](#9-参考与致谢)

## TODO
- [x] 适配voxcpm2
- [x] 适配MOSS-TTS-Nano(0.1 B)
- [ ] 精致定制音色


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
| **VoxCPM2 2.38B** | ~0.166 s | 0.372 | 7.04 s |
| **Moss-TTS-Nano 0.1B(CPU)** | ~0.26 s | 0.42 | 7.36 s |
| **Moss-TTS-Nano 0.1B(GPU)** | ~0.06 s | 0.23 | 7.12 s |

**注意：** voxcpm2 DmlExecutionProvider会比CUDA慢两倍左右，windows且有N卡用户极力推荐安装**Windows + NVIDIA CUDA**

> **提示：** 实际耗时会受输入文本长度、参考音频长度、系统后台负载及具体硬件状态影响，上述数据仅供选型参考。


### 1.2 资源占用评估 (VRAM & Memory)

这是模型 adapter 中声明的最低/推荐资源档位，以及在 4090 上实测的显存峰值占用：

| 模型 | 最低显存 | 推荐显存 | 最低内存 | 实测峰值进程显存 (RTX 4090) |
| :--- | ---: | ---: | ---: | ---: |
| **Qwen3 TTS 0.6B** | 2 GB | 3 GB | 3 GB | **~1892 MB** |
| **Qwen3 TTS 1.7B** | 3 GB | 4 GB | 4 GB | **~2534 MB** |
| **S2-Pro 4B** | 7 GB | 8 GB | 6 GB | **~6580 MB** |
| **VoxCPM2 2.38B** | 6 GB | 8 GB | 4 GB | **~5963 MB** |
| **MOSS-TTS-Nano 0.1B** | 0 GB | 0 GB | 3 GB | **~0 MB** |

---

## 2. 支持模型矩阵

当前版本 StreamVox 搭载了 5 款业界领先的 TTS 模型，可根据业务场景灵活切换：

| 模型名 (`model`) | 参数规模 | 默认采样率 | 核心定位与适用场景 | 模型能力文档 |
| :--- | ---: | ---: | :--- | :--- |
| `qwen3-tts-clone-0.6b-gguf` | 0.6B | 24000 Hz | **极速轻量**：极低的启动延迟与资源占用，适合高并发对话场景。 | [Qwen3 TTS Clone 系列能力说明](doc/models/qwen3-tts-clone.md) |
| `qwen3-tts-clone-1.7b-gguf` | 1.7B | 24000 Hz | **质量均衡**：兼顾速度与生成质量，提供优秀的单参考音色克隆表现。 | [Qwen3 TTS Clone 系列能力说明](doc/models/qwen3-tts-clone.md) |
| `s2-pro-4b-gguf` | 4B | 44100 Hz | **高保真演播**：高采样率输出，支持**多参考** prompt，适合有声书、专业配音及复杂情感合成。 | [S2-Pro 系列能力说明](doc/models/s2-pro.md) |
| `voxcpm2-gguf` | 2.38B | 48000 Hz | 支持 **保持原有音色讲方言**、副语言控制以及音色设计 | [VoxCPM2 系列能力说明](doc/models/voxcpm2.md) |
| `moss-tts-nano-onnx` | 0.1B | 48000 Hz | **CPU 实时首选**：纯 ONNX 路线，支持单参考音色克隆，适合低成本部署、边缘设备与无独显机器。 | [MOSS-TTS-Nano 能力说明](doc/models/moss-tts-nano.md) |

### 2.2 选型建议

如果你还不确定该选哪个模型，可以直接按下面的原则判断：

* **先要跑通、先要低成本、先要低延迟**：选 `qwen3-tts-clone-0.6b-gguf`
* **想要一个通用默认款，速度和质量都比较稳**：选 `qwen3-tts-clone-1.7b-gguf`
* **更在意副语言能力，能控制情绪、角色感、多说话人，超强拟人**：选 `s2-pro-4b-gguf`
* **更需要方言克隆（音色保持）、简单副语言控制、音色设计、极强的音色还原**：选 `voxcpm2-gguf`
* **需要在纯 CPU 或低资源机器上优先追求实时生成，且希望保留稳定的单参考音色克隆能力**：选 `moss-tts-nano-onnx`
* **需要查看某个系列的完整能力边界、语言覆盖和工程注意事项**：直接进入对应的模型能力文档

---

## 3. 核心优势

* **开箱即用：** 单一入口 `TTSEngine` 初始化，自动处理运行环境。
* **灵活部署：** 支持云端自动拉取模型，也支持纯内网物理隔离的本地 Bundle 加载。
* **极致克隆：** 支持零样本（Zero-shot）音色克隆，并内置角色 Prompt 本地缓存机制，大幅提升复用效率。
* **CPU 友好：** `MOSS-TTS-Nano` 当前 release 中最适合作为 CPU 实时生成默认选项的模型。
* **硬件自适应：** 智能探测平台硬件，支持 `auto`、`cpu`、`gpu:<index>` 级调度。
* **商业护航：** 完善的在线 License 校验机制，未授权自动降级为 Trial 模式供开发测试。

---

## 4. 安装教程

当前发布包以 Python `3.10` 为主。推荐使用 `uv` 创建环境，并直接安装本仓库中的 `pyproject.toml` 依赖，再安装已经编译好的 StreamVox wheel。

### 4.1 克隆发布仓库

```bash
git clone https://github.com/RoversCode/StreamVox.git streamvox_release_github
cd streamvox_release_github
```

如果你拿到的是私有发布仓库或压缩包，只要进入包含 `pyproject.toml`、`uv.lock` 和 wheel 文件的目录即可。

### 4.2 创建 Python 3.10 环境

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Windows PowerShell 使用：

```powershell
uv venv --python 3.10
.venv\Scripts\Activate.ps1
```

### 4.3 安装项目依赖

仓库中的 `pyproject.toml` 已经声明了运行 StreamVox 所需的主要依赖，包括 PyTorch、ONNX Runtime、音频处理和模型相关依赖。

#### Linux

Linux 默认依赖已经包含 `onnxruntime-gpu`：

```bash
uv sync
```

#### Windows + NVIDIA CUDA

如果你的机器使用 NVIDIA 显卡，并且希望优先走 `CUDAExecutionProvider`：

```powershell
uv sync --extra windows-cuda
```

#### Windows + DirectML

如果你的机器使用 AMD / Intel / 核显，或你希望优先走 `DmlExecutionProvider`：

```powershell
uv sync --extra windows-dml
```

#### Windows + CPU

如果你只打算在 Windows 上使用 CPU：

```powershell
uv sync --extra windows-cpu
```

### 4.4 安装 StreamVox wheel

前往[StreamVox Releases v1.0.0](https://github.com/RoversCode/StreamVox/releases/tag/v1.0.0)下载与你的**操作系统、Python 版本、后端类型**匹配的 wheel：

下载时请重点看清楚以下三项：

* **操作系统/平台**：例如 `win_amd64` 表示 Windows x64，`manylinux_2_28_x86_64` 表示 Linux x64
* **Python 版本**：例如 `cp310`、`cp311`、`cp312`、`cp313` 分别对应 Python `3.10`、`3.11`、`3.12`、`3.13`
* **运行后端**：`+vulkan` 表示 Vulkan 后端，`+cuda124` 表示 CUDA 12.4 后端

wheel 文件名命名规则如下：

```text
streamvox-{version}+{backend}-{python_tag}-{abi_tag}-{platform}.whl
```

例如：

```text
streamvox-1.0.0+cuda124-cp310-cp310-win_amd64.whl
```

含义是：

* `1.0.0`：StreamVox 版本号
* `cuda124`：内置 CUDA 12.4 后端
* `cp310`：要求 Python 3.10
* `win_amd64`：适用于 64 位 Windows

再例如：

```text
streamvox-1.0.0+vulkan-cp312-cp312-manylinux_2_28_x86_64.whl
```

含义是：

* `1.0.0`：StreamVox 版本号
* `vulkan`：内置 Vulkan 后端
* `cp312`：要求 Python 3.12
* `manylinux_2_28_x86_64`：适用于 64 位 Linux

选择建议：

* **Windows 用户**：可根据自己的 Python 版本选择 `win_amd64` 包
* **Linux 用户**：可根据自己的 Python 版本选择 `manylinux_2_28_x86_64` 包
* **NVIDIA 用户且希望使用 CUDA 后端**：选择 `+cuda124` 的 Windows wheel
* **希望优先保证兼容性，或使用 AMD / Intel / 核显 / Linux 通用方案**：优先选择 `+vulkan` wheel

下载完成后，在当前环境中安装对应 wheel，例如：

```bash
uv pip install streamvox-1.0.0+vulkan-cp310-cp310-manylinux_2_28_x86_64.whl
```

Windows + CUDA 12.4 示例：

```bash
uv pip install streamvox-1.0.0+cuda124-cp310-cp310-win_amd64.whl
```

如果 wheel 文件放在 `dist/` 目录下：

```bash
uv pip install dist/streamvox-1.0.0+vulkan-cp310-cp310-manylinux_2_28_x86_64.whl
```

安装完成后验证：

```bash
python -c "from streamvox import TTSEngine; print(TTSEngine)"
```

### 4.5 Liunx后端二进制说明：默认 Vulkan，极限性能可自编译 CUDA

由于 CUDA 版本兼容关系复杂，且 Linux 发行版、驱动版本、系统库和用户环境差异很大，预编译 CUDA 后端很难保证在大多数机器上稳定运行。为了优先保证开箱即用和跨硬件兼容性，当前提供的 StreamVox 预编译 wheel 默认内置的是基于 `llama.cpp Vulkan` 后端编译的运行库。

Vulkan 后端的优势是兼容面更广：NVIDIA、AMD、Intel 独显，以及部分核显/集显环境都更容易跑通。它的取舍是峰值性能通常会低于 CUDA 后端；在同等 NVIDIA 显卡环境下，Vulkan 相比 CUDA 可能会有约 **30%** 的性能差距，具体差异仍以本机驱动、显卡型号、模型尺寸和输入长度的实测结果为准。

如果你使用的是 NVIDIA 显卡，并且希望压榨极限推理性能，可以自行编译 CUDA 版本的 `llama.cpp` 运行库：

1. 前往 [`llama.cpp` b8683 相关 release](https://github.com/ggml-org/llama.cpp/releases?q=b8683&expanded=true) 下载对应源码。
2. 在本机按 `llama.cpp` 官方方式启用 CUDA 后端并完成编译。
3. 找到编译产物中的全部 `.so` 动态库文件。
4. 找到当前pyhton环境下site-packages里的streamvox，并清空 `streamvox/bin` 目录下的旧运行库文件。
5. 将自行编译得到的全部 `.so` 文件复制到 `streamvox/bin` 目录。

替换完成后，StreamVox 就会自动使用基于cuda编译好的llama.cpp去推理了。

### 4.6 准备模型文件

StreamVox 支持传入模型名，也支持传入本地模型目录。生产和离线环境推荐使用本地模型目录。
[模型下载地址](https://modelscope.cn/profile/ChengHee)

## 5. 快速开始

下面示例会完成初始化模型、创建临时音色 prompt、流式合成文本并保存 wav 文件。

```python
import numpy as np
import soundfile as sf

from streamvox import TTSEngine


engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    llama_backend=None,
    verify_model_sha256=False,
)

try:
    # 构建prompt data
    prompt = engine.make_prompt(
        role_name="demo_role",
        audio_path="example/Condition3.wav",
        prompt_text="所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。",
        persist=False,
    )

    # 流式输出
    chunks = engine.stream(
            text="你好，这里是 StreamVox 的快速开始示例。",
            role_name=prompt,
            language="chinese",
            track_performance=False,
    )

    audios = []
    for chunk in chunks;
        # 输出的每一个chunk块
        audios.append(chunk)

    audio = np.concatenate(audios, axis=-1)
    sf.write("output.wav", audio, engine.runtime.sample_rate)
finally:
    engine.shutdown()
```

说明：

- `license_key` 为空或无效时会进入 trial 模式，输出音频会被限制处理。
- 非永久授权会通过本地在线 token 缓存运行，并在 token 到期后的下一次合成前自动回服务器复核；永久授权保持长期 token 行为。
- `verify_model_sha256=False` 可以跳过初始化时的模型文件哈希校验，适合本地测试提速；正式分发场景建议按需要开启。
- `device="auto"` 会由 runtime 根据当前平台和已打包后端自动选择。
- `llama_backend` 只控制 GGUF / `llama.cpp` 后端；不传或传 `None` 时保持自动选择。

## 6. 完整功能文档

请阅读：[StreamVox 完整功能使用文档](doc/usage.md)


## 8. 常见问题

### 1.安装 wheel 后仍然无法导入 `streamvox`

先确认当前 shell 已经激活 `.venv`，再执行：

```bash
which python
python -c "import streamvox; print(streamvox.__file__)"
```

如果导入失败，重新执行：

```bash
uv pip install streamvox-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

### 2.授权失败会不会导致程序不能运行

授权失败会进入 trial 模式。trial 模式仍保持相同的流式接口，但输出音频会经过限制处理，不能作为正式结果使用。非永久授权 token 到期后若无法完成服务器复核，也会进入 trial 模式。

### 3.应该使用模型名还是本地模型目录

开发和试用阶段可以直接传模型名。生产、离线或需要严格控制模型版本时，建议传本地模型 bundle 目录。

### 4.多卡机器应该怎么选 GPU

建议直接使用 `device="gpu:<物理卡编号>"`，例如 `device="gpu:1"`。如果你还想显式固定 GGUF backend，可以额外传 `llama_backend="vulkan"` 或 `llama_backend="cuda"`。GGUF/Vulkan 后端的设备编号不一定遵循 `CUDA_VISIBLE_DEVICES` 的重映射。

补充说明：

- `llama_backend` 只影响 GGUF / `llama.cpp`，不会改变 ONNX decoder 的 CUDA / DML / CPU 选择。
- 如果 `device="cpu"`，显式传入的 `llama_backend` 会被忽略，并打印警告日志。
- 如果显式传入 `llama_backend="cuda"`，但当前宿主机不是 NVIDIA / CUDA 不可用，SDK 会警告并忽略该请求，回到默认自动选择；若 Vulkan 可用，日志会明确说明已默认选择 Vulkan。

### 5.S2-Pro 和 Qwen3 的 Prompt 能混用吗

不能。Prompt 按模型隔离，不同模型的 Prompt 资产和本地缓存不能互相复用。

### 6.生成音频听起来异常

先检查日志里是否提示 trial 模式。如果授权未生效，输出音频会被限制处理。确认授权状态后，再检查参考音频质量、参考文本是否准确、模型选择和采样参数。

## 9. 参考与致谢
- [fish speech](https://github.com/fishaudio/fish-speech)
- [Qwen3 TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Moss TTS](https://github.com/OpenMOSS/MOSS-TTS-Nano)

感谢开源社区对语音合成、模型推理和音频处理基础设施的持续贡献。
