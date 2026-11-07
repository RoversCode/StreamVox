# StreamVox

StreamVox 是一套在本机运行的 Python 语音合成库。程序把目标文本、可选的参考录音或音色描述交给 `TTSEngine`，引擎按所选模型生成音频块，你可以逐块播放、发送，或保存成 WAV 文件。

项目接入 **Qwen3-TTS、S2-Pro、VoxCPM2、MOSS-TTS-Nano 和 Breeze TTS 2**。安装包提供运行代码和计算库，首次按模型名调用时再下载对应权重。下载完成后，语音合成在本机执行。

当前发行版本为 **2.1.2，使用 CPython 3.12**。Windows x64 使用预编译安装包，Linux x86_64 从源码运行。StreamVox 自有代码采用 Apache-2.0，模型权重和第三方组件按各自条款使用。[SDK 许可证](LICENSE) · [模型许可](docs/model-licenses.md)

- [性能与资源占用](#性能与资源占用)
- [支持模型与选型](#支持模型与选型)
- [安装](#安装)
- [Linux 源码运行](#linux-源码运行)
- [快速开始](#快速开始)
- [后端选择](#后端选择)
- [完整使用文档](docs/usage.md)
- [模型许可与来源](docs/model-licenses.md)
- [开发、测试与发行](docs/development.md)
- [接入新模型](docs/model-adapter-guide.md)
- [维护 ModelScope 模型仓库](docs/modelscope-hub-cheatsheet.md)
- [参与贡献](CONTRIBUTING.md)

## 主要特性

- **流式输出** 让调用方在第一块音频到达后开始处理，再通过迭代取得后续部分。
- **统一调用** 用同一个 `TTSEngine` 管理模型加载、参考录音、生成和关闭。声音设计、克隆、续写、多说话人及表现控制按各模型实际能力开放。
- **语言选择** 由模型负责。可选择覆盖中文、英语、日语、韩语、法语、德语、西班牙语、俄语等语言的模型，具体名单和语言参数见各自指南。[模型选型](#支持模型与选型)
- **计算后端选择** 让同一套调用使用 CPU 或 GPU。后端是执行模型运算的运行库，llama.cpp 读取 GGUF 格式的模型，ONNX Runtime 执行 ONNX 计算图。MOSS 使用纯 ONNX 路径，其他家族使用两者组合。
- **独立运行依赖** 使用 llama.cpp 和 ONNX Runtime 完成正常合成。需要 PyTorch 的参考导出与数值对照在单独开发环境运行。
- **模型下载与离线使用** 按模型名下载资产、核对清单，后续复用本地目录。已准备好完整模型目录时，也可直接传目录路径运行。
- **参考音色复用** 把参考录音先编码成参考音色资产（Prompt），之后多次生成复用这一结果，减少重复的参考处理。

## 性能与资源占用

### 流式推理速度

下表汇总 **Windows 11、NVIDIA GeForce RTX 4090 24 GB、Intel Core i7-13700KF** 上已有的单请求实测，逐行标注实测 SDK 版本。每种模式预热 1 次后测量 3 次，取各指标的中位数。

表中的 Q8_0 表示把大部分生成权重保存成 8 位近似值，以减小权重体积，同时会引入量化误差。Breeze 的完整数值对照和任务验收范围见模型指南。[Breeze 精度与验证](docs/models/breeze-tts-2.md#默认-q8-模型与自动下载)

| 模型与模式 | 实测 SDK | 合成后端 | 首包延迟 | RTF | 相对实时速度 |
| --- | --- | --- | ---: | ---: | ---: |
| Breeze TTS 2 Q8_0，参考克隆 | 2.1.1 源码 | CUDA | **105.1 ms** | 0.345 | 2.90× |
| Breeze TTS 2 Q8_0，声音设计 | 2.1.1 源码 | CUDA | **156.3 ms** | 0.380 | 2.63× |
| Qwen3-TTS 0.6B | 2.0.0 | CUDA | **186 ms** | 0.148 | 6.76× |
| Qwen3-TTS 1.7B | 2.0.0 | CUDA | **187 ms** | 0.159 | 6.29× |
| S2-Pro | 2.0.0 | CUDA | **401 ms** | 0.347 | 2.88× |
| VoxCPM2 | 2.0.0 | CUDA | **113 ms** | 0.344 | 2.91× |
| MOSS-TTS-Nano | 2.0.0 | ONNX CUDA | **83 ms** | 0.234 | 4.27× |
| MOSS-TTS-Nano | 2.0.0 | ONNX CPU | **90 ms** | 0.262 | 3.81× |

首包延迟衡量开始合成后多久取得第一块波形。表中从开始迭代 `stream()` 到收到首块计时，模型加载、参考准备和预热分别进行。首包描述等待时间，每块音频的时长描述这块波形能播放多久。

RTF 是完整合成耗时除以生成音频时长，越小越快。例如 RTF 为 0.2，生成 1 秒音频约需 0.2 秒，对应约 5 倍实时速度。表中的相对实时速度使用未舍入的原始 RTF 计算。

CUDA 行使用 CUDA 合成后端，部分参考处理仍按模型设计使用 CPU。Breeze TTS 2 使用 Q8_0 GGUF 主干与深度解码器，其余 ONNX 组件使用 FP32，也就是 32 位浮点数。文本主干和声码器选择 CUDA，默认每块输出 **160 ms** 音频。完整初始化与校验耗时 11.60 秒，克隆参考准备耗时 0.707 秒，均单独计时。[Breeze 实测记录](docs/models/breeze-tts-2-validation.json) · [测量条件](docs/models/breeze-tts-2.md#当前-q8-性能)

Breeze 的设计与克隆使用同一句固定中文文本，克隆参考为本轮设计生成的 5.60 秒音频。其余六行沿用 SDK 2.0.0 的同一段参考音频和另一句固定中文文本，MOSS 两行使用相同的 `seed=42`。输入、参考长度和音频块长度按各自记录执行，实际表现随这些条件及硬件变化。[SDK 2.0.0 实测记录](docs/performance-2.0.0.json)

### 硬件资源建议

| 模型 | 显存起步 / 推荐 | 内存起步 / 推荐 |
| --- | ---: | ---: |
| Breeze TTS 2 Q8_0 | 起步待实测 / 12 GB | 起步待实测 / 16 GB |
| Qwen3-TTS 0.6B | 2 / 3 GB | 3 / 5 GB |
| Qwen3-TTS 1.7B | 3 / 4 GB | 4 / 6 GB |
| S2-Pro | 7 / 8 GB | 6 / 8 GB |
| VoxCPM2 | 7 / 8 GB | 8 / 12 GB |
| MOSS-TTS-Nano | CPU 运行，或 GPU 推荐 2 GB | 3 / 5 GB |

起步列来自当前硬件档位提示，推荐列为预留资源建议，显存列针对 GPU 运行。更长的文本、参考录音和其他程序占用会改变实际需求。VoxCPM2 的生成模型采用 GGUF Q8，音频编码与生成过程另用 FP32 和 FP16 组件，具体分工见模型指南。[VoxCPM2](docs/models/voxcpm2.md)

Breeze TTS 2 默认 Q8_0 权重在 CUDA 文本配置下，实测整卡显存峰值相对加载前冷基线增加 **6130 MiB，约 5.99 GiB**，进程工作集峰值约 6308 MiB。表中建议为该配置预留 12 GB 显存与 16 GB 内存，最低容量待分别实测。资源测量口径与适用范围见模型指南。[Breeze 指南](docs/models/breeze-tts-2.md#当前-q8-性能)

## 支持模型与选型

| 模型名 | 采样率 | 主要能力与适用场景 | 指南 |
| --- | ---: | --- | --- |
| `qwen3-tts-clone-0.6b-gguf` | 24000 Hz | 单参考音色克隆，适合优先考虑资源占用的场景 | [Qwen3](docs/models/qwen3-tts-clone.md) |
| `qwen3-tts-clone-1.7b-gguf` | 24000 Hz | 单参考音色克隆、多语言生成，适合通用语音应用 | [Qwen3](docs/models/qwen3-tts-clone.md) |
| `s2-pro-4b-gguf` | 44100 Hz | 多参考、多说话人和表现控制，适合角色对白与配音 | [S2-Pro](docs/models/s2-pro.md) |
| `voxcpm2-gguf` | 48000 Hz | 音色设计、参考克隆和续写，适合需要声音控制的场景 | [VoxCPM2](docs/models/voxcpm2.md) |
| `moss-tts-nano-onnx` | 48000 Hz | 轻量 ONNX 路线，适合无独显机器与 CPU 部署 | [MOSS](docs/models/moss-tts-nano.md) |
| `breeze-tts-2-gguf` | 24000 Hz | GGUF+ONNX，中英文克隆、声音设计与表现控制。默认从 ModelScope 下载 Q8_0，权重限研究与非商业使用 | [Breeze](docs/models/breeze-tts-2.md) · [ModelScope](https://www.modelscope.cn/models/ChengHee/Breeze-TTS-2-GGUF) |

没有独立显卡时，可从 MOSS-TTS-Nano 的 CPU 路径入手。已有一段参考录音、希望复用其音色时，可选择 Qwen3-TTS。要用文字设计声音或基于参考录音续写，可查看 VoxCPM2。中英文声音设计与克隆可使用 Breeze TTS 2，多说话人参考组织可使用 S2-Pro。

S2-Pro 权重采用 Fish Audio Research License，商业用途需要 Fish Audio 单独书面许可。Breeze TTS 2 权重、衍生模型及自托管输出限研究与非商业使用。Qwen3-TTS、VoxCPM2 和 MOSS-TTS-Nano 当前默认转换权重采用 Apache-2.0。使用与再分发按所选权重版本的完整条款执行。[模型许可表](docs/model-licenses.md)

## 安装

### Windows 发行包

uv 管理 Python 环境和依赖。先安装 uv，再创建一个只供本项目使用的 `.venv` 环境，随后安装的包会放在这个环境里。[uv 安装说明](https://docs.astral.sh/uv/)

```powershell
uv venv --python 3.12 --managed-python
.venv\Scripts\Activate.ps1
```

从发行页下载 `streamvox-2.1.2-cp312-none-win_amd64.whl`。wheel 是 Python 安装包，文件名中的 `cp312` 和 `win_amd64` 分别标明 CPython 3.12 与 Windows x64。将文件放在当前工作目录，再按硬件选择一行命令。[GitHub Releases v2.1.2](https://github.com/RoversCode/StreamVox/releases/tag/v2.1.2)

| 场景 | 安装命令 |
| --- | --- |
| CPU | `uv pip install "./streamvox-2.1.2-cp312-none-win_amd64.whl[cpu]"` |
| AMD / Intel / 核显，或使用 DirectML | `uv pip install "./streamvox-2.1.2-cp312-none-win_amd64.whl[directml]"` |
| NVIDIA 的 ONNX 与 llama 都使用 CUDA | `uv pip install "./streamvox-2.1.2-cp312-none-win_amd64.whl[onnx-cuda,llama-cuda]"` |
| ONNX 使用 DirectML，llama 使用 CUDA | `uv pip install "./streamvox-2.1.2-cp312-none-win_amd64.whl[directml,llama-cuda]"` |

方括号里的名称是可选依赖组。`cpu`、`directml`、`onnx-cuda` 分别安装不同的 ONNX Runtime 发行包，它们提供同一个 Python 模块，因此一个环境选择其中一组。`llama-cuda` 补充 llama.cpp 所需的 NVIDIA 运行依赖，可与任一 ONNX 组组合。CUDA 和 Vulkan 还依赖操作系统中已经安装的显卡驱动。

运行下面的命令确认当前解释器能导入安装包，正常时会输出版本号。

```powershell
python -c "import streamvox; print(streamvox.__version__)"
```

### Linux 源码运行

以下以 Ubuntu 22.04 及以上的 x86_64 CPU 环境为例。Linux 运行仓库内的 Python 源码，并从同版本 llama.cpp 取得 `.so` 共享库。依赖仍装入独立虚拟环境。当前项目的 wheel 构建面向 Windows，因此同步时使用 `--no-install-project`，让 uv 安装依赖并保留从源码导入的方式。

**1. 准备系统依赖与 Python 环境**

先安装 uv，再执行以下命令。系统依赖提供音频文件读取和 C/C++ 运行库，`uv sync` 根据项目锁文件准备 Python 依赖。[uv 安装](https://docs.astral.sh/uv/getting-started/installation/)

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates libsndfile1 libgomp1 libstdc++6

git clone https://github.com/RoversCode/StreamVox.git
cd StreamVox
uv sync --locked --python 3.12 --managed-python --extra cpu --group dev --no-install-project
source .venv/bin/activate
python -c "import streamvox; print(streamvox.__version__)"
```

**2. 下载 GGUF 模型使用的预编译动态库**

Qwen3-TTS、S2-Pro、VoxCPM2 和 Breeze TTS 2 需要 llama.cpp 动态库，MOSS-TTS-Nano 使用 ONNX Runtime。动态库负责执行 Python 调用的原生计算，库内接口需要与项目绑定一致。下面使用固定的 **b8683** CPU 归档，同一发布页还提供 Vulkan 版本。[llama.cpp b8683](https://github.com/ggml-org/llama.cpp/releases/tag/b8683)

```bash
mkdir -p .artifacts/llama-cpu-b8683 streamvox/bin
curl -fL --retry 3 \
  https://github.com/ggml-org/llama.cpp/releases/download/b8683/llama-b8683-bin-ubuntu-x64.tar.gz \
  -o .artifacts/llama-b8683-bin-ubuntu-x64.tar.gz
tar -xzf .artifacts/llama-b8683-bin-ubuntu-x64.tar.gz -C .artifacts/llama-cpu-b8683
cp -a .artifacts/llama-cpu-b8683/llama-b8683/lib*.so* streamvox/bin/
export LD_LIBRARY_PATH="$PWD/streamvox/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

复制全部共享库并保留软链接，包括 `libllama.so*`、`libggml.so*`、`libggml-base.so*` 和 `libggml-cpu-*.so` 等后端库，保持它们来自同一个归档。CPU 预编译包要求 glibc 至少为 2.34，且系统 C++ 运行库提供 `GLIBCXX_3.4.29`。每次打开新终端，进入仓库根目录后重新激活环境并设置上述 `LD_LIBRARY_PATH`。`streamvox.native_assets` 下载的是 Windows DLL，Linux 使用上面的预编译 `.so` 下载方式。

**3. 从源码直接运行**

在仓库根目录运行模块入口，Python 就会从当前仓库找到 `streamvox`。先查看帮助，再把参考路径与转写替换为自己的素材执行克隆。

```bash
python -m example.synthesize --help
python -m example.synthesize \
  --model qwen3-tts-clone-0.6b-gguf \
  --reference-audio /path/to/reference.wav \
  --reference-text "参考音频的实际内容。" \
  --text "你好，欢迎使用 StreamVox。" \
  --device cpu --llama-backend cpu --onnx-provider cpu \
  --output output.wav
```

首次按模型名运行会下载权重。离线使用时，将 `--model` 换成已准备好的模型目录（bundle），目录中包括清单及它声明的资产。上面的单参考命令也可使用 MOSS 的注册名，具体模式见其指南。通过 `uv run` 执行时加 `--no-sync`，继续使用已准备好的环境。[MOSS 指南](docs/models/moss-tts-nano.md) · [测试与发行验证](docs/development.md)

需要 GPU 时，按下表选择运行库和依赖组。切换原生库时先备份旧目录，再把选定归档的整套库复制到新的 `streamvox/bin/`，使公共库和计算后端保持同一版本。

| 场景 | 环境与构建调整 | 运行参数 |
| --- | --- | --- |
| llama Vulkan，ONNX CPU | 保留 `--extra cpu`，下载同版 Linux Vulkan 预编译包，将全部 `lib*.so*` 放到 `streamvox/bin/`。glibc 至少为 2.34，C++ 运行库提供 `GLIBCXX_3.4.30`，系统安装 `libvulkan1` 和驱动。[预编译包](https://github.com/ggml-org/llama.cpp/releases/download/b8683/llama-b8683-bin-ubuntu-vulkan-x64.tar.gz) | `--device gpu:0 --llama-backend vulkan --onnx-provider cpu` |
| llama CUDA，ONNX CPU | 同步依赖时额外加 `--extra llama-cuda`，再准备匹配 b8683 的 Linux CUDA 运行库，也可按可选步骤构建。[编译步骤](docs/development.md#linux-原生库自编译可选) | `--device gpu:0 --llama-backend cuda --onnx-provider cpu` |
| llama 与 ONNX 均使用 CUDA | 将 `--extra cpu` 换成 `--extra onnx-cuda --extra llama-cuda`，再准备同版 Linux CUDA 运行库。 | `--device gpu:0 --llama-backend cuda --onnx-provider cuda` |

Vulkan 包包含 `libggml-vulkan.so`，CUDA 包包含 `libggml-cuda.so`，两者各自携带匹配的公共库与 CPU 后端。b8683 官方发布页提供 Linux CPU 和 Vulkan 包。需要 CUDA、定制构建或系统兼容性调整时，可从同一提交自行编译。[原生库自编译说明](docs/development.md#linux-原生库自编译可选)

CUDA 运行库默认从当前虚拟环境的 NVIDIA 依赖中发现。自行提供这些库时，通过 `STREAMVOX_NVIDIA_LIB_DIRS` 指定目录，Linux 的多个目录用冒号分隔。DirectML 用于 Windows。现有性能数据来自 Windows，Linux 的完整模型合成与 GPU 组合按目标环境分别验证。

## 快速开始

### Qwen3-TTS 参考音色克隆

将下面代码保存为 `qwen_clone.py`，把参考录音放在运行时的工作目录，填写与录音一致的 `prompt_text`，再执行 `python qwen_clone.py`。录音由你自行准备并确认使用权。

```python
import numpy as np
import soundfile as sf

from streamvox import TTSEngine


if __name__ == "__main__":
    engine = TTSEngine(
        model="qwen3-tts-clone-0.6b-gguf",
        device="cpu",
        llama_backend="cpu",
        onnx_provider="cpu",
    )
    try:
        prompt = engine.make_prompt(
            role_name="demo",
            audio_path="your-reference.wav",
            prompt_text="请替换为参考音频中实际说出的内容。",
            persist=False,
        )
        chunks = list(engine.stream(
            text="你好，欢迎使用 StreamVox。",
            role_name=prompt,
            language="chinese",
        ))
        if not chunks:
            raise RuntimeError("本次合成没有生成音频。")
        audio = np.concatenate(chunks, axis=-1)
        sf.write("output.wav", audio, engine.runtime.sample_rate)
    finally:
        engine.shutdown()
```

命令行示例作为独立脚本随源码提供。使用 Windows wheel 安装环境时，在源码仓库之外的工作目录下载该脚本，再使用当前环境的解释器运行。参考路径和转写同样需要换成自己的素材。[命令行示例](example/synthesize.py)

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/RoversCode/StreamVox/v2.1.2/example/synthesize.py -OutFile synthesize.py
python synthesize.py --model qwen3-tts-clone-0.6b-gguf --reference-audio C:/audio/reference.wav --reference-text "参考音频的实际内容。" --text "你好，欢迎使用 StreamVox。" --output output.wav
```

`stream()` 的每个返回块都是 `float32` NumPy 数组，里面保存 PCM 波形采样值。`engine.runtime.sample_rate` 说明每秒有多少个采样点，保存或播放时使用这个值。上例为写 WAV 收集了全部块，实时接入可以在循环中处理每个块。

`TTSEngine` 除 `model` 外的初始化参数按名称传入。将 `model` 换成完整的本地模型目录，可以直接加载已经准备好的资产。参考音色缓存、长音频逐块写入和不同模型的参数说明见使用文档。[使用文档](docs/usage.md)

### Breeze TTS 2 声音设计与自动下载

安装 NVIDIA CUDA 依赖后，可通过自然语言描述生成音色。首次使用 `model="breeze-tts-2-gguf"` 时，SDK 从 `ChengHee/Breeze-TTS-2-GGUF` 自动下载并校验完整 Q8_0 模型，后续复用本地缓存。Windows 默认模型目录为 `%APPDATA%/StreamVox/ckpts`，可通过 `STREAMVOX_HOME` 指定数据根目录。[ModelScope 模型](https://www.modelscope.cn/models/ChengHee/Breeze-TTS-2-GGUF)

将下面代码保存为 `breeze_design.py`，在已准备好的环境中运行 `python breeze_design.py`，会得到 `breeze-output.wav`。

```python
import numpy as np
import soundfile as sf

from streamvox import TTSEngine


if __name__ == "__main__":
    engine = TTSEngine(
        model="breeze-tts-2-gguf",
        device="gpu:0",
        llama_backend="cuda",
        onnx_provider="cuda",
    )
    try:
        chunks = list(engine.stream(
            text="你好，欢迎使用语音合成。",
            instruction="一位声音清晰、语气自然的年轻女性。",
            cfg_scale=4.0,
            seed=42,
        ))
        if not chunks:
            raise RuntimeError("本次合成没有生成音频。")
        sf.write("breeze-output.wav", np.concatenate(chunks), engine.runtime.sample_rate)
    finally:
        engine.shutdown()
```

参考音色克隆、CPU 运行、采样参数与组件分工见模型指南。Q8_0 的有损数值超限、内容复核和试听记录随模型提供。[Breeze TTS 2 指南](docs/models/breeze-tts-2.md)

## 后端选择

`device` 给出 CPU 或 GPU 以及默认设备编号。`llama_backend` 选择 GGUF 生成部分的运行库，`onnx_provider` 选择 ONNX 图的执行提供程序。两个部分可以采用不同后端。例如，在安装 `[directml,llama-cuda]` 的 NVIDIA 环境中，可采用如下组合。

```python
from streamvox import TTSEngine


if __name__ == "__main__":
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

显式选择的后端无法按要求建立时，引擎会抛出异常。`engine.actual_backends` 提供模型实际选择及可见的 ONNX 会话配置，多卡编号和分阶段 CPU/GPU 组合见设备说明。[设备与实际后端](docs/usage.md#设备与实际后端)

## 使用建议

- 频繁复用的音色先调用 `make_prompt()`，将参考音频处理与正式合成分开，减少请求等待。
- 对数字、日期、金额和缩写有明确读法要求时，先改写成期望朗读的词语。长文本可以按句或短段组织。
- 音色还原和表现力取决于模型、参考音频与输入内容。出现吞字或异常停顿时，可先检查参考转写、调整断句，或更换清晰的参考音频。
- 遇到安装或运行问题，可先查阅 [故障排查](docs/usage.md#故障排查)，再通过 [GitHub Issues](https://github.com/RoversCode/StreamVox/issues) 提供版本、模型、后端和可复现示例。

## 许可证与致谢

StreamVox 自有代码采用 Apache-2.0，原生库和模型权重分别按其对应条款使用。完整许可及归属材料见项目说明。[LICENSE](LICENSE) · [NOTICE](NOTICE) · [第三方声明](THIRD_PARTY_NOTICES.md)

感谢 [llama.cpp](https://github.com/ggml-org/llama.cpp)、[ONNX Runtime](https://github.com/microsoft/onnxruntime)、[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[Fish Speech](https://github.com/fishaudio/fish-speech)、[VoxCPM](https://github.com/OpenBMB/VoxCPM)、[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 和 [Breeze TTS](https://github.com/breezeblue-ai/breeze-tts) 及其贡献者。

## 支持开发者

如果 StreamVox 对你有帮助，欢迎给开发者一点鼓励。无论是一个 Star，还是一份随心的赞赏，都非常感谢。赞赏完全自愿，不影响 SDK 的任何功能。

<img src="docs/assets/wechat-support.png" alt="微信赞赏码" width="300">
