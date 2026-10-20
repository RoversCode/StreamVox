# StreamVox

StreamVox 是一个开源的本地流式语音合成 SDK，用统一的 `TTSEngine` 接入 **Qwen3-TTS、S2-Pro、VoxCPM2 和 MOSS-TTS-Nano**，适用于语音助手、实时对话、配音和离线播报。音频按块生成，可以边合成边播放，无需等待整段完成。

**当前版本：2.0.0 · CPython 3.12**。预编译发行包面向 Windows x64；Linux x86_64 的环境准备与直接运行方式见 [Linux 源码运行](#linux-源码运行)。SDK 全部功能免费提供，采用 [Apache-2.0](LICENSE)；模型权重和第三方组件遵循各自许可证。

- [性能与资源占用](#性能与资源占用)
- [支持模型与选型](#支持模型与选型)
- [安装](#安装)
- [Linux 源码运行](#linux-源码运行)
- [快速开始](#快速开始)
- [后端选择](#后端选择)
- [完整使用文档](docs/usage.md)
- [模型许可与来源](docs/model-licenses.md)

## 主要特性

- **流式合成**：逐块输出音频，便于接入播放器、语音助手和实时通信。
- **多模型、统一接口**：支持参考音色克隆，并按模型提供音色设计、续写、多说话人和情绪控制等能力。
- **多语言支持**：支持中文、英语、日语、韩语、法语、德语、西班牙语、俄语等主流语言；具体语言覆盖因模型而异，详情与使用方式见 [Qwen3-TTS](docs/models/qwen3-tts-clone.md)、[S2-Pro](docs/models/s2-pro.md)、[VoxCPM2](docs/models/voxcpm2.md) 和 [MOSS-TTS-Nano](docs/models/moss-tts-nano.md) 的模型说明。
- **多种硬件后端**：支持 CPU，以及 NVIDIA、AMD、Intel 显卡；同一个安装包包含 llama.cpp 的 CPU、CUDA、Vulkan 后端，ONNX 组件可选择 CPU、CUDA 或 DirectML。
- **轻量运行环境**：使用 llama.cpp 与 ONNX Runtime，运行依赖无需安装 PyTorch。
- **本地与离线部署**：内置模型下载和文件校验；模型文件完整时，可以直接从本地目录离线合成。
- **参考音色复用**：参考音频可预先编码为 Prompt，并按模型缓存，减少重复处理。

## 性能与资源占用

### 流式推理速度

以下为 **SDK 2.0.0** 在 **Windows 11、NVIDIA GeForce RTX 4090 24 GB、Intel Core i7-13700KF** 上的单请求实测。各模型使用同一段参考音频和固定中文短句进行音色克隆，预热 1 次后测量 3 次，表中取各指标的中位数。

| 模型 | 合成后端 | 首包延迟 | RTF | 相对实时速度 |
| --- | --- | ---: | ---: | ---: |
| Qwen3-TTS 0.6B | CUDA | **186 ms** | 0.148 | 6.76× |
| Qwen3-TTS 1.7B | CUDA | **187 ms** | 0.159 | 6.29× |
| S2-Pro | CUDA | **401 ms** | 0.347 | 2.88× |
| VoxCPM2 | CUDA | **113 ms** | 0.344 | 2.91× |
| MOSS-TTS-Nano | ONNX CUDA | **83 ms** | 0.234 | 4.27× |
| MOSS-TTS-Nano | ONNX CPU | **90 ms** | 0.262 | 3.81× |

首包延迟从发起 `stream()` 到收到第一个音频片段计时，不包含模型加载、参考音频编码和预热。**RTF = 完整合成耗时 / 生成音频时长**，越小越快；例如 RTF 为 0.2，表示生成 1 秒音频约需 0.2 秒，即约 5 倍实时速度。

CUDA 行使用 CUDA 合成后端，部分模型的参考编码组件仍按模型设计使用 CPU。MOSS 两行使用相同的 `seed=42`。实际表现会随输入文本、参考音频、后端和硬件变化，完整条件与逐次数据见 [性能实测记录](docs/performance-2.0.0.json)。

### 硬件资源建议

| 模型 | 显存起步 / 推荐 | 内存起步 / 推荐 |
| --- | ---: | ---: |
| Qwen3-TTS 0.6B | 2 / 3 GB | 3 / 5 GB |
| Qwen3-TTS 1.7B | 3 / 4 GB | 4 / 6 GB |
| S2-Pro | 7 / 8 GB | 6 / 8 GB |
| VoxCPM2 | 7 / 8 GB | 8 / 12 GB |
| MOSS-TTS-Nano | CPU 无需显存；GPU 推荐 2 GB | 3 / 5 GB |

以上为当前模型配置的经验建议，显存列针对 GPU 运行。长文本、较长参考音频及其他程序的资源占用会影响实际需求。VoxCPM2 当前使用 GGUF Q8、LocEnc / LocDiT FP32 和 AudioVAE FP16 组合。

## 支持模型与选型

| 模型名 | 采样率 | 主要能力与适用场景 | 指南 |
| --- | ---: | --- | --- |
| `qwen3-tts-clone-0.6b-gguf` | 24000 Hz | 单参考音色克隆，适合优先考虑资源占用的场景 | [Qwen3](docs/models/qwen3-tts-clone.md) |
| `qwen3-tts-clone-1.7b-gguf` | 24000 Hz | 单参考音色克隆、多语言生成，适合通用语音应用 | [Qwen3](docs/models/qwen3-tts-clone.md) |
| `s2-pro-4b-gguf` | 44100 Hz | 多参考、多说话人和表现控制，适合角色对白与配音 | [S2-Pro](docs/models/s2-pro.md) |
| `voxcpm2-gguf` | 48000 Hz | 音色设计、参考克隆和续写，适合需要声音控制的场景 | [VoxCPM2](docs/models/voxcpm2.md) |
| `moss-tts-nano-onnx` | 48000 Hz | 轻量 ONNX 路线，适合无独显机器与 CPU 部署 | [MOSS](docs/models/moss-tts-nano.md) |

没有独立显卡时，可以先尝试 **MOSS-TTS-Nano**；需要参考音色克隆时，可从 **Qwen3-TTS** 入手；需要音色设计与续写时选择 **VoxCPM2**；多说话人和表现控制需求可以考虑 **S2-Pro**。

SDK 包不包含模型权重。**S2-Pro 转换权重采用 Fish Audio Research License，商业用途需要 Fish Audio 单独书面许可**；其余三类当前默认转换权重采用 Apache-2.0。模型能力、来源和许可边界见对应指南及 [模型许可表](docs/model-licenses.md)。

## 安装

### Windows 发行包

先安装 [uv](https://docs.astral.sh/uv/)，创建独立环境：

```powershell
uv venv --python 3.12 --managed-python
.venv\Scripts\Activate.ps1
```

从 [GitHub Releases v2.0.0](https://github.com/RoversCode/StreamVox/releases/tag/v2.0.0) 下载 `streamvox-2.0.0-cp312-none-win_amd64.whl`，按硬件选择安装命令：

| 场景 | 安装命令 |
| --- | --- |
| CPU | `uv pip install "./streamvox-2.0.0-cp312-none-win_amd64.whl[cpu]"` |
| AMD / Intel / 核显，或使用 DirectML | `uv pip install "./streamvox-2.0.0-cp312-none-win_amd64.whl[directml]"` |
| NVIDIA：ONNX 与 llama 都使用 CUDA | `uv pip install "./streamvox-2.0.0-cp312-none-win_amd64.whl[onnx-cuda,llama-cuda]"` |
| ONNX 使用 DirectML，llama 使用 CUDA | `uv pip install "./streamvox-2.0.0-cp312-none-win_amd64.whl[directml,llama-cuda]"` |

`cpu`、`directml`、`onnx-cuda` 三者互斥，同一环境只安装一种 ONNX Runtime。`llama-cuda` 可以与其中任意一组组合，仅补充 llama CUDA 所需的 NVIDIA 运行依赖。CUDA 和 Vulkan 仍需要宿主机安装相应显卡驱动。

检查安装：

```powershell
python -c "import streamvox; print(streamvox.__version__)"
```

### Linux 源码运行

以下以 Ubuntu 22.04 及以上的 x86_64 CPU 环境为例，其他 Linux 发行版需满足预编译库的系统依赖。直接运行仓库中的 Python 源码，依赖安装到独立虚拟环境；当前 `setup.py` 的 wheel 构建只接受 Windows，因此 Linux 使用 `--no-install-project` 跳过项目自身安装，无需执行 `pip install .`。

**1. 准备系统依赖与 Python 环境**

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再执行：

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

Qwen3-TTS、S2-Pro 和 VoxCPM2 需要 llama.cpp 动态库；仅使用 MOSS-TTS-Nano 时可以跳过。默认从 [llama.cpp b8683 发布页](https://github.com/ggml-org/llama.cpp/releases/tag/b8683) 下载已经编译好的 Linux 库，无需本机编译。CPU 与 Vulkan 包均可从该页面取得，版本应与项目固定的 **b8683** 保持一致。

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

在仓库根目录运行模块入口，使 Python 能找到本地 `streamvox` 源码：

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

将参考音频路径与转写替换为自己的素材。首次使用模型名会下载权重；离线运行时，将 `--model` 替换为已准备好的本地 bundle 目录。使用 MOSS 时，将模型名换成 `moss-tts-nano-onnx` 即可。若使用 `uv run` 执行命令，需加 `--no-sync`，避免自动安装项目并触发 Windows wheel 构建。测试与发行验证见 [开发文档](docs/development.md)。

需要 GPU 时，可按后端选择运行库。切换时先备份并移出旧的 `streamvox/bin/`，再将选定归档的整套库复制到新目录：

| 场景 | 环境与构建调整 | 运行参数 |
| --- | --- | --- |
| llama Vulkan，ONNX CPU | 保留 `--extra cpu`；下载 [Linux Vulkan 预编译包](https://github.com/ggml-org/llama.cpp/releases/download/b8683/llama-b8683-bin-ubuntu-vulkan-x64.tar.gz)，将其中全部 `lib*.so*` 放到 `streamvox/bin/`；glibc 至少为 2.34，C++ 运行库需提供 `GLIBCXX_3.4.30`，并安装 `libvulkan1` 与显卡驱动 | `--device gpu:0 --llama-backend vulkan --onnx-provider cpu` |
| llama CUDA，ONNX CPU | 同步依赖时额外加 `--extra llama-cuda`；准备匹配 b8683 的 Linux CUDA 运行库，或按 [可选编译步骤](docs/development.md#linux-原生库自编译可选) 构建 | `--device gpu:0 --llama-backend cuda --onnx-provider cpu` |
| llama 与 ONNX 均使用 CUDA | 将 `--extra cpu` 换成 `--extra onnx-cuda --extra llama-cuda`；同样准备 Linux CUDA 运行库 | `--device gpu:0 --llama-backend cuda --onnx-provider cuda` |

Vulkan 包需要包含 `libggml-vulkan.so`，CUDA 包需要包含 `libggml-cuda.so`，并各自携带匹配的公共库与 CPU 后端。b8683 官方发布页提供 Linux CPU 和 Vulkan 包；需要 CUDA、定制构建或预编译包与系统不兼容时，可使用 [原生库自编译说明](docs/development.md#linux-原生库自编译可选)。CUDA 运行库默认从当前虚拟环境的 NVIDIA 依赖中发现；自行提供时，通过 `STREAMVOX_NVIDIA_LIB_DIRS` 指定所在目录，多个目录用冒号分隔。DirectML 仅用于 Windows。现有性能记录来自 Windows，Linux 完整模型合成与 GPU 组合仍需在目标环境验证。

## 快速开始

以下示例使用你自行准备、并有权使用的参考音频。`prompt_text` 应与音频内容一致。

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

可运行的命令行示例见 [example/synthesize.py](example/synthesize.py)：

```powershell
python example/synthesize.py --model qwen3-tts-clone-0.6b-gguf --reference-audio C:/audio/reference.wav --reference-text "参考音频的实际内容。" --text "你好，欢迎使用 StreamVox。" --output output.wav
```

`stream()` 逐块返回 `float32` 的 `numpy.ndarray`，采样率从 `engine.runtime.sample_rate` 读取。上面的示例为方便保存 WAV 而汇总所有片段；接入实时播放时，可以在迭代过程中逐块播放或发送音频。

`TTSEngine` 除 `model` 外的初始化参数必须按名称传入。模型名可替换为已下载模型的目录；离线部署建议使用本地目录并保留默认哈希校验。Prompt 缓存、长音频写入与更多调用方式见 [使用文档](docs/usage.md)。

## 后端选择

`device` 表达 CPU/GPU 和设备编号；`llama_backend` 与 `onnx_provider` 分别选择 GGUF 和 ONNX 后端。例如，在安装 `[directml,llama-cuda]` 的 NVIDIA 环境中：

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

显式请求的后端不可用时会抛出异常。可以用 `engine.actual_backends` 查看实际选择；多卡配置与各模型的后端说明见 [设备说明](docs/usage.md#设备与实际后端)。

## 使用建议

- 频繁复用的音色先调用 `make_prompt()`，将参考音频处理与正式合成分开，减少请求等待。
- 对数字、日期、金额和缩写有明确读法要求时，先改写成自然语言；长文本可以按句或短段组织。
- 音色还原和表现力取决于模型、参考音频与输入内容。出现吞字或异常停顿时，可先检查参考转写、调整断句，或更换清晰的参考音频。
- 遇到安装或运行问题，可先查阅 [故障排查](docs/usage.md#故障排查)，再通过 [GitHub Issues](https://github.com/RoversCode/StreamVox/issues) 提供版本、模型、后端和可复现示例。

## 许可证与致谢

[LICENSE](LICENSE) 授予 StreamVox 自有代码的 Apache-2.0 权利。[NOTICE](NOTICE) 与 [第三方声明](THIRD_PARTY_NOTICES.md) 保留第三方权利边界；随包原生库和模型权重的许可不会因 SDK 开源而改变。

感谢 [llama.cpp](https://github.com/ggml-org/llama.cpp)、[ONNX Runtime](https://github.com/microsoft/onnxruntime)、[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[Fish Speech](https://github.com/fishaudio/fish-speech)、[VoxCPM](https://github.com/OpenBMB/VoxCPM) 和 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 及其贡献者。

## 支持开发者

如果 StreamVox 对你有帮助，欢迎给开发者一点鼓励。无论是一个 Star，还是一份随心的赞赏，都非常感谢。赞赏完全自愿，不影响 SDK 的任何功能。

<img src="docs/assets/wechat-support.png" alt="微信赞赏码" width="300">
