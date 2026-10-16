# StreamVox

StreamVox 是一个开源的本地流式语音合成 SDK，用统一的 `TTSEngine` 接入 Qwen3-TTS、S2-Pro、VoxCPM2 和 MOSS-TTS-Nano。它提供模型下载、本地 bundle 校验、音色 Prompt 缓存、设备选择与逐块音频输出。

**2.0.0** 首个开源版本面向 **Windows x64 / CPython 3.12**。Python 源码可读，SDK 全部功能免费提供，不需要产品激活。StreamVox 自有代码采用 [Apache-2.0](LICENSE)，模型权重和第三方组件遵循各自许可证。

- [安装与快速开始](#安装)
- [完整使用文档](docs/usage.md)
- [模型许可与来源](docs/model-licenses.md)
- [模型适配指南](docs/model-adapter-guide.md)
- [构建与验证](docs/development.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)

## 能力与发行范围

- 逐块返回 `numpy.ndarray`，数据类型为 `float32`；采样率从 `engine.runtime.sample_rate` 读取。
- GGUF 模型通过 llama.cpp 运行，ONNX 模型组件通过 ONNX Runtime 运行。SDK 运行依赖不包含 `torch`、`torchaudio`、`torchvision`。
- 单个 Windows wheel 同时包含 llama.cpp 的 **CPU、CUDA、Vulkan** 动态库；通过参数选择后端，不需要更换 wheel。
- Prompt 可以临时使用，也可以按模型持久化，便于复用参考音色。
- 支持传入本地模型目录；传入已注册模型名时，可从 ModelScope 获取模型。

发行文件名为 `streamvox-2.0.0-cp312-none-win_amd64.whl`。其中 `none` 表示 Python 模块不依赖特定扩展 ABI；包内包含 Windows 原生动态库，仍限 Windows x64，解释器要求 CPython 3.12。其他系统与 Python 版本不在本次发行支持范围内。

## 安装

先安装 [uv](https://docs.astral.sh/uv/)，创建独立环境：

```powershell
uv venv --python 3.12 --managed-python
.venv\Scripts\Activate.ps1
```

从 [GitHub Releases](https://github.com/RoversCode/StreamVox/releases) 下载同版本 wheel，按硬件选择一个 ONNX Runtime 依赖组：

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

源码开发安装：

```powershell
git clone https://github.com/RoversCode/StreamVox.git
cd StreamVox
uv sync --managed-python --extra cpu --group dev
uv run python -m streamvox.native_assets --project-root .
```

公开 Git 仓库不存放 DLL。`native_assets` 从固定版本、固定哈希的官方 llama.cpp 归档恢复三种后端，并将 OpenMP 运行库替换为固定公开的 conda-forge LLVM 构建，供源码运行和本地构建使用；详见 [构建文档](docs/development.md)。

## 快速开始

以下示例使用你自行准备、并有权使用的参考音频。`prompt_text` 应与音频内容一致；项目不提供没有明确再分发许可的参考录音。

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

`TTSEngine` 除 `model` 外的初始化参数必须按名称传入。模型名可替换为已下载 bundle 的目录；离线部署建议使用本地目录并保留默认哈希校验。详见 [使用文档](docs/usage.md)。

## 支持模型

| 模型名 | 采样率 | 主要能力 | 指南 |
| --- | ---: | --- | --- |
| `qwen3-tts-clone-0.6b-gguf` | 24000 Hz | 单参考音色克隆，较低资源需求 | [Qwen3](docs/models/qwen3-tts-clone.md) |
| `qwen3-tts-clone-1.7b-gguf` | 24000 Hz | 单参考音色克隆，多语言生成 | [Qwen3](docs/models/qwen3-tts-clone.md) |
| `s2-pro-4b-gguf` | 44100 Hz | 多参考、多说话人和表现控制 | [S2-Pro](docs/models/s2-pro.md) |
| `voxcpm2-gguf` | 48000 Hz | 音色设计、参考克隆和续写 | [VoxCPM2](docs/models/voxcpm2.md) |
| `moss-tts-nano-onnx` | 48000 Hz | 轻量 ONNX 路线，单参考克隆与续写 | [MOSS](docs/models/moss-tts-nano.md) |

模型能力取决于对应上游和所用权重。SDK 包不包含模型权重。**S2-Pro 转换权重采用 Fish Audio Research License，商业用途需要 Fish Audio 单独书面许可**；其余三类当前默认转换权重采用 Apache-2.0。完整来源和边界见 [模型许可表](docs/model-licenses.md)。

## 后端选择

`device` 表达 CPU/GPU 和设备编号；`llama_backend` 与 `onnx_provider` 分别选择 GGUF 和 ONNX 后端。例如，在安装 `[directml,llama-cuda]` 的 NVIDIA 环境中：

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

显式请求的后端不可用或初始化时发生不允许的回退，会抛出异常。`actual_backends` 返回实际后端与主进程可见 ONNX 会话；MOSS 的 DirectML codec decode step 保留已知 CPU 兼容策略。多卡选择和子进程会话范围见 [设备说明](docs/usage.md#设备与实际后端)。

## 验证状态与开发

2.0.0 调整 SDK 公开接口和发行方式。**全模型数值对照与人工试听尚未验收**，已有数值失败证据仍需继续排查；源码测试和安装检查不能代替模型数值一致性与音质验证。不同模型、输入和硬件上的速度应以本机测试为准。

基础检查：

```powershell
uv run python -m pytest tests -q
```

参与开发时，请先阅读 [模型适配指南](docs/model-adapter-guide.md) 和 [构建与验证](docs/development.md)。提交问题时附上 SDK 版本、模型名、实际后端、硬件、可复现输入和经过脱敏的日志。

## 许可证与致谢

[LICENSE](LICENSE) 授予 StreamVox 自有代码的 Apache-2.0 权利。[NOTICE](NOTICE) 与 [第三方声明](THIRD_PARTY_NOTICES.md) 保留第三方权利边界；随包原生库和模型权重的许可不会因 SDK 开源而改变。

感谢 [llama.cpp](https://github.com/ggml-org/llama.cpp)、[ONNX Runtime](https://github.com/microsoft/onnxruntime)、[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[Fish Speech](https://github.com/fishaudio/fish-speech)、[VoxCPM](https://github.com/OpenBMB/VoxCPM) 和 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) 及其贡献者。

## 支持开发者

如果 StreamVox 对你有帮助，欢迎给开发者一点鼓励。无论是一个 Star，还是一份随心的赞赏，都非常感谢。赞赏完全自愿，不影响 SDK 的任何功能。

<img src="docs/assets/wechat-support.png" alt="微信赞赏码" width="300">
