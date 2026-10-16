# 第三方组件与权利声明

StreamVox 自有代码采用根目录 [Apache-2.0](LICENSE)，署名为 StreamVox Developers。第三方原生库、依赖、移植或派生内容继续适用对应上游条款；不能用根目录许可证替代它们的版权、归属和分发要求。

## 原生库

Windows wheel 包含用于 GGUF 推理的 llama.cpp / ggml CPU、CUDA、Vulkan 运行库及其运行所需文件。

| 组件 | 原始来源 | 许可边界与副本 |
| --- | --- | --- |
| llama.cpp / ggml | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 上游采用 [MIT](third_party/licenses/llama-cpp-MIT.txt)；保留原始版权和许可全文。 |
| LLVM / OpenMP 运行库 | [LLVM OpenMP](https://openmp.llvm.org/)，本次采用 conda-forge 的 LLVM OpenMP 15.0.5 公开构建 | 适用 [Apache-2.0 with LLVM-exception](third_party/licenses/llvm-OpenMP.txt)；保留其原文和例外，不能简写成根 SDK 许可。 |
| NVIDIA CUDA 运行依赖 | [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) | `llama-cuda`、`onnx-cuda` 由安装器拉取的 NVIDIA 包适用 NVIDIA 自有条款；包名不表示其采用 Apache-2.0。 |

本次将官方 llama.cpp 归档中的 OpenMP 运行库替换为上述 conda-forge LLVM 包内的 `libomp.dll`，并按 ggml 导入名重命名为 `libomp140.x86_64.dll`；它不是 llama.cpp 归档原样附带的文件。原生准备工具固定归档和 DLL 的 SHA256，并在 `native-manifest.json` 中记录替换来源。该替换仅涉及运行库。

原生 DLL 的具体来源和版本应以本次发行记录为准。修改运行库、重新编译或增减随包文件后，应同步本表及对应许可证副本。源码目录中的 ctypes 绑定按 llama.cpp 接口工作，SDK 的许可声明不授予对上游原生库的额外权利。

## Python 运行依赖

ONNX Runtime、NumPy、librosa、SoundFile、ModelScope、gguf、tokenizers 等依赖由安装器分别安装，继续使用各自发行包携带的 LICENSE/NOTICE。依赖清单和准确版本见 [pyproject.toml](https://github.com/RoversCode/StreamVox/blob/v2.0.0/pyproject.toml) 与 [uv.lock](https://github.com/RoversCode/StreamVox/blob/v2.0.0/uv.lock)。SDK 的 Apache-2.0 声明不更改这些依赖的许可证。

`torch`、`torchaudio`、`torchvision` 不属于 SDK 的运行依赖。离线导出或数值对照工具使用的参考项目与工具依赖，应在对应开发环境中遵循其原条款。

## 模型和适配来源

StreamVox 适配 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[Fish Speech / S2-Pro](https://github.com/fishaudio/fish-speech)、[VoxCPM](https://github.com/OpenBMB/VoxCPM) 和 [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)。模型权重不包含在 SDK wheel 中，转换后的 GGUF/ONNX 权重也继续继承上游许可。

模型名称、原始权利人、下载来源和商业使用边界集中维护在 [模型许可表](https://github.com/RoversCode/StreamVox/blob/v2.0.0/docs/model-licenses.md)。其中 S2-Pro 的 Fish Audio Research License 与 SDK 的 Apache-2.0 分开适用。Built with Fish Audio。

S2-Pro 原许可要求的模型归属声明为：

> This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.

若纳入上游源码、派生实现或新的权重转换，维护者应记录实际来源提交、保留原始声明，并按照该提交对应的许可证履行义务。上游当前主分支的许可不能自动替换历史版本的许可。尚未核实来源的材料不应被统一标注为 StreamVox 自有代码。

## 示例与参考音频

公开示例代码属于 StreamVox。运行示例所需的参考录音由使用者自行准备，并自行确认录音、声音及文本的使用权。公开源码与 SDK 包不分发未确认来源许可的历史参考录音。
