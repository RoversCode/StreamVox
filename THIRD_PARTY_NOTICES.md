# 第三方组件与权利声明

StreamVox 自有代码采用 Apache-2.0，署名为 StreamVox Developers。一次安装还会使用第三方运行库和 Python 依赖，调用模型时另行取得模型权重。每类材料分别适用其原始条款，下面记录这些来源和随包许可文件。[SDK 许可原文](LICENSE)

## 原生库

原生库是由 Python 加载的已编译计算代码。Windows 安装包（wheel）携带 llama.cpp / ggml 的 CPU、CUDA、Vulkan 运行库及所需文件，用于执行 GGUF 模型的计算。

| 组件 | 原始来源 | 许可与随包副本 |
| --- | --- | --- |
| llama.cpp / ggml | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT，保留原始版权与完整条款。[原文副本](third_party/licenses/llama-cpp-MIT.txt) |
| LLVM / OpenMP 运行库 | [LLVM OpenMP](https://openmp.llvm.org/)，采用 conda-forge 的 LLVM OpenMP 15.0.5 公开构建 | Apache-2.0 with LLVM-exception，保留原文及例外条款。[原文副本](third_party/licenses/llvm-OpenMP.txt) |
| NVIDIA CUDA 运行依赖 | 通过 `llama-cuda`、`onnx-cuda` 依赖组安装的 NVIDIA 包 | 按 NVIDIA 自有条款使用与分发。[CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) |

项目的原生资产准备流程采用 conda-forge LLVM 包中的 `libomp.dll` 替换官方 llama.cpp 归档所带的 OpenMP 运行库。保存时使用 ggml 导入的文件名 `libomp140.x86_64.dll`，使加载器能找到该运行库。准备工具固定归档和每个 DLL 的 SHA256，并在 `native-manifest.json` 中记录替换来源。

发行记录保存具体原生库版本与文件摘要。维护者重新编译、替换或增减运行库时，应同步来源和适用的许可证副本。Python 源码中的 ctypes 绑定负责调用这些原生接口。

## Python 运行依赖

ONNX Runtime 执行 ONNX 计算图，NumPy 和 SoundFile 等依赖负责数组与音频处理，ModelScope 负责模型仓库访问。安装器分别取得这些 Python 发行包，它们的 LICENSE 和 NOTICE 由各自发行包携带。准确依赖和锁定版本见项目配置。[pyproject.toml](pyproject.toml) · [uv.lock](uv.lock)

SDK 的正常运行使用 llama.cpp 与 ONNX Runtime。需要 Torch、torchaudio、torchvision 的参考导出和数值对照工具使用独立开发环境，并按各自依赖的原始条款使用。

## 模型和适配来源

StreamVox 接入 Qwen3-TTS、Fish Speech / S2-Pro、VoxCPM、MOSS-TTS-Nano 和 Breeze TTS 2。模型权重由模型目录提供，GGUF 与 ONNX 转换后的权重仍按原始模型许可使用。注册名、原始权利人、下载来源及商业使用范围集中记录在模型许可表。[模型许可与来源](docs/model-licenses.md)

S2-Pro 的 Fish Audio Research License 要求单独取得商业用途许可，并保留相应模型归属说明。Built with Fish Audio。原许可要求的声明如下。

> This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.

维护者纳入上游源码或派生实现时，应记录实际使用的提交，并保存该版本的原始归属与许可。上游后续分支的许可证变化按新版本分别核对。

## Breeze TTS 2 派生运算

`streamvox/models/breeze/` 与 `tools/breeze_reference/` 中的模型公式、模板、GGUF 布局转换和流式状态参考固定上游提交 `daa6cb2efe5471f09369229f322d20bb956cb3c3`。上游模型实现保留以下版权文字，源码采用 Apache License, Version 2.0。[上游源码](https://github.com/breezeblue-ai/breeze-tts/tree/daa6cb2efe5471f09369229f322d20bb956cb3c3) · [许可原文副本](third_party/licenses/breeze-tts-Apache-2.0.txt)

> Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.

音频编解码运算参考 Alibaba Cloud / Qwen Team 的 Qwen3-TTS，对应 `qwen-tts==0.1.1`，采用 Apache-2.0。[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

> Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

`breeze-tts-2-gguf` 使用固定检查点 `3e28c5151381a722f1d8661b4118c298caa77aa4`。Qwen3 主干及深度解码器保存为 GGUF，保留十六码本嵌入和十五组位置输出头。文本、音频嵌入、深度条件投影和音频编解码器保存为 FP32 ONNX。[固定检查点](https://huggingface.co/BreezeBlue/breeze-tts-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4)

深度码本表先采用等价投影 `E' = E @ P.T`，把每次重复执行的线性投影合并到查表数据中。基础导出可使用 FP32 或原始 BF16，默认 ModelScope 版本再以 Q8_0 量化主干和深度生成权重。量化会引入近似误差，数值对照及任务验收范围随模型保存。

该检查点的权重、衍生模型及自托管输出限研究与非商业使用。分发转换后的模型目录时，保存原始 `MODEL_LICENSE` 和归属材料，模型条款与源码条款分别履行。[Breeze 模型许可](docs/model-licenses.md#breeze-tts-2)

## 示例与参考音频

公开示例代码属于 StreamVox。示例使用的参考录音由使用者自行准备，并确认录音、声音和文本的使用权。运行步骤使用占位路径和对应转写，替换为自己的素材后执行。
