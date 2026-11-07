# 模型许可与来源

[返回 README](../README.md) · [使用文档](usage.md) · [第三方声明](../THIRD_PARTY_NOTICES.md)

一次 StreamVox 部署会使用三类材料。SDK 源码实现统一调用和模型适配，运行库负责执行计算，模型权重保存训练得到的参数。三类材料分别适用各自的许可证，使用时应按实际分发的版本保留相应条款。

StreamVox 自有代码采用 Apache-2.0。安装包（wheel）提供代码和原生运行库，模型权重在首次调用时另行下载。将权重转换为 GGUF 或 ONNX 后，它仍属于原模型的派生材料，使用范围由该权重版本的许可证确定。

## 默认模型

下表列出 SDK 注册名、官方来源、默认下载位置及权重的主要许可范围。前五行依据 2026-10-16 的来源核对记录，Breeze 依据固定检查点和 2026-11-06 的发布记录。部署时保存所选提交携带的完整许可文件，更换权重来源或提交后重新核对。

| SDK 模型 | 原始来源与权利人 | 默认转换仓库 | 权重许可与商业范围 |
| --- | --- | --- | --- |
| `qwen3-tts-clone-0.6b-gguf` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)，Alibaba Cloud / Qwen Team | [ChengHee/qwen3-tts-clone-0.6b-gguf](https://modelscope.cn/models/ChengHee/qwen3-tts-clone-0.6b-gguf) | Apache-2.0，商业使用按完整条款履行义务。 |
| `qwen3-tts-clone-1.7b-gguf` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)，Alibaba Cloud / Qwen Team | [ChengHee/qwen3-tts-clone-1.7b-gguf](https://modelscope.cn/models/ChengHee/qwen3-tts-clone-1.7b-gguf) | Apache-2.0，商业使用按完整条款履行义务。 |
| `s2-pro-4b-gguf` | [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro)，39 AI, INC. / Fish Audio | [ChengHee/s2-pro-4b-gguf](https://modelscope.cn/models/ChengHee/s2-pro-4b-gguf) | Fish Audio Research License（2026-03-07）。研究与非商业使用按原文，商业用途需要 Fish Audio 单独书面许可。 |
| `voxcpm2-gguf` | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) / [VoxCPM](https://github.com/OpenBMB/VoxCPM)，OpenBMB | [ChengHee/voxcpm2-gguf](https://modelscope.cn/models/ChengHee/voxcpm2-gguf) | Apache-2.0，商业使用按完整条款履行义务。 |
| `moss-tts-nano-onnx` | [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) / [100M 权重](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)，OpenMOSS Team、Fudan University、SII、MOSI | [ChengHee/moss-tts-nano-onnx](https://modelscope.cn/models/ChengHee/moss-tts-nano-onnx) | Apache-2.0，商业使用按完整条款履行义务。 |
| `breeze-tts-2-gguf` | [BreezeBlue/breeze-tts-2](https://huggingface.co/BreezeBlue/breeze-tts-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4)，BreezeBlue | [ChengHee/Breeze-TTS-2-GGUF](https://www.modelscope.cn/models/ChengHee/Breeze-TTS-2-GGUF) | 模型权重、衍生模型及自托管输出仅供研究与非商业用途，完整条款见模型目录的 `MODEL_LICENSE`。 |

## Breeze TTS 2

该模型采用固定检查点 `3e28c5151381a722f1d8661b4118c298caa77aa4`，转换实现参考源码提交 `daa6cb2efe5471f09369229f322d20bb956cb3c3`。主干和深度解码器保存为 Q8_0 GGUF，其余组件保存为 FP32 ONNX。[检查点](https://huggingface.co/BreezeBlue/breeze-tts-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4) · [参考源码](https://github.com/breezeblue-ai/breeze-tts/tree/daa6cb2efe5471f09369229f322d20bb956cb3c3)

该检查点把源码许可与模型许可分开。`LICENSE` 保存源码的 Apache-2.0 条款，`MODEL_LICENSE` 保存模型权重、衍生模型和自托管输出的研究与非商业使用条款，`NOTICE` 保存归属说明。FP32、BF16 和 Q8_0 描述的是数值存储方式，这些转换后的模型继续适用同一模型许可。再分发时保留对应的法律文件。

## S2-Pro

默认转换模型携带 Fish Audio Research License。它规定研究与非商业使用的条件，并要求商业用途另行取得 Fish Audio 书面许可。分发、归属和使用限制按该版本原文执行。[固定许可原文](https://huggingface.co/fishaudio/s2-pro/blob/1de9996b6be38b745688de084d87a5633f714e4e/LICENSE.md)

模型要求的归属说明需随相关分发保留。Built with Fish Audio。上游源码或后续主分支采用其他许可证时，已下载权重仍按其所选版本的条款使用。

## 已知的外部说明错误

2026-10-16 的核对记录发现，VoxCPM2 默认转换仓库的 NOTICE 曾把模型及权利人误写成 Qwen3-TTS、Alibaba Cloud。MOSS 默认转换仓库的 NOTICE 也曾沿用 Alibaba Cloud / Qwen Team 和其他模型的转换描述。

上表按官方模型来源区分这些权利人。维护者再分发某个具体提交前，应把该提交的权重来源、LICENSE 和 NOTICE 对照核实，再修正与实际内容不符的归属描述。上述日期记录用于指出已发现的来源差异，具体版本以它携带的文件为准。

## 下载与再分发

模型目录（bundle）包含运行资产和随模型提供的说明材料。运行清单 `manifest.json` 记录模型名、文件路径和内容摘要，SDK 加载时检查 `ckpt/` 中的运行资产。许可证和归属文件随模型保存，音频运算本身使用权重和计算图。

按模型名自动下载会取得默认仓库的快照。固定版本下载同时传 `revision` 和 `expected_manifest_sha256`，分别选择提交并校验原始清单。当前源码还按同一提交的文件树取得 LICENSE、MODEL_LICENSE、NOTICE、COPYING 及识别的扩展名版本，并校验这些文件的摘要。[固定下载方式](usage.md#模型下载与固定版本)

再分发时，保存与所用权重相同版本的完整许可和归属文件。若原仓库采用其他法律文件名，应由维护者将这些原文一并纳入分发目录。原生库与 Python 依赖的权利说明见第三方声明。[第三方组件](../THIRD_PARTY_NOTICES.md)
