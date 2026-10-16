# 模型许可与来源

[返回 README](../README.md) · [使用文档](usage.md) · [第三方声明](../THIRD_PARTY_NOTICES.md)

StreamVox 自有代码采用 Apache-2.0。模型权重、词表、音频编解码器权重以及它们的 GGUF/ONNX 转换文件具有独立许可；SDK 免费提供不改变这些条款。SDK wheel 不包含模型权重，模型下载在使用时单独进行。

## 默认模型

以下记录以 2026-10-16 核对的默认 ModelScope 仓库 LICENSE/NOTICE 和官方原始来源为依据。实际使用时应保存所选提交附带的许可文件；更换权重来源或 revision 后应重新核对。

| SDK 模型 | 原始来源与权利人 | 默认转换仓库 | 权重许可与商业边界 |
| --- | --- | --- | --- |
| `qwen3-tts-clone-0.6b-gguf` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)，Alibaba Cloud / Qwen Team | [ChengHee/qwen3-tts-clone-0.6b-gguf](https://modelscope.cn/models/ChengHee/qwen3-tts-clone-0.6b-gguf) | Apache-2.0；商业使用仍须遵守其完整条款。 |
| `qwen3-tts-clone-1.7b-gguf` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)，Alibaba Cloud / Qwen Team | [ChengHee/qwen3-tts-clone-1.7b-gguf](https://modelscope.cn/models/ChengHee/qwen3-tts-clone-1.7b-gguf) | Apache-2.0；商业使用仍须遵守其完整条款。 |
| `s2-pro-4b-gguf` | [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro)，39 AI, INC. / Fish Audio | [ChengHee/s2-pro-4b-gguf](https://modelscope.cn/models/ChengHee/s2-pro-4b-gguf) | Fish Audio Research License（2026-03-07）；研究和非商业使用按原条款，**商业用途需要 Fish Audio 单独书面许可**。 |
| `voxcpm2-gguf` | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) / [VoxCPM](https://github.com/OpenBMB/VoxCPM)，OpenBMB | [ChengHee/voxcpm2-gguf](https://modelscope.cn/models/ChengHee/voxcpm2-gguf) | Apache-2.0；商业使用仍须遵守其完整条款。 |
| `moss-tts-nano-onnx` | [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) / [100M 权重](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)，OpenMOSS Team、Fudan University、SII、MOSI | [ChengHee/moss-tts-nano-onnx](https://modelscope.cn/models/ChengHee/moss-tts-nano-onnx) | Apache-2.0；商业使用仍须遵守其完整条款。 |

## S2-Pro

当前默认转换仓库附带的是 [Fish Audio Research License 原文](https://huggingface.co/fishaudio/s2-pro/blob/1de9996b6be38b745688de084d87a5633f714e4e/LICENSE.md)。该许可没有授予商业使用权，要求商业用途另行获得 Fish Audio 书面许可；其分发、归属声明和使用限制也继续适用。Built with Fish Audio。

Fish Speech 仓库后续版本可能采用其他许可。不要因为 SDK 已采用 Apache-2.0，或上游主分支后来更换了许可证，就认为已下载权重的条款自动改变。这里的许可表也不构成 Fish Audio 的商业授权。

## 已知的外部说明错误

核对时，默认 `voxcpm2-gguf` 仓库的 NOTICE 将模型误写为 Qwen3-TTS、原权利人误写为 Alibaba Cloud；`moss-tts-nano-onnx` 仓库的 NOTICE 同样误写了 Alibaba Cloud / Qwen Team，并存在沿用其他模型转换描述的情况。本表按上面的官方模型来源区分权利人，不沿用这些错误归属。

这次 SDK 开源只同步本项目和公开源码仓库的文档，没有修改外部 ModelScope 模型仓库。分发这些模型前，应核对实际权重来源、保留原许可，并修正与实际转换内容不符的 NOTICE；不能把错误归属当作新的授权证明。

## 下载与再分发

用于部署的 bundle 应保留 `manifest.json`、运行权重和对应来源的 LICENSE/NOTICE。法律文件不参与音频推理，不应因 README 文案变化阻断模型运行，但再分发时仍须保留适用的法律声明。

生产环境可通过 `download_model(..., revision=..., expected_manifest_sha256=...)` 固定模型版本，两项必须同时提供。具体调用和目录语义见 [使用文档](usage.md#模型下载与固定版本)。
