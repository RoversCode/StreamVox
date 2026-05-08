# MOSS-TTS-Nano 能力说明

本文档面向希望接入 StreamVox 的客户，集中说明 `MOSS-TTS-Nano` 的模型定位、核心能力、适用场景与工程注意事项。  
当前 StreamVox 接入型号为：

* `moss-tts-nano-onnx`

## 1. 系列定位

MOSS-TTS-Nano 是当前 StreamVox 中最偏**低资源、低成本、纯 CPU 实时生成**路线的模型。  
它最大的价值不在于“参数规模最小”本身，而在于：

* 不要求显卡
* 纯 ONNX 路线即可完成实时流式合成
* 支持稳定的单参考音色克隆
* 适合边缘部署、低成本部署和 CPU-only 服务

如果你的业务优先级是：

* 先在没有独显的机器上跑通实时语音
* 更关注单位部署成本
* 更关注单机 CPU 实时能力
* 需要一个工程接入简单、行为稳定的单参考克隆模型

那么 `moss-tts-nano-onnx` 应该是当前最值得优先试的模型。

## 2. 支持语言与文本风格

MOSS-TTS-Nano 目前支持 20 种语言：中文、英文、德语、西班牙语、法语、日语、意大利语、匈牙利语、韩语、俄语、波斯语、阿拉伯语、波兰语、葡萄牙语、捷克语、丹麦语、瑞典语、希腊语、土耳其语。					

MOSS-TTS-Nano 更适合作为**中短文本、日常播报、助手回复、低成本实时交互**的语音生成模型。  
在 StreamVox 当前适配中，我们建议把它理解为：

* 主打中文场景
* 适合清晰、直接、结构明确的输入文本
* 更适合单说话人实时输出
* 不以复杂多说话人、细粒度副语言控制为主

对于数字、日期、金额、缩写、单位等复杂文本，仍建议在进入推理前先做文本归一化，避免模型按字面形式播报。

## 3. 核心能力

### 3.1 单参考音色克隆

MOSS-TTS-Nano 支持基于**一段参考音频 + 对应参考文本**快速构建角色 Prompt。  
构建完成后，这个 Prompt 可以被反复复用，用于后续实时流式合成。

这是它最推荐的使用方式，也是当前 StreamVox 中 CPU 路线下最实用的能力组合之一。

示例：

参考音频文本：

```text
你好，欢迎使用 StreamVox。今天我们来看一个轻量级实时语音模型的部署示例。
```

目标生成文本：

```text
好的，下面我将继续介绍这个模型在纯 CPU 环境下的实际表现。
```

### 3.2 continuation 与 voice_clone 两种模式

MOSS-TTS-Nano 在 StreamVox 中支持两种核心推理模式：

* `voice_clone`
  * 需要参考音频
  * 适合零样本单参考音色克隆
* `continuation`
  * 可用于无参考音频的纯文本 continuation
  * 也可用于“参考文本 + 参考音频 + 目标文本”的延续式生成

这意味着它既可以作为一个轻量克隆模型使用，也可以作为一个更基础的流式 TTS 模型使用。

### 3.3 实时流式输出

MOSS-TTS-Nano 在 StreamVox 当前适配中，默认就是按流式方式工作。  
也就是说，它不是先整段生成完再一次性返回，而是会持续产出音频 chunk。

这使它非常适合：

* 实时语音助手
* 低成本对话系统
* 边缘设备播报
* CPU-only 的服务端推理

### 3.4 Prompt 可复用

`make_prompt(...)` 完成后，参考音频会先被编码为可复用的 Prompt 资产。  
如果你的业务角色会被频繁复用，建议：

* 在角色创建阶段就完成 `make_prompt(...)`
* 高频角色使用 `persist=True` 落盘缓存
* 正式推理时直接传角色名或 Prompt 对象

这样做通常能显著降低正式请求的首包压力。

## 4. 工程侧推荐用法

### 4.1 推荐：先 make_prompt，再正式 stream

对 MOSS-TTS-Nano，推荐工作流是：

1. 先执行 `make_prompt(...)`
2. 角色复用时直接使用缓存 Prompt
3. 正式推理只做文本预处理、TTS 推理与流式输出

原因很简单：  
MOSS 的参考音频编码本身就需要一定成本。如果把 prompt 构建和正式合成混在同一个请求里，首个 chunk 延迟会被明显拉长。

### 4.2 推荐：CPU 场景优先尝试

如果你的机器没有独显，或者业务希望压低 GPU 成本，可以优先尝试：

```python
engine = TTSEngine(
    model="moss-tts-nano-onnx",
    license_key="YOUR_SDK_KEY",
    device="cpu",
    verify_model_sha256=False,
)
```

这也是当前 release 中最推荐的 MOSS 接入方式。

### 4.3 推荐：中短文本优先

MOSS-TTS-Nano 更适合：

* 一次一句
* 一次一小段
* 连续对话回复
* 结构清晰的通知/播报类文本

如果文本非常长、情绪变化复杂、需要多说话人组织，优先考虑 `S2-Pro` 或其他更重型模型会更稳。

## 5. 不适合的场景

虽然 MOSS-TTS-Nano 很适合作为 CPU 实时路线的首选，但它并不适合所有场景。  
以下需求更建议选择其他模型：

* **多说话人对白**：更适合 `S2-Pro`
* **复杂情绪控制 / 副语言控制**：更适合 `S2-Pro` 或 `VoxCPM2`
* **方言克隆与音色设计**：更适合 `VoxCPM2`
* **更强调高保真成片感**：更适合 `S2-Pro`

## 6. 典型示例

### 6.1 voice_clone

```python
from streamvox import TTSEngine


engine = TTSEngine(
    model="moss-tts-nano-onnx",
    license_key="YOUR_SDK_KEY",
    device="cpu",
    verify_model_sha256=False,
)

prompt = engine.make_prompt(
    role_name="moss_voice",
    audio_path="reference.wav",
    prompt_text="这是参考音频的转写文本。",
    persist=False,
)

chunks = engine.stream(
    text="这是一段 MOSS-TTS-Nano 的实时流式语音合成示例。",
    role_name=prompt,
    mode=0,
    track_performance=True,
)
```

### 6.2 continuation

```python
from streamvox import TTSEngine


engine = TTSEngine(
    model="moss-tts-nano-onnx",
    license_key="YOUR_SDK_KEY",
    device="cpu",
    verify_model_sha256=False,
)

chunks = engine.stream(
    text="这是一段 continuation 模式示例。",
    role_name="role name",
    mode=1,
    track_performance=True,
)
```

## 8. 参考与致谢

感谢 MOSS-TTS-Nano 原始团队在轻量级实时语音生成方向上的持续研究与开源贡献。  
