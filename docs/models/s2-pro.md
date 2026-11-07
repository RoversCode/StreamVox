# S2-Pro 系列能力说明

[返回 README](../../README.md) · [使用文档](../usage.md) · [模型许可与来源](../model-licenses.md)

S2-Pro 根据参考录音合成新语音，并允许在正文中标记说话人、情绪和说话方式。StreamVox 当前接入 `s2-pro-4b-gguf`。要让两个角色轮流说话，先用两段录音建立角色对应关系，再在目标文本中用相同编号标记各人的台词。

模型目录（bundle）保存权重、词表和 `manifest.json`。参考音色资产（Prompt）保存参考录音的音频编码、对应文本以及提前计算好的模型状态。`make_prompt()` 完成这些准备，后续 `stream()` 复用同一份资产。

权重适用 Fish Audio Research License（2026-03-07）。研究和非商业使用遵循原条款，商业用途需要 Fish Audio 单独书面许可。SDK 自有代码的 Apache-2.0 与模型权重分别适用。Built with Fish Audio。完整来源与条款入口见 [模型许可与来源](../model-licenses.md)。

## 1. 系列定位

S2-Pro 的生成组件先预测随时间变化的语音编码，再补出每个时间位置的声音细节，最后由音频解码器还原成波形。当前 StreamVox 使用两组 GGUF 生成权重和 ONNX 音频编解码器运行这条链路。GGUF、ONNX 是这些组件的模型文件格式和执行链路。

在接入选择上，S2-Pro 提供单参考克隆、多参考角色组织、正文中的风格控制和多说话人对白。当前接口需要参考录音及其转写文本。它适合把角色台词和局部表达意图写进同一次合成请求。

当前模型规格的经验档位为最低 / 推荐显存 7 / 8 GB、最低 / 推荐内存 6 / 8 GB。模型名中的 `4b` 沿用系列命名，规格层按整条生成链路约 45 亿参数估算资源。实际延迟、资源用量和听感应使用目标硬件、参考录音与业务正文测量。

## 2. 支持语言

上游模型卡声明支持超过 80 种语言，并按下列层级列出语言。这里沿用上游覆盖范围，StreamVox 当前没有逐语种试听验收记录。调用时直接填写目标语言正文，适配器公开参数中没有单独的 `language` 开关。[S2-Pro 官方模型卡](https://huggingface.co/fishaudio/s2-pro)

### 2.1 一线主语言

上游列为第一层级的语言是中文 `zh`、英文 `en` 和日文 `ja`。这些代码用于说明语言，正文仍直接使用相应文字。

### 2.2 重点支持语言

第二层级包括韩文 `ko`、西班牙文 `es`、葡萄牙文 `pt`、阿拉伯文 `ar`、俄文 `ru`、法文 `fr` 和德文 `de`。

### 2.3 全球覆盖语言

上游列出的其他语言包括意大利文 `it`、土耳其文 `tr`、荷兰文 `nl`、丹麦文 `da`、波兰文 `pl`、印地语 `hi`、乌尔都语 `ur`、泰语 `th`、越南语 `vi`、印尼语 `id`、乌克兰语 `uk`、希腊语 `el`、罗马尼亚文 `ro`、塞尔维亚文 `sr`、泰卢固语 `te`、旁遮普语 `pa`、卡纳达语 `kn`、马拉地语 `mr` 和古吉拉特语 `gu` 等。完整集合见 [官方模型卡的语言列表](https://huggingface.co/fishaudio/s2-pro#supported-languages)。

## 3. 核心能力

### 3.1 单参考与多参考角色构建

每条参考录音都需要对应的 `prompt_text`。录音提供声音样本，转写文本告诉模型这段声音说了什么。转写应与实际内容一致。`make_prompt()` 会提前编码音频，并计算参考条件的前缀状态，合成时可以从该状态继续计算。

单参考时传一条文件路径和一条转写字符串。多参考时可传路径列表，并用等长的文本列表逐项对齐。对于内存音频，传入 `audio_data` 数组列表和采样率，采样率可以是所有录音共用的整数，也可以是与录音等长的整数列表。文件路径和内存音频是两种互斥来源。

例如，准备两段分别属于不同人的 `speaker-a.wav` 和 `speaker-b.wav`。若文本列表未手动写说话人标记，模型会按列表顺序补上 `<|speaker:0|>` 和 `<|speaker:1|>`。目标正文用这些编号选择相应角色。多段录音也可以描述同一角色，需在参考文本中明确使用一致的说话人标记，并试听实际效果。

多参考音频还允许搭配一条完整的转写字符串。此时由调用方在这个字符串中组织说话人标记和全部参考内容，SDK 会按输入顺序拼接音频编码。逐项文本列表更容易核对录音与台词的对应关系。

下面是可独立运行的双人对白示例。先按 [使用文档](../usage.md) 安装 SDK 和准备模型，两段录音放在脚本工作目录中，并分别录下示例中的参考文本。项目不附带这些录音。

将下面代码保存为 `s2_dialogue_demo.py`。

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from streamvox import TTSEngine


def main():
    """构建两个人的参考音色并保存轮流发声的对白。

    Returns:
        无返回值，合成音频保存为 s2-dialogue.wav。
    """
    engine = TTSEngine(
        model="s2-pro-4b-gguf",
        device="auto",
        verify_model_sha256=True,
    )
    try:
        prompt = engine.make_prompt(
            role_name="presenters",
            audio_path=["speaker-a.wav", "speaker-b.wav"],
            prompt_text=[
                "你好，今天由我介绍产品亮点。",
                "接下来我来补充部署方式。",
            ],
            persist=False,
        )
        chunks = list(engine.stream(
            text=(
                "<|speaker:0|>[professional broadcast tone]欢迎来到产品发布会。"
                "<|speaker:1|>大家好，接下来由我介绍部署流程。"
                "<|speaker:0|>[excited]让我们开始演示！"
            ),
            role_name=prompt,
            track_performance=True,
        ))
        audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("本次合成未返回音频")
        output = Path("s2-dialogue.wav")
        sf.write(output, audio, engine.runtime.sample_rate)
        print(output.resolve())
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
```

`list()` 会实际消费音频生成器，脚本结束后得到 44,100 Hz 的 `s2-dialogue.wav`。每块音频是一维 `float32` 的 NumPy 数组。需要实时播放时，在 `for` 循环中逐块消费。`device="auto"` 由运行时结合已安装后端选择设备，实际后端可通过 `engine.actual_backends` 查看。

要改成单参考，将 `audio_path` 和 `prompt_text` 分别换成一条路径和字符串，并只使用 `<|speaker:0|>`。把 `persist` 改为 `True` 会保存角色，后续同一模型可用 `role_name="presenters"` 读取缓存。Prompt 的音频、文本和预计算状态共同占用模型上下文，参考过长时可能在构建阶段超出窗口。

### 3.3 多说话人对白生成

`<|speaker:0|>` 这样的标记用于指定后续台词由哪个角色说。每次换人，都在其台词前写一个标记。编号应与参考文本中的编号对应。例如上面的列表按顺序建立 0、1 两个角色，目标文本也使用 0、1。

含标记的文本会按角色片段拆分，再按长度预算合成批次，单批最多组织 5 个不同说话人。单人正文省略全部标记时，预处理会按 `<|speaker:0|>` 处理。多角色正文应从第一个说话人标记开始，再写台词，确保每句话都有明确归属。

```text
<|speaker:0|>你好，欢迎来到产品发布会。
<|speaker:1|>大家好，接下来由我介绍多语言功能。
<|speaker:0|>最后我们一起看看部署方式。
```

### 3.4 细粒度自然语言内联控制

内联控制是在正文相应位置写一段英文方括号描述，让模型把描述与附近台词一起用于生成。说话人标记决定角色，`[whisper]` 等描述给出这一处的说话方式。SDK 会保护方括号片段，避免清洗和分段把它拆散。

```text
<|speaker:0|>[whisper]别出声，他们已经到门口了。
<|speaker:1|>[excited]太好了，我们终于上线了！
<|speaker:0|>[professional broadcast tone]下面播报一则正式通知。
```

这些描述由模型解释，标签是否产生预期语气需试听确认。当前适配器的公开合成参数中没有 `control_text`，局部风格直接写在正文中。

### 3.5 丰富的风格与情绪标签

上游说明提到超过 15,000 种独特标签，并允许自由文字描述。SDK 原样保护方括号中的内容，没有维护一份逐标签校验的固定表。[S2-Pro 官方内联控制说明](https://huggingface.co/fishaudio/s2-pro#fine-grained-inline-control)

可尝试的短标签包括 `[pause]`、`[emphasis]`、`[laughing]`、`[inhale]`、`[chuckle]`、`[tsk]`、`[singing]`、`[excited]`、`[angry]`、`[whisper]`、`[screaming]`、`[shouting]`、`[surprised]`、`[short pause]`、`[exhale]`、`[delight]`、`[sad]`、`[shocked]`、`[with strong accent]`、`[volume up]` 和 `[volume down]`。

较长的描述可写为 `[whisper in small voice]`、`[professional broadcast tone]` 或 `[pitch up]`。把标签放在希望发生变化的台词附近，例如下面四行分别尝试短停顿、笑着说、愤怒和悲伤。

```text
[short pause]我理解你的担心。[calm]这次升级已经完成测试。
[laughing tone]你刚才那个反应，也太真实了吧。
[angry]我已经说过很多次了，这个参数需要先测试！
[sad]很遗憾，这次我们没能赶上截止时间。
```

### 3.6 长文本与演播型内容

长文本会先拆为批次。一次 `stream()` 内，运行时会把已生成的批次保留在上下文中，再接着处理后续台词。上下文接近 `n_ctx` 容量时，会恢复到参考 Prompt 状态后继续，因此更早生成的对白历史会离开窗口。内容仍按输入顺序输出。

初始化时 `n_ctx` 默认 12,288，显式值会被限制在 4,096 至 32,768。参考录音编码、参考文本、当前台词和生成语音共同使用这个窗口，增加正文分段长度也会减少单批可用的生成空间。

| 请求参数 | 默认值 | 作用 |
| --- | --- | --- |
| `max_length` | `400` | 分段预算。预处理先按全文平均 UTF-8 字节密度折算，再按 tokenizer 的 token 长度拆分，并将折算值封顶为 300。token 是分词器输出的模型输入单位。 |
| `min_length` | `40` | 合并较短片段的阈值，分段时会与有效上限一起调整。须为非负整数。 |
| `temperature` | `0.9` | 调整生成候选的概率分布，小于等于 0 时直接选择最高分候选。 |
| `top_p` | `0.8` | 按概率从高到低保留累计概率阈值内的候选。 |
| `top_k` | `30` | 保留概率最高的若干候选，小于等于 0 时关闭这一步截断。 |
| `remove_meaningless_chars` | `True` | SDK 默认清理控制字符和允许范围外字符，保留说话人及方括号控制片段。 |
| `track_performance` | `False` | 开启文本预处理、生成和解码等阶段的性能日志。 |

`max_length` 是当前实现的分段预算参数，最终批次使用 token 计数，实际 UTF-8 字节数会随文字和分词结果变化。传入正整数上限和非负整数下限，并结合实际输入查看分段日志。当前适配固定使用流式解码，完整消费生成器后可将音频块拼成文件。

## 4. 参考与致谢

Fish Audio 团队提供了 S2-Pro 原始模型。上游公开的语言覆盖与标签说明见 [官方模型卡](https://huggingface.co/fishaudio/s2-pro)，StreamVox 当前接口和参数见 [使用文档](../usage.md)。
