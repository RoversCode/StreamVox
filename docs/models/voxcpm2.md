# VoxCPM2 系列能力说明

[返回 README](../../README.md) · [使用文档](../usage.md) · [模型许可与来源](../model-licenses.md)

VoxCPM2 把正文和声音条件一起转换为语音。声音条件可以来自一段文字描述，也可以来自一条参考录音。StreamVox 当前接入 `voxcpm2-gguf`，通过 `mode` 选择纯文本生成、参考音色克隆或带已有语音上下文的续写。

模型目录（bundle）保存权重、词表和 `manifest.json`。参考音色资产（Prompt）保存从录音提前提取的特征，以及可选的录音转写文本。`make_prompt()` 为同一条录音计算参考音色和续写两条路径需要的特征，`stream()` 再按所选模式使用它们。

默认转换权重采用 Apache-2.0，SDK 与模型分别适用各自许可。原始权利人和转换来源见 [模型许可与来源](../model-licenses.md)。

## 1. 模型能力

### 1.1 四种生成模式

音色设计是用文字描述要生成的声音。参考克隆是从录音提取声音特征，让该声音说新内容。续写则同时使用已有录音及其转写，让新内容接在这段语音上下文之后。当前四种模式的输入关系如下。

| 模式名 | 数字值 | 参考音频 | `prompt_text` | `control_text` | 使用的条件 |
| --- | ---: | --- | --- | --- | --- |
| `text` | `0` | 无需 | 无需 | 生效 | 正文和可选的声音描述。 |
| `ref` | `1` | 必需 | 无需 | 生效 | 录音的音色特征和可选的表达描述。 |
| `continuation` | `2` | 必需 | 必需 | 忽略并记录警告 | 已有录音及其转写上下文。 |
| `ref_continuation` | `3` | 必需 | 必需 | 忽略并记录警告 | 同一录音的音色特征与续写上下文。 |

`mode` 也接受 `"0"` 至 `"3"` 的数字字符串。省略时，引擎根据 Prompt 推导模式。

- 没有音频时使用 `text`。
- 有音频且有非空 `prompt_text` 时使用 `continuation`。
- 有音频但没有转写时，按当前默认配置使用 `ref`。

显式请求 `continuation` 或 `ref_continuation`，但 Prompt 只有音频而没有转写时，适配器会警告并回退到 `ref`。缺少音频仍会报错。为了让每次请求使用明确的条件，以下示例都写出模式名。

`control_text` 是描述声音或表达方式的一段文字。`text` 和 `ref` 模式会把它包在英文圆括号中，放到正文前面交给模型解释。续写两种模式会丢弃这一字段，声音与表达主要沿用录音上下文。

### 1.2 多语种与文本覆盖

上游模型卡列出 30 种语言，包含下列集合。StreamVox 的字符过滤规则保留相应文字，公开合成参数中没有单独的语言开关，调用时直接填写目标语言正文。[VoxCPM2 官方模型卡](https://huggingface.co/openbmb/VoxCPM2)

- 亚洲语言包括中文、日语、韩语、印地语、泰语、越南语、印尼语、马来语、他加禄语、高棉语、老挝语和缅甸语。
- 欧洲与美洲语言包括英语、法语、德语、西班牙语、意大利语、俄语、葡萄牙语、荷兰语、波兰语、瑞典语、丹麦语、芬兰语、挪威语和希腊语。
- 中东与非洲语言包括阿拉伯语、土耳其语、希伯来语和斯瓦希里语。

字符预处理只决定哪些文字能进入模型。各语言的实际发音与音色表现需要使用目标文本生成并试听，当前 StreamVox 尚无逐语种试听验收记录。

### 1.3 方言与非语言标签

非语言标签用于在正文中提示笑声、叹气、迟疑或拟声等表达。当前预处理会原样保护下面这组标签，避免字符清洗或分段把标签拆散。按列表保留英文大小写。

```text
[laughing] [sigh] [Uhm] [Shh]
[Question-ah] [Question-ei] [Question-en] [Question-oh]
[Surprise-wa] [Surprise-yo] [Dissatisfaction-hnn]
```

标签被保留后，仍由模型决定如何生成相应声音。需要笑声时可写 `[laughing]你刚才那个反应，也太真实了吧。`，需要低声提醒时可写 `[Shh]先别出声，他们已经到门口了。`。

方言正文应使用当地实际说法。当前 SDK 不负责把普通话改写成粤语、四川话、东北话或河南话。正文中的词句、参考录音和 `control_text` 一起提供方言线索，先整理台词再试听生成结果。

## 2. 最佳实践

### 2.1 先把正文写对，再谈控制

先确定希望听到的台词，再整理数字、日期、缩写和标点的朗读方式。若要方言表达，先把正文改写成当地说法。之后仅在需要拟声或语气变化的位置加标签，最后选择模式和 `control_text`。

例如，业务数据 `2026-11-07 15:00` 若应读成中文通知，可以先写为 `十一月七日下午三点`。文字明确了要读出的内容，控制描述再负责声音和表达方式。

### 2.2 做 Voice Design 时，优先用 `text`

Voice Design 即音色设计。没有参考录音时，使用 `mode="text"`，将台词放在 `text`，声音描述放在 `control_text`。描述可包括人物年龄或身份、音高与质感、语速和表达状态，以及具体播报场景。

例如 `沉稳的中年男性讲解员，声音偏低，语速平稳，用清晰的旁白口吻介绍科技产品。` 同时给出声音质感和说话方式。也可尝试英文描述，例如下句针对低音、沙哑和缓慢的历史旁白。

```text
A quiet, raspy elderly woman with a low-pitched voice and a grainy texture. Speaks slowly at low volume for historical narration.
```

### 2.3 做音色克隆时，优先用 `ref`

已经有目标音色录音，需要让同一个角色说新台词时，使用 `mode="ref"`。这一模式使用录音的音色特征，同时允许 `control_text` 提示更快的语速、更明亮的声线或具体情绪。参考转写可以省略。

录音应尽量保持单人、清楚且没有持续背景声音。可先尝试 5 秒以上的参考片段，再用同一段目标台词比较结果，这个时长是准备素材时的建议，接口没有检查 5 秒门槛。需要保留角色音色并调整表达时，描述应围绕本次台词的说法编写。

### 2.4 做续写时，别把 `continuation` 当成加强版 `ref`

已有一段录音且知道准确转写，需要在它的语音上下文后继续生成时，使用 `mode="continuation"`。模型接收 `prompt_text + text` 组成的文本和已有录音的特征，新生成部分从这个上下文继续。

`prompt_text` 应只写参考录音实际说出的内容，`text` 写后续要说的新内容。录音边界、转写和新台词的衔接关系共同决定模型得到的上下文。此模式会忽略 `control_text`，需要通过文字描述调整新台词表达时可选择 `ref`。

### 2.5 `ref_continuation` 的正确理解

当前公开入口每个 Prompt 接受一条音频。`make_prompt()` 对同一条音频分别计算音色参考特征和续写特征，`ref_continuation` 同时使用这两份特征及转写。准备一条清楚的单人录音和准确的 `prompt_text`，再传入后续正文即可。

这适合既要提供明确音色条件，又要沿用该段语音上下文的请求。输入仍为同一条录音，此模式同样忽略 `control_text`。

### 2.6 标签要少而准

标签会参与模型输入，应放在真正需要的表达位置。先用一处标签验证效果，再决定是否增加。例如以下三行分别演示低声提醒、笑声和迟疑。

```text
[Shh]先别出声，他们已经到门口了。
[laughing]你刚才那个反应，也太真实了吧。
[Uhm]我再想一下，这个方案还可以调整。
```

`text` 或 `ref` 中需要整句的声音设定时，将描述放入 `control_text`，正文保留台词和必要标签。

## 3. 典型使用方式

先按 [使用文档](../usage.md) 安装 SDK 和准备 `voxcpm2-gguf` 模型目录。下例是完整独立脚本，使用 `device="auto"` 按已安装后端选择设备。示例会完整消费生成器，再将一维 `float32` 的 NumPy 音频块拼接成 WAV。边生成边播放时，可改用循环逐块交给播放器，采样率取 `engine.runtime.sample_rate`。

### 3.1 纯文本 Voice Design

这段脚本无需参考录音，运行后写出 `voxcpm2-text.wav`。

将下面代码保存为 `voxcpm2_design_demo.py`。

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from streamvox import TTSEngine


def main():
    """根据文字设计声音并保存合成结果。

    Returns:
        无返回值，合成音频保存为 voxcpm2-text.wav。
    """
    engine = TTSEngine(
        model="voxcpm2-gguf",
        device="auto",
        verify_model_sha256=True,
    )
    try:
        chunks = list(engine.stream(
            text="欢迎来到语音控制台，接下来我将介绍部署流程。",
            mode="text",
            control_text="沉稳的中年男性讲解员，声音偏低，语速平稳，表达清晰。",
        ))
        audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("本次合成未返回音频")
        output = Path("voxcpm2-text.wav")
        sf.write(output, audio, engine.runtime.sample_rate)
        print(output.resolve())
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
```

### 3.2 单参考音色克隆

先准备自己有权使用的 `reference.wav`，放在脚本工作目录中。项目不附带参考录音。在上例 `try` 内，将创建 `chunks` 的语句替换为下面这段，并把输出文件名改为 `voxcpm2-ref.wav`，后续拼接、保存和资源清理保持相同流程。

```python
prompt = engine.make_prompt(
    role_name="demo_role",
    audio_path="reference.wav",
    persist=False,
)
chunks = list(engine.stream(
    text="您好，您的订单已经发货，预计明天下午送达。",
    role_name=prompt,
    mode="ref",
    control_text="语速稍快，声音明亮，语气亲切。",
))
```

`persist=False` 返回仅在内存中使用的 Prompt。需要后续复用时改为 `True`，随后同一模型可通过 `role_name="demo_role"` 读取已保存的参考音色资产。

### 3.3 continuation 续写

准备 `reference.wav`，其中只录下 `今天我们先讨论两个问题。`。转写必须按你的实际录音修改。在完整脚本的 `try` 内替换创建 `chunks` 的部分，并将输出文件名改为 `voxcpm2-continuation.wav`。

```python
prompt = engine.make_prompt(
    role_name="continuation_role",
    audio_path="reference.wav",
    prompt_text="今天我们先讨论两个问题。",
    persist=False,
)
chunks = list(engine.stream(
    text="接下来，我们继续讨论第三个问题。",
    role_name=prompt,
    mode="continuation",
))
```

输出保存本次新生成的音频块。需要把已有录音和新语音做成一个连续文件时，由应用读取参考录音、转换到输出采样率，再按所需边界衔接。

### 3.4 `ref_continuation`

沿用上一节的录音及准确转写，在完整脚本的 `try` 内使用下面片段，并将输出文件名改为 `voxcpm2-ref-continuation.wav`。同一条录音同时提供音色和续写条件。

```python
prompt = engine.make_prompt(
    role_name="reference_continuation_role",
    audio_path="reference.wav",
    prompt_text="今天我们先讨论两个问题。",
    persist=False,
)
chunks = list(engine.stream(
    text="如果继续往下分析，第三个问题与前面两个问题有共同原因。",
    role_name=prompt,
    mode="ref_continuation",
))
```

这四种模式共用下面的请求参数。未列在当前适配器公开集合中的参数会在参数规范化阶段被过滤，调用时应使用下表及上面的模式、控制字段。

| 参数 | 默认值 | 作用和条件 |
| --- | --- | --- |
| `max_length` | `300` | 输入分段的 token 上限，须为正整数。token 是分词器输出的模型输入单位。同时将每段异常生成保护上限设为该值的两倍音频片段数。 |
| `min_length` | `20` | 合并较短文本段的 token 阈值，须为非负整数。与上限同时指定时不能大于上限。 |
| `seed` | `None` | 整数随机种子。省略时每次使用随机状态，多段文本会在给定种子基础上逐段递增。 |
| `remove_meaningless_chars` | `True` | 按允许字符规则清理正文，并保护列出的特殊标签。 |
| `track_performance` | `False` | 开启预处理、生成、解码和音频块耗时日志。 |

初始化的 `n_ctx` 默认 8,192，决定生成组件可保留的上下文容量。扩散步数 `inference_timesteps=10`、条件引导强度 `cfg_value=2.0` 和流式前缀长度 `streaming_prefix_len=4` 由当前模型配置固定，当前公开 `stream()` 参数中只开放表中这些相关控制。

## 4. 参考与致谢

OpenBMB 团队提供了原始 VoxCPM2 模型。声音设计、克隆及续写的上游说明见 [官方最佳实践](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html)，本页调用方式以 StreamVox 当前适配器为准。
