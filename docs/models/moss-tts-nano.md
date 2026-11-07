# MOSS-TTS-Nano 能力说明

[返回 README](../../README.md) · [使用文档](../usage.md) · [模型许可与来源](../model-licenses.md)

MOSS-TTS-Nano 将文字逐步生成为语音，StreamVox 接入型号为 `moss-tts-nano-onnx`。它可以在 CPU 上运行，支持用一条录音提取音色，也支持直接从文字生成语音。

模型目录（bundle）保存权重、词表和 `manifest.json`。参考音色资产（Prompt）保存参考录音经过编码得到的模型输入，以及可选的录音转写。先调用 `make_prompt()` 建立 Prompt，后续合成即可复用同一个声音，省去重新读取和编码参考录音的步骤。

默认转换权重采用 Apache-2.0，SDK 与模型分别适用各自许可。原始权利人和转换来源见 [模型许可与来源](../model-licenses.md)。

## 1. 系列定位

MOSS-TTS-Nano 的语音生成主模型约有 1 亿参数，另配套一个把波形与模型音频编码互相转换的音频编解码器。当前适配通过 ONNX Runtime 运行这些模型文件。ONNX Runtime 是执行已导出模型计算的运行库，SDK 推理链路无需加载 PyTorch。

没有独立显卡、需要本地助手回复或设备播报时，可用 `device="cpu"` 启动。流式输出表示模型逐步返回音频块，是否能跟上实时播放还取决于 CPU、参考录音长度和输入内容，需在目标机器上测量。

当前规格给出的 CPU 内存经验档位为最低 3 GB、推荐 5 GB。`auto` 和 `gpu` 也属于可接受的设备值，实际后端由运行环境选择，初始化后可通过 `engine.actual_backends` 查看。CPU 示例见后文。

## 2. 支持语言与文本风格

上游具名列出的语言包括中文、英文、德语、西班牙语、法语、日语、意大利语、匈牙利语、韩语、俄语、波斯语、阿拉伯语、波兰语、葡萄牙语、捷克语、丹麦语、瑞典语、希腊语和土耳其语。上游标题写为 20 种语言，当前表格实际列出上述 19 种。本页保留具名名单，语言效果需结合自己的输入验证。[MOSS-TTS-Nano 官方语言表](https://github.com/OpenMOSS/MOSS-TTS-Nano#supported-languages)

StreamVox 当前没有为 MOSS 暴露 `language` 参数，直接填写目标语言正文。适配包含轻量字符清理和长文本分段，数字、日期、金额、缩写及单位的读法应在业务侧明确。例如希望金额 `¥12.50` 读作中文时，可以先整理成 `十二元五角`。

助手回复和通知可按一句或一小段提交，便于确认台词与声音。长文本会自动分段，具体分段与生成上限见第 4 节。

## 3. 核心能力

### 3.1 单参考音色克隆

`voice_clone` 从单条参考音频提取音色条件，让这个声音说新的 `text`。当前实现的这一模式只消费参考音频编码，`prompt_text` 可以省略，传入后也不会加入此模式的生成请求。

参考录音可以说 `你好，欢迎使用语音服务。`，待生成文本可以是 `好的，下面我继续介绍部署流程。`。参考音频提供音色，新文本提供本次台词。录音与待生成台词可分别准备。

`make_prompt()` 只接受一条参考音频，路径、音频数组、采样率和参考文本均使用单值。文件输入传 `audio_path`，内存输入传 `audio_data` 和明确的 `sample_rate`，两种音频来源互斥。参考音频会转换到编解码器需要的采样率和声道布局。

### 3.2 continuation 与 voice_clone 两种模式

`continuation` 表示在生成序列中继续产生语音。无参考录音时，它直接从目标文本生成。有参考录音时，模型将录音作为已生成的语音前缀，同时把 `prompt_text` 和目标 `text` 相接，再生成后续语音。

| 模式 | 数字值 | 音频和转写要求 | 默认选择条件 |
| --- | ---: | --- | --- |
| `voice_clone` | `0` | 必须有参考音频，转写可省略且不参与该模式生成。 | 使用 `make_prompt()` 创建的参考音色资产时默认采用。 |
| `continuation` | `1` | 可以只有目标文本。使用参考条件时，音频与非空 `prompt_text` 必须同时提供。 | `role_name=None` 且没有参考输入时默认采用。 |

`mode` 也接受数字字符串 `"0"` 和 `"1"`。只有音频的 Prompt 切换到 `continuation` 会因缺少转写而报错，此时需要重建带准确转写的 Prompt。无参考条件时应保持 `role_name=None`，引擎才会构造临时的纯文本输入。

### 3.3 实时流式输出

`stream()` 返回一个需要迭代的生成器。模型生成一批音频编码后，解码器就把对应波形交给调用方，后面仍可继续生成。只调用方法并保存生成器变量时，合成尚未执行。

当前 MOSS 适配始终逐块输出一维 `float32` 的 NumPy 数组。配套编解码器按 48,000 Hz 工作，SDK 将其音频输出转换为单声道。播放或保存时读取 `engine.runtime.sample_rate`，让播放器和文件使用实际采样率。

### 3.4 Prompt 可复用

`make_prompt()` 会先读取录音并运行参考音频编码器，把结果放进 Prompt。传 `role_name=prompt` 可在内存中重复使用同一份资产。默认 `persist=True` 还会写入本地角色缓存，后续使用同一模型时，可传角色名字符串读取。

如果角色会多次发声，将 Prompt 构建放在角色创建阶段，正式请求只提交新文本。这样首个音频块前就能省去这次参考编码的计算。首次引擎初始化还包含模型加载和预热，应与正式合成耗时分别观察。

## 4. 工程侧推荐用法

### 4.1 推荐先 make_prompt，再正式 stream

先准备清楚的单人录音，使用 `make_prompt()` 构建参考音色资产。需要复用时保留 Prompt 对象，或用 `persist=True` 保存角色。每次新请求通过 `stream()` 传入新台词，并逐块播放或完整收集。

`voice_clone` 可仅提供录音。需要做带参考的续写时，在构建 Prompt 时同步提供准确转写，并在合成时显式传 `mode="continuation"`。先分清这两种输入结构，可以避免把缺少转写的克隆资产直接用于续写。

### 4.2 推荐CPU 场景优先尝试

按 [使用文档](../usage.md) 安装 SDK 并选择 CPU 依赖后，使用 `TTSEngine(model="moss-tts-nano-onnx", device="cpu")` 初始化。通过模型名启动会按 SDK 下载规则获取模型目录，也可传入已准备好的本地模型目录。

当前 MOSS 全部模型组件使用 ONNX 链路，初始化参数 `llama_backend` 不参与设备选择。运行在 CPU 时，可以先用下方纯文本示例检查输入到文件的完整流程，再加入参考音色。

### 4.3 推荐中短文本优先

每次先提交一句或一小段，便于检查数字读法、标点停顿和角色音色。较长正文由当前模型配置按 `max_length=80`、`min_length=12` 分段，长度按分词器输出的 token 计数，token 是模型读取文本的基本单位。

每个文本分段会重新构造请求并重置该段的音频编解码状态，生成上限为 `max_new_frames=375`。这里的帧是模型生成的音频编码时间单元。长文本跨段时，前一段生成的声音历史会结束，下一段重新使用参考条件。

| 当前请求参数 | 默认值 | 作用 |
| --- | --- | --- |
| `mode` | 随 Prompt 选择 | `voice_clone` 或 `continuation`，也可传 0、1。 |
| `seed` | `None` | 给定整数时重置音频采样的随机数状态。省略时沿用运行时的随机状态。 |
| `remove_meaningless_chars` | `True` | 按 MOSS 的字符范围过滤正文。 |
| `track_performance` | `False` | 开启后记录预处理、参考条件请求构造、首包及音频块等耗时。 |

分段长度、最大生成帧数和采样参数由当前 YAML 配置管理。`TTSEngine.stream()` 的通用 `max_length`、`min_length` 在 MOSS 适配器中会被过滤，对本模型不生效。需要控制单次输入长度时，可在应用侧按句提交。当前默认采样采用 `fixed` 模式，`do_sample=True`、温度 0.8、`top_p=0.95`、`top_k=25`、重复惩罚 1.2。`fixed` 使用导出时固定采样参数的整帧计算图，公开请求仅通过 `seed` 重置其中的随机过程。

## 5. 不适合的场景

当前 MOSS 接口围绕一个目标声音组织请求，每份 Prompt 只接收一条录音，公开参数提供模式与随机种子控制。多角色对白需要的说话人标记入口可查看 [S2-Pro](s2-pro.md)，通过文字设计音色和自然语言控制表达的入口可查看 [VoxCPM2](voxcpm2.md)。

选择这些接口时，先依据所需输入结构和控制方式确定模型，再用同一组台词、参考录音和目标设备比较声音及延迟。

## 6. 典型示例

先按 [使用文档](../usage.md) 安装 SDK 并准备模型。下面第一个脚本还需要工作目录中存在一段自己有权使用的 `reference.wav`，项目不附带这段录音。两个脚本均使用主入口守卫启动，并在结束或异常时释放引擎。

### 6.1 voice_clone

这段脚本使用录音提取音色，完整消费生成器后保存 `moss-clone.wav`。由于使用 `voice_clone`，构建 Prompt 时可以省略转写。

将下面代码保存为 `moss_clone_demo.py`。

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from streamvox import TTSEngine


def main():
    """用单条参考录音的音色合成并保存新台词。

    Returns:
        无返回值，合成音频保存为 moss-clone.wav。
    """
    engine = TTSEngine(
        model="moss-tts-nano-onnx",
        device="cpu",
        verify_model_sha256=True,
    )
    try:
        prompt = engine.make_prompt(
            role_name="moss_voice",
            audio_path="reference.wav",
            persist=False,
        )
        chunks = list(engine.stream(
            text="好的，下面我继续介绍这个模型的部署流程。",
            role_name=prompt,
            mode="voice_clone",
            track_performance=True,
        ))
        audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("本次合成未返回音频")
        output = Path("moss-clone.wav")
        sf.write(output, audio, engine.runtime.sample_rate)
        print(output.resolve())
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
```

### 6.2 continuation

纯文本生成无需参考录音。在上面脚本的 `try` 内，将创建 Prompt 和 `chunks` 的语句一起替换为下面片段，输出文件名改为 `moss-text.wav`。

```python
chunks = list(engine.stream(
    text="这段语音只使用目标文本生成。",
    role_name=None,
    mode="continuation",
    track_performance=True,
))
```

需要沿着一段已有语音续写时，准备 `reference.wav` 并录下 `今天我们先介绍部署环境。`，把转写改成实际录音内容。将上面脚本创建 Prompt 和 `chunks` 的部分替换为以下片段，输出文件名改为 `moss-continuation.wav`。

```python
prompt = engine.make_prompt(
    role_name="moss_continuation",
    audio_path="reference.wav",
    prompt_text="今天我们先介绍部署环境。",
    persist=False,
)
chunks = list(engine.stream(
    text="接下来，我们开始安装运行依赖。",
    role_name=prompt,
    mode="continuation",
    track_performance=True,
))
```

录音与准确转写共同提供前缀上下文，`text` 只填写后续台词。脚本保存本次生成的音频块。要边生成边播放，可将 `list(...)` 换为循环，并逐块送入相同采样率的播放器。

## 8. 参考与致谢

OpenMOSS 团队提供了原始 MOSS-TTS-Nano 及配套音频编解码模型。上游 CPU 与 ONNX 工作流见 [官方仓库](https://github.com/OpenMOSS/MOSS-TTS-Nano)，本页说明 StreamVox 当前单参考和续写接口。
