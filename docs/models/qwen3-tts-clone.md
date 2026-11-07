# Qwen3 TTS Clone 系列能力说明

[返回 README](../../README.md) · [使用文档](../usage.md) · [模型许可与来源](../model-licenses.md)

Qwen3 TTS Clone 根据一段录音提取说话人的声音特征，再让这个声音朗读新文本。StreamVox 接入 `qwen3-tts-clone-0.6b-gguf` 和 `qwen3-tts-clone-1.7b-gguf`，两者共用调用方式。

模型目录（bundle）保存权重、词表和 `manifest.json`，引擎从中加载合成需要的文件。参考音色资产（Prompt）保存从参考录音提取的音色特征、音频编码和解码初始状态。`make_prompt()` 负责制作这份资产，`stream()` 使用它生成新的语音。同一个 Prompt 可以重复使用，省去每次请求重新编码录音的开销。

默认转换权重采用 Apache-2.0，SDK 与模型分别适用各自许可。原始权利人和转换来源见 [模型许可与来源](../model-licenses.md)。

## 1. 支持语言

上游 Qwen3-TTS 声明支持中文、英文、日文、韩文、德文、法文、俄文、葡萄牙文、西班牙文和意大利文这 10 种主语言。当前 StreamVox 的 `language` 接受下表中的小写字符串。[Qwen3-TTS 官方说明](https://github.com/QwenLM/Qwen3-TTS)

| 目标语言 | `language` 值 |
| --- | --- |
| 中文 | `chinese` |
| 英文 | `english` |
| 日文 | `japanese` |
| 韩文 | `korean` |
| 德文 | `german` |
| 法文 | `french` |
| 俄文 | `russian` |
| 葡萄牙文 | `portuguese` |
| 西班牙文 | `spanish` |
| 意大利文 | `italian` |

适配器还接受 `sichuan_dialect` 和 `beijing_dialect`，分别映射到四川话和北京话的语言标记。发音和方言表现需要结合参考录音与目标文本试听确认。

`language` 默认是 `auto`。当前文本处理流程保留这个值，运行时随后将它回退为中文语言标记，流程中没有语言检测。因此，生成英文时应显式传 `language="english"`，生成中文时传 `language="chinese"`。这一参数描述本次待合成文本的语言。

## 2. 从参考录音生成一段新语音

先按 [使用文档](../usage.md) 安装 SDK 并准备所选模型目录。下面的独立脚本使用 CPU，首次通过模型名初始化时会按 SDK 的下载规则获取模型。准备当前目录下的 `reference.wav`，录音内容须与 `prompt_text` 对应。项目不附带这段录音。

将下面代码保存为 `qwen_clone_demo.py`。

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from streamvox import TTSEngine


def main():
    """从单条参考录音构建音色并保存新语音。

    Returns:
        无返回值，合成音频保存为 qwen-clone.wav。
    """
    engine = TTSEngine(
        model="qwen3-tts-clone-0.6b-gguf",
        device="cpu",
        verify_model_sha256=True,
    )
    try:
        prompt = engine.make_prompt(
            role_name="order_voice",
            audio_path="reference.wav",
            prompt_text="你好，欢迎使用语音服务。",
            persist=False,
        )
        chunks = list(engine.stream(
            text="您好，您的订单已经发货，预计明天下午送达。",
            role_name=prompt,
            language="chinese",
        ))
        audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("本次合成未返回音频")
        output = Path("qwen-clone.wav")
        sf.write(output, audio, engine.runtime.sample_rate)
        print(output.resolve())
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
```

`stream()` 返回可迭代的音频块。脚本通过 `list()` 完整消费生成器后，才会完成合成并写出 `qwen-clone.wav`。每块是一维 `float32` 的 NumPy 数组，采样率为 24,000 Hz。需要边生成边播放时，在循环里把每块交给播放器即可，播放器应使用相同采样率。

把 `persist` 改为 `True` 会将 Prompt 写入本地角色缓存。后续使用同一模型初始化引擎后，可传 `role_name="order_voice"` 读取已保存角色。内存中的 Prompt 对象则通过 `role_name=prompt` 直接复用。

## 3. 核心能力

### 3.1 单参考音色克隆

`make_prompt()` 每次只接受一条参考录音。音频路径、音频数组、采样率和参考文本都不能使用列表。文件输入使用 `audio_path`，内存输入使用 `audio_data` 并显式给出 `sample_rate`，引擎会按模型采样率重采样。

默认 `icl=False` 时，模型使用从录音提取的说话人特征，参考转写文本可以省略。`icl=True` 会同时把参考转写和参考音频编码放入生成上下文，让目标文本接在参考文本之后。使用这一方式时，应准备与录音一致的 `prompt_text`。在上例的 `engine.stream(...)` 中加入 `icl=True` 即可切换。

当前 Clone 接口要求参考录音或可复用的 Prompt。公开参数中提供 `language` 和 `icl`，未提供自然语言音色设计入口。需要用文字描述设计新声音时，可查看 [VoxCPM2](voxcpm2.md)。

| 合成参数 | 默认值 | 作用和条件 |
| --- | --- | --- |
| `language` | `"auto"` | 选择目标语言标记，取值见语言表。当前 `auto` 最终回退中文。 |
| `icl` | `False` | 为 `True` 时将参考文本和参考音频编码加入上下文。 |
| `stream` | `True` | 控制模型内部的文本与音频条件组织方式。SDK 对外仍通过生成器返回音频块。 |
| `max_length` | `300` | 文本分段的 token 上限，token 是分词器把文本切成的模型输入单位，须为正整数。 |
| `min_length` | `30` | 合并短文本片段的 token 阈值，须为非负整数。与上限同时指定时应小于等于上限。 |
| `remove_meaningless_chars` | `True` | SDK 默认按该模型允许的字符范围过滤正文。关闭后保留更多原始字符。 |
| `track_performance` | `False` | 开启后记录合成阶段耗时，便于观察当前输入和设备的表现。 |

`max_length` 和 `min_length` 管理的是输入分段长度。单段的文本、参考条件和生成内容仍需装入模型上下文。初始化参数 `n_ctx` 决定可保留的上下文单位数，默认 4,096，运行时将显式值限制在 4,096 至 32,768 的范围内。

### 3.2 文本理解与自然韵律

模型根据正文和标点生成停顿、语调及重音。例如 `今天会议改到下午三点，别忘了提前十分钟到。` 中，逗号给出两个信息段之间的停顿线索。正文应直接写希望被读出的内容，日期、缩写和业务编号按实际朗读方式整理。

相同文字在不同录音、`icl` 设置和语言标记下可能有不同语气。对于通知、疑问或感叹等文本，用目标句式和标点表达意图，再对生成结果试听确认。

### 3.4 多语言生成

生成多地区版本时，可以复用参考音色，再分别传入目标语言正文和对应的 `language`。例如英文文本 `Your order has shipped.` 配 `language="english"`，中文文本配 `language="chinese"`。

对于 `欢迎来到控制台。Please verify your device before deployment.` 这样的混合文本，应先试听所选语言标记的结果。需要分别指定每部分语言时，将正文拆为中文和英文两次请求并按顺序播放。当前 `auto` 会使用中文语言标记，混合输入同样遵循这一规则。

## 4. 两个版本的差异

### 4.1 Qwen3 TTS Clone 0.6B

`0.6B` 表示约 6 亿参数。当前版本由 GGUF 生成模型和 ONNX 音频编解码组件共同完成合成，适合先验证较低资源占用的部署方案。GGUF 和 ONNX 是运行这些模型组件使用的文件格式与执行链路。

### 4.2 Qwen3 TTS Clone 1.7B

`1.7B` 表示约 17 亿参数，与 0.6B 共用克隆接口和语言参数。参数更多会改变模型的计算及存储需求。音色相似度、自然度和长句表现应使用同一组参考录音与目标文本进行比较。

### 4.3 选型建议

下表来自当前模型规格中的经验硬件档位，用于准备运行资源。实际用量还随上下文、后端和参考录音长度变化。

| 型号 | 显存最低 / 推荐 | 内存最低 / 推荐 |
| --- | --- | --- |
| `qwen3-tts-clone-0.6b-gguf` | 2 / 3 GB | 3 / 5 GB |
| `qwen3-tts-clone-1.7b-gguf` | 3 / 4 GB | 4 / 6 GB |

资源受限时可先运行 0.6B。需要比较两个版本的输出时，只替换示例的 `model` 值，为每个模型分别创建 Prompt，再记录同一设备下的延迟并试听。

## 5. 参考与致谢

Qwen 团队提供了原始 Qwen3-TTS 模型及相关工作。本页描述 StreamVox 当前 Clone 接入，其他上游模型的接口和能力见 [Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS)。
