# StreamVox 使用文档

[返回 README](../README.md) · [模型许可](model-licenses.md) · [构建与开发](development.md)

StreamVox 用 `TTSEngine` 管理一次语音合成所需的模型、参考音色和音频输出。通常先加载模型，再把参考录音编码成可复用的音色资产，最后迭代读取生成的音频块。声音设计模式直接使用文字描述音色，具体能力由模型决定。

本文对应 2.1.2 的公开调用方式。Windows x64 / CPython 3.12 使用发行安装包，Linux x86_64 从源码运行。两种环境的安装步骤见 README。下文的参考音频路径和转写需要替换为自己的素材。

## 初始化

下面是完整的 Qwen3-TTS CPU 克隆脚本。`engine` 持有已加载的模型，`prompt` 保存参考录音的编码结果，`chunks` 收集本次生成的波形。保存为 `clone.py`，准备好 `your-reference.wav` 并填写实际转写后，运行 `python clone.py` 会得到 `result.wav`。

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
        verify_model_sha256=True,
    )
    try:
        prompt = engine.make_prompt(
            role_name="demo",
            audio_path="your-reference.wav",
            prompt_text="请填写参考录音中实际说出的内容。",
            persist=False,
        )
        chunks = list(engine.stream(
            text="你好，这是一段新的文本。",
            role_name=prompt,
            language="chinese",
        ))
        if not chunks:
            raise RuntimeError("本次合成没有生成音频。")
        sf.write("result.wav", np.concatenate(chunks, axis=-1), engine.runtime.sample_rate)
    finally:
        engine.shutdown()
```

Windows 上部分模型会创建工作子进程。`if __name__ == "__main__":` 让子进程导入脚本时跳过主程序，避免再次加载整套模型。`finally` 在合成结束或异常时释放引擎占用的资源。完整命令行实现见项目示例。[synthesize.py](../example/synthesize.py)

后文参考音色和音频块处理的局部片段沿用已初始化的 `engine`，放在完整脚本的 `try` 块内。初始化替换和独立下载脚本会另外说明。`language="chinese"` 是 Qwen3 参数，其他模型的参数见对应指南。

除 `model` 外，初始化参数均须按名称传入。

| 参数 | 作用与取值 |
| --- | --- |
| `model` | 模型注册名或本地模型目录。注册名让 SDK 定位缓存并按需下载，本地目录让 SDK 读取该目录中的清单。 |
| `device` | `auto`、`cpu`、`gpu`、`gpu:<index>`，指定默认设备类型与编号。 |
| `llama_backend` | 选择 GGUF 模型的计算后端。Windows 安装包提供 `cpu`、`cuda`、`vulkan`，省略时由模型选择。 |
| `onnx_provider` | 选择 ONNX 图的执行后端，可传 `cpu`、`directml`、`cuda`，省略时由模型选择。 |
| `n_ctx` | 向模型请求上下文窗口长度。上下文是生成时可读取的先前文本或音频位置，具体单位、默认值和边界由模型定义。 |
| `verify_model_sha256` | 默认 `True`。读取清单中的运行资产并比较 SHA256 内容摘要，缺失或摘要不符时拒绝加载。 |
| `registry` | 自定义模型注册表，将模型名映射到适配器。普通使用者沿用默认注册表即可。 |

初始化会解析模型目录、检查资产、创建计算后端，并执行模型声明的初始化预热。它的耗时应与每次 `stream()` 的首包耗时分别记录。

## 模型与本地目录

模型目录也称 bundle，里面包含权重、配置、分词器、音频编解码文件和 `manifest.json`。清单记录模型名、资产路径及内容摘要，SDK 据此检查目录是否完整。安装包提供运行代码和原生库，模型目录提供合成所需的模型数据。

按模型名初始化时，SDK 依次检查注册表和默认模型目录。目录中已有对应清单时复用该目录，加载时再按校验开关检查运行资产。缺少本地清单时，从默认 ModelScope 仓库下载快照，校验完成后发布到正式模型目录。

若已有本地模型，将完整脚本初始化调用中的 `model` 改为 `"./ckpts/qwen3-tts-clone-0.6b-gguf"`。该目录的 `manifest.json` 必须声明已注册的模型名。相对路径以启动脚本时的工作目录为基准。

五个模型家族的模式、参考要求和参数分别见模型指南。

- Qwen3-TTS 提供单参考音色克隆，包含 0.6B 和 1.7B 两个注册名。[Qwen3-TTS](models/qwen3-tts-clone.md)
- S2-Pro 使用参考音频与说话人标记组织多参考、多说话人合成。[S2-Pro](models/s2-pro.md)
- VoxCPM2 提供声音设计、参考克隆和参考续写，模式决定参考音频如何参与生成。[VoxCPM2](models/voxcpm2.md)
- MOSS-TTS-Nano 使用纯 ONNX 路径，可进行无参考生成、音色克隆和续写。[MOSS-TTS-Nano](models/moss-tts-nano.md)
- Breeze TTS 2 提供中英文声音设计和参考克隆。2.1.1 起按 `breeze-tts-2-gguf` 初始化，会下载 Q8_0 GGUF 与 FP32 ONNX 的完整模型目录。[Breeze TTS 2](models/breeze-tts-2.md)

Windows 默认数据根目录为 `%APPDATA%/StreamVox`，Linux 为 `~/.streamvox`。模型放在 `ckpts/`，持久参考音色放在 `prompts/`。在启动 Python 前设置 `STREAMVOX_HOME`，可让模型和角色缓存统一存到指定位置。

SDK 运行环境使用 llama.cpp 和 ONNX Runtime。离线导出与数值对照工具另用 Torch 环境，环境准备见开发文档。模型权重的使用与再分发条款见许可说明。[开发文档](development.md) · [模型许可](model-licenses.md)

## Prompt 与缓存

参考音色资产（Prompt）保存模型从参考录音中提取的编码结果。`make_prompt()` 先完成这一步，之后多次 `stream()` 可以复用同一份资产，将重复参考处理移出正式合成请求。

临时资产通过 `persist=False` 创建，直接把返回对象传给 `role_name`。对象由当前 Python 进程持有。

```python
prompt = engine.make_prompt(
    role_name="temporary_voice",
    audio_path="your-reference.wav",
    prompt_text="请填写参考录音中实际说出的内容。",
    persist=False,
)
for chunk in engine.stream(text="使用同一个音色朗读新文本。", role_name=prompt):
    print(chunk.shape)
```

要在后续引擎实例或程序重启后按名字调用音色，改用 `persist=True`。SDK 会按模型保存该角色，之后可用 `role_name="角色名称"` 读取。角色名在当前模型内应唯一，替换已有角色前可调用 `del_roles()`。每个模型使用自己的参考编码格式，缓存目录按模型隔离。

录音已在内存中时，使用 `audio_data` 和实际采样率 `sample_rate`。例如先从文件读出波形，再把这两个值一起交给 SDK。一次调用从文件路径和内存波形中选择一种输入方式。

```python
reference_audio, reference_rate = sf.read("your-reference.wav", dtype="float32", always_2d=True)
reference_audio = reference_audio.mean(axis=1, dtype=np.float32)
prompt = engine.make_prompt(
    role_name="memory_voice",
    audio_data=reference_audio,
    sample_rate=reference_rate,
    prompt_text="请填写参考录音的实际转写。",
    persist=False,
)
```

`always_2d=True` 让读入结果固定为“采样点数 × 通道数”，再按通道取平均得到单声道波形。`sample_rate` 使用文件实际采样率，SDK 才能按正确的时间尺度处理录音。

`stream()` 也可通过 `prompt_audio_path`，或 `prompt_audio_data` 与 `prompt_audio_sample_rate` 即时接收参考音频。在尚无可复用 Prompt 时，调用会构建临时参考音色，处理时间计入首包等待。已有 Prompt 对象或命中角色缓存时，优先复用该资产，即时音频和本次 `prompt_text` 会被忽略。要更换参考录音或转写，应重新创建资产。参考文本是否必需，由模型和模式决定。

角色管理可以查看名称、读取保存的参考资料或删除角色。

```python
print(engine.list_roles())

role = engine.get_prompt_data("saved_voice")
if role is not None:
    print(role.role_name, role.sample_rate, role.prompt_text)
    print([reference.audio.shape for reference in role.references])

deleted = engine.del_roles("saved_voice")
print(deleted)
```

`get_prompt_data()` 返回 `PromptData`，其中包括参考文本列表、按位置对应的参考音频，以及模型编码后的 `prompt`。无需转写的位置可保存 `None`，较早生成的缓存也可能只有模型编码而缺少公开参考资料。找不到角色时返回 `None`。`del_roles()` 接受一个名称或名称列表，返回实际删除的名称列表。

## 流式输出与保存

`engine.stream()` 返回一个逐步生成音频的迭代器，开始迭代后才执行合成。每个音频块是 `float32` 类型的 NumPy 数组，保存未经 WAV 编码的 PCM 波形采样值。现有模型输出单声道，采样率从 `engine.runtime.sample_rate` 读取。

完整脚本用 `list()` 收集所有块，再用 `np.concatenate()` 按先后顺序拼接，适合一次保存成 WAV。实时播放或网络发送应在 `for chunk in engine.stream(...)` 的循环中处理每一块，第一块到达后即可开始消费。

长音频可以边生成边写入文件，避免把全部波形同时留在内存。下面片段沿用已经准备好的 `prompt`。

```python
stream = engine.stream(text="这是一段逐块保存的示例。", role_name=prompt)
try:
    with sf.SoundFile(
        "long-result.wav", mode="w", samplerate=engine.runtime.sample_rate,
        channels=1, subtype="PCM_16",
    ) as output:
        for chunk in stream:
            output.write(chunk)
finally:
    stream.close()
```

提前停止读取时调用 `stream.close()`，让底层合成会话释放本次请求资源。程序结束时再调用 `engine.shutdown()`，释放引擎。命令行示例也采用逐块写入方式。[synthesize.py](../example/synthesize.py)

若要通过网络传输原始浮点波形，可把块转换为 `chunk.tobytes()`。接收端需要另外知道采样率、通道数、字节序和数据类型，才能把这些字节还原成相同的波形。`track_performance=True` 可让支持的模型记录本次请求的性能分解。

## 设备与实际后端

后端是执行模型运算的运行库。llama.cpp 读取 GGUF 格式的生成模型，ONNX Runtime 执行 ONNX 格式的计算图。一个模型可以把不同阶段交给不同后端，因此需要分别选择 `llama_backend` 与 `onnx_provider`。

| `device` | 默认设备请求 |
| --- | --- |
| `auto` | 由模型结合可用设备选择。 |
| `cpu` | 请求 CPU。 |
| `gpu` | 请求默认 GPU，等价于 `gpu:0`。 |
| `gpu:<index>` | 请求指定编号，例如 `gpu:1`。 |

`device` 提供默认设备类型和编号，显式后端参数分别覆盖对应部分。例如 `device="cpu", llama_backend="cpu", onnx_provider="cuda"` 表达 CPU 生成与 CUDA ONNX 的组合。MOSS 的各阶段全部采用 ONNX，使用 `onnx_provider` 选择执行后端。

下面的初始化语句适用于已经安装 `[directml,llama-cuda]` 依赖的 NVIDIA 环境。将完整脚本中的初始化调用替换为这一段，再使用 VoxCPM2 指南中的生成模式和参数。

```python
engine = TTSEngine(
    model="voxcpm2-gguf",
    device="gpu:0",
    llama_backend="cuda",
    onnx_provider="directml",
)
```

查看 `engine.actual_backends` 可以取得实际选择的 `llama`、`onnx` 后端，以及主进程可见的 `onnx_sessions` 列表。列表中的 provider 是会话启用的执行提供程序，决定哪些后端可以接收图中的计算节点。一个会话可同时启用 GPU 和 CPU provider，具体节点按支持情况分配。纯 ONNX 模型会省略 `llama` 键，子进程里的会话通过初始化日志核对。

显式请求的后端不可用，或初始化及合成时核对到后端与请求不符，SDK 会抛出异常。省略后端参数时，选择规则由各模型运行时决定。ONNX 会话可能有按模型设计保留的 CPU 部分。例如 MOSS 的 DirectML 路径用 CPU 执行音频解码步骤，Breeze 的音频嵌入、深度条件投影和参考编码也使用 CPU。

Breeze 默认文本策略为 `auto`。声码器使用 CUDA，且清单声明全部三个快速文本资产时，运行时用 CPU ONNX 查表，再用 CUDA ONNX 执行文本主干。其余情况下沿用完整 CPU 文本图。这样可以让大文本运算使用 GPU，同时将嵌入表留在主机内存。组件选择条件和显式覆盖方法见模型指南。[Breeze 指南](models/breeze-tts-2.md)

多卡环境需要结合日志和系统设备列表确认编号。Vulkan 枚举设备的顺序可能与 CUDA 不同，`CUDA_VISIBLE_DEVICES` 的重映射也不能直接当作 Vulkan 编号。

安装依赖时，`cpu`、`directml`、`onnx-cuda` 三组互斥，因为它们安装不同的 ONNX Runtime 发行包。`llama-cuda` 补充 llama.cpp 使用 CUDA 所需的 NVIDIA 运行依赖，可与其中任一组组合。Windows 安装包始终携带 CPU、CUDA、Vulkan 三套 llama.cpp 动态库，运行时按请求加载对应后端。

## 模型参数

`stream(..., **model_kwargs)` 将模型私有参数交给当前适配器检查。参数名相同也可能有不同含义，应按模型指南选择。上面的完整脚本使用 Qwen3 的 `language`，Breeze 的声音描述使用 `instruction`。

| 模型 | 常用参数及作用 |
| --- | --- |
| Qwen3 | `language` 指定目标语言标记，`auto` 当前回退中文标记。`icl` 默认 `False`，开启后结合参考文本和音频上下文。`stream` 选择模型内部的流式生成方式。 |
| S2-Pro | `temperature`、`top_p`、`top_k` 调整候选编码的采样，speaker 标记组织不同说话人的参考和目标文本。 |
| VoxCPM2 | `mode` 选择设计、克隆或续写。`control_text` 在 `text`、`ref` 模式描述声音，另外两种续写模式会警告后忽略它。 |
| MOSS | `mode` 选择无参考生成、克隆或续写，`seed` 固定当前请求的随机种子。 |
| Breeze TTS 2 | `instruction` 描述声音，`cfg_scale` 控制条件引导，两级采样参数分别控制主干与深度解码器，`chunk_frames` 设置返回块包含的音频帧数。 |

`max_length`、`min_length`、`remove_meaningless_chars` 由模型解释，前两项通常用于文本预处理分段。Qwen3 和 S2-Pro 可按其指南设置这些字段，MOSS 的分段阈值来自配置，调用中的这些通用字段会被过滤。Breeze 保留原始文本段和事件标记，其音频生成长度使用 `max_new_tokens`，文本分段参数不会代替这一限制。数字、日期、金额和缩写有固定读法要求时，可直接把输入改写成期望朗读的词语。

Breeze 默认 `chunk_frames=2`，每帧表示 80ms 音频，通常返回 160ms 的块，末块可以为 80ms。调整它会改变累计多少帧后向调用方返回一次，主干和声码器仍连续维护上下文。[模型参数和流式边界](models/breeze-tts-2.md#采样与流式边界)

### 多参考音频

S2-Pro 支持多参考音频。音频和转写按列表位置一一对应，参考文本及目标文本使用相同的 speaker 标记，使模型知道每段话归属哪个说话人。下面片段应放到 S2-Pro 引擎的 `try` 块中。

```python
prompt = engine.make_prompt(
    role_name="two_speakers",
    audio_path=["your-speaker-a.wav", "your-speaker-b.wav"],
    prompt_text=[
        "<|speaker:0|>第一位说话人的实际转写。",
        "<|speaker:1|>第二位说话人的实际转写。",
    ],
    persist=False,
)
chunks = list(engine.stream(
    text="<|speaker:0|>你好。<|speaker:1|>很高兴见到你。",
    role_name=prompt,
))
```

Qwen3、VoxCPM2、MOSS 和 Breeze 当前接收单参考音频。Breeze 克隆还需要参考录音的完整转写。S2-Pro 权重的使用条件见独立许可。[S2-Pro 许可](model-licenses.md#s2-pro)

## 模型下载与固定版本

普通模型名初始化和普通 `download_model()` 使用默认 ModelScope 仓库的最新快照。`target` 指定下载到哪个根目录，最终模型目录位于 `target/model`。手工下载到自选目录后，使用返回的 `target_dir` 初始化引擎。

```python
from streamvox.download import download_model


if __name__ == "__main__":
    result = download_model(model="qwen3-tts-clone-0.6b-gguf", target="./ckpts")
    print(result.target_dir)
```

要在不同机器上取得同一份资产，需要同时固定提交与清单。提交是仓库一次上传的 40 位标识，清单 SHA256 是该次 `manifest.json` 内容的 64 位摘要。前者选择版本，后者核对清单是否与发布记录一致。

下面采用 Breeze Q8 的已发布记录。SDK 先核对该提交的原始清单，再下载清单中的运行资产和识别到的法律文件，逐项校验后将临时目录切换为正式目录。

```python
from streamvox.download import download_model


if __name__ == "__main__":
    result = download_model(
        model="breeze-tts-2-gguf",
        target="./ckpts",
        revision="58e63c94b99a2f6f2883aa02ebc0c3e54f7a278a",
        expected_manifest_sha256="910d5a4e80d5fd7cf51986822c83c7f8bfc1c38ce88826bbce2457f0a7c2312e",
    )
    print(result.target_dir)
```

用 `TTSEngine(model=str(result.target_dir), ...)` 加载上述目录，可继续使用该固定版本。单独调用模型名入口时，SDK 仍按默认目录规则解析模型。

`progress_callback` 接收下载阶段和进度，`cancel_event` 允许在文件处理边界取消，`cache_dir` 设置可复用的下载缓存。取消后缓存保留，正式目录的切换发生在校验完成后。固定版本下载保留其原始清单，普通快照下载会为本地运行资产重建清单并记录下载来源。[下载实现](../streamvox/download.py) · [再分发许可](model-licenses.md#下载与再分发)

## 故障排查

先记录模型名、SDK 版本、平台和 `engine.actual_backends`，再按异常出现的阶段检查。

| 阶段或现象 | 检查与处理 |
| --- | --- |
| 安装失败 | 核对 CPython 3.12、Windows x64 和 wheel 文件名。Linux 按源码步骤使用 `--no-install-project`。同一环境选择一种 ONNX Runtime。[安装说明](../README.md#安装) |
| 找不到 DLL 或共享库 | Windows 源码环境先恢复原生资产，Windows wheel 使用包内 DLL。Linux 下载同版本 `.so` 并在启动前设置 `LD_LIBRARY_PATH`。[原生库说明](development.md#windows-原生运行库) |
| 模型目录无效或摘要不符 | 检查清单里的模型名、`ckpt/` 文件和来源版本，重新取得缺失或损坏的资产。 |
| 角色不存在 | 跨实例调用的角色先以 `persist=True` 保存，临时资产直接传返回对象。 |
| 后端不可用 | 核对依赖组、驱动、原生库和显式后端参数，并从日志确认可用设备。 |
| 吞字、异常音或音色漂移 | 核对转写与参考录音是否对齐，检查模型模式、目标文本与采样参数，用最短输入复现。 |

验收记录按模型、权重精度和后端分别解释。Breeze Q8 已完成对应固定语料内容复核及两段用户试听，其严格数值对照仍保留超限记录。其他模型和平台的状态见各自记录，单元测试只能证明测试覆盖的行为。[Breeze 验收记录](models/breeze-tts-2-validation.json) · [SDK 2.0.0 性能记录](performance-2.0.0.json)
