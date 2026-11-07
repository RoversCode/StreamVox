# Breeze TTS 2

[返回 README](../../README.md) · [模型许可](../model-licenses.md) · [开发与验证](../development.md)

Breeze TTS 2 把中文或英文文本合成为语音，在 StreamVox 中的注册名为 `breeze-tts-2-gguf`。它支持三种输入方式。

- 直接传入 `text`，使用模型生成声音。
- 同时传入 `text` 和 `instruction`，用自然语言描述音色、语气或语速，进行声音设计。
- 先提供一段参考音频及其完整转录，调用 `make_prompt()` 创建参考音色资产（Prompt），再让模型用这段参考声音朗读新的 `text`。克隆时也能加入 `instruction` 控制表现。

参考音色资产保存参考转录和音频编码，可以按角色名持久化复用。模型权重、分词器、运行图和校验清单则放在模型目录（bundle）中。默认转换仓库 `ChengHee/Breeze-TTS-2-GGUF` 提供可直接加载的完整模型目录，日常合成按下文准备 SDK 环境即可。[ModelScope 模型](https://www.modelscope.cn/models/ChengHee/Breeze-TTS-2-GGUF)

一次合成按以下顺序进行。

1. 文本编码器把目标文本及可选指令变成模型可计算的向量。克隆时，参考转录单独编码，参考音频先转换成整数音频编码，两者共同组成参考前缀。
2. 主干读取参考前缀、目标文本和已经生成的音频历史，为下一帧预测第一个音频编码，或预测结束符。
3. 深度解码器结合主干状态和当前帧已有编码，补齐这一帧其余十五个编码。
4. 声码器把完整一帧的十六个编码还原成波形。相同编码还会转换成向量送回主干，让下一帧接着生成。SDK 按指定帧数合并波形并逐块返回。

输出是 24kHz 单声道音频，每秒包含 24,000 个采样值，数组用 `float32` 单精度浮点数保存。一个音频帧对应 80ms，默认每块合并两帧，包含 160ms 可播放音频。结束时若只剩一帧，末块为 80ms。音频块的播放时长与等待首块的耗时分别计量，具体边界见[采样与流式边界](#采样与流式边界)。

## 固定来源与许可

下列版本固定了转换时采用的模型公式、权重和原生运行接口，离线重建时应使用同一组来源。

| 来源 | 固定版本 |
| --- | --- |
| [Breeze 源码](https://github.com/breezeblue-ai/breeze-tts/tree/daa6cb2efe5471f09369229f322d20bb956cb3c3) | `daa6cb2efe5471f09369229f322d20bb956cb3c3` |
| [BreezeBlue/breeze-tts-2 检查点](https://huggingface.co/BreezeBlue/breeze-tts-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4) | `3e28c5151381a722f1d8661b4118c298caa77aa4` |
| [llama.cpp b8683](https://github.com/ggml-org/llama.cpp/tree/d0a6dfeb28a09831d904fc4d910ddb740da82834) | `d0a6dfeb28a09831d904fc4d910ddb740da82834` |

源码采用 Apache-2.0。**检查点权重、衍生模型和自托管输出仅供研究及非商业用途**。GGUF 与 ONNX 转换不改变权重许可。部署和再分发应保留对应的完整模型许可及归属文件，SDK 的 Apache-2.0 不授予该模型的商业使用权。

## 组件分工

模型用十六个码本共同表示一个 80ms 音频帧。码本是一张可查找的声音表示表，每个码本有 2048 个有效编号。每帧从每个码本选一个编号，得到十六个整数，声码器再将这组整数转换成声音。主干负责沿时间推进，深度解码器负责在同一帧内依次补齐码本，因此两者分别维护时间上下文和帧内上下文。

GGUF 是 llama.cpp 加载的模型权重格式。ONNX 图保存其他组件的计算步骤，由 ONNX Runtime 执行。这里的混合后端把两级生成交给 llama.cpp，把文本和音频转换交给 ONNX Runtime。文本先经过分词器变成词元（token，即分词器切出的编号单元），文本编码器再把每个词元转换成包含上下文信息的向量。

本页用 FP32 表示每个数占 32 位的单精度浮点格式，BF16 表示上游生成权重采用的 16 位浮点存储格式。Q8_0 则按分组量化，用 8 位整数及组内缩放值近似表达权重，从而缩小文件并改变部分计算结果。

| 组件 | 资产与后端 | 输入、输出及作用 |
| --- | --- | --- |
| Qwen3 主干 | `backbone.gguf`，llama.cpp，默认 Q8_0 | 根据完整因果上下文预测下一帧的首码本及结束类别，同时输出供深度解码器使用的隐藏状态。因果上下文只读取当前位置及过去的信息，用于计算注意力匹配分数的查询与键向量，仍按原公式归一化以调整幅度。 |
| Llama 深度解码器 | `depth.gguf`，llama.cpp，默认 Q8_0 | 接收主干状态和当前帧已有编号，依次生成剩余十五个码本。资产包含完整十六码本嵌入和十五组位置输出头，每组输出头负责对应位置的编号预测。 |
| T5Gemma2 文本编码器 | 完整 CPU `text_encoder.onnx`，或 CPU ONNX 查表加 CUDA ONNX 文本主干 | 独立编码参考转录与目标文本，利用双向局部和全局注意力结合前后文，再投影成生成主干所需维度。全部使用 FP32，选择条件见[使用](#使用)。 |
| 音频嵌入 | `audio_embedding.onnx`，CPU | 按码本偏移分别查表，再对十六行向量求和，把完整音频帧变成下一次主干输入。 |
| 深度条件投影 | `depth_conditioner.onnx`，CPU | 每帧开始时执行一次 `h @ P.T`，把主干的 2048 维隐藏状态转换成深度解码器使用的 1024 维条件向量。 |
| 参考音频编码器 | `audio_encoder.onnx`，按需加载到 CPU | 读取完整参考波形，通过原始因果注意力和残差量化选择声音表示，各级量化继续表达前一级留下的差异，输出每帧十六码本编号。 |
| 流式声码器 | `decoder_step.onnx`，ONNX CPU、CUDA 或 DirectML | 接收一帧完整码本及上一帧状态，输出连续波形和新状态。状态保留原生注意力窗口及卷积所需历史，供下一帧继续解码。 |

文本编码器的局部注意力窗口覆盖 512 个词元，其中当前位置左侧 255 个、右侧 256 个。b8683 的 `gemma-embedding` 使用对称窗口，与原模型的边界不同。直接替换架构名称会改变哪些词元能互相读取，因此该组件保留 ONNX 图。

深度解码器也需要把码本嵌入从主干维度转换到深度维度。导出时先对整张嵌入表应用等价变换 `E' = E @ P.T`，运行时就能直接查到投影后的向量。默认发布版本进一步将这张投影表按 Q8_0 量化。`depth_conditioner.onnx` 因而只处理每帧起始的主干状态，后续码本读取 GGUF 中的投影表。导出保留全部十六码本和十五组输出头，仅排除推理不读取的 `codec_model` 与 `embed_text_tokens`。

快速文本路径把查表与上下文计算分开。CPU 通过只读内存映射访问原始嵌入表，将它零复制注入 ONNX `Gather` 查表算子，再把查出的向量送到 CUDA ONNX 文本主干。这样嵌入表留在主机内存，显存主要增加文本主干权重、执行工作区和缓存。`text_encoder_lookup.onnx`、`text_encoder_lookup.json` 与 `text_encoder_trunk.onnx` 共同描述这条路径。

## 默认 Q8 模型与自动下载

默认主干和深度解码器使用官方 llama.cpp b8683 的 `Q8_0` 量化策略，用较少位数近似存储权重。二维权重保存为 Q8_0，归一化权重和旋转位置编码（RoPE，用于向模型表示位置）的频率保留 FP32。两个 GGUF 合计从原 BF16 的约 3.44GiB 降到约 1.80GiB。模型目录中的发布资产还包括全部 ONNX 图、外部参数、原始分词器和快速文本资产，315 个资产合计 7,395,969,548 字节，约 7.40GB。目录另含 `manifest.json` 清单。

StreamVox 2.1.1 起，首次按 `model="breeze-tts-2-gguf"` 初始化时，会从 ModelScope 自动下载完整资产。下载器先使用快照缓存，逐文件核对 SHA256 内容摘要，全部校验完成后再发布可加载目录。后续初始化复用本地文件并执行校验。Windows 默认保存到 `%APPDATA%/StreamVox/ckpts`，`STREAMVOX_HOME` 可指定数据根目录。也可以将 `model` 设为已核验的本地模型目录。2.1.0 安装需升级到 2.1.1 才有该模型的自动下载映射。默认快照下载跟随转换仓库的 `master`，下方性能记录通过清单摘要绑定当次资产，复现时应使用对应的已核验模型目录。[SDK 安装](../../README.md#安装)

Q8_0 会产生权重近似和量化计算误差。本次发布采用有损任务验收，分别记录结构是否完整、请求能否执行、合成是否结束和固定内容是否可识别。严格数值对照仍保留失败状态。六条真实任务覆盖全部原生调用和缓存操作，记录 21,575 个实际消费的输出，包含十六码本及条件引导。条件引导会组合带指令与移除指令的预测分数，以调节指令影响。形状、有限值、缓存操作和词表位置检查正常，但按原单图门限统计，57 个随机探针中 37 项超限，20,912 个真实消费事件中 6,005 项超限，其中包含条件引导放大后的差异。[Q8 验收汇总](breeze-tts-2-validation.json)

八次自由合成覆盖两个随机种子、声音设计、参考克隆及长参考克隆，均输出有限音频并自然结束。同一 Whisper-small 语音识别器复核六条固定文本时，先统一简体并归一化标点，再统计字符编辑错误。BF16 为 12/198，Q8 为 8/198，各条 Q8 错误数均未高于基线。这些分数只描述该识别器在这六条语料上的内容核对。两段 Q8 女声与克隆音频还取得本次用户“声音正常”的试听反馈。[Q8 验收汇总](breeze-tts-2-validation.json)

模型目录中的 `ckpt/q8-validation.json` 保存完整误差、ASR 原始转录、试听音频摘要和量化器来源。`strict_numerical_gates_passed=false` 表示原严格数值门限未通过，`numerical_equivalence_claimed=false` 表示这次有损验收没有建立数学等价结论。原 BF16 导出与验收另存于 `ckpt/source-export-report.json`。公开汇总位于验收 JSON 的 `q8_distribution` 字段。[原始数据索引](breeze-tts-2-validation.json)

## 准备运行与导出环境

运行 SDK 和重建模型使用两个环境。SDK 环境安装 ONNX Runtime、分词器和 llama.cpp 原生运行库，用于加载模型并合成。离线导出环境另外安装 Torch、Transformers、gguf 等工具，用于读取上游检查点、转换资产和计算数值参考。将两者分开，可以保持 SDK 的固定运行依赖，SDK 环境无需 Torch。

Windows 源码环境在仓库根执行以下命令，准备 CPU 依赖并恢复固定原生库。通过安装包（wheel）使用 SDK 的用户，按 README 安装对应依赖组即可。[Windows 开发环境](../development.md#windows-开发环境) · [安装命令](../../README.md#安装)

```powershell
uv sync --managed-python --extra cpu --group dev
uv run python -m streamvox.native_assets --project-root .
```

下文完整合成示例让生成组件和 ONNX 声码器均使用 NVIDIA CUDA。源码环境应将同步命令改为以下组合，再执行同一条原生库恢复命令。

```powershell
uv sync --managed-python --extra onnx-cuda --extra llama-cuda --group dev
uv run python -m streamvox.native_assets --project-root .
```

`llama-cuda` 为 llama.cpp CUDA 提供依赖，`onnx-cuda` 为 ONNX CUDA 提供依赖，两者使用项目固定的 CUDA 12.4 组件。Windows DirectML 使用 `directml`。`cpu`、`onnx-cuda`、`directml` 三组选装依赖互斥，因为它们分别安装不同的 ONNX Runtime 发行包。`llama-cuda` 可与其中任意一组组合，模型的两条执行链分别选择后端。Linux 源码运行还需准备同一版本的 llama.cpp 共享库。[Linux 源码运行说明](../../README.md#linux-源码运行)

以下离线重建步骤使用独立 CPython 3.12 环境，在仓库根执行。先固定源码和原始检查点，再运行转换，输出到新的模型目录。

1. 获取并检出上表固定的 Breeze 源码。导出工具会核对这个提交，确保参考公式与转换实现对应。

   ```powershell
   git clone https://github.com/breezeblue-ai/breeze-tts.git .artifacts/breeze-source
   git -C .artifacts/breeze-source checkout daa6cb2efe5471f09369229f322d20bb956cb3c3
   ```

2. 创建导出环境并安装固定依赖。下列 `requirements.txt` 同时提供读取权重、导出 ONNX 和写入 GGUF 所需工具。

   ```powershell
   uv venv --managed-python --python 3.12 .artifacts/breeze-export
   uv pip install --python .artifacts/breeze-export/Scripts/python.exe -r tools/breeze_reference/requirements.txt
   ```

3. 下载固定版本的完整检查点到本地。此步骤完成后，导出工具只读取本地文件。

   ```powershell
   .artifacts/breeze-export/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BreezeBlue/breeze-tts-2', revision='3e28c5151381a722f1d8661b4118c298caa77aa4', local_dir='.artifacts/breeze-checkpoint')"
   ```

4. 导出并验证各组件。`--runtime-python` 指向前面已安装 SDK 依赖和原生库的解释器，工具会用它实际调用 llama.cpp，再与导出环境中的 Torch 参考结果比较。成功后，`ckpts/breeze-tts-2-gguf` 包含可加载的模型清单及各组件报告。

   ```powershell
   .artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.export --source-dir .artifacts/breeze-source --checkpoint .artifacts/breeze-checkpoint --output ckpts/breeze-tts-2-gguf --precision float32 --runtime-python .venv/Scripts/python.exe --llama-backend cpu
   ```

Linux 将环境路径中的 `Scripts/python.exe` 换为 `bin/python`。上述命令输出 FP32 基准模型，默认下载仓库中的 Q8_0 是基于已验收 BF16 资产另行量化并验收的派生版本。离线导出按组件加载权重，`--runtime-python` 所指的 SDK 解释器无需安装 Torch。

| 导出参数 | 用途与条件 |
| --- | --- |
| `--checkpoint` | 本地完整检查点，真实模型导出必填。 |
| `--source-dir` | 固定提交的本地 Breeze 源码，必填。 |
| `--output` | 输出模型目录的根路径，必填。应为本次导出准备独立目录。 |
| `--audio-tokenizer` | 可选本地音频分词器目录，默认读取检查点内的 `audio_tokenizer`。 |
| `--precision float32` | 默认基准，把原始 BF16 权重精确提升为 FP32。 |
| `--precision bfloat16` | 保留生成组件原始 BF16 权重存储，投影查表、归一化及 ONNX 组件仍使用 FP32。 |
| `--runtime-python` | 实际执行 llama.cpp 数值验收的解释器，默认当前解释器。分离环境时显式传入 SDK 环境路径。 |
| `--llama-backend cpu` / `cuda` | GGUF 数值验收的执行后端，默认 CPU。选择 CUDA 时，SDK 环境需先准备相应依赖、原生库和驱动。 |
| `--tiny` | 用随机 FP32 小模型检查导出链路，不读取真实检查点，也不导出真实分词器或音频组件。 |

需要保留原始 BF16 存储时，可将上面的最后一条命令改为 `--precision bfloat16 --output ckpts/breeze-tts-2-gguf-bf16`。BF16 和 FP32 导出都以原始权重精确提升后的 Torch FP32 前向作为数学参考，报告记录这一参考精度。BF16 存储与原始 GPU BF16 前向的计算精度口径不同，缓存和全部内部计算也不能由权重存储精度推断。FP32 与 BF16 分别执行既有门限，任一组件验收失败都不写入可加载的 `manifest.json`，门限保持固定。

完整真实模型导出会自动包含三个快速文本资产。已有通过数值验收、尚未包含快速文本资产的旧版混合模型目录，可以在独立导出环境派生到新目录。

```powershell
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.prepare_text_bundle --source ckpts/old-bundle --output ckpts/new-fast-bundle
```

运行前确保 `ckpts/old-bundle` 是完整真实模型目录，`ckpts/new-fast-bundle` 尚不存在且位于源目录之外。工具校验并复用已验收权重，将文本图拆成查表和文本主干，重建文件摘要及模型清单，成功后打印新清单路径。大资产优先使用硬链接，跨卷或文件系统不支持时复制。构建失败时会清理本次创建的半成品目录，以便重试。已经包含快速文本资产的模型目录可直接加载。

所有图、GGUF、外部参数和必要元数据均进入文件摘要及清单，使加载时能核对文件是否与导出记录一致。重新分发时保留 `LICENSE`、`MODEL_LICENSE`、相关 NOTICE 和原始归属说明。

## 使用

先准备前述 SDK CUDA 环境和可用的 NVIDIA 驱动。将以下脚本保存在仓库根并运行，首次初始化会下载默认模型，随后按文字描述设计声音，最终在当前目录写入 `output.wav`。`list()` 在此收集所有音频块，便于演示完整文件写入。需要边生成边播放时，可在循环中逐块消费 `engine.stream()`。

```python
from streamvox import TTSEngine
import numpy as np
import soundfile as sf


if __name__ == "__main__":
    engine = TTSEngine(
        model="breeze-tts-2-gguf",
        device="gpu:0",
        llama_backend="cuda",
        onnx_provider="cuda",
    )
    try:
        chunks = list(engine.stream(
            text="[笑] 欢迎来到今晚的故事时间。",
            instruction="一位温柔、清晰的年轻女性，语气亲切。",
            cfg_scale=4.0,
            seed=42,
        ))
        if chunks:
            sf.write("output.wav", np.concatenate(chunks), engine.runtime.sample_rate)
    finally:
        engine.shutdown()
```

`device` 给出默认设备意图和设备编号。`llama_backend` 选择主干与深度解码器的后端，`onnx_provider` 选择 ONNX 声码器的后端。上例两条执行链均使用 CUDA。全 CPU 运行改为 `device="cpu", llama_backend="cpu", onnx_provider="cpu"`，并使用 CPU 依赖组。Vulkan 生成配 DirectML 声码器可设为 `device="gpu:0", llama_backend="vulkan", onnx_provider="directml"`，并在 Windows 准备 DirectML 依赖和 Vulkan 驱动。CPU 生成也可与 CUDA 声码器组合，两个显式后端分别生效。

SDK 原生链路支持 CPU、CUDA、Vulkan，ONNX 链路支持 CPU、CUDA、Windows DirectML。运行环境需具备所选后端的运行库和驱动。明确请求的后端不可用，或初始化发生不允许的后端回退时，会报错。音频嵌入、深度条件投影和参考音频编码器固定使用 CPU ONNX。

文本后端由 `BreezeTTS2GGUFAdapter(text_encoder_backend=...)` 单独选择，默认值为 `"auto"`。

| 选择 | 生效条件与结果 |
| --- | --- |
| `"auto"` | 声码器已选择 CUDA，且清单声明全部三个快速文本资产时，使用 CPU 查表和 CUDA 文本主干。其他情况使用完整 CPU 文本图，包括缺少快速资产的旧模型目录。 |
| `"cpu"` | 始终使用完整 CPU 文本图，可降低显存占用，生成和声码器仍按各自参数运行。 |
| `"cuda"` | 显式使用 CPU 查表和 CUDA 文本主干。必须具备全部快速文本资产及可用 CUDA 运行环境，缺少快速资产会报错。 |

因此，`text_encoder_backend="cuda"` 仍会在 CPU 查嵌入表。CUDA 只执行查表后的文本主干。初始化后可读取 `engine.actual_backends` 核对实际生成后端和各 ONNX 会话的执行提供程序，并从 `engine.runtime_info.metadata["text_encoder_backend"]` 查看文本主干选择。

需要降低显存占用时，在上面的完整示例中将引擎初始化部分替换为以下代码，保留原来的合成、写文件和 `finally` 关闭部分。它创建一份独立注册表，把这个引擎的文本编码设为 CPU，生成和声码器仍使用 CUDA。本地路径须指向已准备好的完整模型目录，也可改回自动下载模型名。

```python
from streamvox.models.breeze import BreezeTTS2GGUFAdapter
from streamvox.models.registry import ModelRegistry


registry = ModelRegistry()
registry.register(BreezeTTS2GGUFAdapter(text_encoder_backend="cpu"))
engine = TTSEngine(
    model="ckpts/breeze-tts-2-gguf",
    device="gpu:0",
    llama_backend="cuda",
    onnx_provider="cuda",
    registry=registry,
)
```

克隆前准备一条可读取的 `reference.wav`，并将 `prompt_text` 换成整段参考音频的准确转录。将以下代码放入完整示例的 `try` 块，替换声音设计和写文件部分。`make_prompt()` 返回的参考音色资产可直接传给 `role_name`，`persist=True` 还会按 `breeze_voice` 保存角色，输出写入 `clone.wav`。

```python
prompt = engine.make_prompt(
    role_name="breeze_voice",
    audio_path="reference.wav",
    prompt_text="这里必须是整段参考音频的准确转录。",
    persist=True,
)
chunks = list(engine.stream(text="很高兴再次听到你的声音。", role_name=prompt))
if chunks:
    sf.write("clone.wav", np.concatenate(chunks), engine.runtime.sample_rate)
```

后续请求可用 `role_name="breeze_voice"` 读取角色。内存音频可改用互斥的 `audio_data` 与 `sample_rate`，允许单条单声道或双声道数组。Breeze 每个参考音色资产要求一条音频和一段非空完整转录，列表形式的多参考输入会报错。编码会将音频规整为 24kHz 单声道，并保留整段参考上下文。

克隆时可加入 `instruction="语速放慢，语气克制而认真。", cfg_scale=4.0` 控制表现。`cfg_scale` 是条件引导（CFG）强度，运行时比较带指令和移除指令的两组预测，再按该系数组合分数，以加强指令对结果的影响。`cfg_scale=1.0` 直接使用单分支预测，不传指令时保持该值。具体分支和参数边界见下一节。

参考文本与目标文本分段独立编码，避免文本编码器的双向注意力让两段互相读取。`[笑]`、`[叹气]`、`(laugh)` 等事件标记原样传入模型，适配器保持原始标记和分段，不执行额外清理或自动分段。

创建参考音色资产与准备运行时缓存有不同的存续时间。`make_prompt()` 先编码参考音频，再自动调用 `prepare_reference()`，提前编码参考转录，并为单分支克隆建立主干参考前缀。这里的预填充指一次读取已知前缀，将后续生成要用的注意力键和值保存为 KV 缓存，以便生成新音频时复用。

运行时只缓存最近一段参考转录的编码，且其数组不超过 16MiB。它也只保留最近一个可复用的单分支参考前缀 KV 状态。CUDA 原生计算按两词对齐，奇数长度参考的最后一个位置会随目标后缀重算。每个请求仍独立编码和预填充目标文本。CFG 多分支请求保留完整参考上下文，为各分支执行独立的完整预填充和缓存。

持久化角色保存参考音色资产，重新读取角色后，若希望把参考准备耗时放到角色选择阶段，需要在正式请求前显式调用 `prepare_reference()`。以下代码同样放在已初始化引擎的 `try` 块中，前提是 `breeze_voice` 已保存。它先完成准备，再合成并写入 `clone.wav`。

```python
prompt_data = engine.get_prompt_data("breeze_voice")
if prompt_data is None:
    raise KeyError("找不到角色 breeze_voice")
engine.runtime.prepare_reference(prompt_data.prompt)
chunks = list(engine.stream(text="很高兴再次听到你的声音。", role_name="breeze_voice"))
if chunks:
    sf.write("clone.wav", np.concatenate(chunks), engine.runtime.sample_rate)
```

## 采样与流式边界

主干和深度解码器分别采样。采样先把模型分数转换成概率，再按参数筛选可选编号并抽取一个结果。两级参数独立设置，避免修改主干时同时改变帧内细节的分布。

| 参数 | 默认值与作用 |
| --- | --- |
| `instruction` | 空字符串。为目标文本附加声音设计或表现指令。 |
| `cfg_scale` | `1.0`，允许有限非负数。没有指令时必须为 `1.0`。有指令且为 `0.0` 时只使用移除指令的分支，为 `1.0` 时只使用带指令分支，其他值使用两个独立分支。 |
| `temperature`、`top_k`、`top_p` | `0.9`、`50`、`1.0`。控制首码本的分布平滑程度、高分候选裁剪和累积概率范围。温度须大于零，`top_k` 为非负整数且零表示关闭，`top_p` 在 `(0, 1]`。 |
| `depth_temperature`、`depth_top_k`、`depth_top_p` | `0.9`、`50`、`1.0`。分别控制剩余十五个码本，取值边界与主干相同。 |
| `repetition_penalty` | `1.1`，须大于零。只调整已经生成过的第一码本编号的分数。 |
| `do_sample` | `True`。设为 `False` 时选择最高分编号，仍执行重复惩罚和保留符号屏蔽，温度与概率裁剪不参与这一贪心路径。 |
| `seed` | `None`。也可传非负整数，为当前请求创建独立随机数流。 |
| `max_new_tokens` | `750`，允许 `1..750`。限制新生成音频帧数，默认上限对应 60 秒，遇结束符可提前结束。 |
| `chunk_frames` | `2`，允许 `1..25`。每个返回块包含的音频帧数，只改变波形分组。 |

`top_k` 启用截断时，普通采样按第 `top_k` 高分设置裁剪阈值，并列项都会保留，因此候选数可能超过 `top_k`。CFG 深度采样启用截断时，按稳定编号顺序处理阈值并列，先保留指定数量的候选。后续 `top_p` 仍可继续减少候选。

适配器拒绝未知合成参数，也会拒绝非有限数值和不符合上述边界的类型。它保留固定源码 `model.generate()` 与 Transformers 4.57.3 普通生成路径的温度、截断分布、首码本重复惩罚、结束符及条件引导公式。深度条件引导保留上游先计算概率再筛选的规则。官方命令行流式入口另有深度采样路径，边界行为并不完全相同。NumPy 与 Torch 的随机数流不同，相同种子无法保证生成同一段音频，分布和随机数流需分别核对。

`chunk_frames` 控制返回块的大小，模型始终逐帧生成。默认一块包含两帧、160ms 音频，修改为 `1` 时每块包含 80ms。首包延迟从发起合成请求算到首块返回，包含该请求的目标文本编码、预填充和首块音频生成。初始化负责加载、校验及相应预热，角色准备负责参考编码和前缀准备，两段耗时在性能记录中另计。

`n_ctx` 限制每个条件分支的上下文位置数。当前默认模型目录允许 `1..40960`，省略时取 `40960`。运行时根据本次提示和生成预算按需扩容原生缓存，初始化预热也可能提前分配部分容量。参考转录、参考音频帧及其结束帧、目标文本以及计划生成的音频帧都会占用位置，音频帧与文本词元按各自输入计数。提示长度加 `max_new_tokens` 超出 `n_ctx` 时，请求报错，调用方需缩短请求或在上述范围内增大上下文。主干保留完整因果上下文，声码器保留原生注意力窗口和全部所需卷积历史。

条件引导各分支使用独立序列和 KV 缓存，深度缓存每帧重置。正常请求完成后移除目标后缀，保留可复用的单分支参考前缀。异常或取消后清空请求缓存。同一运行时串行执行请求，常用角色可复用参考编码。若提前停止消费音频，先调用生成器的 `close()`，再调用 `engine.shutdown()`，让在途请求先完成清理。

CUDA 声码器在运行时持有的固定线程上执行，将上一帧波形解码与下一帧码本生成重叠。加载时在该线程预捕获单个 CUDA Graph，即记录可反复提交的设备计算流程。输入、输出及全部状态使用固定设备地址，状态回写和音频回传使用同一条专用非阻塞流。首包所需的两帧码本完成后立即等待并取回对应音频，后续继续跨帧流水执行。取消或推理失败时先等待在途声码器完成，再清理原生缓存。CPU 和 DirectML 使用同步调度。

当生成与声码器均使用 CUDA 时，`TTSEngine` 初始化还会调用 `warmup_for_init()`，提前运行文本、主干以及 CFG=1、CFG=4 两种深度分支形状。这样常见短提示所需的执行资源在初始化阶段创建，初始化计时包含这些预热步骤。

原生 KV 缓存固定为 FP32。b8683 的部分 CUDA 小矩阵内核会直接使用 TF32。TF32 采用较少有效位参与矩阵运算，这条内核路径会绕过 cuBLAS 矩阵库的普通精度开关。本实现将原生物理微批次设为两个词，避开该计算路径，逻辑条件分支和完整上下文保持原样。融合注意力会把 FP32 键和值转换为 FP16 半精度浮点数，因此此处关闭。这些设置作用于词级计算，80ms 音频帧和 `chunk_frames` 仍按前述方式计量。

## 验证与性能范围

导出链路可以先用随机小模型检查，确认格式、接口和状态传递能工作，再处理真实权重。前提是已准备独立导出环境、固定 Breeze 源码，以及带原生库的 SDK 解释器。以下命令分别验证生成及辅助图、音频编解码器，报告写入指定的两个输出目录。

```powershell
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.export --tiny --source-dir .artifacts/breeze-source --output .artifacts/breeze-tiny --runtime-python .venv/Scripts/python.exe --llama-backend cpu
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.verify_codec --output .artifacts/breeze-codec-tiny
```

Tiny 仅支持 FP32，产物用于数值测试，加载器会拒绝将其作为真实 TTS 模型运行。生成组件验证完整码本、预填充、连续增量解码、独立条件分支和缓存重置。ONNX 验证覆盖文本动态窗口、音频嵌入、深度条件投影和声码器状态对照，音频小模型还核对跨窗口的连续解码。真实模型导出在这些组件检查之外读取实际检查点，结果记录在模型目录的 `ckpt/export-report.json`。[数值校验实现](../../tools/breeze_reference/numerics.py)

下列性能记录分别属于当前默认 Q8 资产和历史 BF16 实现。首包表示等待第一块音频的耗时，实时率（RTF）等于合成耗时除以生成音频时长。例如 RTF 为 `0.38`，表示生成 1 秒音频约消耗 0.38 秒，数值越小，合成越快。各表保留各自的权重、输入和计时范围，便于核对比较条件。

### 当前 Q8 性能

2026-11-06 在 Windows 11、RTX 4090 24GB、Core i7-13700KF、ONNX Runtime 1.21.1 上测量默认 Q8_0 模型，生成、文本主干和声码器均使用 CUDA。每种模式先执行一次完整请求，再测量三次，默认每包包含 160ms 音频。

| 模式 | 首次请求首包 | 三次首包中位数 | RTF 中位数 |
| --- | ---: | ---: | ---: |
| 声音设计，CFG=4 | 161.6ms | 156.3ms | 0.3804 |
| 参考准备后的克隆，CFG=1 | 107.0ms | 105.1ms | 0.3452 |

首次请求列记录各模式第一次实际合成。首包和 RTF 均从合成请求开始计时，初始化与角色准备在请求前完成并另计。完整初始化与校验耗时 11.60 秒，角色准备耗时 0.707 秒。整卡显存峰值相对冷基线增加 6130MiB，约 5.99GiB，工作集峰值约 6308MiB，私有提交量峰值约 12153MiB。显存按约 0.2 秒间隔采样整张物理 GPU 的占用增量。

固定文本、设计指令和种子与下方 BF16 历史测量一致，参数为 `seed=42, max_new_tokens=300, chunk_frames=2`。本次设计和克隆各生成 5.60 秒音频，克隆使用第三次 Q8 设计输出作为参考。八次请求全部自然结束，各模式三次测量的码本及完整浮点音频摘要一致。权重量化和参考长度会改变随机生成轨迹，因此比较跨精度 RTF 和克隆首包时，也需核对生成时长及参考长度。完整记录见验收 JSON 的 `q8_distribution` 字段。[验收记录](breeze-tts-2-validation.json)

### 历史 BF16 首包优化

这组记录属于 2026-11-06 的 BF16 资产，用于说明快速文本路径及参考准备带来的时延变化。硬件仍为 Windows 11、RTX 4090 24GB、Core i7-13700KF，ONNX Runtime 为 1.21.1。生成权重使用原始 BF16 存储，原生 KV 缓存和全部 ONNX 组件使用 FP32，生成与声码器均使用 CUDA。测试分别选择默认 CUDA 文本主干和显式 CPU 文本，数值对照与缓存验证保存于 `ttfa_optimization` 字段。[验收记录](breeze-tts-2-validation.json)

| 文本主干 | 模式 | 首次请求首包 | 三次首包中位数 | RTF 中位数 |
| --- | --- | ---: | ---: | ---: |
| CUDA，默认优先首包 | 声音设计，CFG=4 | 210.8ms | 206.5ms | 0.4893 |
| CUDA，默认优先首包 | 参考准备后的克隆，CFG=1 | 138.9ms | 130.0ms | 0.4359 |
| CPU，降低显存占用 | 声音设计，CFG=4 | 310.1ms | 311.6ms | 0.4824 |
| CPU，降低显存占用 | 参考准备后的克隆，CFG=1 | 196.9ms | 188.7ms | 0.4283 |

首次请求列为各模式第一次实际合成，随后测量三次。每包保持默认两帧、160ms，首包从合成请求开始至首块音频返回，初始化与角色准备分别记录。固定 `seed=42, max_new_tokens=300, chunk_frames=2`，文本和设计指令与下方历史记录相同。克隆使用对应配置第三次设计音频及完整转录。CUDA 文本设计输出为 5.12 秒，克隆为 5.52 秒，CPU 文本分别为 6.24 秒和 6.16 秒，全部自然结束。参考长度也随配置变化，跨文本后端的克隆时延比较包含这一差异。各配置内部三次测量的码本和完整音频均一致。

默认 CUDA 文本配置的初始化耗时为 13.7546 秒，包含文件校验和初始化预热，`make_prompt()` 的参考准备耗时为 0.7503 秒。CPU 文本配置分别为 13.9440 秒和 0.8391 秒。读取已持久化角色后，也要显式提前准备参考，才能采用表中的参考准备后首包口径。

默认配置的整卡显存峰值相对加载前冷基线增加 7834MiB，约 7.65GiB。相较上一轮 CPU 文本历史记录的 4672MiB，增加 3162MiB，约 3.09GiB，其中约 2671MiB、2.61GiB 来自文本主干 FP32 权重，其余包含执行工作区、KV 缓存和初始化资源。本轮显式 CPU 文本配置在初始化预分配后的峰值增量为 4898MiB，约 4.78GiB，较上一轮增加 226MiB。默认 CUDA 文本配置相较本轮 CPU 文本增加 2936MiB。测量仍按约 0.2 秒间隔采样物理 GPU 整卡占用，并扣除各自加载前冷基线。

CUDA 文本编码与 CPU FP32 图对照时，最大绝对误差为 `2.86102294921875e-5`，相对二范数误差约 `1.2e-6`，通过既有 FP32 门限。相对二范数误差衡量整个输出向量的差异相对参考向量的大小。浮点差异会影响后续随机采样，因此跨文本后端的码流和音频可以不同。

CUDA 原生参考缓存回归 12/12 项通过，预填充及其后三步增量的隐藏状态和输出分数最大绝对误差均为零。提前准备参考的 3/3 项还通过这些输出的逐位对照。两项都在同一个原生解码器内，比较启用参考缓存与完整重新预填充的结果。

本轮 CPU 文本配置还重放了历史五段试听音频，完整波形和分块长度均逐位一致。既有人工试听反馈对应 CPU 文本路径，这组 BF16 CUDA 文本新音频在该轮记录中尚待人工试听。当前 Q8 版本的两段试听记录见前文。

### 历史接入与吞吐优化记录

以下记录对应 2026-11-04 至 2026-11-05 的 CPU 文本实现，用于追溯接入和吞吐优化。首包调度、初始化预热、参考缓存策略及资源数据均对应各次测量时的实现。

2026-11-04 晚在上述 Windows 硬件和软件环境完成真实混合链路测量，2026-11-05 完成复核。生成权重保留原始 BF16，缓存和全部 ONNX 组件使用 FP32，llama.cpp 与声码器均使用 CUDA，文本、音频嵌入及条件投影在 CPU。逐次结果、源码与模型版本、摘要及误差保存在验收记录中。[验收记录](breeze-tts-2-validation.json)

| 模式 | 首包中位数 | RTF 中位数 | 阶段显存峰值增量 |
| --- | ---: | ---: | ---: |
| 声音设计，CFG=4 | 373ms | 0.658 | 4628MiB |
| 参考克隆，CFG=1 | 454ms | 0.654 | 4615MiB |

每种模式预热一次、测量三次，固定 `seed=42, max_new_tokens=300, chunk_frames=2`。文本为“你好，欢迎使用语音合成。今天我们一起测试自然、清晰的中文播报。”，设计指令为“一位声音清晰、语气自然的年轻女性。”，克隆使用第三次设计音频及其完整转录。设计为 78 帧、6.24 秒，克隆为 77 帧、6.16 秒，各模式三次码本与浮点音频摘要一致，全部请求自然结束。首包不含加载和预热，RTF 为耗时除以音频时长，这组初次接入记录约为 1.52 倍实时速度。

完整加载与 SHA256 校验约 12.7 秒，全流程显存峰值相对冷基线增加 4628MiB，约 4.52GiB。Windows 工作集峰值约 8.13GiB，私有提交量约 11.03GiB。显存按约 0.2 秒间隔记录整卡相对加载前基线的增量，克隆在同一进程设计之后测量。GPU 进程退出后，整卡占用恢复至测试前水平。

FP32 和 BF16 GGUF 均分别通过 CPU、CUDA 的 7 个主干与 32 个深度样本对照。FP32 CUDA 分数最大误差约为主干 `5.96e-6`、深度 `1.14e-5`，参考为原始权重精确提升的 Torch FP32 前向。BF16 依据原有相对二范数误差、合法词表概率分布的 KL 差异和最大概率差门限验收，同时保留逐元素超限审计。KL 在这里衡量参考与实际预测概率分布的偏离程度，迁移过程保持原门限。

两条真实条件前缀还单独核对了同一 FP32 数学参考。BF16 原生正、负分支均通过单图门限，CFG=4 后最大概率差约 `0.0810`，首选符号一致。条件引导会组合分支差异，因此该值单独作为数值诊断记录，单图门限不直接决定引导后的音质。跨精度、跨后端或使用不同随机数流时，结果可能出现逐位差异。同配置三次可重复的记录，也只覆盖该配置与输入，与旧全 ONNX 码流的一致性需单独对照。

2026-11-05 在相同硬件、模型清单和上述输入上，用优化前源码快照及当时优化后的实现重新测量，每组仍预热一次、测量三次。

| 模式 | 优化前耗时 / RTF | 优化后耗时 / RTF | 首包变化 |
| --- | ---: | ---: | ---: |
| 声音设计，CFG=4 | 4.469 秒 / 0.716 | 3.243 秒 / 0.520 | 371ms → 380ms |
| 参考克隆，CFG=1 | 4.116 秒 / 0.668 | 2.869 秒 / 0.466 | 446ms → 436ms |

这组对照的合成速度分别提升约 38% 和 43%。优化使辅助小图使用单线程、空闲线程及时休眠，并采用条件引导采样的稳定分区筛选、声码器固定绑定和单图重放，以及跨帧流水线。该阶段整卡显存峰值增量从 4628MiB 变为 4672MiB，增加 44MiB，最终约 4.56GiB，加载与文件校验约 12.9 秒。完整记录位于 `speed_optimization` 字段。[验收记录](breeze-tts-2-validation.json)

两种模式各三次测量的码本和完整音频均与优化前逐位一致。用户此前试听的清晰女声、低沉男声、活泼女声、故事讲述及合成参考克隆也完成重放，五段完整波形和音频分块长度均逐位一致，用户对这五段的反馈为“声音是正常的”。

### 测量范围与复现

当前和历史性能记录覆盖上述固定中文短句，历史五段试听重放覆盖约 7 至 16 秒的音频。全 CPU、Vulkan、DirectML、Linux CUDA 和更长上下文的完整合成性能仍需分别测量。适配器建议预留 16GB 内存与 12GB 显存，这一建议来自已测短句的资源情况，最低可运行硬件门槛尚待验证，外部说话人音色质量评测也尚未完成。

资源复现可使用现有基准工具。先准备已核验的本地完整模型目录和选定的运行环境，再在仓库根执行下列命令。示例显式选择物理 GPU 0，工具会据此使用 `device="gpu:0"`，随后跳过参考创建，测量带声音指令的合成，报告默认写入 `tests/output/`。基准工具加载时关闭模型 SHA256 复核，因此它的启动耗时需与前面包含校验的完整初始化耗时分开记录。工具的设备和输入选项可先通过 `uv run --no-sync python tools/benchmark_streamvox_models.py --help` 查看。

```powershell
uv run --no-sync python tools/benchmark_streamvox_models.py --model ckpts/breeze-tts-2-gguf --model-key breeze-tts-2 --physical-gpu-index 0 --skip-make-prompt --stream-kwargs-json '{"instruction":"平静清晰的成年女性。","cfg_scale":4,"seed":42}'
```

复现时固定模型版本、硬件、后端、目标文本、参考音频、参数和种子，分别保存预热及逐次结果。换成自己的文本或参考音频后，分别核对数值、生成效果和性能，才能确认该输入下的行为。单元测试验证程序约束，本机短句记录描述已测输入，两者各自的覆盖范围见相应报告。
