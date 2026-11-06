# Breeze TTS 2

[返回 README](../../README.md) · [模型许可](../model-licenses.md) · [开发与验证](../development.md)

模型注册名为 `breeze-tts-2-gguf`，支持中英文、单参考音色克隆、自然语言声音设计和表现控制。主干与深度解码器使用 llama.cpp，其余组件使用 ONNX Runtime。输出为 24kHz 单声道 `float32` 音频块，默认每块两帧、160ms，结束时不足两帧的末块可以为 80ms。

默认转换仓库为 `ChengHee/Breeze-TTS-2-GGUF`，提供 Q8_0 生成权重和 FP32 ONNX 的完整 bundle。StreamVox 2.1.1 起，首次按模型名初始化时自动下载并校验完整资产，后续复用本地模型。也可传入已核验的本地目录。[ModelScope 模型](https://www.modelscope.cn/models/ChengHee/Breeze-TTS-2-GGUF)

## 固定来源与许可

| 来源 | 固定版本 |
| --- | --- |
| [Breeze 源码](https://github.com/breezeblue-ai/breeze-tts/tree/daa6cb2efe5471f09369229f322d20bb956cb3c3) | `daa6cb2efe5471f09369229f322d20bb956cb3c3` |
| [BreezeBlue/breeze-tts-2 检查点](https://huggingface.co/BreezeBlue/breeze-tts-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4) | `3e28c5151381a722f1d8661b4118c298caa77aa4` |
| [llama.cpp b8683](https://github.com/ggml-org/llama.cpp/tree/d0a6dfeb28a09831d904fc4d910ddb740da82834) | `d0a6dfeb28a09831d904fc4d910ddb740da82834` |

源码采用 Apache-2.0。**检查点权重、衍生模型和自托管输出仅供研究及非商业用途**。GGUF 与 ONNX 转换不改变权重许可。部署和再分发应保留对应的完整模型许可及归属文件，SDK 的 Apache-2.0 不授予该模型的商业使用权。

## 组件分工

| 组件 | 资产与后端 | 保留的计算 |
| --- | --- | --- |
| Qwen3 主干 | `backbone.gguf`，llama.cpp，默认 Q8_0 | 完整因果上下文、查询与键归一化、首码本及结束类别预测。 |
| Llama 深度解码器 | `depth.gguf`，llama.cpp，默认 Q8_0 | 十六码本嵌入与十五组位置输出头，逐帧生成剩余十五个码本。 |
| T5Gemma2 文本编码器 | 默认按资产和声码器后端选择 CPU ONNX 查表与 CUDA ONNX 主干，或完整 CPU `text_encoder.onnx` | 独立编码参考转录与目标文本，保留双向局部及全局注意力和文本投影，全部使用 FP32。 |
| 音频嵌入 | `audio_embedding.onnx`，CPU | 按码本偏移查表，对十六行嵌入求和，组成下一帧的主干输入。 |
| 深度条件投影 | `depth_conditioner.onnx`，CPU | 每帧开始时将主干隐藏状态经一次 `h @ P.T` 投影，从 2048 维进入 1024 维深度空间。 |
| 参考音频编码器 | `audio_encoder.onnx`，按需加载到 CPU | 保留完整参考波形、原始因果注意力与残差量化，输出十六码本。 |
| 流式声码器 | `decoder_step.onnx`，ONNX CPU、CUDA 或 DirectML | 保留原生注意力窗口及卷积历史，逐帧生成连续波形。 |

文本编码器的 512 词局部窗口允许左侧 255 词、右侧 256 词，b8683 的 `gemma-embedding` 使用对称窗口，边界与原模型不同。因此文本编码器保留 ONNX，不能只替换架构名称迁入该原生图。

深度解码器的音频查表先在导出时应用等价投影 `E' = E @ P.T`，默认发布版本再按 Q8_0 量化该投影表。生成后续码本时直接使用 GGUF 中的投影表，`depth_conditioner.onnx` 只负责每帧起始条件，避免再次投影码本嵌入。完整十六码本和十五组输出头均提供。导出时只排除推理不读取的 `codec_model` 与 `embed_text_tokens`。

快速文本路径将原始嵌入表以只读内存映射零复制注入 CPU ONNX `Gather`，再由 CUDA ONNX 执行原始 FP32 文本主干。嵌入表保留在主机内存，新增显存主要来自文本主干权重、执行工作区与缓存。`text_encoder_lookup.onnx`、`text_encoder_lookup.json` 与 `text_encoder_trunk.onnx` 共同声明这条路径。

## 默认 Q8 模型与自动下载

主干与深度解码器使用官方 llama.cpp b8683 默认 `Q8_0` 策略，二维权重保存为 Q8_0，归一化及 RoPE 频率使用 FP32。两个 GGUF 总量从原 BF16 的约 3.44GiB 降到约 1.80GiB。全部 ONNX 图和外部参数、原始分词器及快速文本资产随模型一起提供，完整 bundle 约 7.38GB。

SDK 运行环境按下文准备后，可直接使用 `model="breeze-tts-2-gguf"`。下载器沿用项目现有的 ModelScope 快照、缓存、逐文件 SHA256 校验和完成后发布机制。Windows 默认模型目录为 `%APPDATA%/StreamVox/ckpts`，可通过 `STREAMVOX_HOME` 指定数据根目录。已有 2.1.0 安装需要升级到 2.1.1 才能取得新增的自动下载映射。

Q8_0 会引入权重近似与量化计算误差。本次发布按有损任务范围验收，保留了原始随机压力和真实轨迹的严格数值超限记录。六条真实任务覆盖全部原生调用与缓存、21,575 个消费输出、全部十六码本及条件引导。形状、有限值、缓存操作和词表位置检查正常，按原单图门限统计时，57 个随机探针中 37 项超限，20,912 个真实消费事件中 6,005 项超限，包含 CFG 放大后的差异。

八次自由合成覆盖两个随机种子、声音设计、参考克隆及长参考克隆，均输出有限音频并自然结束。同一 Whisper-small 对六条固定文本的内容复核中，统一简体和标点归一化后的字符错误率为 BF16 的 12/198 与 Q8 的 8/198，各条 Q8 错误数均未高于基线。该指标用于固定语料的内容核对，两段 Q8 女声与克隆音频还取得本次用户“声音正常”的试听反馈。

完整误差、ASR 原始转录、试听音频摘要与量化器来源位于模型的 `ckpt/q8-validation.json`。`strict_numerical_gates_passed=false` 和 `numerical_equivalence_claimed=false` 明确记录有损验收范围，原 BF16 导出与验收另存于 `ckpt/source-export-report.json`。

## 准备运行与导出环境

SDK 运行环境不需要 Torch。Windows 源码环境可先按开发文档准备 CPU 依赖和固定原生库。[开发文档](../development.md#windows-开发环境)

```powershell
uv sync --managed-python --extra cpu --group dev
uv run python -m streamvox.native_assets --project-root .
```

NVIDIA llama CUDA 使用现有 `llama-cuda` extra，ONNX CUDA 使用 `onnx-cuda`，对应项目固定的 CUDA 12.4 依赖。Windows DirectML 使用 `directml`。`cpu`、`onnx-cuda`、`directml` 三者互斥，`llama-cuda` 可与其中任意一组组合。Linux 还需准备同一版本的 llama.cpp 共享库。[安装命令](../../README.md#安装) · [Linux 源码运行说明](../../README.md#linux-源码运行)

Torch、Transformers、gguf 及导出依赖放在独立 CPython 3.12 环境。

```powershell
git clone https://github.com/breezeblue-ai/breeze-tts.git .artifacts/breeze-source
git -C .artifacts/breeze-source checkout daa6cb2efe5471f09369229f322d20bb956cb3c3
uv venv --managed-python --python 3.12 .artifacts/breeze-export
uv pip install --python .artifacts/breeze-export/Scripts/python.exe -r tools/breeze_reference/requirements.txt
.artifacts/breeze-export/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BreezeBlue/breeze-tts-2', revision='3e28c5151381a722f1d8661b4118c298caa77aa4', local_dir='.artifacts/breeze-checkpoint')"
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.export --source-dir .artifacts/breeze-source --checkpoint .artifacts/breeze-checkpoint --output ckpts/breeze-tts-2-gguf --precision float32 --runtime-python .venv/Scripts/python.exe --llama-backend cpu
```

Linux 将环境路径中的 `Scripts/python.exe` 换为 `bin/python`。导出工具只读取本地文件，核对固定源码后逐组件加载权重。`--runtime-python` 指向已安装 SDK 依赖与原生库的独立解释器，用于实际 llama.cpp 数值验收，该解释器不需要 Torch。

| 导出参数 | 用途 |
| --- | --- |
| `--checkpoint` | 本地完整检查点，真实模型导出必填。 |
| `--source-dir` | 固定提交的本地 Breeze 源码，必填。 |
| `--output` | 新 bundle 的根目录，必填。 |
| `--audio-tokenizer` | 可选本地音频分词器目录，默认读取检查点内的 `audio_tokenizer`。 |
| `--precision float32` | 默认基准，把原始 BF16 权重精确提升为 FP32。 |
| `--precision bfloat16` | 保留生成组件原始 BF16 权重存储，投影查表、归一化及 ONNX 组件仍使用 FP32。 |
| `--runtime-python` | 数值验收解释器，默认当前解释器。 |
| `--llama-backend cpu` / `cuda` | 实际 GGUF 数值验收使用的后端，默认 CPU。 |
| `--tiny` | 仅用随机 FP32 小模型验证导出链路，不读取真实检查点，不导出真实分词器或音频组件。 |

原始 BF16 存储可输出到独立目录，例如在上述导出命令中使用 `--precision bfloat16 --output ckpts/breeze-tts-2-gguf-bf16`。其数值参考仍为同一原始权重精确提升后的 Torch FP32 数学基准，报告明确记录参考精度，不能将它描述为原始 GPU 的同精度前向。FP32 与 BF16 分别执行已有门限，任一组件验收失败都不写入可加载的 `manifest.json`，也不自动放宽门限。BF16 存储不代表缓存或所有内部计算均使用 BF16。

完整真实模型导出会自动包含上述三个快速文本资产。已有通过数值验收的旧版混合 bundle，可在独立导出环境派生到新目录。

```powershell
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.prepare_text_bundle --source ckpts/old-bundle --output ckpts/new-fast-bundle
```

源目录必须是尚未包含快速文本资产的真实 bundle，目标目录必须尚不存在且位于源目录之外。工具复用已验收权重、生成文本拆分资产并重建文件摘要与模型清单。大资产优先使用硬链接，跨卷或文件系统不支持时复制。构建失败后清理本次创建的半成品目录，允许重试。已包含快速资产的 bundle 可直接加载。

所有图、GGUF、外部参数和必要元数据均进入文件摘要与清单。重新分发时保留 `LICENSE`、`MODEL_LICENSE`、相关 NOTICE 与原始归属说明。

## 使用

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

`device` 给出默认设备意图和编号，`llama_backend` 与 `onnx_provider` 分别选择生成组件和 ONNX 声码器的后端。上例使用 CUDA 生成和声码器。Vulkan 生成配 DirectML 声码器可设置 `device="gpu:0", llama_backend="vulkan", onnx_provider="directml"`。CPU 生成也可与 CUDA 声码器组合，显式选择不会被另一条链路的 CPU 意图覆盖。

SDK 原生链路支持 CPU、CUDA、Vulkan，ONNX 链路支持 CPU、CUDA、Windows DirectML，目标环境需具备相应运行库及驱动。明确请求的后端不可用或发生不允许的回退时会报错。音频嵌入、深度条件投影及参考音频编码器使用 CPU ONNX，文本路径按以下规则选择。`engine.actual_backends` 分别列出实际后端与各 ONNX 会话的提供程序，`engine.runtime_info.metadata["text_encoder_backend"]` 给出实际文本主干后端。

默认适配器使用 `text_encoder_backend="auto"`。当声码器选择 CUDA 且模型清单声明全部三个快速文本资产时，自动使用 CPU ONNX 查表与 CUDA ONNX 文本主干，优先降低首包延迟。旧版 bundle 缺少快速资产时自动使用完整 CPU 文本图。显式选择 `text_encoder_backend="cuda"` 时，缺少快速资产会报错。

需要降低显存占用时，可通过独立注册表明确选择 CPU 文本，同时保留 CUDA 生成和声码器。

```python
from streamvox import TTSEngine
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

全 CPU 运行可将引擎参数设为 `device="cpu", llama_backend="cpu", onnx_provider="cpu"`。以上引擎使用完成后均需调用 `engine.shutdown()`。

单参考克隆可在已初始化引擎的 `try` 块中创建角色并合成。

```python
prompt = engine.make_prompt(
    role_name="breeze_voice",
    audio_path="reference.wav",
    prompt_text="这里必须是整段参考音频的准确转录。",
    persist=True,
)
chunks = list(engine.stream(text="很高兴再次听到你的声音。", role_name=prompt))
```

后续也可用 `role_name="breeze_voice"` 读取缓存。加入 `instruction="语速放慢，语气克制而认真。", cfg_scale=4.0` 可控制表现，不使用指令时保持 `cfg_scale=1.0`。参考音频须有完整转录，编码时不裁去原始上下文。参考文本与目标文本分段独立编码，`[笑]`、`[叹气]`、`(laugh)` 等事件标记原样传入，不额外清理或自动分段。

`make_prompt()` 会在角色准备阶段自动调用 `prepare_reference()`，提前完成参考转录编码与单分支参考前缀预填充。运行时只缓存最近一段参考转录，以及 CFG=1 时最近一个参考前缀的原生 KV 状态。目标文本在各请求中独立编码和预填充。CFG 多分支始终使用完整、独立的前填充与缓存，保留全部参考上下文。

持久角色读取只恢复提示资产，提前计算参考需在正式请求前显式调用 `prepare_reference()`，将这部分耗时移到角色选择阶段。

```python
prompt_data = engine.get_prompt_data("breeze_voice")
if prompt_data is None:
    raise KeyError("找不到角色 breeze_voice")
engine.runtime.prepare_reference(prompt_data.prompt)
chunks = list(engine.stream(text="很高兴再次听到你的声音。", role_name="breeze_voice"))
```

## 采样与流式边界

默认 `temperature=0.9, top_k=50, top_p=1.0`，深度解码器独立使用 `depth_temperature=0.9, depth_top_k=50, depth_top_p=1.0`。`repetition_penalty=1.1` 只作用于已经生成的第一码本，`do_sample=True`，`max_new_tokens=750`，`chunk_frames=2`。未知参数会报错。

保留固定源码 `model.generate()` 与 Transformers 4.57.3 普通生成路径的温度、截断分布、首码本重复惩罚、结束符及条件引导公式，深度条件引导保留上游先概率化再筛选的规则。官方命令行流式入口另有深度采样路径，边界行为并不完全相同。NumPy 与 Torch 的随机数流不同，相同种子不保证生成相同音频，分布一致性与随机流须分别核对。

每个音频帧为 80ms，`chunk_frames` 允许 1 到 25，它只改变返回音频的分组，不改变逐帧生成。主干保留原始因果上下文，声码器保留原生窗口及全部所需卷积历史。若提示长度加请求生成长度超出 `n_ctx`，请求会报错，不通过截断上下文完成合成。

条件引导分支使用独立序列和缓存，深度缓存逐帧重置。请求正常结束后移除目标后缀，只保留可复用的单分支参考前缀，异常或取消后清空请求缓存。同一运行时串行执行请求，常用角色可复用参考编码。提前停止消费音频时先关闭生成器，再关闭引擎。

CUDA 声码器在运行时持有的固定线程上执行，与下一帧码本生成重叠。加载时在该线程预捕获单个 CUDA Graph，输入、输出及全部状态使用固定设备地址，状态回写和音频回传使用同一条专用非阻塞流。首包所需的两帧码本完成后立即收割对应音频并返回，后续音频继续使用跨帧流水线。取消或推理失败时先等待在途声码器完成，再清原生缓存。CPU 和 DirectML 使用同步调度。

当生成与声码器均使用 CUDA 时，`TTSEngine` 初始化会调用运行时的 `warmup_for_init()`，提前预热文本、主干和 CFG=1、CFG=4 两种深度分支形状。初始化耗时包含这部分预热，将常见短提示的执行资源创建移入初始化阶段。

原生缓存固定为 FP32。b8683 的部分 CUDA 小矩阵内核直接使用 TF32，绕过普通 cuBLAS 精度开关。本实现使用两个词的原生物理微批次避开该路径，逻辑条件分支和完整上下文保持不变。融合注意力因会把 FP32 键和值转换为 FP16 而关闭。这些是词级计算设置，不改变 80ms 音频帧或 `chunk_frames` 的含义。

## 验证与性能范围

可先在独立导出环境运行无真实权重的检查。

```powershell
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.export --tiny --source-dir .artifacts/breeze-source --output .artifacts/breeze-tiny --runtime-python .venv/Scripts/python.exe --llama-backend cpu
.artifacts/breeze-export/Scripts/python.exe -m tools.breeze_reference.verify_codec --output .artifacts/breeze-codec-tiny
```

Tiny 仅支持 FP32，其产物不能作为真实 TTS bundle 加载。生成组件验证包含完整码本、预填充与连续增量解码、独立条件分支及缓存重置，ONNX 验证保留文本动态窗口、音频嵌入、深度条件投影及声码器状态对照。导出结果记录在 bundle 的 `ckpt/export-report.json`。[数值校验实现](../../tools/breeze_reference/numerics.py)

### 当前 Q8 性能

2026-11-06 在 Windows 11、RTX 4090 24GB、Core i7-13700KF、ONNX Runtime 1.21.1 上测量默认 Q8_0 模型，生成、文本主干和声码器均使用 CUDA。每种模式先执行一次完整请求，再测量三次，默认每包 160ms 音频。

| 模式 | 首次请求首包 | 三次首包中位数 | RTF 中位数 |
| --- | ---: | ---: | ---: |
| 声音设计，CFG=4 | 161.6ms | 156.3ms | 0.3804 |
| 参考准备后的克隆，CFG=1 | 107.0ms | 105.1ms | 0.3452 |

完整初始化与校验耗时 11.60 秒，角色准备耗时 0.707 秒，分别单独计时。整卡显存峰值相对冷基线增加 6130MiB，约 5.99GiB，工作集峰值约 6308MiB，私有提交量峰值约 12153MiB。显存按约 0.2 秒间隔采样整张物理 GPU 的占用增量。

固定文本、设计指令、种子与下方 BF16 历史测量一致，参数为 `seed=42, max_new_tokens=300, chunk_frames=2`。本次设计和克隆各生成 5.60 秒音频，克隆使用第三次 Q8 设计输出作为参考，八次请求全部自然结束，各模式三次码本及完整浮点音频摘要一致。量化和参考长度差异会影响随机生成轨迹，跨精度 RTF 及克隆首包比较应结合这些条件。完整记录见验收 JSON 的 `q8_distribution` 字段。[验收记录](breeze-tts-2-validation.json)

### 历史 BF16 首包优化

2026-11-06 在 Windows 11、RTX 4090 24GB、Core i7-13700KF、ONNX Runtime 1.21.1 上完成真实模型验收。生成权重使用原始 BF16 存储，原生 KV 缓存与全部 ONNX 组件为 FP32。生成与声码器均使用 CUDA，分别测量默认 CUDA 文本主干和显式 CPU 文本两种配置。详细数据、数值对照和缓存验证保存在验收 JSON 的 `ttfa_optimization` 字段。[验收记录](breeze-tts-2-validation.json)

| 文本主干 | 模式 | 首次请求首包 | 三次首包中位数 | RTF 中位数 |
| --- | --- | ---: | ---: | ---: |
| CUDA，默认优先首包 | 声音设计，CFG=4 | 210.8ms | 206.5ms | 0.4893 |
| CUDA，默认优先首包 | 参考准备后的克隆，CFG=1 | 138.9ms | 130.0ms | 0.4359 |
| CPU，降低显存占用 | 声音设计，CFG=4 | 310.1ms | 311.6ms | 0.4824 |
| CPU，降低显存占用 | 参考准备后的克隆，CFG=1 | 196.9ms | 188.7ms | 0.4283 |

首次请求列为各模式第一次实际合成，随后测量三次。首包保持默认两帧、160ms，计时从合成请求开始至首块音频返回，初始化与角色准备分别记录。固定 `seed=42, max_new_tokens=300, chunk_frames=2`，文本和设计指令与下方历史记录相同。克隆使用对应配置第三次设计音频及完整转录。CUDA 文本设计输出为 5.12 秒，克隆为 5.52 秒，CPU 文本分别为 6.24 秒和 6.16 秒，全部自然结束。两种配置的克隆参考长度也随之变化，跨文本后端的克隆时延比较包含这一差异。各配置内部三次测量的码本与完整音频均一致。

默认 CUDA 文本配置的初始化耗时为 13.7546 秒，包含文件校验和初始化预热，`make_prompt()` 的参考准备耗时为 0.7503 秒。CPU 文本配置分别为 13.9440 秒和 0.8391 秒。已持久化角色也需显式提前准备，才能使用表中的参考准备后首包口径。

默认配置的整卡显存峰值相对加载前冷基线增加 7834 MiB，约 7.65 GiB。相较上一轮 CPU 文本历史记录的 4672 MiB，增加 3162 MiB，约 3.09 GiB，其中约 2671 MiB、2.61 GiB 为文本主干 FP32 权重，其余包含执行工作区、KV 缓存和初始化资源。当前显式 CPU 文本配置在初始化预分配后的整卡峰值增量为 4898 MiB，约 4.78 GiB，较历史记录增加 226 MiB。默认 CUDA 文本配置相较本轮 CPU 文本配置增加 2936 MiB。显存仍按约 0.2 秒间隔采样物理 GPU 整卡占用，并扣除各自加载前冷基线。

CUDA 文本编码与原 CPU FP32 图的最大绝对误差为 `2.86102294921875e-5`，相对二范数误差约 `1.2e-6`，通过原有 FP32 门限。浮点差异会影响随机采样，跨文本后端的码流和音频可以不同。CUDA 原生参考缓存回归 12/12 项通过，预填充及其后三步增量的隐藏状态和输出分数最大绝对误差均为零。提前准备参考的 3/3 项还通过这些输出的逐位对照，比较范围均为同一原生解码器启用缓存与完整预填充。

当前 CPU 文本配置还重放了历史五段试听音频，完整波形和分块长度均逐位一致。既有人工试听反馈对应 CPU 文本路径，本轮默认 CUDA 文本的新音频尚待人工试听。

### 历史接入与吞吐优化记录

以下记录对应 2026-11-04 至 2026-11-05 的 CPU 文本实现，保留用于追溯接入和吞吐优化过程。其首包调度、初始化预热与参考缓存策略属于当时实现，资源数据也对应各次独立测量。

2026-11-04 晚在上述 Windows 硬件和软件环境上完成真实混合链路测量，2026-11-05 完成复核。表中生成权重保留原始 BF16，缓存及全部 ONNX 组件为 FP32，llama.cpp 与声码器均使用 CUDA，文本、音频嵌入及条件投影在 CPU。逐次结果、源码与模型版本、摘要及误差保存在验收记录中。[验收记录](breeze-tts-2-validation.json)

| 模式 | 首包中位数 | RTF 中位数 | 阶段显存峰值增量 |
| --- | ---: | ---: | ---: |
| 声音设计，CFG=4 | 373ms | 0.658 | 4628 MiB |
| 参考克隆，CFG=1 | 454ms | 0.654 | 4615 MiB |

每种模式预热一次、测量三次，固定 `seed=42, max_new_tokens=300, chunk_frames=2`。文本为“你好，欢迎使用语音合成。今天我们一起测试自然、清晰的中文播报。”，设计指令为“一位声音清晰、语气自然的年轻女性。”，克隆使用第三次设计音频及其完整转录。设计为 78 帧、6.24 秒，克隆为 77 帧、6.16 秒，各模式三次码本与浮点音频摘要一致，全部请求自然结束。首包不含加载和预热，RTF 为耗时除以音频时长。这组初次接入记录约为 1.52 倍实时速度。

完整加载与 SHA256 校验约 12.7 秒，全流程显存峰值相对冷基线增加 4628 MiB，约 4.52 GiB。Windows 工作集峰值约 8.13 GiB，私有提交量约 11.03 GiB。显存按约 0.2 秒间隔记录整卡相对加载前基线的增量，不使用 Windows 不支持的进程显存读数，克隆在同一进程设计之后测量。GPU 进程退出后整卡占用恢复至测试前水平。

FP32 和 BF16 GGUF 均分别通过 CPU、CUDA 的 7 个主干与 32 个深度样本对照。FP32 CUDA 分数最大误差约为主干 `5.96e-6`、深度 `1.14e-5`，参考是原始权重精确提升的 Torch FP32 前向。BF16 使用原有相对二范数、合法词表 KL 和最大概率差门限，同时保留逐元素超限审计，未因迁移而放宽。

两条真实条件前缀也单独核对了同一 FP32 数学参考。BF16 原生正、负分支均通过单图门限，CFG=4 后最大概率差约 `0.0810`，首选符号一致。这项引导后的测量是数值诊断，不能把单图门限直接当作引导后的音质界限。跨精度、跨后端或不同随机数流不保证逐位一致，三次可重复也不代表与旧全 ONNX 码流相同。

2026-11-05 在相同硬件、模型清单和上述输入上，分别用优化前源码快照与当时优化后的实现重新测量，每组仍预热一次、测量三次。

| 模式 | 优化前耗时 / RTF | 优化后耗时 / RTF | 首包变化 |
| --- | ---: | ---: | ---: |
| 声音设计，CFG=4 | 4.469 秒 / 0.716 | 3.243 秒 / 0.520 | 371ms → 380ms |
| 参考克隆，CFG=1 | 4.116 秒 / 0.668 | 2.869 秒 / 0.466 | 446ms → 436ms |

这组对照的合成速度分别提升约 38% 和 43%。优化包括辅助小图单线程、空闲线程休眠、条件引导采样的稳定分区筛选、声码器固定绑定与单图重放，以及跨帧流水线。该阶段整卡显存峰值相对冷基线的增量从 4628 MiB 变为 4672 MiB，增加 44 MiB，约 4.56 GiB。加载与文件校验约 12.9 秒。完整记录位于验收 JSON 的 `speed_optimization` 字段。

两种模式各三次测量的码本与完整音频均与优化前逐位一致。用户此前试听的五段清晰女声、低沉男声、活泼女声、故事讲述及合成参考克隆也完成重放，五段完整波形和音频分块长度均逐位一致。用户对这五段的反馈为“声音是正常的”。

### 测量范围与复现

当前及历史性能记录覆盖上述固定中文短句，历史五段试听重放覆盖约 7 至 16 秒的音频。全 CPU、Vulkan、DirectML、Linux CUDA 与更长上下文的完整合成性能仍需各自测量。资源建议不代表已验证的最低硬件门槛，外部说话人音色质量评测尚未完成。

资源测量可使用现有工具，固定模型、硬件、后端、输入、参考音频、参数及种子，分别记录预热与逐次结果。

```powershell
uv run --no-sync python tools/benchmark_streamvox_models.py --model ckpts/breeze-tts-2-gguf --model-key breeze-tts-2 --skip-make-prompt --stream-kwargs-json '{"instruction":"平静清晰的成年女性。","cfg_scale":4,"seed":42}'
```

使用自己的文本与参考音频时，应分别核对数值、生成效果和性能，不能用单元测试或本机短句记录替代相应验收。
