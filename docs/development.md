# 构建与验证

[返回 README](../README.md) · [使用文档](usage.md) · [模型适配](model-adapter-guide.md)

本页覆盖从源码开发到发行附件的过程。开发环境用于修改和运行 SDK，离线导出环境用于把上游模型转换成 SDK 可加载的文件，发行环境用于从确定的源码提交构建安装包。三者的依赖和输入各有边界。

StreamVox 2.1.2 的安装包（wheel）面向 Windows x64 / CPython 3.12，Python 模块以可读源码分发。文件名中的 `cp312-none-win_amd64` 分别声明解释器版本、Python 扩展 ABI 和平台。ABI 是扩展模块与解释器交换二进制数据的约定，`none` 表示包内 Python 模块无需特定扩展 ABI，`win_amd64` 对应随包提供的 Windows 原生动态库。源码归档（sdist）也携带这套 DLL，可在相同发行环境中重建 wheel。Linux x86_64 通过仓库源码与 Linux 共享库运行。

## 仓库与贡献

项目唯一源码仓库为 `RoversCode/StreamVox`，保留 `dev` 和 `master` 两个分支。日常开发与文档修改提交到 `dev`，验证后从 `dev` 向 `master` 提交 PR。合并后将 `master` 快进同步回 `dev`，维护者再选择确定的 `master` 提交作为发行输入。外部贡献先进入 `dev`，具体流程和提交范围见参与贡献说明。[源码仓库](https://github.com/RoversCode/StreamVox) · [参与贡献](../CONTRIBUTING.md)

## Windows 开发环境

在仓库根目录准备 CPython 3.12、CPU 版 ONNX Runtime 和开发工具，再恢复公开源码中省略的 DLL。

```powershell
git clone https://github.com/RoversCode/StreamVox.git
cd StreamVox
uv sync --locked --managed-python --extra cpu --group dev
uv run --no-sync python -m streamvox.native_assets --project-root .
```

`cpu` extra 为 ONNX 组件安装 CPU 执行依赖。ONNX 文件保存可由 ONNX Runtime 执行的模型计算图，GGUF 文件保存 llama.cpp 读取的模型权重。GGUF 组件通过 `streamvox/bin/` 中的 llama.cpp 运行，原生资产恢复命令会准备 CPU、CUDA 和 Vulkan 三种后端所需的 DLL。模型目录（bundle）包含权重、分词器和模型清单，需要另行准备，位置与下载方法见使用文档。[模型与本地目录](usage.md)

如果要开发 Windows DirectML 与 llama CUDA 的组合，用下面的命令替换依赖同步步骤。

```powershell
uv sync --locked --managed-python --extra directml --extra llama-cuda --group dev
```

ONNX 的 `cpu`、`directml`、`onnx-cuda` 三组互斥，因为它们分别安装不同的 ONNX Runtime 发行包。`llama-cuda` 提供 llama.cpp 使用的 NVIDIA 依赖，可以与其中任意一组组合。使用 ONNX CUDA 时，将 ONNX extra 换为 `onnx-cuda`。运行时仍需通过 `device`、`llama_backend` 和 `onnx_provider` 选择执行位置，安装依赖本身不会强制选择后端。[后端参数](usage.md)

SDK 运行依赖不包含 Torch。离线模型导出和数值对照工具可能使用 Torch、Transformers 及上游源码，应放在单独虚拟环境中。以 Breeze 为例，`tools/breeze_reference/requirements.txt` 为该独立环境固定依赖，导出工具通过 `--runtime-python` 调用 SDK 环境中的解释器完成原生数值对照。上游源码提交、检查点版本和导出精度也属于导出输入，具体命令见模型文档。[Breeze 运行与导出环境](models/breeze-tts-2.md#准备运行与导出环境)

## Linux 源码环境

Linux x86_64 使用 `uv sync --no-install-project` 安装依赖。当前 `setup.py` 只允许在 Windows 构建 wheel，因此 Linux 从仓库根目录直接导入源码，使用已激活环境中的 `python` 运行命令。预编译共享库下载、系统库要求、环境激活和示例命令见 README。[Linux 源码运行](../README.md#linux-源码运行)

```bash
uv sync --locked --python 3.12 --managed-python --extra cpu --group dev --no-install-project
source .venv/bin/activate
python -m example.synthesize --help
python -m pytest tests/test_release_artifacts.py -q
```

需要 llama.cpp 的模型还须准备与 b8683 配套的全部 `.so` 文件，并在当前终端设置 `LD_LIBRARY_PATH`。仅使用 MOSS-TTS-Nano 时可以跳过 llama.cpp。通过 `uv run` 执行已有环境时加 `--no-sync`，防止再次同步时尝试安装项目。

其他测试按修改内容与平台选择。全量测试包含 Windows 路径、发行宿主和 wheel 标签检查，这些用例需要在 Windows 验证。下文 `native_assets` 恢复 Windows DLL，`package_release` 和 `release_matrix` 构建 Windows 发行产物。

## Linux 原生库自编译（可选）

需要 Linux CUDA 后端、定制 CPU 指令集，或预编译包无法满足系统库要求时，可以从项目固定的 llama.cpp 版本编译。以下命令在仓库根目录执行，生成 CPU 共享库。

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config
git clone --depth 1 --branch b8683 https://github.com/ggml-org/llama.cpp.git .artifacts/llama.cpp
test "$(git -C .artifacts/llama.cpp rev-parse HEAD)" = "d0a6dfeb28a09831d904fc4d910ddb740da82834"

cmake -S .artifacts/llama.cpp -B .artifacts/llama.cpp/build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build .artifacts/llama.cpp/build-cpu --parallel 4

mkdir -p streamvox/bin
cp -a .artifacts/llama.cpp/build-cpu/bin/lib*.so* streamvox/bin/
export LD_LIBRARY_PATH="$PWD/streamvox/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

CPU 配置中的 `GGML_NATIVE=OFF` 关闭按构建机自动启用本机指令的选项，ggml 默认的 AVX2、FMA、F16C 和 BMI2 等选项仍需与目标 CPU 匹配。较旧机器应按其指令能力调整配置。

CUDA 构建需要 NVIDIA 驱动和 CUDA 工具链，在独立的 `build-cuda` 目录将 `GGML_CUDA` 设为 `ON`。Vulkan 构建需要 Vulkan 开发工具、头文件、loader 和驱动，在独立的 `build-vulkan` 目录将 `GGML_VULKAN` 设为 `ON`。构建后，从对应目录复制全部 `lib*.so*` 并保留软链接。公共库和后端库须来自同一构建，运行参数和 Python 依赖按 README 的后端表配置。[Linux 后端选择](../README.md#linux-源码运行)

## Windows 原生运行库

`streamvox.native_assets` 从两份官方 llama.cpp b8683 Windows 归档中合并 CPU、CUDA 和 Vulkan 库，源码将 b8683 固定到提交 `d0a6dfeb28a09831d904fc4d910ddb740da82834`。归档自带的微软 OpenMP 文件替换为固定的 conda-forge LLVM OpenMP 15.0.5 包内 `libomp.dll`，并按 ggml 导入名保存为 `libomp140.x86_64.dll`。

工具先核对下载归档 SHA256，只提取批准清单中的 DLL，再核对每个 DLL 的摘要。同名 DLL 只有内容一致才能合并。全部文件通过校验后，工具把来源地址、固定版本、OpenMP 替换信息和文件摘要写入 `streamvox/bin/native-manifest.json`。

```powershell
uv run --no-sync python -m streamvox.native_assets --project-root . --cache-dir .artifacts/native/b8683
```

默认缓存就是 `.artifacts/native/b8683`。缓存文件摘要不匹配时，工具会报错。目标 `streamvox/bin/` 含未知文件时也会报错，开发者须先确认这些文件的来源和用途，再决定如何处理。原生库来源与许可材料应随这套库保留。[第三方声明](../THIRD_PARTY_NOTICES.md) · [许可证目录](../third_party/licenses/)

## 发行构建

发行输入包含确定的源码提交、Windows x64 CPython 3.12 构建环境和固定原生归档。准备完成后，在检出的发行提交上取得完整 40 位 SHA。

发行参数 `--source-commit` 指 StreamVox 仓库的提交号。模型导出使用的上游源码提交、检查点版本，以及原生运行库的 llama.cpp 提交各自独立。SDK 发行清单记录本次 SDK 源码和原生库来源，模型清单与导出报告记录模型文件及转换输入，复现时需要同时核对这些记录。

```powershell
git rev-parse HEAD
git status --short
```

将下一条命令的 `<公开仓库40位提交>` 替换为选定的 SHA，选择 `package_release` 构建两份归档，或选择 `release_matrix` 一次完成构建和附件汇总。

```powershell
uv run --no-sync python -m streamvox.package_release --project-root . --python .venv/Scripts/python.exe --source-commit <公开仓库40位提交> --native-cache .artifacts/native/b8683
```

`package_release` 按下面的顺序确定产物来源。

1. 核对构建解释器为 Windows x64 CPython 3.12，且具备 `build`、`wheel` 和 `setuptools>=77`。
2. 要求 `source-commit` 精确等于当前 `HEAD`，并拒绝已跟踪文件的未提交修改。源码通过 `git archive` 从该提交导出到独立暂存目录，未跟踪或已忽略的本机文件不会被读取。
3. 核对提交中的文件类型与路径，拒绝 DLL、符号链接、子模块及禁止随发行携带的内容。在暂存目录恢复固定原生库。
4. 使用指定解释器执行 `python -m build --sdist --wheel --no-isolation`，再核对归档中的 Python 源码摘要、三种 llama 后端及其来源清单、版本、平台和许可材料。元数据检查还核对 Python 范围，并拒绝旧授权依赖。
5. 全部审计通过后，将归档公布到 `dist/windows-py312/2.1.2/artifacts/`。该目录已有产物时构建会失败，工具会保留现有文件供核对。

本地构建检查 `HEAD`、已跟踪文件状态及归档内容。选择“已合并到公开仓库 `master` 的提交”属于发行流程要求，维护者须在运行前确认其合并与公开状态。

两份产物分别为 `streamvox-2.1.2-cp312-none-win_amd64.whl` 与 `streamvox-2.1.2.tar.gz`。需要同时生成校验附件时，从尚未构建该版本产物的状态运行下列命令。

```powershell
uv run --no-sync python -m streamvox.release_matrix --project-root . --python .venv/Scripts/python.exe --source-commit <公开仓库40位提交> --native-cache .artifacts/native/b8683
```

`release_matrix` 会调用同一构建流程，再将 wheel、sdist、`SHA256SUMS.txt` 和 `release-metadata.json` 汇总到 `dist/release/2.1.2/`。汇总目录须为空，底层构建目录也须满足前述要求。元数据记录源码仓库、提交号、源码与构建配置摘要、原生资产来源以及附件摘要。因此应在两条入口中选择一条完成本轮构建。

只有显式提供 GitHub 参数时，`release_matrix` 才会上传附件。上传流程要求目标为公开的 `RoversCode/StreamVox`，当前账号具有写入权限，`v2.1.2` 标签已经存在并精确指向同一源码提交，而且该版本尚无 GitHub Release。上传前会重新核对附件摘要，并从声明的提交再次导出源码审计包内内容。具体参数可通过帮助查看。

```powershell
uv run --no-sync python -m streamvox.package_release --help
uv run --no-sync python -m streamvox.release_matrix --help
```

## 验证

源码修改先运行相关测试，Windows 提交前运行全量测试。下面的 `twine` 命令用于检查已经构建完成的发行文件元数据。

```powershell
uv run --no-sync python -m pytest tests -q
uv run --no-sync python -m twine check dist/windows-py312/2.1.2/artifacts/*
```

发行验收还包括安装和重建两条路径。

1. 在干净的 Windows x64 CPython 3.12 环境中安装最终 wheel，并按 README 选择一组 ONNX 依赖。核对 SDK 版本、导入路径、CPU/CUDA/Vulkan llama DLL、依赖树和示例入口。
2. 准备目标模型目录及有权使用的参考音频，按支持的后端执行实际合成，记录模型版本、运行参数、硬件、输出和失败情况。
3. 解包最终 sdist，在干净且具备构建依赖的 Windows x64 CPython 3.12 环境中进入解包目录，运行 `python -m build --wheel`，再安装并检查重建的 wheel。标准重建路径使用 sdist 随附的 DLL，所需输入均在源码归档及声明的构建依赖中。

文档修改应检查相对链接与 Python 代码块语法。帮助命令可以先检查入口和参数名，实际合成还需要对应权重和参考资料。

```powershell
uv run --no-sync python -m example.synthesize --help
uv run --no-sync python tools/benchmark_streamvox_models.py --help
```

## 模型资源测试

`tools/benchmark_streamvox_models.py` 通过公开 `TTSEngine` 入口测量初始化、参考音色资产（Prompt）准备、首包、合成耗时和资源占用。Prompt 保存参考音频提取出的说话人信息、音频编码或可复用状态，具体内容由模型决定。`--stream-kwargs-json` 把模型私有参数传给 `engine.stream()`。

例如，下面测量 VoxCPM2 的无参考音频文本模式。`--skip-make-prompt` 跳过参考准备，适配器需要支持默认 Prompt。示例显式选择物理 GPU 0，工具会据此使用 `device="gpu:0"`，多卡环境按实际编号调整。先准备已核验的模型目录，基准工具加载时关闭模型 SHA256 复核，其初始化耗时需与包含校验的完整初始化分开记录。

```powershell
uv run --no-sync python tools/benchmark_streamvox_models.py --model ckpts/voxcpm2-gguf --model-key voxcpm2 --physical-gpu-index 0 --skip-make-prompt --stream-kwargs-json '{"mode":"text","control_text":"清晰、平静的成年女声。"}'
```

报告默认写入 `tests/output/`。每组记录应一同保存模型清单或版本、SDK 提交、原生库版本、硬件、实际后端、输入文本、参考音频长度、私有参数和预热方法。首包是请求开始后第一块音频到达的时间，实时率（RTF）是合成耗时除以音频时长。分块大小和初始化、参考准备是否计入测量，会改变这些数值的比较范围。

## 当前验收边界

测试、安装验收、数值对照、人工试听和性能测量分别回答不同问题。测试核对接口与状态处理，安装验收核对归档在目标环境中能否运行，数值对照核对指定输入下的计算误差，试听核对具体音频效果，性能测量核对给定模型与环境的耗时和资源。每种结论都须保留模型版本、精度、后端和输入范围。

Breeze Q8_0 已按有损任务范围完成验收，并有两段 Q8 女声及克隆音频的用户“声音正常”反馈。严格数值记录仍有超限，`strict_numerical_gates_passed=false` 与 `numerical_equivalence_claimed=false` 保存在验收材料中。这些记录支持其已声明的任务范围，数值等价性仍未成立。历史 BF16、文本后端优化和试听记录须按各自版本阅读。[Breeze 验证范围](models/breeze-tts-2.md#验证与性能范围) · [原始验收记录](models/breeze-tts-2-validation.json)

其他模型及新的精度、硬件、后端、长上下文或输入类型，应以对应记录为准。已有失败证据需要保留并说明，模型权重、计算公式或采样配置变化后，应重新验证受影响的数值、效果和性能。
