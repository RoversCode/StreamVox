# 构建与验证

[返回 README](../README.md) · [使用文档](usage.md) · [模型适配](model-adapter-guide.md)

2.0.0 的发行目标为 Windows x64 / CPython 3.12。Python 模块以可读源码分发，wheel tag 为 `cp312-none-win_amd64`。其中 `none` 表示 Python 模块不依赖特定扩展 ABI；包内包含 Windows 原生动态库，因此仍限 Windows x64 / CPython 3.12。构建同时生成带原生 DLL 的 sdist，以便重建安装包；公开 Git 仓库不存放 DLL。

## 仓库与贡献

[RoversCode/StreamVox](https://github.com/RoversCode/StreamVox) 是项目唯一的源码维护仓库。日常开发与外部贡献都从 `master` 创建分支，通过 PR 审查并合并；本地工作目录可以使用任意名称。提交范围与个人 AI 配置的本地排除方式见 [参与贡献](../CONTRIBUTING.md)。

## Windows 开发环境

```powershell
git clone https://github.com/RoversCode/StreamVox.git
cd StreamVox
uv sync --managed-python --extra cpu --group dev
uv run python -m streamvox.native_assets --project-root .
```

使用 DirectML 和 llama CUDA 时，将同步命令换成：

```powershell
uv sync --managed-python --extra directml --extra llama-cuda --group dev
```

`cpu`、`directml`、`onnx-cuda` 互斥。SDK 运行环境不需要 Torch；离线参考导出与数值对照工具应使用单独环境，避免改变 SDK 运行依赖。

## Linux 源码环境

Linux x86_64 的依赖安装、预编译共享库下载、库路径与运行命令见 [README：Linux 源码运行](../README.md#linux-源码运行)。Linux 使用 `uv sync --no-install-project` 准备依赖，从仓库根运行 `python -m example.synthesize`。通过 `uv run` 执行时需加 `--no-sync`。

激活虚拟环境后，可先运行与发行来源和附件校验有关的测试：

```bash
python -m pytest tests/test_release_artifacts.py -q
```

其他用例按改动范围与平台选择。全量测试包含 Windows 专用的路径与 wheel 标签检查，该部分需要在 Windows 上验证。

当前 wheel 构建与发行审计面向 Windows，Linux 源码环境不执行下述 Windows 原生资产恢复与发行构建步骤。

## Linux 原生库自编译（可选）

通常使用 README 中的 b8683 预编译 CPU 或 Vulkan 包即可。需要 Linux CUDA 后端、定制 CPU 指令集，或预编译包与系统库不兼容时，可从同一版本自行编译。以下命令在仓库根目录执行，默认生成 CPU 库：

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

该 CPU 配置保留 ggml 默认的 AVX2、FMA、F16C 和 BMI2 选项，需要 CPU 支持；较旧机器应按其能力调整选项。原生库应作为同一套部署，避免混用不同构建的公共库和后端。

CUDA 构建需准备 NVIDIA 驱动与匹配项目运行依赖的 CUDA 工具链，在独立的 `build-cuda` 目录启用 `-DGGML_CUDA=ON`；Vulkan 构建需准备 Vulkan 开发工具、头文件、loader 与驱动，在独立的 `build-vulkan` 目录启用 `-DGGML_VULKAN=ON`。构建完成后，复制对应构建目录的全部 `lib*.so*` 到 `streamvox/bin/`。Python 依赖与运行参数见 README 的后端选择表。

## Windows 原生运行库

`native_assets` 使用固定的官方 llama.cpp b8683 归档及 SHA256，恢复 CPU、CUDA 和 Vulkan 运行库到 `streamvox/bin/`。其中 OpenMP 替换为固定公开的 conda-forge LLVM OpenMP 15.0.5 包内的 `libomp.dll`，按 ggml 的导入名保存为 `libomp140.x86_64.dll`。该步骤只替换运行库，不涉及模型权重和采样配置；校验归档及 DLL 哈希后，将完整来源和文件摘要写入 `native-manifest.json`。下载缓存默认位于 `.artifacts/native/b8683`；可以显式指定缓存目录：

```powershell
uv run python -m streamvox.native_assets --project-root . --cache-dir .artifacts/native/b8683
```

合并官方归档时，同名 DLL 只有内容哈希一致才可复用。不要混用来源不明、接口版本不同或散落在其他 Python 环境中的动态库。原生组件许可见 [第三方声明](../THIRD_PARTY_NOTICES.md) 与 [许可证目录](../third_party/licenses/)。

## 发行构建

先选定本仓库已合并到 `master` 的发行提交，确认检出目录干净，取得其完整 40 位 SHA，替换下面的占位内容：

```powershell
uv run python -m streamvox.package_release --project-root . --python .venv/Scripts/python.exe --source-commit <公开仓库40位提交> --native-cache .artifacts/native/b8683
```

产物默认位于 `dist/windows-py312/2.0.0/artifacts/`，包括：

- `streamvox-2.0.0-cp312-none-win_amd64.whl`
- `streamvox-2.0.0.tar.gz`

构建要求干净的 Git 源码目录，`source-commit` 必须等于该目录当前提交。暂存区直接从这个提交导出，不读取未跟踪或被忽略的工作区文件。Python 源码和第三方许可文件保留在安装包中；来源清单记录源码、构建配置及许可材料的摘要。构建和产物审计核对三组 llama 后端、版本、依赖、许可证及包内文件，拒绝旧产物或源码漂移。

需要统一汇总产物和校验文件时：

```powershell
uv run python -m streamvox.release_matrix --project-root . --python .venv/Scripts/python.exe --source-commit <公开仓库40位提交> --native-cache .artifacts/native/b8683
```

默认只进行本地构建和汇总，输出 `dist/release/2.0.0/`。外部发布使用额外的 GitHub 参数，并要求 tag 已存在且准确指向同一个公开源码提交；不能用旧提交或旧附件冒充本次构建。实际选项以各模块的 `--help` 为准。

## 验证

```powershell
uv run python -m pytest tests -q
uv run python -m twine check dist/windows-py312/2.0.0/artifacts/*
```

再在干净 CPython 3.12 环境中安装最终 wheel，并按 README 选择一种 ONNX 依赖组。核对版本、源码导入路径、三种 llama 动态库、依赖树和示例入口；准备模型及自己有权使用的参考音频后执行实际合成。还应从 sdist 解包目录执行标准 `python -m build --wheel` 重建并安装，验证分发文件完整；这条标准重建路径使用随包 DLL，不需要 Git 仓库或重新下载原生库。

文档改动应检查相对链接和 Python 代码块语法；示例可先运行 `python example/synthesize.py --help`，实际合成还需要权重和参考音频。

## 模型资源测试

`tools/benchmark_streamvox_models.py` 支持模型名、本地 bundle 和模型私有合成参数，输出资源及性能记录。可以先查看命令帮助：

```powershell
uv run python tools/benchmark_streamvox_models.py --help
```

例如，测量 VoxCPM2 无参考音频的文本模式：

```powershell
uv run python tools/benchmark_streamvox_models.py --model ckpts/voxcpm2-gguf --model-key voxcpm2 --skip-make-prompt --stream-kwargs-json '{"mode":"text","control_text":"清晰、平静的成年女声。"}'
```

工具的报告默认位于 `tests/output/`。硬件、模型版本、输入文本和 Prompt 长度应随记录保存，以便比较。

## 当前验收边界

全模型数值对照与人工试听尚未验收，已有数值失败证据不能被源码、打包或单元测试结果覆盖。模型权重、精度、公式或采样配置发生变化时，推理性能、音质和跨后端数值一致性需要各自的验证记录。
