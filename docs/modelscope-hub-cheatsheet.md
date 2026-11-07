# ModelScope 模型仓库常见用法备忘

[返回 README](../README.md) · [模型许可](model-licenses.md) · [模型目录与下载](usage.md#模型下载与固定版本)

本文供发布模型的维护者使用。ModelScope 仓库保存权重、配置、清单和法律文件，上传会产生一个或多个提交。StreamVox 使用者通常只需传模型名，由 SDK 下载已经发布的模型目录。

以下 Python API 按项目固定的 `modelscope==1.30.0` 核对。示例中的 `namespace/model-name`、本地目录和令牌需要换成自己的值。执行上传前，先准备包含完整资产和许可文件的目录。

## 1. 环境与认证

在独立维护环境安装项目使用的 ModelScope 版本，避免升级平台工具时同时改变推理环境依赖。

```powershell
uv venv --python 3.12 .modelscope-maintenance
uv pip install --python .modelscope-maintenance/Scripts/python.exe "modelscope==1.30.0"
```

在 ModelScope 账户设置中创建有目标仓库写入权限的 SDK 令牌，再通过进程环境变量 `MODELSCOPE_API_TOKEN` 提供给下面的脚本。令牌用于身份认证，模型名用于选择上传位置。令牌值应由终端或部署环境注入。[访问令牌设置](https://modelscope.cn/my/myaccesstoken)

## 2. 登录

保存为 Python 脚本后，用上面维护环境的解释器运行。`api.login()` 读取环境中的 SDK 令牌，登录成功后保存供后续 API 和 Git 操作使用的认证资料。

```python
import os

from modelscope.hub.api import HubApi


api = HubApi()
api.login(access_token=os.environ["MODELSCOPE_API_TOKEN"])
```

后续 Python 片段沿用已经登录的 `api`。`os.environ[...]` 在未设置令牌时会直接指出缺少的变量，便于在上传前发现环境问题。

## 3. 上传文件 / 目录（推荐 API）

上传完整目录前，先在网页中创建目标模型仓库，填写可见性和该模型实际采用的许可证。`create_model()` 也能建仓，但其许可证默认值为 Apache License 2.0，调用时需要显式核对并传入正确值。模型权重、派生模型与源码可能分别适用不同条款。[模型许可](model-licenses.md)

StreamVox 模型目录应包含 `manifest.json` 和清单里的全部运行资产。模型卡 `README.md` 说明来源、格式、用法与验证范围，`configuration.json` 提供平台展示所需的元信息，模型的 LICENSE、MODEL_LICENSE、NOTICE 等法律文件随目录保存。

```python
api.upload_folder(
    repo_id="namespace/model-name",
    folder_path="./model-bundle",
    path_in_repo="",
    revision="master",
    commit_message="发布完整模型目录和校验清单",
    max_workers=8,
)
```

`repo_id` 选择远端仓库，`folder_path` 选择本地目录，`path_in_repo=""` 表示把目录内容放到仓库根部。`revision` 在此选择要写入的分支，下面的例子使用已有 `master`。`max_workers` 控制平台上传任务的并发数。

只替换一个文件时，用 `upload_file()` 明确指定本地文件和远端路径。

```python
api.upload_file(
    repo_id="namespace/model-name",
    path_or_fileobj="./model-bundle/README.md",
    path_in_repo="README.md",
    revision="master",
    commit_message="更新模型说明",
)
```

`allow_patterns` 和 `ignore_patterns` 可以限制目录上传的文件范围。文件较多时，`upload_folder()` 会分批提交，返回单个 `CommitInfo` 或其列表。所有批次完成后，再取得目标分支的最终完整提交，核对文件树、字节数和 SHA256。发布 StreamVox 模型时，远端清单应与实际上传资产对应，上传与音频质量验收分别记录。

## 4. 旧版 SDK`push_model`（降级方案）

旧脚本可能使用 `push_model()`。该接口在 1.30.0 已标记弃用，现役发布流程统一使用上一节的 `upload_folder()` 与 `upload_file()`。迁移时将旧脚本的 `model_id` 对应到 `repo_id`，`model_dir` 对应到 `folder_path`，建仓的可见性与许可证在创建仓库时填写。

这样，建仓负责决定谁能访问和采用哪些条款，上传负责提交已准备好的文件。上传目录中的 `configuration.json` 和法律文件仍需由维护者提供。

## 5. 版本模型分支、Tag、Commit

分支记录一条持续更新的版本线，例如 `master`。每个提交的完整标识确定一次仓库内容，目录上传可按批次产生多个提交。标签（tag）给某个提交取一个便于记忆的名字，例如 `v1.0.0`。

| 操作 | `revision` 选择的内容 |
| --- | --- |
| `upload_folder()` / `upload_file()` | 写入目标分支。发布时选择已经存在的分支。 |
| `snapshot_download()` | 读取指定分支、标签或提交的文件快照。 |
| `streamvox.download.download_model()` 固定下载 | 同时传完整 40 位提交与预期清单 SHA256。 |

要复现一次发布，应记录完整提交标识与原始清单摘要。分支名会随着后续提交移动，标签的实际目标也应核对。[StreamVox 固定版本下载](usage.md#模型下载与固定版本)

## 6. 开新分支（SDK + Git）

需要单独准备下一版模型时，可从已有仓库克隆并创建分支。先安装 Git 和 Git LFS，再初始化大文件支持。

```bash
git lfs install
git clone https://www.modelscope.cn/namespace/model-name.git
cd model-name
git switch -c feat/next-model
```

修改并检查模型文件后，选择要提交的文件，提交并推送这个分支。Git 凭据由本机凭据管理配置，`HubApi.login()` 保存的资料用于 SDK 的仓库工具。

```bash
git status --short
git add README.md configuration.json manifest.json ckpt/ LICENSE NOTICE
git commit -m "Prepare next model version"
git push -u origin feat/next-model
```

上例中的文件名需要按实际目录选择，独立模型许可文件也应纳入提交。Git LFS 把大文件内容存到专用存储，并在 Git 提交里记录对应对象，推送时需要一并完成大文件传输。

如果维护脚本使用 `Repository`，可显式选择本地和远端分支。下面先把已修改、且内容与远端版本不同的 `model-bundle/README.md` 写入克隆目录，再提交这项改动。

```python
from pathlib import Path
import shutil

from modelscope.hub.repository import Repository


repo = Repository("./model-repository", clone_from="namespace/model-name", revision="master")
shutil.copyfile("./model-bundle/README.md", Path(repo.model_dir) / "README.md")
repo.push("准备下一版模型", local_branch="master", remote_branch="feat/next-model")
```

`Repository.push()` 在该版本也会提示弃用。原生 Git 步骤把分支和提交过程展开，便于检查上传了哪些文件。直接向一个尚未创建的分支调用 `upload_folder()` 是否成功还取决于服务端，发布流程先创建分支。

## 7. 打 Tag

发布确认后，把标签指向已经验证的完整提交，再推送标签。先在克隆目录中运行 `git rev-parse HEAD` 取得提交值，替换下面的占位内容。

```bash
git tag -a v1.0.0 FULL_COMMIT_SHA -m "Model release v1.0.0"
git push origin v1.0.0
```

SDK 的仓库对象也提供相同操作。

```python
repo.tag_and_push("v1.0.0", "Model release v1.0.0", ref="FULL_COMMIT_SHA")
```

`HubApi.create_model_tag(model_id, tag_name)` 在 1.30.0 固定以 `master` 为目标，函数签名没有选择任意提交的 `ref` 参数。要准确绑定某个提交，使用上面的 Git 或 `Repository.tag_and_push()` 路径。

## 8. 「回退」的两种含义

应用需要继续使用旧模型时，下载旧标签或完整提交即可。例如下面取 `v0.9.0` 的文件快照。

```python
from modelscope.hub.snapshot_download import snapshot_download


model_dir = snapshot_download("namespace/model-name", revision="v0.9.0", cache_dir="./model-cache")
print(model_dir)
```

若要修正共享仓库当前内容，可在克隆目录使用 `git revert BAD_COMMIT_SHA` 创建一个抵消错误提交的新提交，再正常推送。也可以重新上传修正后的文件，两种做法都在历史末尾留下修复记录。

`git reset --hard` 配合强制推送会移动分支指针并改写共享历史，其他协作者需要重新对齐分支。`HubApi` 的这些上传接口没有独立的 `revert()` 方法，常规内容恢复按新提交处理。

## 9. 查看分支与 Tag

```python
branches, tags = api.get_model_branches_and_tags_details(
    "namespace/model-name", use_cookies=True,
)
print(branches)
print(tags)
```

`branches` 与 `tags` 分别列出分支和标签的详情。查看分支当前完整提交，也可使用只读 Git 命令。

```bash
git ls-remote https://www.modelscope.cn/namespace/model-name.git refs/heads/master
```

## 10. CLI 简要（可选）

ModelScope 1.30.0 的 CLI 提供 `login`、`download`、`upload` 等子命令，具体参数由该版本的 `--help` 列出。项目旧备忘中的 `modelscope model -act ...` 应迁移到当前命令或本文的 Python API。

在本项目使用 setuptools 84.0.0 的环境中，CLI 启动会因依赖旧的 `pkg_resources` 而报错，`HubApi` 导入和上传 API 可独立使用。维护脚本可采用本文已核对签名的 Python 接口，避免将 CLI 的依赖调整带入推理环境。

## 11. 前置依赖（Git 路线）

Git 路线需要 Git、Git LFS 和目标仓库的写入权限。SDK 的 HTTP 上传接口自行处理平台传输，使用 `Repository` 或原生 Git 时则由 Git 工具管理本地提交及大文件对象。

上传前检查目标仓库、分支、完整文件目录和适用许可。上传后保存远端提交值与清单摘要，供使用者固定下载该版本。

## 12. 参考链接

- 创建访问令牌，用于身份认证与仓库写入。[账户设置](https://modelscope.cn/my/myaccesstoken)
- 查阅固定版本实现，核对本地实际可用的参数。[ModelScope 1.30.0 源码](https://github.com/modelscope/modelscope/tree/v1.30.0)
- 查看平台 SDK 说明，了解当前服务端的仓库操作。[官方 SDK 文档](https://modelscope.cn/docs/SDK%E6%96%87%E6%A1%A3)
