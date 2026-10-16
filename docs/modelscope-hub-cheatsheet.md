# ModelScope 模型仓库常见用法备忘

本文汇总通过 **Python SDK** 与 **CLI** 管理 ModelScope 上模型仓库时的常见操作：上传、分支、Tag、版本与回退思路。模型在平台上的形态本质是 **Git 仓库**（大文件经 **Git LFS**）。

---

本文属于模型仓库维护备忘，不是 StreamVox 安装步骤。上传示例中的许可证字段仅为占位示例，必须按实际模型许可填写；SDK 的 Apache-2.0 不会替代模型权重条款。使用模型请先阅读 [模型许可与来源](model-licenses.md)。

## 1. 环境与认证

- **安装**：`pip install -U "modelscope>=1.26"`（过旧版本可能没有 `HubApi.upload_folder` / `upload_file`）。
- **SDK 令牌**：[modelscope.cn 访问令牌](https://modelscope.cn/my/myaccesstoken) 创建 **SDK 令牌**。
- **推荐**：`export MODELSCOPE_API_TOKEN='...'`，避免把密钥写进脚本或提交到 Git。

---

## 2. 登录

**命令行：**

```bash
modelscope login --token YOUR_SDK_TOKEN
```

**Python：**

```python
from modelscope.hub.api import HubApi

api = HubApi()
api.login(access_token="YOUR_SDK_TOKEN")  # 已设置 MODELSCOPE_API_TOKEN 时可省略参数
```

---

## 3. 上传文件 / 目录（推荐 API）

适用于较新的 `modelscope` 版本。

```python
from modelscope.hub.api import HubApi

api = HubApi()
api.login()

# 整目录上传；repo_id 格式为 owner/repo
api.upload_folder(
    repo_id="namespace/model-name",
    folder_path="/path/to/local_folder",
    path_in_repo="",  # 可选：仓库内子目录
    commit_message="说明",
    revision="master",  # 分支或已有 tag，见下文
)

# 单文件
api.upload_file(
    path_or_fileobj="/path/to/file.bin",
    path_in_repo="weights/file.bin",
    repo_id="namespace/model-name",
    commit_message="上传单文件",
    revision="master",
)
```

**说明：**

- 若仓库不存在，部分接口会尝试创建；模型卡展示仍建议自备 `README.md`、`configuration.json` 等。
- 可用 `allow_patterns` / `ignore_patterns` 控制上传范围（以 SDK 文档为准）。

---

## 4. 旧版 SDK：`push_model`（降级方案）

当 `upload_folder` 不存在时，可用整目录推送（已标记 Deprecated，但旧环境仍可用）。目录需有 **`configuration.json`**；若缺失，部分版本会自动生成默认配置。

```python
api.push_model(
    model_id="namespace/model-name",
    model_dir="/path/to/local_folder",
    visibility=5,  # 1 私有，5 公开
    license="Apache License 2.0",
    chinese_name="显示名",
    commit_message="说明",
    revision="master",
    # 大文件可尝试 lfs_suffix，如 ["*.gguf", "*.safetensors"]
)
```

文档说明：若远端 **不存在** `revision` 对应分支，会 **新建分支** 再推送。

---

## 5. 版本模型：分支、Tag、Commit

| 概念 | 含义 |
|------|------|
| **分支（revision 常为分支名）** | 如 `master`，持续开发线；每次上传通常产生新 **commit**。 |
| **Tag** | 给某次提交打固定名字（如 `v1.0.0`），便于引用稳定版本。 |
| **Commit SHA** | 精确到某次提交；下载时可作为 `revision`（若平台支持）。 |

下载侧统一通过 **`revision`** 指定分支名、tag 名或 commit（与 `snapshot_download` 等 API 一致）。

---

## 6. 开新分支（SDK + Git）

**推荐：本地克隆后推到新远程分支名**（官方测试用例用法）。

```python
from modelscope.hub.repository import Repository

model_id = "namespace/model-name"
local_dir = "/path/to/clone_dir"

repo = Repository(local_dir, clone_from=model_id, revision="master")
# 修改文件后：
repo.push("提交说明", remote_branch="feat/my-branch")
```

注意：源码中 `Repository.push` 可能带 DeprecationWarning，长期可改用 **原生 git** 完成 `add/commit/push`，逻辑相同。

**仅 `push_model` 时**：设置 `revision="新分支名"`，远端无该分支时会创建。

**仅 `upload_folder` 时**：`revision="新分支名"` 是否能在「分支尚不存在」时创建，取决于 **服务端行为**；失败则改用克隆 + `git push` 或 `Repository.push`。

---

## 7. 打 Tag

**本地仓库打 Tag 并推送**（可指定指向分支或某次提交）：

```python
repo.tag_and_push("v1.0.0", "release v1.0.0", ref="master")
# ref 也可为 commit hash：
# repo.tag_and_push("v1.0.0", "release", ref="abc1234...")
```

**不克隆时**：`HubApi.create_model_tag(model_id, tag_name, ...)`（参数与 `Ref` 行为以本机 SDK 的 `help` / 源码为准）。

---

## 8. 「回退」的两种含义

**（1）使用旧版本（不改仓库）**

下载时指定旧 **tag** 或 **commit**：

```python
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download("namespace/model-name", revision="v0.9.0", cache_dir="...")
```

**（2）修正远端历史（需 Git）**

- 协作友好：`git revert <坏提交>` 再推送。
- 改写分支指针：`git reset --hard <旧 SHA>` + `git push --force`（需权限，**慎用**）。
- 或再上传一版文件，把内容恢复成旧状态（**前进式**修复，历史中仍保留错误提交）。

平台侧通常 **没有** 单独的 `HubApi.revert` 一键接口。

---

## 9. 查看分支与 Tag

```python
# 示例：列出与 revision 相关的 API（名称以本机 SDK 为准）
api.list_model_revisions("namespace/model-name", use_cookies=True)
# 或 get_model_branches_and_tags_details 等
```

具体方法请以 `dir(HubApi)` / `help(HubApi)` 为准。

---

## 10. CLI 简要（可选）

```bash
modelscope model -act create -gid 'GROUP' -mid 'MODEL_ID' -vis 1 -lic 'MIT' -ch '中文名'
modelscope model -act upload -gid 'GROUP' -mid 'MODEL_ID' -md ./modelfiles/ -vt 'v0.0.1' -vi '说明'
```

参数以本机 `modelscope model --help` 为准。

---

## 11. 前置依赖（Git 路线）

使用 `Repository` 或原生 git 推送时：

- 安装 **Git** 与 **Git LFS**（`git lfs install`）。
- 已 `HubApi.login` 或配置好令牌，以便远程 URL 带认证。

---

## 12. 参考链接

- 令牌：<https://modelscope.cn/my/myaccesstoken>
- 上游仓库：<https://github.com/modelscope/modelscope>
- Hub 上传相关 PR：<https://github.com/modelscope/modelscope/pull/1152>

---

*文档为项目内备忘，若平台 API 变更请以官方文档与当前 `modelscope` 版本为准。*
