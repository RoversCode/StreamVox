# 参与贡献

[返回 README](README.md) · [开发与验证](docs/development.md)

## 仓库与协作流程

[RoversCode/StreamVox](https://github.com/RoversCode/StreamVox) 是 SDK 源码、测试、示例、公开文档和构建脚本的唯一维护仓库。

1. 从最新的 `master` 创建独立分支。外部贡献者先 Fork 仓库，在自己的分支上开发。
2. 修改一个明确的问题或功能，并同步受影响的测试、示例和文档。
3. 按 [开发与验证](docs/development.md) 准备 Windows x64 / CPython 3.12 环境，运行与修改相关的验证；源码变更应运行 `uv run python -m pytest tests -q`。
4. 向本仓库的 `master` 提交 PR，说明问题、修改后的行为、验证命令与结果，以及仍未验证的部分。
5. 维护者审查并合并后，从已合并的确定提交构建发行产物。源码提交、发行标签与构建来源必须一致。

涉及模型行为、后端或性能的修改，还应说明模型版本、硬件、输入和测量方法。单元测试通过不能替代实际合成、数值对照或人工试听。

## 提交内容边界

提交与项目维护直接相关的源码、测试、示例、文档、许可证、依赖配置和构建脚本。模型权重、个人参考音频、虚拟环境、缓存、日志和发行产物保留在本地；原生运行库通过项目提供的下载恢复流程取得。

个人 AI 规则与编辑器配置由本机管理，例如 `AGENTS.md`、`CLAUDE.md`、`.codex/`、`.claude/` 和 `.vscode/`。将这些路径写入当前克隆的 `.git/info/exclude`；该文件不会提交到仓库。例如：

```gitignore
AGENTS.md
AGENTS.override.md
CLAUDE.md
.codex/
.claude/
.cursor/
.vscode/
```

`.gitignore` 维护所有贡献者通用的生成文件规则。个人排除规则只对未跟踪文件生效；已跟踪的个人文件需要先取消跟踪并保留本地副本。提交前检查 `git status --short` 与暂存区差异，确认只包含本次贡献所需的文件。
