# CLAUDE.md

本文件是 `E:\Experiment\boxmot` 的唯一协作说明源。

- 后续涉及仓库规范、环境、测试、文档同步时，统一读取并修改本文件。
- `AGENTS.md` 仅作为入口指引保留。
- 若 `AGENTS.md` 与本文件冲突，以本文件为准。

## 0. 每次会话的上下文恢复与收尾

- 开始项目工作前，若本地存在 [项目状态](docs/research/project-state.md)，先读取它，从当前目标、证据状态和下一步恢复上下文；已记录的背景无需再请用户重复。
- 涉及实验运行、结果比较或复现时，若文件存在则继续读 [实验台账](docs/research/experiments.md)；涉及调研、选题或改进设计时，继续读 [调研与方向](docs/research/research-log.md) 及其中相关专题。
- 本地台账存在时，完成有实质进展的会话后更新对应台账及项目状态中的下一步；运行失败、否定结论、暂停方案也要记录。纯解释且无新事实时无需追加流水账。
- 记录采用绝对日期，区分用户确认、文件证据、运行验证和待验证假设。历史产物缺少配置或来源时保持待追溯，不能补写成已复现实验。
- 当前目标以项目状态为唯一入口；协议细节维护在已有专题中，指标与运行证据维护在实验台账中，其他文档只放指针。

## 1. 基本规则

- 所有代码注释、技术文档、README 片段、排障说明统一使用中文。
- 技术名词保留英文原文，不强制翻译。
- 注释优先解释“为什么这样做”，不要只重复代码表面行为。
- 需要进行 git 操作时，遵循 `git-workflow`。
- 分支名默认使用 `codex/<short-topic>`。
- commit message 前缀使用 `feat:`、`fix:`、`refactor:`、`docs:`、`ci:`、`perf:`。
- commit message 中不要加入 `Co-Authored-By`。

## 2. Python 环境与运行方式

- conda 环境名：`detr`
- Python 路径：`D:\anaconda3\envs\detr\python.exe`
- pip 路径：`D:\anaconda3\envs\detr\Scripts\pip.exe`

统一要求：

- 使用 `D:\anaconda3\envs\detr\python.exe` 执行 Python 命令。
- 不要使用 `conda run`。
- 本仓库本地协作不要使用 `uv` 执行命令。

安装命令：

```bash
D:\anaconda3\envs\detr\Scripts\pip.exe install -e .
```

始终从仓库根目录以模块方式运行：

```bash
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli --help
```

避免：

```bash
python boxmot/engine/cli.py --help
PYTHONPATH=. python boxmot/engine/cli.py --help
```

原因：本仓库若干导入依赖 package context，直接跑脚本容易出现导入或路径行为不一致。

## 3. 当前仓库事实

- 项目名：`boxmot`
- 当前版本：`16.0.9`
- 许可证：`AGPL-3.0`
- Python 要求：`>=3.9,<3.13`
- CLI 入口：`boxmot.engine.cli:main`

当前 CLI 子命令：

- `track`
- `generate`
- `eval`
- `tune`
- `export`
- `version`
- `help`

当前支持的跟踪器、检测器分流、导出格式等实现细节，不要在这里维护长列表，直接以这些代码为准：

- `boxmot/engine/cli.py`
- `boxmot/trackers/tracker_zoo.py`
- `boxmot/detectors/__init__.py`
- `boxmot/reid/core/__init__.py`

补充：

- `track` 使用单个检测器和单个 ReID 权重。
- `generate`、`eval`、`tune` 支持多个 `--yolo-model` / `--reid-model`。
- `strongsort` 在 `create_tracker()` 中不走 `per_class`，不要假设所有跟踪器都支持该行为。

## 4. 代码修改约束

- 新增或修改函数、类时，优先补充 type hints 与 docstring。
- import 保持精简和有序，删除未使用项。
- 不要用 `try/except` 包 import，除非确有必要且在代码中说明原因。
- 库代码统一使用现有 `LOGGER` / `loguru logger`，不要用 `print`。
- CLI 入口如确有必要可以使用 `click.echo()` 或 `print`，但优先沿用现有日志风格。
- 修改 CLI 时，优先复用现有装饰器和解析辅助函数，不要重复造参数处理逻辑。

## 5. 文档同步原则

文档分层：

- `CLAUDE.md`：面向仓库协作代理，记录规则、环境、入口、陷阱。
- `README.md`：面向外部使用者，记录安装、快速开始、能力概览。
- `GUIDE.md`：面向中文深度说明，记录结构、用法、排障、经验。
- `docs/`：面向 MkDocs 站点。
- `docs/research/`：不纳入基准提交的本地实验知识库；存在时用于恢复项目状态，需自行备份。

同步原则：

- 改 CLI 参数或默认值时，至少同步检查 `README.md`、`GUIDE.md` 和相关示例。
- `docs/modes/*.md` 当前主要通过 `mkdocs-click` 反映 CLI；如果 CLI 没改，只改单独的 mode 文档通常没有实际意义。
- 改跟踪器列表、检测器分流、导出格式时，以代码为源头，再同步 README / GUIDE 中的示例和说明。

## 6. 测试与验证

默认测试命令：

```bash
D:\anaconda3\envs\detr\python.exe -m pytest
```

常用最小相关测试：

```bash
D:\anaconda3\envs\detr\python.exe -m pytest tests\test_config.py
D:\anaconda3\envs\detr\python.exe -m pytest tests\unit\test_trackers.py
D:\anaconda3\envs\detr\python.exe -m pytest tests\unit\test_postprocessing.py
D:\anaconda3\envs\detr\python.exe -m pytest tests\unit\test_reidbackend.py
```

如果无法完成验证，不要伪造结果；要明确写出执行的命令、成功/失败状态和失败原因。

## 7. 环境验证边界

`2026-11-09` 在本工作区运行以下命令成功（退出码 0）：

```bash
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli --help
```

历史上曾在导入 `ultralytics`、读取用户目录 `settings.yaml` 时出现 `PermissionError`；本次未复现。帮助命令成功不代表完整检测、跟踪与评估链路已复现。切换环境或机器后重新检查实际包路径；具体验证与实验状态见实验台账。

## 8. 维护约定

- 后续更新仓库协作规范时，优先修改本文件。
- `AGENTS.md` 不再维护完整副本，只保留最小跳转说明。
