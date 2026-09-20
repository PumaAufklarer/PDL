# PDL · 教学型深度学习工程库

一个用来学深度学习的**真实工程仓库**。

底层用 PyTorch 的张量和自动微分，**上层全部自己写**：层、损失、优化器、数据管道、训练循环。
这不是在造框架，而是在把框架里真正需要理解的部分亲手做一遍。

教材参照[《动手学深度学习》](https://zh.d2l.ai/)（D2L），工程规范参照本仓库自己的 `CONTRIBUTING.md`。

> 如果你是这个仓库的学生：**先读 [`docs/roadmap.md`](docs/roadmap.md) 的第 3 节「数学怎么补」和第 6 节「学习节奏」**，再读当前 Stage 的条目。
> 知识点不懂就翻 [`docs/handbook/`](docs/handbook/README.md)；当前要做的 Issue 名字在 roadmap 附录 A。

## 快速开始

前置条件：Python ≥ 3.10、[uv](https://docs.astral.sh/uv/)、git。

```bash
git clone <this-repo>
cd PDL

uv sync --all-groups     # 创建 .venv 并安装依赖（含 dev 依赖）
make check               # 提 PR 前必须通过：格式 + lint + 类型 + 测试
```

`make help` 可以看到所有可用命令。

> torch 走的是 **CPU 索引**（`download.pytorch.org/whl/cpu`，约 190 MB）。
> 默认索引在 Linux 上会拖一整套 CUDA 运行时（2 GB 以上），CI 时间会差一个数量级。
> 配置在 `pyproject.toml` 的 `[[tool.uv.index]]` 里，不要改掉。

## 「从零」的边界

| | |
|---|---|
| ✅ 直接用 | `torch.Tensor` 的全部张量运算、`torch.autograd`、`torch.nn.Parameter`、`torch.device` |
| ❌ 自己写 | `Module`/`Linear`/激活/`Dropout`/`Sequential`、损失、优化器、初始化、`Dataset`/`DataLoader`、训练循环、指标、梯度校验器 |
| ⛔ Stage 7 前禁用 | `torch.nn`、`torch.nn.functional`、`torch.optim`、`torch.nn.init`、`torch.utils.data` |

完整说明见 [`docs/roadmap.md`](docs/roadmap.md) 第 1 节和第 4 节。

## 目录结构

```
docs/
├── roadmap.md            学习路线图：学什么、按什么顺序、每个 Issue 叫什么名字
├── handbook/             知识点手册：通俗讲清概念，只讲必要的数学
├── notes/               你自己的推导笔记
├── experiments/         实验记录：一份 config + 一张结果表，必须可复现
└── journal/             每次学习的日志，三行即可

src/pdl/                  库本体，一个子包对应一组 Stage
├── utils/                Stage 1  数值梯度校验器、随机种子、实验记录
├── data/  loss/  optim/  Stage 2  数据管道、损失、优化器
├── trainer/              Stage 2–4 训练循环、验证、早停
├── metrics/              Stage 3  准确率、混淆矩阵
└── nn/                   Stage 4–5 层与模型（Module、Linear、激活、Dropout、Sequential）

tests/                    与 src/pdl/ 同构的测试目录
examples/                 端到端可运行脚本（跑完能看到数字的那种）
scripts/                  工具脚本（commit message 校验等）
```

## 这个仓库怎么运转

- **每个任务一个 Issue**，用「任务」模板开，五段必填（目标/背景/验收标准/提示/不要做）
- **一个 Issue 一个 PR**，用 PR 模板，其中「我怎么验证的」必须贴真实命令与输出
- **CI 是第一个 reviewer**：格式、lint、类型、测试、覆盖率、commit message 全过才能合并
- **Review 意见分四级**：`[blocking]` / `[suggestion]` / `[nitpick]` / `[question]`，见 `CONTRIBUTING.md`
- **导师不写实现**，只开 Issue、写测试、给 review 意见

## 当前状态

骨架阶段：包结构与工具链已就绪，`src/pdl/` 下所有子包都是空的。
下一步是 `docs/roadmap.md` 里的 Stage 0 与 Stage 1。

<!-- TODO(first-pr): 把这一行整行删掉就行，别改别的地方。任务说明见 docs/roadmap.md 的 Stage 0。 -->
