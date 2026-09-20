# AGENTS.md · 本仓库对 AI 协作者（含 coding agent）的额外约束

本文件覆盖全局默认规则。在本仓库工作的 AI agent 必须遵守以下条款。

## 绝对禁止

1. **不得替学生写实现。** `src/pdl/` 下的功能代码必须由学生自己完成。
   agent 可以帮助：解释概念、写测试脚手架、修工具链配置、review 代码。
2. **不得创建 Commit、推送分支、创建或合并 PR**，除非用户（导师）在当次对话中明确要求。
3. **不得直接改 `main`**，不得执行 `amend` / `rebase` / `squash` / `force-push`，
   除非用户明确要求且已确认目标是未合入的功能分支。
4. **不得为了让检查通过而弱化检查**：不要放宽 `pyproject.toml` 里的 ruff / mypy / 覆盖率配置，
   不要 `# type: ignore` 掩盖真实问题，不要删测试。

## 「从零」的边界（最重要的一条）

本仓库的定位是「**PyTorch 张量 + torch.autograd 当底层，上层全部自己写**」。
动 `src/pdl/` 之前先确认代码落在哪一边：

**允许直接用**

- `torch.Tensor` 及其全部张量运算：`@` / `matmul`、广播、`exp` / `log`、`sum` / `mean`、
  `reshape` / `transpose`、索引切片
- `torch.autograd`：`requires_grad`、`backward()`、`.grad`、`torch.no_grad()`、`detach()`
- `torch.nn.Parameter`
- `torch.device` 与 `.to(device)`

**必须自己写**

`Module` 基类与参数注册、`Linear`、`ReLU`/`Sigmoid`/`Tanh`、`Dropout`、`Sequential`、
Xavier/He 初始化、`MSELoss`、`CrossEntropyLoss`、`Optimizer` 基类、`SGD`/`Momentum`/`Adam`、
梯度裁剪、`Dataset`/`DataLoader`、训练循环、早停、指标、数值梯度校验器。

**`src/pdl/` 里永久禁止**

`torch.nn`、`torch.nn.functional`、`torch.optim`、`torch.nn.init`、`torch.utils.data`。
`src/pdl/` 是"从零实现区"，这些模块永远不会出现在里面——不是"到 Stage 7 才解禁"。
Stage 7 的对照代码放 `examples/`，那里不受限制。

**测试里用 `torch.nn.functional` 当 oracle 是允许且鼓励的**——禁令只针对自己实现的部分，
不针对验证的部分。

这条边界由 `tests/test_import_boundary.py` 用 AST 强制（含 `import torch.nn as nn` 的别名写法），
CI 会拦住越界。**不要为了让代码通过而修改或绕过这个测试。**

理由：这个项目要练的是参数组织、梯度使用、数值稳定性和训练循环设计，不是张量加法和矩阵乘。
完整说明见 `docs/roadmap.md` 第 1 节和第 4 节。

## 必须遵守

1. 任何代码改动前先确认对应的 **Issue ID**；没有 Issue ID 就不要动 `src/pdl/`。
2. 改动后必须真实运行 `make check`，并如实报告结果。没有实际跑过就不要说"应该能过"。
3. 新增/修改行为必须补测试；梯度相关代码必须做数值梯度校验；
   涉及 `log` / `exp` 的必须做数值稳定性测试（大输入不能出 `inf` / `nan`）。
4. 文档与代码不一致时，以 `docs/roadmap.md` 的 Stage 划分为准；变动要同时更新两边。
5. 面向学生的解释用中文，语气直白，先给直觉再给公式，公式只保留能落到代码的部分。

## 本仓库的教学约定

- 一次只引入一个新概念；不要顺手重构学生没学过的部分。
- 命名与接口要直白，不用设计模式包装。
- 拒绝过度抽象：三个相似函数胜过一个提前设计的基类。
- 统一 `torch.float32` 与 `torch.int64`（标签），不要引入第二套数组库语义。
