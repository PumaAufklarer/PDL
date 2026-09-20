# PDL 学习路线图

> **这份文档是写给你的。** 它回答三个问题：我们要走到哪、按什么顺序走、每一步怎么算走完了。
>
> 建议先读第 0、2、3、5 节（不到十分钟），然后每次开工前只看当前 Stage 那一段。
> 具体某个知识点看不懂，去翻 [`docs/handbook/`](handbook/README.md)。

**配套文档**

| 文档 | 作用 |
|------|------|
| 本文档 `docs/roadmap.md` | 学什么、按什么顺序学、怎么验收、每个 Issue 叫什么名字 |
| `docs/handbook/` | 知识点手册：通俗讲清概念，并指出每个概念对应哪一行代码 |
| `CONTRIBUTING.md` | 流程规范：Issue / 分支 / Commit / PR / Review |
| `README.md` | 仓库怎么跑起来 |

---

## 0. 这门课要带你走到哪

用一个**真实可维护的 Python 库**（`pdl`）当载体，让你从零走完这条链路：

```
张量与自动微分 → 线性回归 → Softmax 分类 → 泛化与正则 → 多层感知机 → 深度学习计算
```

教材参照[《动手学深度学习》](https://zh.d2l.ai/)（D2L），路线和它基本一致。
区别在于：**D2L 用 Notebook 讲一遍，我们用工程仓库做一遍。** 同样的东西，你要写成有类型注解、有测试、有 CI、能被 review 的代码。

走到终点时，你应该能：

- 用大白话解释反向传播、梯度下降、过拟合、Softmax、MLP 分别在干什么；
- 独立完成 `读 Issue → 建分支 → 小步实现 → 自己写测试 → make check → 提 PR → 回应 review → 修 CI → 合并` 的完整闭环；
- 拿到一个没做过的功能，知道先问清验收标准，而不是直接开写。

**这里没有考试，也没有时间表。** 验收标准是"这个 Issue 的 DoD 全部满足、CI 全绿、review 通过"。

---

## 1. 项目定位：`pdl` 是什么

`pdl` 是一个**教学型深度学习工程库**。

它用 PyTorch 的张量和自动微分当底层，但从零实现**上层的一切**：层、损失、优化器、数据管道、训练循环。
换句话说——**张量运算是借来的，模型和训练是你自己搭的。**

这不是偷懒。这正是 D2L 的做法：它的"从零实现"章节里，`w`、`b` 是 `torch` 张量，梯度来自 `l.backward()`，
但**损失、参数更新、训练循环全是手写的**。D2L 从没让人写过自动微分引擎，我们也不写。

```
src/pdl/
├── utils/      Stage 1    数值梯度校验器、随机种子、实验记录
├── data/       Stage 2    Dataset / DataLoader / 批采样（不用 torch.utils.data）
├── loss/       Stage 2–3  MSE、交叉熵（含 log-sum-exp）
├── optim/      Stage 2–4  优化器基类、SGD、Momentum、Adam、权重衰减、梯度裁剪
├── trainer/    Stage 2–4  训练循环、验证、早停、日志
├── metrics/    Stage 3    准确率、混淆矩阵
└── nn/         Stage 4–5  层与模型：Module、Linear、激活、Dropout、Sequential、初始化
```

**接口像框架（能复用、能测试），实现像教材（直白、无魔术）。**

### "从零"的边界

这条线必须画死，否则很容易不知不觉滑成"调包"。完整清单见第 4 节。

| | |
|---|---|
| ✅ **允许** | `torch.Tensor` 及其全部张量运算（`@`、广播、`exp`/`log`、`sum`/`mean`、`reshape`/`transpose`、索引）<br>`torch.autograd`：`requires_grad`、`backward()`、`.grad`、`torch.no_grad()`、`detach()`<br>`torch.nn.Parameter`（只是"这个张量要训练"的标记）<br>`torch.device` 与 `.to(device)` |
| ❌ **必须自己写** | 层：`Module` 基类与参数注册、`Linear`、`ReLU`/`Sigmoid`/`Tanh`、`Dropout`、`Sequential`<br>损失：`MSELoss`、`CrossEntropyLoss`<br>优化器：`SGD`、`Momentum`、`Adam`、梯度裁剪<br>初始化：Xavier、He<br>数据：`Dataset`、`DataLoader`、批采样、shuffle<br>训练：训练循环、验证、早停、指标、实验记录<br>测试：数值梯度校验器 |
| ⛔ **`src/pdl/` 里永久禁止** | `torch.nn` 里的层实现（`Linear`/`ReLU`/`Sequential`/`Module`）<br>`torch.nn.functional` 里的 `cross_entropy`/`mse_loss`/`relu`<br>`torch.optim.*`、`torch.nn.init.*`、`torch.utils.data.DataLoader`<br>由 `tests/test_import_boundary.py` 强制。测试里当 oracle 用是允许的 |

**为什么这条线划在这里？** 因为在这个项目里，真正有工程含量、也真正需要你理解的是上表第二行：
参数怎么组织、梯度怎么用、数值怎么才稳定、训练循环怎么设计。张量加法和矩阵乘不是。

---

## 2. 开始之前：确认你已经会这些

不需要全会才开工，但下面这些要能看懂。有不会的先说，我们补上再开始。

- [ ] Python 语法：变量、条件、循环、函数、列表/字典、类的基本用法
- [ ] 会用 `pip install`，见过 `ModuleNotFoundError`
- [ ] 能读懂 Python 的报错信息（哪怕读得慢）
- [ ] 电脑上能装软件、能用命令行 `cd` / `ls` / `mkdir`

**这份路线图不假设你会**（这些正是要学的）：

- git 分支 / PR / code review
- 单元测试、CI、lint、类型注解
- 微积分和线性代数（**这一条见第 3 节，不是门槛**）

---

## 3. 数学怎么补：按需，永远挂在一个例子上

**先说结论：你不需要先去补数学。**

单独上数学课对你不适用——不是因为数学不重要，而是因为脱离用途的符号没有锚点，听不下去是正常的。
这里换一种方式：**每一个数学概念都在它第一次要被用到的那一刻才出现，而且一定绑定在一个具体的代码例子上。**

三条规则：

1. **用到哪补哪。** 不看"微积分入门"这类通讲材料，只查当前这一行代码需要的那一个概念。
2. **先跑通代码，再回头看公式。** handbook 每章的「最小数学」都可以先跳过，写完代码回来读，会顺很多。
3. **允许反复横跳。** 同一节读三遍是正常的，不是笨。

### 你需要补的数学清单（就这些）

| 数学概念 | 什么时候用到 | 具体挂在哪个例子上 | 学到什么程度算够 |
|----------|--------------|--------------------|------------------|
| 矩阵乘 | Stage 1 | `X @ W`：一批样本乘权重矩阵 | 会判断形状能不能乘、结果形状是多少 |
| 求和记号 Σ | Stage 1 | `loss.mean()` 把一批损失压成一个数 | 看得懂 Σ 就是 for 循环累加 |
| 广播 | Stage 1 | `X + b`：偏置加到每一行 | 会手推形状，知道维度 1 会被拉伸 |
| 导数 | Stage 1 | "`w` 动一点点，`loss` 动多少" | 会算 $x^n$、$e^x$、$\log x$ 的导数就够，**不需要极限的严格定义** |
| 偏导 | Stage 1 | 多参数时，一次只动一个 | 把其他变量当常数 |
| 梯度 | Stage 1 | `w -= lr * w.grad` | 知道它指向上升最快的方向，所以要减 |
| 链式法则 | Stage 1（Stage 5 深化） | 手算 `(3x+1)**2` 的导数；Stage 5 手推一层反向时是同一件事 | 会算 $(f(g(x)))' = f'(g(x))\,g'(x)$，能连套三层 |
| 向量范数 | Stage 2 | MSE 里的 $\|y-\hat y\|^2$；权重衰减 | 知道 $\|w\|^2$ 就是各分量平方和 |
| 期望与方差 | Stage 2、4 | 数据里的噪声；参数初始化的尺度 | 知道"平均水平"和"波动大小"说的是什么 |
| 条件概率与似然 | Stage 3 | softmax 输出的 $\hat p_y$ 和交叉熵 | 知道"给定输入属于某类的概率"，最大似然 = 最小化负对数 |
| 指数与对数 | Stage 3 | softmax 的数值稳定 | 记得 $\log(ab)=\log a+\log b$、$e^{\log x}=x$ |
| 乘法法则 | Stage 5 | MLP 一层的反向 | 会算 $(uv)' = u'v + uv'$ |

**这份表就是全部的数学前置。** 除此之外遇到的公式，都可以先跳过。

---

## 4. "从零"的边界（写死，不要临场发挥）

### 4.1 你可以直接用 torch 的部分

- 张量的创建、变形、索引、广播、全部逐元素运算
- `@` / `torch.matmul`、`sum` / `mean` / `max` 等归约
- `torch.autograd` 的全部公开接口
- `torch.nn.Parameter`
- `torch.device` / `.to(device)` / `torch.cuda.is_available()`

### 4.2 你必须自己写的部分

| 模块 | 要写的东西 | 为什么它值得自己写 |
|------|-----------|-------------------|
| `utils/` | `numerical_grad`、`check_gradient`、`seed_everything` | 这是你验证一切代码的地基，也是理解"梯度到底是什么"的最佳入口 |
| `data/` | `Dataset`、`DataLoader`、批采样、shuffle | 数据管道出 bug 是最高频的坑，自己写过才知道坑在哪 |
| `loss/` | `MSELoss`、`CrossEntropyLoss` | 交叉熵的数值稳定实现是本章最重要的工程技巧 |
| `optim/` | `Optimizer` 基类、`SGD`、`Momentum`、`Adam`、`clip_grad_norm_` | 优化器最容易写错又最难发现，必须自己写才能审出来 |
| `trainer/` | 训练循环、验证、早停、日志 | 这是把上面所有零件接起来的地方 |
| `metrics/` | `accuracy`、`confusion_matrix` | 简单，但要养成"指标和损失是两件事"的习惯 |
| `nn/` | `Module` 基类、`Linear`、激活、`Dropout`、`Sequential`、Xavier/He 初始化 | 参数注册和 `train()`/`eval()` 传播是核心工程设计 |
| 测试 | 数值梯度校验器、每一层的梯度测试 | 见 `CONTRIBUTING.md` 第 6 节 |

### 4.3 `src/pdl/` 里永久禁止的

`torch.nn`（层）、`torch.nn.functional`（损失与激活）、`torch.optim`、`torch.nn.init`、`torch.utils.data`。

**注意：不是"到 Stage 7 才解禁"。** `src/pdl/` 是"从零实现区"，这些模块永远不会出现在里面。
Stage 7 的对照代码放在 `examples/` 下，那里不受这条限制。

**用法不是"替换"，而是"对照"**：用它们写一遍同一个 MLP，然后回答"我的实现和它的差异在哪、为什么"。
用别人的实现当标准答案来测自己的实现，是这一阶段最重要的练习——**永远不要拿自己的代码测自己**。

> **测试里用 `torch.nn.functional` 当 oracle 是允许且鼓励的**——那正是"对照"的正确用法。
> 禁令只针对 `src/pdl/`（自己实现的部分），不针对 `tests/`（验证的部分）。

**这条边界由 `tests/test_import_boundary.py` 强制。** 它用 AST 扫描 `src/pdl/` 下每一条 import 和属性访问，
CI 会拦住任何越界——包括 `import torch.nn as nn` 之后用 `nn.Linear` 这种别名写法。
约定只写在文档里会被忘记，写成测试才会真的被执行。

---

## 5. 七条设计原则

| # | 原则 | 为什么 |
|---|------|--------|
| 1 | **张量借、模型自己搭** | 详细边界见第 4 节。把精力花在真正需要理解的地方 |
| 2 | **每个概念都必须落成可测试的代码** | 知识如果不能被断言，就说明还没学明白 |
| 3 | **数学按需补，永远挂在一个例子上** | 见第 3 节 |
| 4 | **一次只引入一个新东西** | 不要同时上新概念 + 新工具 + 新 API，认知带宽不够 |
| 5 | **工程标准不打折** | 你被当成正式参与项目的实习生：类型检查 strict、覆盖率有硬门槛、CI 强制 |
| 6 | **节奏按 Issue 推进，不按日历推进** | 时间是碎片的不影响——只要每个 Issue 都能独立走完闭环 |
| 7 | **导师不写实现** | 导师只开 Issue、写测试、给 review 意见。他一旦动手写实现，你就变成旁观者了 |

关于原则 5 和"降级预案"的关系：Stage 里写的降级预案**只降知识广度和实现复杂度，不降工程标准**。
比如"暂不支持嵌套 Sequential"是可以的；但"这个 Stage 测试先不写"不行。

---

## 6. 学习节奏：按 Issue 推进，不按日历推进

时间不固定没关系。真正需要固定的不是日期，而是**每个 Issue 都走完同一个闭环**。

### 一个 Issue 的完整生命周期

| 阶段 | 谁 | 做什么 |
|------|-----|--------|
| 开工前 | 导师 | 开 Issue：目标 / 背景 / 验收标准 / 提示 / **明确不要做什么** |
| 开工前 | 你 | 用自己的话复述一遍验收标准；有疑问当天问，不要猜 |
| 实现中 | 你 | 写代码 + 写测试。卡住就在 Issue 评论区留言，不要憋着 |
| 提交前 | 你 | 本地 `make check` 全绿，`make msg` 校验 commit message |
| 提 PR | 你 | 填 PR 模板，**「我怎么验证的」必须贴真实命令和输出** |
| Review | 导师 | 按四级标签给意见（第 7.4 节） |
| 修改 | 你 | 每条评论都要回应：改，或说明为什么不改 |
| 合并 | 导师 | CI 全绿 + 所有 `[blocking]` 解决，squash 后合并 |
| 收尾 | 你 | 写 `docs/journal/YYYY-MM-DD.md`，三行 |

### 三条硬规则

- **一个 Issue 一个 PR**，不夹带无关改动；**diff 超过 400 行就要拆**（这条对导师同样生效）
- **一次学习回合（2–4 小时）应该能推进一个 Issue 到"可提交"状态。** 撑不下就说明它该拆小——来找导师拆
- **宁可一次做完一个，也不要同时开三个做一半。** 分支保持短命，合并即删

收尾的 journal 只要三行：

```markdown
- 推进了什么：
- 卡在哪：
- 下次想搞懂什么：
```

别小看它。间隔几天再回来，这三行能省掉半小时的"我上次做到哪了"。

---

## 7. 协作系统怎么运转

### 7.1 Issue 是"任务说明书"

标题格式：`stage<N>: <动词开头的英文小写描述>`，纯文档/修复用 `docs:` / `fix:`。

正文用模板（`.github/ISSUE_TEMPLATE/task.yml`）强制填五段：

| 字段 | 说明 |
|------|------|
| 目标 | 一句话，可验收 |
| 背景 | 为什么现在做这个，和上一阶段什么关系 |
| 验收标准 (DoD) | 3–5 条，最好是**可执行**的（能跑哪个命令、看哪个数字） |
| 提示 | 一两条思路，不给答案 |
| 不要做 | 明确划出范围外的东西，防止跑偏和过度设计 |

**附录 A 里有每个 Stage 的完整 Issue 清单，开 Issue 时把标题原样抄进去就行。**

### 7.2 PR 是"交付说明"

模板（`.github/pull_request_template.md`）固定四段 + 一个自检清单：

- `Closes #<issue>`（CODEOWNERS 会自动把导师加为 reviewer）
- 我做了什么（对应 DoD 逐条勾）
- **我怎么验证的**（这条最重要：跑了什么命令、看到什么结果）
- 我拿不准 / 遗留的问题
- 自检清单：跑过 `make check` / 加了测试 / 没夹带无关改动 / commit message 合规

### 7.3 CI 是不会疲倦的审稿人

每次 push 和 PR 上自动跑，六项全过才算绿：

1. `ruff format --check` — 格式
2. `ruff check` — lint
3. `mypy` — 类型（strict）
4. `pytest --cov --cov-fail-under=90` — 测试 + 覆盖率门槛
5. `scripts/check_commit_msg.py` — commit message 格式
6. Python 3.10 / 3.11 / 3.12 三档矩阵

本地对应的命令就是 `make check`（加 `make test-cov` 看覆盖率）。
**不要"先推上去让 CI 看看"**——本地能跑的检查，本地跑完再提。

### 7.4 Review 意见分四级（重要）

导师在每条评论前加标签，你必须学会区别对待：

| 标签 | 含义 | 你该做什么 |
|------|------|------------|
| `[blocking]` | 必须改，否则不合并 | 改，然后回复改在哪 |
| `[suggestion]` | 我认为更好，但你可以不同意 | 改，或**说明理由后不改** |
| `[nitpick]` | 吹毛求疵，纯风格 | 随手改掉，不必讨论 |
| `[question]` | 我在问你，不是让你改 | 用文字回答 |

两条规则：

- **每条评论都要有回应**——要么改，要么给出不改的理由。沉默视为未处理。
- **所有技术讨论都留在 Issue / PR 评论区**，不要用私聊。这样答案可以被搜索、被回看，三个月后你还找得到。

---

## 8. Stage 分解

每个 Stage 的结构：**知识输入 → 工程技能 → 交付物 → 验收 (DoD) → 降级预案**。

---

### Stage 0 · 工程地基

| 项 | 内容 |
|----|------|
| 知识 | 无深度学习内容。git 三区模型、分支、`uv sync`、CI 是什么、Issue/PR 长什么样 |
| 工程 | 装好环境，跑通 `make check`，走完一次完整的 PR 流程 |
| 交付 | 第一个 PR：**删掉 `README.md` 里那行占位标记**（一行 HTML 注释，全文搜 `TODO(first-pr)` 就能找到） |
| DoD | 那行标记没了、**其他内容一行没动**；CI 全绿；PR 模板四段都填了；commit message 通过校验；review 通过并合并 |
| 为什么这样设计 | 第一个任务故意做到最简：**内容上没有任何判断，考点全在流程** |

> **这一阶段不考内容。** 你不需要读懂任何一行深度学习代码，甚至不需要写代码——
> 只需要把「建分支 → 改文件 → commit → 推上去 → 提 PR → 等 CI → 回应 review → 合并」这条流程走完一遍。
> 之所以选"删一行"这种任务：**内容零风险，你才能专心对付工具。**
>
> 第一次 CI 变红、第一次收到 `[blocking]` 意见，都该发生在这一阶段——那时候出错的成本最低。

### Stage 1 · 张量与梯度

| 项 | 内容 |
|----|------|
| D2L | 2.1 数据操作、2.2 数据预处理、2.3 线性代数、2.4 微积分、2.5 自动微分 |
| 知识 | 张量语义（shape / dtype / device）；广播；矩阵乘；导数与链式法则；`backward()` 到底做了什么 |
| 工程 | **手写数值梯度校验器**：学会用"完全独立的参照物"验证自己的代码。这是全课程最重要的工程习惯 |
| 实现 | `pdl.utils.gradcheck`（`numerical_grad` + `check_gradient`），支持任意标量表达式、广播和归约 |
| 交付 | `pdl/utils/gradcheck.py` + `examples/gradient_basics.py` + `docs/notes/gradient.md`（你自己的笔记） |
| DoD | 能手算 `x**2`、`(2x+1)**2` 这类标量导数并用 torch 对上；校验器自己先过"已知导数对"的测试；校验器能验任意标量表达式 |
| 降级预案 | 校验器"还原被扰动元素"总写错 → 先写一个只支持单元素张量的版本，跑通再泛化 |
| 可选加练 | 标量版 autograd 引擎（micrograd 路线，约 150 行）——**只求看懂机制，不要求支撑整个库** |

> 这一阶段**刻意不碰矩阵级的反向推导**。手推一整层的反向放在 Stage 5——
> 那时你手边真的有 `Linear` 可以推，而不是对着一个还不存在的层做矩阵体操。
> 对应 handbook 第 02、03 章。

### Stage 2 · 第一个能学习的模型

| 项 | 内容 |
|----|------|
| D2L | 3.1 线性回归、3.2 从零实现、3.3 简洁实现（选做） |
| 知识 | 模型 = 假设空间；损失 = 打分标准；优化 = 搜索过程。三者要分开看 |
| 工程 | 配置与代码分离、随机种子、可复现性、脚本化入口（`examples/`） |
| 实现 | `Dataset` / `DataLoader`、`MSELoss`、`Optimizer` 基类与 `SGD`、`Trainer` 训练循环 |
| 交付 | 合成数据上的线性回归端到端脚本 + 测试 |
| DoD | `uv run python examples/linear_regression.py --seed 0` 两次结果完全一致；loss 下降到噪声下界附近；学到的 `w` 逼近解析解 |
| 降级预案 | `DataLoader` 的迭代器协议太绕 → 先写成纯函数 `iterate_batches(X, y, bs)`，Stage 3 再重构成类 |

### Stage 3 · 分类与 Softmax

| 项 | 内容 |
|----|------|
| D2L | 3.4 softmax 回归、3.5 图像分类数据集、3.6 从零实现、3.7 简洁实现（选做） |
| 知识 | 为什么分类不用 MSE；logits → softmax → 交叉熵；交叉熵为什么等价于最大似然 |
| 工程 | **数值稳定性测试**：输入 1e4 量级的 logits 时不能出现 `inf`/`nan` |
| 实现 | `CrossEntropyLoss`（内部做 log-sum-exp）、`accuracy`、`confusion_matrix`、Fashion-MNIST 读取与缓存 |
| 交付 | 从零实现的 softmax 回归 + 训练脚本 + 数值稳定性测试 |
| DoD | Fashion-MNIST 测试集准确率 > 0.80；极端 logits 下 loss 有限且梯度有限；与 `torch.nn.functional.cross_entropy` 对得上 |
| 降级预案 | 数据集接口不顺 → 用 `data/synthetic.py` 的合成分类数据，**接口保持一致**，训练脚本一行不用改 |

### Stage 4 · 泛化与模型选择

| 项 | 内容 |
|----|------|
| D2L | 4.4 模型选择/欠拟合过拟合、4.5 权重衰减、4.6 暂退法、4.9 环境和分布偏移（节选） |
| 知识 | 训练误差 ≠ 泛化误差；验证集和测试集分工；正则化在"惩罚什么" |
| 工程 | **实验记录**：每个实验一份 config + 结果表，能复现、能对比；一次只改一个变量 |
| 实现 | train/val/test 划分、早停、`weight_decay`、`Dropout` |
| 交付 | `docs/experiments/` 下第一份实验报告：过拟合曲线 + 正则化前后对比 |
| DoD | 能画出训练/验证两条 loss 曲线并解释 gap；能说明权重衰减和 Dropout 的机制差异 |
| 降级预案 | 实验太占时间 → 缩小数据规模，只要曲线形状对即可 |

### Stage 5 · 多层感知机 ★

| 项 | 内容 |
|----|------|
| D2L | 4.1 多层感知机、4.2 从零实现、4.3 简洁实现、4.7 正向/反向传播、**5.1 层和块、5.2 参数管理** |
| 知识 | 为什么需要非线性激活；万能近似定理的直觉（不是证明）；Xavier / He 初始化在解决什么 |
| 工程 | **模块化**：层与块、参数注册、`Sequential`、模型结构可打印；重构已有代码而不破坏测试 |
| **有界手写** | **手推 `Linear` 和 `ReLU` 的反向**，用 `torch.no_grad()` + 张量运算写出 `manual_backward()`，断言它与 `torch.autograd` 的结果在 1e-6 内一致 |
| 实现 | `Module` 基类（`parameters()` / `train()` / `eval()` / `__repr__`）、`Linear`、`ReLU`/`Sigmoid`/`Tanh`、`Dropout`、`Sequential`、初始化方案 |
| 交付 | 从零实现的 MLP + Fashion-MNIST 训练脚本 + 完整测试 + `examples/manual_backward.py` |
| DoD | 测试集准确率 > 0.88；每一层都有梯度校验测试；XOR 测试通过且线性模型必然失败；**手推反向与 `torch.autograd` 一致** |
| 降级预案 | 手推反向卡住 → 先只做 `Linear`（不含激活），梯度是三个矩阵乘，推导最短 |

> **这是全课程的分水岭。** 到这里你要么真的懂了反向传播，要么只是会调 `backward()`。
> review 的重点不是代码对不对，而是你能不能在 PR 里用自己的话把它讲一遍。
> 之所以放在这一阶段而不是 Stage 1：手边得先有真实的 `Linear` / `ReLU` 可推。

### Stage 6 · 深度学习计算（D2L 第 5 章）

| 项 | 内容 |
|----|------|
| D2L | 5.3 延后初始化、5.4 自定义层、5.5 读写文件、5.6 GPU |
| 知识 | 参数是什么时候、由谁创建的；为什么"保存和加载"是工程问题而不是琐事；CPU 和 GPU 的差别在哪 |
| 工程 | 参数的生命周期管理、序列化格式的稳定性、**设备无关的代码**（device-agnostic） |
| 实现 | 延后初始化、无参数自定义层、带参数自定义层、参数的保存与加载、全链路 `device` 支持 |
| 交付 | `nn/` 与 `utils/` 的扩展 + 设备无关的训练脚本 |
| DoD | 保存再加载后模型输出逐位一致；自定义层能参与 `parameters()` 和梯度校验；所有脚本接受 `device` 参数并在无 GPU 的机器上自动回落 CPU，测试必须通过 |
| 关于 GPU | 代码写成设备无关的，有 GPU 就跑 GPU、没有就跑 CPU。**CI 上没有 GPU，所以测试必须在 CPU 上通过**——这本身就是最重要的那条工程约束。D2L 5.6 讲的概念（显存、数据传输、多卡）要读 |
| 降级预案 | 序列化格式设计纠结 → 先用最直白的 `state_dict` 式的名字→张量字典，把"逐位一致"的测试写死 |

### Stage 7 · 框架对照与复盘

| 项 | 内容 |
|----|------|
| D2L | 3.3 / 3.7 / 4.3 的"简洁实现"章节集中看 |
| 知识 | 自己写的东西和工业框架差在哪：性能、工程抽象、边界处理 |
| 工程 | `uv build` 打包、版本号、CHANGELOG；写一份技术复盘 |
| 实现 | 用 `torch.nn` + `torch.optim` 把同一个 MLP 重写一遍（代码放 `examples/`，不放 `src/pdl/`） |
| 交付 | PyTorch 高层实现 + `docs/retrospective.md` |
| DoD | 两版准确率差距 < 1%；复盘能指出至少 3 个"我的实现 vs PyTorch"的具体差异 |
| 出口 | **在这里决定下一阶段方向**：CNN / 序列模型 / Transformer / 工程深化。见 handbook 第 11 章的方向表 |

---

## 9. 每个 Stage 结束的自查

不用交作业，口头过一遍就行：

1. 用一句话说清这个 Stage 的核心概念
2. 这个 Stage 的代码里，哪一段如果不写会直接坏掉？为什么？
3. 如果让你重新设计接口，你会改哪里？
4. 动手：现场加一个小功能（5–10 行），不查之前的代码

4 题全过 → 进下一个 Stage；1–2 题卡住 → 补一个针对性小 Issue；3 题以上卡住 → 停下来重做本 Stage 的降级版。

**卡住不是失败。** 走降级版走完一遍，比抄完一遍强得多。

---

## 10. 已经定下来的教学设定

这些是已经确定的方向，不要在实现时反复摇摆：

| 设定 | 内容 |
|------|------|
| 底层 | PyTorch 张量 + `torch.autograd`；不手写张量运算，不手写 autograd 引擎 |
| 上层 | 层、损失、优化器、数据管道、训练循环、指标，全部自己写。边界见第 4 节 |
| 手写反向 | Stage 5 有一个**有界**的练习：手推 `Linear` / `ReLU` 的反向并与 `torch.autograd` 对照。可选加练标量版 autograd 引擎 |
| 覆盖 D2L 第 5 章 | 5.1/5.2 在 Stage 5；5.3–5.6 在 Stage 6，GPU 用设备无关代码实现并在 CPU 上测试 |
| 工程标准 | mypy strict、覆盖率 ≥ 90%、CI 强制、commit message 校验。不打折 |
| 数据集 | 真实的 Fashion-MNIST 等；合成数据只作为 CI 的快速离线路径 |
| 数学 | 按需补，挂在具体例子上。见第 3 节 |
| 节奏 | 按 Issue 推进，不设日历 |
| 依赖 | torch 走 **CPU 索引**（`download.pytorch.org/whl/cpu`），避免 CI 下载 2 GB 的 CUDA 运行时 |

---

## 11. 风险与对策

| 风险 | 症状 | 对策 |
|------|------|------|
| 滑成调包 | 顺手用了 `nn.Linear` 或 `F.cross_entropy` | 第 4 节的边界写死在 roadmap 和 `AGENTS.md` 里；review 会盯这一条 |
| 只懂调 API | 代码对但讲不出为什么 | Stage 5 的手推反向是硬性 DoD；review 要求你口述 |
| 被工程细节压垮 | 花三小时调 ruff，没写一行模型代码 | **工具链报错直接找导师**，环境配置由导师修；你的精力花在模型代码上 |
| 数学卡住 | 看到 ∂ 就跳读 | 按第 3 节随用随补；允许先写代码再看公式；公式只保留能落到代码的那些 |
| 讨论散落丢失 | 同一个问题问三遍 | 所有讨论留在 Issue / PR 评论区，不用私聊 |
| PR 越滚越大 | 一个 PR 改了 30 个文件 | 400 行 diff 硬规则 + DoD 逐条勾选；超了当场拆 |
| 隔久了接不上 | 忘了上次做到哪 | 每个 Issue 收尾必须写 journal；分支保持短命；Issue 描述里写清"上次到哪" |
| 导师忍不住给实现 | review 里出现能直接粘贴的代码 | review 只写"哪里不对、往哪想"，要具象化就写伪代码 |
| 走到一半没方向 | Stage 7 结束就停了 | Stage 7 出口强制做方向决策；handbook 第 11 章给了备选路线 |

---

## 12. Stage 总览

| Stage | 主题 | 对应 D2L | Issue 数 | 关键交付 |
|-------|------|----------|---------|----------|
| 0 | 工程地基 | — | 1 | 第一个 PR 合并 |
| 1 | 张量与梯度 ★ | 2.1–2.5 | 6 (+1 可选) | 梯度校验器 |
| 2 | 线性回归 | 3.1–3.3 | 6 | 端到端训练脚本 |
| 3 | 分类与 Softmax | 3.4–3.7 | 5 | Fashion-MNIST > 0.80 |
| 4 | 泛化与正则 | 4.4–4.6, 4.9 | 6 | 第一份实验报告 |
| 5 | 多层感知机 ★ | 4.1–4.3, 4.7, 5.1–5.2 | 11 | Fashion-MNIST > 0.88 + 手推反向 |
| 6 | 深度学习计算 | 5.3–5.6 | 5 | 设备无关 + 参数存取 |
| 7 | 框架对照与复盘 | 3.3/3.7/4.3 | 3 | PyTorch 对照 + 方向决策 |

**合计 43 个 Issue（另有 2 个可选加练）。**

---

## 附录 A · Issue 清单

开 Issue 时把下面的标题**原样复制**到 title 里，前面的 `stageN:` 不要改——
它让我们一眼看出这个 Issue 属于哪个阶段。

> **注意**：这里写的是 **Issue 标题**，不是 commit message。
> commit message 要把 `stageN:` 换成对应的 type，并在末尾加上 `(#<issue-id>)`。
> 转换方法和对照表见 `CONTRIBUTING.md` 第 3.3 节，这里给个例子：
>
> ```
> Issue 标题：  stage2: implement mse loss with the one half factor
> Commit 标题： feat(loss): implement mse loss with the one half factor (#9)
> ```

### Stage 0 · 工程地基

1. `docs: remove the placeholder marker line from the readme`

### Stage 1 · 张量与梯度

2. `stage1: add a numerical gradient checker to pdl utils`
3. `stage1: test the gradient checker against known derivative pairs`
4. `stage1: add gradient checks for broadcasting and reduction operators`
5. `stage1: run the gradient comparison in float64 and document why`
6. `stage1: add a scalar gradient example script that matches torch`
7. `docs: write a note explaining what a derivative is in your own words`

### Stage 2 · 线性回归

8. `stage2: implement dataset and dataloader with deterministic shuffling`
9. `stage2: implement mse loss with the one half factor`
10. `stage2: implement the optimizer base class and sgd`
11. `stage2: implement the trainer loop and per epoch records`
12. `stage2: add an end to end linear regression example with a fixed seed`
13. `stage2: assert convergence to the closed form solution in tests`

### Stage 3 · 分类与 Softmax

14. `stage3: implement cross entropy loss with log sum exp`
15. `stage3: implement accuracy and confusion matrix metrics`
16. `stage3: implement the fashion mnist dataset with local caching`
17. `stage3: add numerical stability tests for extreme logits`
18. `stage3: add a softmax regression training example`

### Stage 4 · 泛化与模型选择

19. `stage4: implement a deterministic train validation split`
20. `stage4: implement early stopping with best weight restore`
21. `stage4: add weight decay with separate weight and bias handling`
22. `stage4: implement dropout with train and eval modes`
23. `stage4: record each run as a reproducible config and result row`
24. `docs: write the overfitting and regularisation experiment report`

### Stage 5 · 多层感知机

25. `stage5: implement the module base class and parameter registration`
26. `stage5: implement the linear layer with configurable bias`
27. `stage5: implement relu sigmoid and tanh activation modules`
28. `stage5: implement the sequential container`
29. `stage5: hand derive the backward pass of linear and relu`
30. `stage5: implement a manual backward and match torch autograd`
31. `stage5: implement xavier and he initialisation`
32. `stage5: add a gradient check test for every layer`
33. `stage5: add an xor test that a linear model must fail`
34. `stage5: implement parameter counting and structure printing`
35. `stage5: add the mlp training example on fashion mnist`

### Stage 6 · 深度学习计算

36. `stage6: implement deferred initialisation for layers`
37. `stage6: implement a custom layer without parameters`
38. `stage6: implement a custom layer with parameters`
39. `stage6: save and load model parameters in a stable format`
40. `stage6: make every model and script device agnostic`

### Stage 7 · 框架对照与复盘

41. `stage7: reimplement the mlp with torch nn and torch optim`
42. `build: package the library with a version number and changelog`
43. `docs: write the technical retrospective and choose the next direction`

### 可选加练（不在 43 个之内）

- `stage1: implement a scalar autograd engine as an optional reading exercise`
- `stage2: implement adam and compare it against sgd on the same task`

---

## 附录 B · 这一版包含什么

这不是一份写完就冻结的文档。学到一半发现某个 Stage 太难、某个 Issue 该拆、某章该重写，都会直接改正文——
**你看到的永远是最新版本，正文就是唯一事实来源。**

所以：不要留旧版对照，也不用背某个数字（比如"一共多少个 Issue"）。判断"现在该做什么"只看两处——
**当前 Stage 那一节**，和**附录 A 里那个还没关掉的 Issue**。

| 现在这一版 | |
|------------|--|
| Stage | 8 个（0–7），总览见第 12 节 |
| Issue | 43 个 + 2 个可选加练，清单见附录 A |
| 手册 | 12 章（00–11），每章固定 8 节，见 [`docs/handbook/README.md`](handbook/README.md) |
| 教学设定 | 已经定下来的技术方向，见第 10 节 |
| 判定标准 | 每个 Stage 结束的自查见第 9 节；每个 Issue 的判定看它自己的验收标准 (DoD) |
