# 协作规范

这份文档描述的是**流程**，不是知识。它保证"代码能合并"这件事有客观标准，
而不是靠谁记得更牢或者谁脾气更好。

导师和学生都遵守同一套规则。**工程标准按正式项目执行，没有"教学简化版"**——
你会被当成一个真正参与项目的实习生：类型检查是 strict 的，覆盖率有硬门槛，
commit message 由 CI 强制校验。

---

## 1. 环境

```bash
uv sync --all-groups            # 安装依赖（含 dev 依赖）
make check                      # 格式 + lint + 类型 + 测试，提 PR 前必须全绿
make fmt                        # 自动格式化并修复可自动修复的问题
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

最后那条会把检查挂到 `git commit` 上：提交前自动跑格式和 lint，并校验 commit message。

不要用 `pip install` 往这个环境里装东西。要加依赖就改 `pyproject.toml` 里的 `dependencies`，
然后跑 `uv sync`，并把这个改动写进 PR 说明。

**工具链报错不用自己硬扛**：环境、配置、CI 的问题直接找导师，你的时间应该花在模型代码上。
但"代码写得对不对"是你自己的事。

---

## 2. Issue

- 标题格式：`stage<N>: <动词开头的英文小写描述>`；纯文档/修复用 `docs:` / `fix:`
- `docs/roadmap.md` 附录 A 里写了每个 Stage 的完整 Issue 清单，**开 Issue 时把标题原样抄进去**
- 用仓库的「任务」模板，五段必填
- **验收标准 (DoD) 必须可执行**：写"能跑哪条命令、看到什么数字"，不要写"实现得比较优雅"
- **「不要做」不是可选项**：明确排除掉的东西，是防止 PR 膨胀的第一道闸

一个 Issue 只对应一个**可独立交付**的变更。做不完就拆 Issue，不要把半个功能塞进一个 PR。

---

## 3. 分支与 Commit

### 3.1 分支命名

```
main                         受保护，只能通过 PR 合入
feat/12-tensor-add           功能分支：<type>/<issue-id>-<短描述>
docs/20-handbook-ch02
fix/31-broadcast-negative-dim
```

### 3.2 Commit message 格式

```
<type>(<scope>): <英文小写祈使句 summary> (#<issue-id>)
```

| 部分 | 规则 |
|------|------|
| `type` | 只能取：`build` `chore` `ci` `docs` `feat` `fix` `perf` `refactor` `style` `test` |
| `(scope)` | 可省略。建议用子包名：`tensor` `autograd` `data` `loss` `optim` `metrics` `nn` `trainer` `utils` |
| `summary` | 英文、小写字母开头、祈使句、**结尾不加句号**、至少 20 个字符 |
| `(#id)` | 必须放在末尾，`id` 是这个改动对应的 Issue 编号 |
| 整行 | 不超过 100 个字符 |

```
feat(tensor): support broadcasting in elementwise add (#12)
docs: correct a factual error in handbook chapter 01 (#3)
test(nn): add a gradient check test for every layer (#31)
```

禁止：`wip`、`update`、`fix bug`、`fix ci again`、`修改`。

**PR 内允许多个有意义的小 commit**，每个 commit 都应该能通过 `make check`。
合并前由作者 squash 成一个最终 commit。

### 3.3 怎么把 Issue 标题变成 commit message

Issue 标题和 commit 标题**不是同一个字符串**：`stageN:` 不是合法的 commit type，
而且 commit 末尾要多一个 Issue 编号。转换方法只有两步：

```
Issue 标题：  stage1: implement elementwise arithmetic with broadcasting
                    │
                    │  ① 把 `stageN:` 换成对应 type：功能实现 → feat(<子包>)
                    │  ② summary 原样照抄，末尾加 (#<issue-id>)
                    ▼
Commit 标题： feat(tensor): implement elementwise arithmetic with broadcasting (#12)
```

type 的对应关系：

| Issue 前缀 | 这个改动在做什么 | commit type |
|------------|------------------|-------------|
| `stageN:` | 实现新功能 | `feat(<子包>)` |
| `stageN:` | 只补测试 | `test(<子包>)` |
| `fix:` | 修一个缺陷 | `fix(<子包>)` |
| `docs:` | 只改文档 | `docs` |
| `build:` / `ci:` | 构建系统、依赖、CI | `build` / `ci` |

### 3.4 本地自查

不要等 CI 告诉你格式错了。提交前自己跑一遍：

```bash
# 校验一条写好的 message
make msg M="feat(tensor): support broadcasting in elementwise add (#12)"

# 校验一个文件里的 message（和 pre-commit 钩子等价）
python3 scripts/check_commit_msg.py --file .git/COMMIT_EDITMSG

# 校验当前分支上的所有 commit（和 CI 等价）
git log --format=%s main..HEAD | python3 scripts/check_commit_msg.py --stdin
```

---

## 4. Pull Request

- 一个 PR 只解决一个 Issue，描述里写 `Closes #<issue>`
- 用 PR 模板，四段都要填。**「我怎么验证的」必须贴真实命令和输出**
- **diff 超过 400 行就要拆**（这条对导师同样生效）
- 不允许"先推上去让 CI 看看"——本地 `make check` 通过再提
- PR 标题与最终 commit message 保持一致

---

## 5. Review

### 5.1 导师给意见时必须带分级标签

| 标签 | 含义 | 学生该做什么 |
|------|------|--------------|
| `[blocking]` | 必须改，否则不合并 | 改，然后在该条评论下回复改在了哪里 |
| `[suggestion]` | 我认为更好，但你可以不同意 | 改，或**说明理由后不改** |
| `[nitpick]` | 吹毛求疵，纯风格 | 随手改掉，不必展开讨论 |
| `[question]` | 我在问你，不是让你改 | 用文字回答 |

导师只写"哪里不对、往哪个方向想"，**不直接给实现**。需要具象化时写伪代码，不写能直接粘贴的代码。

### 5.2 学生回应规则

- **每条评论都要有回应**：要么改，要么给出不改的理由。沉默 = 未处理
- 不同意 `[suggestion]` 是可以的，但必须说清理由；说不清就改
- 改完之后不需要逐条 @ 导师，但要在 PR 里留一条汇总："已处理 1/3/5，2 保持原样因为……"
- **所有技术讨论都留在 Issue / PR 评论区**，不要用私聊。这样答案可以被搜索、被回看，三个月后你还找得到

### 5.3 什么时候可以合并

- 所有 `[blocking]` 已解决
- CI 全绿（格式 / lint / 类型 / 测试 / 覆盖率 / commit message）
- 覆盖率不低于 90%
- 作者已 squash 成一个 commit

合并由导师执行。

---

## 6. 测试要求

新增或修改行为**必须**带测试。测试至少覆盖三类输入：

1. **正常输入**——典型形状/取值
2. **边界输入**——空、单元素、最大/最小维度、极端数值
3. **错误输入**——该抛异常的地方要断言异常类型和消息

本仓库特有的要求：

- 手写算子必须**和 numpy 对照**（`np.allclose`），numpy 是标准答案
- 涉及梯度的代码必须做**数值梯度校验**（有限差分）
- 涉及概率/指数的代码必须做**数值稳定性测试**（大输入不能出 `inf`/`nan`）
- 随机相关测试必须**固定种子**

---

## 7. 学习节奏与 journal

时间不固定没关系：**节奏按 Issue 推进，不按日历推进**。

- 一次学习回合（2–4 小时）应该能把**一个 Issue** 推进到"可提交"状态
- 撑不下一个回合的 Issue 就是太大了——去找导师拆小
- 不在仓库里留长期挂着的分支：一个 Issue 一个分支，合并即删
- 宁可一次做完一个 Issue，也不要同时开三个做一半

每次收尾写一份 `docs/journal/YYYY-MM-DD.md`，三行就够：

```markdown
- 推进了什么：
- 卡在哪：
- 下次想搞懂什么：
```

---

## 8. 文件大小

单个文件不得超过 1 MB。超过就要先确认如何处理（数据集、模型权重一律不要提交，
用脚本下载或生成，并把生成脚本一起提交）。
