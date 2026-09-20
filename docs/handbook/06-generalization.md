# 05 · 过拟合、欠拟合与模型选择

> **一句话**：在训练集上考高分不算本事，模型得在**没见过**的数据上也行；这章讲怎么判断这件事、怎么让它变好。
>
> **对应 D2L**：4.4 模型选择、欠拟合与过拟合（顺带看一眼 4.9 环境和分布偏移）｜ **对应 Stage**：4 ｜ **对应代码**：`src/pdl/trainer/`、`src/pdl/utils/`

## 为什么需要它

Stage 2、3 的训练脚本最后一定会打印一个很漂亮的数字：`train loss = 0.03`。它只说明一件事：**模型把答案背下来了**。而我们真正要的是它在没见过的数据上也答得对——部署出去以后，模型遇到的每一张图都是"没见过"的。于是问题变成两个：怎么知道模型是"真会"还是"背下来"？知道以后，怎么让它往"真会"的方向走？

这章没有新算子要实现。但从这章起你的身份变了：从"写算子的人"变成"做实验的人"。**重点在工程纪律，不在数学。**

## 直觉

把训练集想成一本习题册，验证集当成模拟考，测试集当成期末考。

- **背答案**：把习题册答案全背下来，册子上全对，换一道新题就废。这是**过拟合**。
- **没学会**：连习题册都做不对，公式都没记住。这是**欠拟合**。
- **学会了**：习题册做得不错，模拟考也差不多，两边接近。这是我们想要的。

判断"会不会"的唯一办法，是拿**没做过的题**考它。习题册上的分数没有信息量——它就是用来学习的，分数高是必然的。

调参就像按模拟考成绩改学习方法：改了十次，模拟考确实涨了，但它已经不能代表期末考——你已经把模拟考"用掉"了。所以测试集只能考一次。

## 最小数学

> 看不懂就跳过，先去做「落到代码」，回来再读。

**1. 训练误差与泛化误差**

$$R_{\text{emp}} = \frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i), y_i) \qquad R = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\ell(f(x),y)\right]$$

$R_{\text{emp}}$ 是你打印的那个数，$R$ 才是你要的。真实分布 $\mathcal{D}$ 永远拿不到，只能抽样本估；两者的差叫 **generalization gap**。

**2. 带 L2 正则的目标**

$$J(w) = \frac{1}{n}\sum_i \ell(f_w(x_i), y_i) + \frac{\lambda}{2}\lVert w\rVert^2
\qquad\Longrightarrow\qquad
w \leftarrow (1-\eta\lambda)w - \eta\nabla\ell$$

右边第一项说明：**梯度为 0 时 $w$ 也会每个 step 乘一个略小于 1 的系数**——这就是 weight decay 这个名字的来历，也直接给你一个能写的测试。它的直觉是：权重小，输入动一点输出只动一点，函数更"平"，没法贴着每个噪声点扭；而随机噪声恰恰是不平滑的，想拟合它就必须用大权重。所以压住 $\lVert w\rVert$，等于禁止模型去背噪声。

**3. K 折交叉验证**

把数据切 $K$ 份，每次留一份当验证、其余训练，跑 $K$ 次：$\hat R = \frac{1}{K}\sum_k R_k$，同时报 $\sigma = \text{std}(R_1..R_K)$。**均值和标准差一起看**，才知道这个估计有多可信。

## 落到代码

**数据划分**：

```python
# src/pdl/utils/split.py
def train_val_split(X, y, val_ratio: float = 0.2, seed: int = 0):
    """切成 (train, val)，返回 ((X_tr, y_tr), (X_val, y_val))。X 形状 (N, ...)，y 形状 (N,)。
    职责：1) 两组下标不重叠，并集恰好是全集；2) 同 (val_ratio, seed) 切出的结果完全一致；
          3) 默认先打乱再切。
    不做：不碰测试集（它只在最后用一次）；不做分层采样，类别不均衡时记 TODO。
    """
```

**早停**：

```python
# src/pdl/trainer/early_stop.py
class EarlyStopping:
    """只看验证损失，判断该不该停；不碰模型参数。"""
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None: ...
    def step(self, val_loss: float) -> bool:
        """喂一个 epoch 的验证损失；返回 True = 这次更好了（调用方据此存快照）。
        patience：连续 patience 个 epoch 都没改善，就该停了。"""
    @property
    def should_stop(self) -> bool: ...      # 该不该停
    @property
    def best_epoch(self) -> int: ...        # 目前最好的 epoch
    @property
    def best_val_loss(self) -> float: ...   # 以及那一个损失值
```

职责切分很重要：`EarlyStopping` 只做判断，"存/恢复最佳权重"归 `Trainer`。这样早停能脱离模型单独测——喂一串人为构造的数字就行。

**训练循环**：

```python
# src/pdl/trainer/trainer.py（在第 04 章那份上长大，别重写一个）
class Trainer:
    def __init__(self, model, loss_fn, optimizer, weight_decay: float = 0.0,
                 decay_bias: bool = False,
                 early_stopping: EarlyStopping | None = None) -> None: ...
    def fit(self, train_loader, epochs: int, val_loader=None) -> "History":
        """签名和 03 章一致；History 就是 list[EpochRecord]（见下），
        每轮在 03 章的基础上多记 val_acc / lr / seconds，并喂早停。"""
    def restore_best(self) -> None:
        """把模型参数回滚到验证损失最好那个 epoch 的快照。"""
```

`weight_decay` 有两个细节：它等价于在梯度上加 $\lambda w$，只在**更新参数时**生效，**别去改 loss**（否则打印的 loss 和正则强度混在一起，后面没法对比）；第 04 章整份参数一起衰减就够了，到这里要**区分权重和偏置**——默认只衰减 `ndim >= 2` 的参数，要关掉这条规则就 `decay_bias=True`。还有：第 04 章的 `SGD(weight_decay=...)` 和这里的 `Trainer(weight_decay=...)` 是同一个旋钮的两个入口，**只开一个**，两个都开就衰减两次；走 `Trainer` 更省事，因为只有它能区分权重和偏置。

**实验记录**：

```python
# src/pdl/utils/experiment.py
@dataclass
class EpochRecord:
    epoch: int; train_loss: float; val_loss: float; val_acc: float; lr: float; seconds: float

@dataclass
class RunRecord:
    name: str          # 配置名，如 "wd=1e-3"
    config: dict       # 完整超参 + seed + 划分参数，能一键复现
    epochs: list[EpochRecord]
    note: str = ""     # 一句话：你从这个配置里看到了什么
    def to_markdown_row(self) -> str: ...   # 配置名 / train loss / val loss / val acc / 备注
```

底线是**把 `config` 原样存下来**。少了这一步，一周后你看着表格上的数字，不知道它是怎么来的。

## 工程视角

**1. 实验记录 = 一份 config + 一张结果表**

```
docs/experiments/overfit-baseline/
├── configs/base.toml  wd_1e-4.toml  wd_1e-3.toml
└── README.md          # 结果表 + 你的结论
```

| 配置名 | train loss | val loss | val acc | 备注 |
|--------|-----------|----------|---------|------|
| base | 0.021 | 0.58 | 0.83 | 明显过拟合，val 第 12 epoch 后开始涨 |
| wd=1e-4 | 0.035 | 0.41 | 0.87 | gap 变小，val 最佳在 epoch 21 |
| wd=1e-3 | 0.052 | 0.44 | 0.865 | 略欠拟合，train loss 降不下去 |

可复现的最低要求：**表里每个数字都能用 `configs/` 里那份 config 重新跑出来**。所以 config 必须写全 seed、划分比例、epoch 数、batch size、学习率——一个都不能靠"默认值"糊过去。

**2. 一次只改一个变量**

上面三份 config 只有 `weight_decay` 不同。如果 `wd=1e-3` 那份顺手把学习率也调了，你最后无法回答"到底是谁起的作用"。写个小测试守住：`assert_single_diff(cfg_a, cfg_b) -> str`，断言两份 config 只有一个 key 不同，多改了就报错。

**3. 跑 3 个种子再下结论**

同一份 config 换个 seed，val acc 差 1~2 个点非常正常。所以只跑一个 seed 拿到更大的数字，**不能**说这个配置更好。每个配置至少 3 个 seed，报**均值 ± 标准差**；差异小于噪声就老实写"没看出差别"。种子要控制的地方不止一处：参数初始化、数据打乱，Stage 4 还有 dropout。

**4. 测试怎么写**

| 要测的 | 断言什么 |
|--------|---------|
| 划分不重叠 | `set(train) & set(val) == set()`，两边内部也没有重复下标 |
| 划分不漏数据 | `len(train) + len(val) == N` 且 `set(train) \| set(val) == set(range(N))` |
| 划分可复现 | 同 seed 切两次，下标数组 `torch.equal`；跨进程也一致 |
| 早停触发时机 | 喂 `[0.5, 0.4, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]`，断言恰在 `best_epoch + patience` 处 `should_stop` 变 True |
| 能恢复最佳权重 | 测试里每个 epoch 自己存一份参数副本，找出 best epoch；`restore_best()` 后逐参数与它 `torch.equal` |
| 权重衰减生效 | 梯度为 0 时走一步，参数约等于 `(1 - lr * wd) * w_old`；`wd=0` 时逐位不变；`decay_bias=False` 时偏置**逐位相等**（不是 `torch.allclose`，要精确比较） |

**5. 早停为什么是"免费"的正则化**
你本来就要算验证损失，早停只是加了几行比较；它不改损失函数、不多算一次梯度、不引入新的超参数（只有一个 patience）。它白拿的好处是自动停在"开始记噪声"之前。代价是你必须留出验证集，而且**要恢复 best epoch 的权重**，不能拿着停下来那一刻已经过拟合的参数去交作业。

**6. 把两条曲线画出来**

症状判断不看数字，看形状。用 matplotlib 画 `RunRecord.epochs` 里的两条 loss，比如 `uv run --extra viz python examples/plot_curves.py <run.json>`。**matplotlib 不进 pdl 的运行时依赖**：它只是可选的展示工具，pdl 核心只要 torch；放进 `[project.optional-dependencies]` 的 `viz` 组（或临时 `uv run --with matplotlib`）。CI 里绝对不能依赖它，否则你会在一个跟画图无关的 PR 上看到 matplotlib 装不上。

```
欠拟合：两条都高、gap 小        过拟合：val 先降后升、gap 大
loss                            loss
 |\                              |\
 | \__                           | \__
 |    \____  train/val           |    \___    train (一路降)
 |         \___                  |        \____
 |     __.--''                   |     _.--''
 +--------------- epoch          +--'----------- epoch
```

- **两条都高、gap 小** → 欠拟合：模型太简单 / 没训够 / 学习率不对
- **train 低、val 先降后升、gap 大** → 过拟合：上正则化 / 加数据 / 早停
- **两条都在降但没收敛** → 还没训完，别急着下结论

## 常见误解

| 误解 | 实际 |
|------|------|
| "验证集效果好就是模型好" | 你按验证集调了 20 次超参，那个"最好"里有相当一部分是**运气**，必须用没参与过决策的测试集复核 |
| "过拟合只是因为数据太少" | 数据少会加重过拟合，但数据不变、把容量降下来同样能治。容量、数据量、正则强度是三个旋钮 |
| "dropout 是正则化，所以一定能涨点" | 它降方差、升偏差。欠拟合的模型加 dropout 只会更差；而且 train/eval 行为不同，忘了切模式会得到莫名其妙的数字 |
| "训练 loss 低说明模型学到了" | 训练 loss 低只说明**记住了**。要判断"学到了"，看 val/test 和 gap 的大小 |
| "测试集多调几轮更稳" | 每用测试集做一次选择，分数就乐观一点。调 N 次等于在 N 个含噪估计里取最大值，最大值天然偏高 |

## 自测题

> 每题的依据都在**本章**，不需要课外知识。答不上来就回去翻，不用猜。这章的题都是"看着数字判断"。
>
> 标记：`[查]` 翻本章的表或清单 · `[算]` 手算一个具体数字 · `[跑]` 写一两行代码看结果 · `[判]` 判断一个说法对不对

1. `[查]` 本章用习题册、模拟考、期末考来比训练集、验证集、测试集。哪个是哪个？三者里哪一个**只能用一次**？
   *怎么确认：本章「直觉」那一节开头三句，以及最后那句"测试集只能考一次"。*

2. `[查]` 拿到两条 loss 曲线，本章列了三种形状。它们分别是什么症状？
   - (a) train 和 val 都高，两条贴得很近
   - (b) train 一路降，val 先降后升，gap 越来越大
   - (c) 两条都还在降，还没变平
   *怎么确认：本章「工程视角」第 6 节那张示意图下面的三条。*

3. `[算]` `patience=5`，第 1 到第 9 轮的验证损失是 `[0.5, 0.4, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36]`。
   - (a) 目前最好的 epoch 是第几轮？
   - (b) 从那一轮往后，连续几轮没改善就该停？
   - (c) `should_stop` 会在第几轮变成 True？
   *怎么确认：本章「工程视角」第 4 节测试表里"早停触发时机"那一行，它把触发点写成了 `best_epoch + patience`。*

4. `[算]` 同一份配置，A 跑三个 seed 得 `0.861 / 0.872 / 0.858`，B 得 `0.869 / 0.871 / 0.874`。
   - (a) 两个配置的均值各是多少？
   - (b) 两个配置的极差（最大值 − 最小值）各是多少？
   - (c) 本章要求报什么？按这个要求，你写"B 更好"还是"没看出差别"？
   *怎么确认：本章「工程视角」第 3 节，注意"差异小于噪声就老实写"那句。*

5. `[跑]` 本章「工程视角」第 1 节那张结果表有三行。写 1 行代码把每行的 `val loss − train loss` 算出来：
   ```python
   rows = {"base": (0.021, 0.58), "wd=1e-4": (0.035, 0.41), "wd=1e-3": (0.052, 0.44)}
   print({k: round(v - t, 3) for k, (t, v) in rows.items()})
   ```
   哪一行 gap 最大？对照本章的判据，这一行是什么症状？
   *怎么确认：数字来自本章「工程视角」第 1 节那张表；gap 的判法在第 6 节，别改数字。*

6. `[判]` "保险起见，我把 `SGD(weight_decay=1e-3)` 和 `Trainer(weight_decay=1e-3)` 都填上。" 本章同意吗？实际上会发生什么？
   *怎么确认：本章「落到代码」里讲 `Trainer.__init__` 的那一段，末尾专门说了"同一个旋钮的两个入口"。*

7. `[判]` 早停触发以后，直接拿那一刻的模型参数去交作业，行不行？本章要你补哪个动作？
   *怎么确认：本章「工程视角」第 5 节最后一句，以及「落到代码」里 `Trainer` 的 `restore_best()`。*

## 延伸阅读

- D2L 4.4 模型选择、欠拟合与过拟合 —— 本章知识来源；4.9 环境和分布偏移讲"训练和测试分布不一致"会怎样
- [Deep Learning Book, Ch.7 Regularization](https://www.deeplearningbook.org/contents/regularization.html) —— 想系统理解正则化时看，7.1 和 7.2 就够
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) —— K 折的实际用法；注意它是 API 视角，机制还得自己想清楚
- 下一章：[07 Softmax 回归](07-softmax.md) —— 把模型从回归搬到分类
