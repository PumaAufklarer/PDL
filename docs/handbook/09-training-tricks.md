# 08 · 训练技巧：初始化、正则化、数值稳定

> **一句话**：网络结构对了不代表训得起来——**初始化决定信号能不能传下去，数值稳定决定会不会炸成 nan，正则化决定学到的是规律还是噪声**。
>
> **对应 D2L**：4.5 权重衰减、4.6 暂退法、4.8 数值稳定性和模型初始化 ｜ **对应 Stage**：4–5 ｜ **对应代码**：`src/pdl/nn/`（初始化、Dropout）、`src/pdl/optim/`（权重衰减、梯度裁剪）
>
> 这一章全是"经验"，但每一条都能写成一个测试或者一个可复现的实验。

---

## 为什么需要它

第 08 章你把 MLP 搭起来了，XOR 也能学会。然后换成 Fashion-MNIST，一跑：loss 是 `nan`。再改一版，loss 一动不动地停在 `2.303`（就是 `log(10)`——十类均匀输出时的交叉熵，说明模型什么都没学）。再改一版，训练集 0.99、测试集 0.85——过拟合了。

三种症状对应本章三块内容：**loss 变 `nan` 或爆炸** → 初始化太大、学习率太大、`exp`/`log` 上溢；**loss 不动** → 初始化全 0 或太小、激活函数饱和；**训练好、验证差** → 没正则化。先学会认症状，再学对应技巧，不然这些技巧就只是一堆玄学咒语。

---

## 直觉

**初始化**：盖楼前先把地基打平。全塌成 0（全零初始化），八根柱子会长得一模一样，等于只有一根；地基给得太高（初始化太大），每层都在放大信号，传到第五层已经爆表；给得太低，信号每过一层就减弱一点，到最后一层只剩噪声。好的初始化就是让信号**进来多少、出去差不多还是多少**。

**数值稳定**：float 像一个格子数固定的计算器。特别大的数放不下就变成 `inf`，特别小的数放不下就变成 0。而深度学习里"连乘"是常态——每层乘一次，几十层就是几十次连乘，任何一点偏差都会被指数放大。

**梯度裁剪**：给车装限速器。不改变方向，只在你踩得太猛时把速度拉回安全范围。治标不治本，但能让你从"跑十步就 nan"变成"至少能跑下去"。

**权重衰减 vs Dropout**：前者像老师说"别把答案背得一字不差，要理解大意"，通过不让权重长得太大，逼模型去学更平滑简单的规律；后者像一个随时有人请假的团队——每个人都可能临时缺席，其他人就必须学会独立干活，不能指望某个大神兜底。

---

## 最小数学

> 看不懂就跳过，先去做「落到代码」，回来再读。

### 1. 初始化：让方差活着传下去

一层 $\mathbf{h} = \phi(W\mathbf{x})$，输入各分量方差 $\sigma_x^2$，$W$ 独立、均值 0、方差 $\sigma_w^2$，则 $\mathrm{Var}(h_i) = \sum_j \mathrm{Var}(w_{ij}x_j) = n_{in}\,\sigma_w^2\,\sigma_x^2$。想让输出方差等于输入方差，就要 $\sigma_w^2 = 1/n_{in}$。两条常用公式从这里来：

- **Xavier**（配 tanh / sigmoid）：在 $n_{in}$ 和 $n_{out}$ 之间折中，$\sigma_w^2 = \dfrac{2}{n_{in}+n_{out}}$
- **He**（配 ReLU）：ReLU 把一半输入压成 0、方差砍半，所以要补回来，$\sigma_w^2 = \dfrac{2}{n_{in}}$

| 场景 | 用哪个 |
|------|--------|
| ReLU / LeakyReLU 家族 | He（`kaiming_normal_`），补回被 ReLU 砍掉的一半方差 |
| tanh / sigmoid 隐藏层 | Xavier（`xavier_uniform_`），$n_{in}$ / $n_{out}$ 两头都要照顾 |
| bias | 一律 `zeros_`：它不在连乘链上，不会引起对称性问题，非零也没有理论好处 |

### 2. 三个必须记住的浮点事实

- float64 最大约 $1.8\times10^{308}$，`exp(1000)` 直接 `inf`；可表示数的间隔随数值增大而增大，所以大数相减会**丢精度**（灾难性抵消）。本课程统一用 `torch.float32`，它的最大约 $3.4\times10^{38}$，`exp(88)` 就溢出
- `log(0) = -inf`，`log(负数) = nan`

**`log-sum-exp`**：朴素地算 $\log\sum_i e^{x_i}$，只要有一个 $x_i > 709$ 就上溢。做法是先减去最大值：$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i-m}$，其中 $m = \max_i x_i$。第 07 章的 `CrossEntropyLoss` 内部就是这么做的，这里不重复，忘了回去看。

**`log(1+e^x)`**（softplus）：$x$ 很大时 `exp(x)` 上溢，分情况算就绕开了——$x>0$ 时用 $x+\log(1+e^{-x})$，$x\le0$ 时用 $\log(1+e^{x})$，两个分支里的指数都是负的。

### 3. 梯度消失与梯度爆炸：连乘效应

反向传播是链式法则连乘。粗略地看，梯度每往回一层就乘一个因子 $s$，$L$ 层就是 $s^{L-1}$：$s<1$（sigmoid 的导数最大才 0.25）时指数衰减到 0，这是**梯度消失**，前几层学不动；$s>1$ 时指数放大，这是**梯度爆炸**，参数一步跨到天上，loss 变 `nan`。ReLU 缓解了第一种，好的初始化让 $s$ 一开始就接近 1，梯度裁剪专门应付第二种。

### 4. Dropout 为什么除以 (1-p)

训练时每个元素以概率 $p$ 置 0、以 $1-p$ 保留。不补偿的话输出期望就变成原来的 $(1-p)$ 倍，训练和推理对不上。所以训练时把保留的元素除以 $(1-p)$，于是 $\mathbb{E}[\text{dropout}(x)] = (1-p)\cdot\frac{x}{1-p} = x$。

这叫 **inverted dropout**：补偿放在训练时做，**推理时什么都不用干**；另一种做法是推理时乘 $(1-p)$，也能work，但改 $p$ 就要动推理代码，容易忘，所以主流用前者。**它为什么等价于集成大量子网络**：每次前向丢掉的组合都不同，相当于同时训练了指数多个共享权重的子网络，推理时用完整网络近似这些子网络的"平均投票"，所以更稳、更不容易过拟合。

---

## 落到代码

```python
# src/pdl/nn/init.py —— 全部原地修改（名字带下划线），返回 None
def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None:
    """按 U(-a,a) 填充，a = gain * sqrt(6/(fan_in+fan_out))。适合 tanh/sigmoid。"""
def kaiming_normal_(tensor: Tensor, mode: str = "fan_in") -> None:
    """按 N(0, std^2) 填充，std = sqrt(2/fan)。适合 ReLU 家族。"""
def zeros_(tensor: Tensor) -> None:
    """全填 0。给 bias 用；绝不要给 W 用。"""

# src/pdl/optim/clip.py
def clip_grad_norm_(parameters: list[Tensor], max_norm: float) -> float:
    """全局 L2 范数裁剪：把所有梯度看成一个长向量，超了就整体缩放。
    返回裁剪前的范数——这个返回值要拿去做监控。"""

# src/pdl/nn/dropout.py
class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None: ...  # p 是丢弃概率
    def forward(self, x: Tensor) -> Tensor: ...      # 训练：置 0 并除以 (1-p)；eval：原样返回

# src/pdl/optim/sgd.py —— 权重衰减属于优化器，不属于层
class SGD:
    def __init__(self, params, lr: float, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        """weight_decay>0 时更新量变成 g + wd*w，等价于损失里加 (wd/2)*||w||^2。"""
```

三个设计要点：① `init_` 系列必须**知道 fan_in / fan_out**，所以 `Linear.__init__` 要把形状信息传下去；② `clip_grad_norm_` 返回**裁剪前的范数**，它是最有用的调试信号之一；③ **按值裁剪**（`clip_grad_value_`）更简单，但对异常值的分布不敏感，一般只在范数裁剪不好用时才考虑。

---

## 工程视角

### 1. 怎么测 Dropout

Dropout 是随机的，所以**必须固定随机种子**，断言要写成统计意义上的：

```python
def test_dropout_expectation():
    x = torch.ones((1000, 1000))
    layer = Dropout(p=0.3); layer.train(); out = layer(x).detach()
    assert abs((out == 0).mean() - 0.3) < 0.01      # 约 30% 被置 0
    assert (out[out != 0] - 1 / 0.7).abs().max() < 1e-6   # 留下的被放大
    assert abs(out.mean() - x.detach().mean()) < 0.01   # 期望约等于输入（inverted 的核心）
```

> **注意**：本仓库**禁止使用 `.data`**——它绕过 autograd 的版本计数，梯度会静默算错；取数值和原地修改一律用 `.detach()`（需要更清楚的写法时可以用 `torch.no_grad()`）。

必测的边界：**`p=0`** 训练模式输出与输入**完全相等**（用 `torch.equal`，不是 `torch.allclose`）；**`p=1`** 输出全 0，建议在 `__init__` 里直接拒绝 `p>=1`，把问题提前到构造阶段；**`eval()`** 连续两次调用逐元素相同且等于输入——这条最重要，很多人只测了训练模式；`train()` / `eval()` 来回切换仍然正确。还有一个容易漏的坑：**Dropout 用的是哪一路随机数**，去动全局 `torch` 随机源会让测试互相干扰、训练不可复现，所以随机源从一开始就要可控（统一走 `pdl.utils` 的种子管理）。

### 2. 怎么测初始化

初始化也是随机的，要测"性质"而不是"数值"：

```python
def test_linear_init_breaks_symmetry():
    h = Linear(4, 8)(torch.randn(16, 4)).detach()
    assert h.std(dim=0).min() > 1e-6       # 8 个隐藏单元不能全一样（全 0 初始化会挂在这）

def test_linear_init_variance_is_stable():
    for seed in range(5):
        pdl.utils.seed_everything(seed)
        layer, x = Linear(1000, 1000), torch.randn(256, 1000)
        assert 0.3 < layer(x).detach().std() < 3.0 and torch.isfinite(layer(x).detach()).all()
```

**对称性检测**写完先故意把初始化改坏一次，确认测试真的能抓到——不然你测的是空气。**量级检测**给宽区间（0.3–3.0），因为种子会波动，看的是"稳定"不是"精确"。边界：`Linear(1, 5)` 的方差公式分母是 1，别写出除零；`zeros_` 单独测，bias 全 0 是正常的、网络仍然能训练。

### 3. 训练时该监控什么

不要只盯着 loss。每若干步打印四组量，绝大多数"训不动"一眼定位：**每层梯度的 L2 范数**（逐层递减到 0 → 梯度消失；逐层指数增长 → 梯度爆炸）；**参数 L2 范数**（突然暴涨 → 学习率太大；缓慢单调增长且验证 loss 上升 → 过拟合）；**每层激活值的均值 / 方差**（方差逐层塌向 0 → 信号死了；逐层放大 → 初始化太大）；**`clip_grad_norm_` 的返回值**（长期贴在阈值附近 → 一直在裁剪，说明学习率或初始化有问题）。实现上放进 `Trainer`，别散落在脚本里；但**只在调试时打开**，逐层统计会明显拖慢训练。

### 4. 把"梯度爆炸"写成一个可复现的实验

不要靠运气复现，要靠构造：

```python
def test_gradient_explosion_and_clip():
    model = Sequential(Linear(64, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 1))
    for layer in model.layers:
        if isinstance(layer, Linear): layer.W.detach().mul_(100.0)   # 故意放大，制造爆炸
    loss = compute_loss(model); loss.backward()      # 前向 + 算 loss + 反向
    norm = lambda: sum(torch.linalg.vector_norm(p.grad) for p in model.parameters())
    assert not torch.isfinite(norm()) or norm() > 1e6              # 先确认它真的爆了
    assert clip_grad_norm_(model.parameters(), max_norm=1.0) > 1.0 and norm() <= 1.0 + 1e-6
```

这个测试有三个价值：证明裁剪**真的在干活**、给出"爆炸长什么样"的基线、说明裁剪**只改幅度不改方向**（可以再断言裁剪前后梯度的余弦相似度 ≈ 1）。注意这是**全局**范数裁剪：所有参数的梯度当成一个长向量；逐层裁剪会把每层都缩放到阈值，等于悄悄改了各层之间的相对比例。

### 5. 权重衰减 vs Dropout：别混为一谈

| | 权重衰减 | Dropout |
|---|---|---|
| 作用位置 | **损失/更新公式**里加惩罚项 $\frac{\lambda}{2}\|W\|^2$ | **前向传播**里随机置 0 |
| 对抗什么 | 权重过大导致的高方差、对个别特征过度敏感 | 神经元之间的**共适应**（互相依赖抄答案） |
| 推理时 | 无影响（惩罚只在训练时算） | 无影响（inverted dropout，原样输出） |
| 随机性 | 没有 | 有 |
| 典型强度 | `1e-4` ~ `1e-2` | `p = 0.2` ~ `0.5` |

**它们是两个机制，可以同时用。** 说"加了 weight_decay 就不用 Dropout"的人，往往没分清它们对付的不是同一件事。

> **附：BatchNorm 只剧透一句。** 它对每个 batch 的激活值做标准化，让网络对初始化和学习率**宽容得多**，是深网络好训的关键组件；细节（训练/推理行为不同、running statistics、和 batch size 的关系）D2L 第 8 章讲，**本课程暂不实现**，但你会反复听到它，先知道它存在。

---

## 常见误解

| 误解 | 实际 |
|------|------|
| "Dropout 推理时也要开，保持一致性" | 推理必须关（`eval()`），否则输出是随机的；inverted dropout 已把缩放做在训练时，推理原样输出 |
| "权重衰减和 Dropout 是一回事" | 一个惩罚权重幅度、一个破坏共适应，作用位置和随机性都不同，可以同时用 |
| "初始化随便设都行，反正会训出来" | 全 0 让网络退化成单神经元，太大直接 nan；初始化是"能不能训起来"的前置条件 |
| "梯度裁剪能治所有 NaN" | 它只压得住"梯度太大"这一类。`log(0)`、除零、`exp` 上溢造成的 nan 与梯度幅度无关，裁剪救不了 |
| "Dropout 放在输入层效果最好" | 输入层丢弃等于随机删特征，通常最差；惯例是放在隐藏层、尤其是较宽的全连接层之后 |

---

## 自测题

> 每题的依据都在**本章**，不需要课外知识。答不上来就回去翻，不用猜。
>
> 标记：`[查]` 翻本章的表或清单 · `[算]` 手算一个具体数字 · `[跑]` 写一两行代码看结果 · `[判]` 判断一个说法对不对

1. `[查]` Xavier 和 He 分别配哪一类激活函数？bias 用哪个初始化函数？
   *怎么确认：本章「最小数学」第 1 节最后那张初始化对照表，三行逐条念一遍。*

2. `[查]` 本章说 `torch.float32` 下 `exp` 大约多大就溢出？`log(0)` 和 `log(负数)` 分别得到什么？
   *怎么确认：本章「最小数学」第 2 节，开头那两条浮点事实一起看。*

3. `[算]` 一个 `Linear(200, 100)`，也就是 $n_{in}=200$、$n_{out}=100$。
   - (a) 配 ReLU 时用 He，$\sigma_w^2 = 2/n_{in}$，算出具体数值。
   - (b) 配 tanh 时用 Xavier，$\sigma_w^2 = 2/(n_{in}+n_{out})$，算出具体数值。
   - (c) 同一个网络里，哪个公式算出来的方差更大？
   *怎么确认：本章「最小数学」第 1 节的两个公式，把两个数代进去各算一遍；旁边那张表也写了"He 补回被 ReLU 砍掉的一半方差"。*

4. `[算]` 要算 `log(1 + exp(x))`，其中 $x=800$。本章给的分情况写法在 $x=800$ 时走哪一支？这一支里指数上带的是 $+x$ 还是 $-x$？在 float32 下估一下这个指数项大概是多少（提示：本章说 float32 在 `exp(88)` 就溢出）。
   *怎么确认：本章「最小数学」第 2 节最后一段给了两个分支；上一条刚写了 float32 的溢出点。*

5. `[跑]` 假设你已经按「落到代码」写好了 `Dropout`。固定种子后把它切到 `train()`，对同一个 `x` 连续调两次 `forward`——两次输出一样吗？再切到 `eval()` 调一次，输出和 `x` 比呢？
   *怎么确认：本章「工程视角」第 1 节「怎么测 Dropout」里的三条断言，重点是最后一条——`eval()` 下输出逐元素等于输入。*

6. `[判]` 小美把隐藏层的 `W` 全初始化成 0，说"反正梯度会把它调出来"。按本章那条链子一步步推：
   - (a) 一层 8 个隐藏单元，全 0 初始化后，它们的前向输出彼此是什么关系？
   - (b) 反向时它们各自收到的梯度是什么关系？
   - (c) 做一次参数更新之后，这 8 个单元还互不相同吗？所以这个网络实际上相当于几个单元在工作？
   *怎么确认：本章「直觉」里"八根柱子会长得一模一样"那段，加「常见误解」里"全 0 让网络退化成单神经元"那一行；「工程视角」第 2 节的 `test_linear_init_breaks_symmetry` 抓的就是这件事。*

7. `[判]` 有人写 Dropout 时，训练阶段只做了"以概率 $p$ 置 0"，忘了除以 $(1-p)$，推理阶段也原样输出。设这一层输入恒为 1、$p=0.5$：训练时输出的平均值大约是多少？推理时输出是多少？两者差几倍？
   *怎么确认：本章「最小数学」第 4 节「Dropout 为什么除以 (1-p)」，前两句就把"不补偿会怎样"写清楚了；再看「常见误解」第一行。*

---

## 延伸阅读

- D2L 4.5 权重衰减、4.6 暂退法、4.8 数值稳定性和模型初始化 —— 本章三块的教材来源
- [Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html) —— Xavier 原始论文，读第 4 节
- [Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html) —— "子网络集成"的直觉在引言里
- 下一章：[10 训练调试手册](10-debugging.md) —— 上面这些都试过还是不行时，按那份清单排查
