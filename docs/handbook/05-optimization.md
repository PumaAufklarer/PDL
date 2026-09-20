# 04 · 梯度下降与优化器

> **一句话**：反向传播只告诉你"每个参数该往哪个方向调"，优化器回答"一次调多少、要不要记历史"——同一份梯度配上不同的优化器，收敛速度能差一个数量级。
>
> **对应 D2L**：3.2 从零实现线性回归 ｜ 11.3–11.10 优化算法 ｜ **对应 Stage**：2 ｜ **对应代码**：`src/pdl/optim/`

## 为什么需要它
第 02 章你学会了求梯度、第 03 章你学会了验证它，第 04 章你把它乘上 `-lr` 加到参数上，训练就跑起来了。看起来优化器就是个乘法，但那句 `w -= lr * w.grad` 里藏着一组要命的权衡：
- 学习率写 `0.001` 还是 `0.1`？差一百倍，一个慢到看不出变化，一个直接把 loss 打成 `nan`。
- 一个 batch 五万条样本，是每步都用完，还是抽 32 条凑合？
- 参数 A 一直朝同一个方向稳步前进，参数 B 在来回横跳，要不要给它俩不同的步长？
- 训练前期梯度大、后期梯度小，学习率要不要跟着变？
这些问题的答案就是一整柜的优化器。**它们本身不难，难的是知道每个优化器在治上面哪一个病。** 所以本章按"它治什么病"来排，不按论文年份排。
还有一句得说在前面：优化器是最好写错、又最难发现写错的模块。前向写错会报形状不匹配，损失写错会算出负数，优化器写错什么都不报，只是收敛变慢或者不收敛，然后你花三天去怀疑数据。所以「工程视角」这节的比重和「最小数学」一样大。

## 直觉
你在浓雾里下山，唯一能做的是感受脚下这块地的坡度。三种做法：
- **全批量**：把山上所有石头摸一遍再决定往哪走。方向最准，但摸完天都黑了——一次更新要扫全部数据。
- **单样本**：随便踢一块石头感受一下就动身。快，但方向噪声极大，路线像醉汉。
- **小批量**：摸一小把再走。这才是实际用的，也是"batch"这个词的由来。
**Momentum（惯性）**：球从山上滚下来，路过小坑不会停，因为它有惯性。SGD 每一步只看当前梯度，撞上"这一步梯度刚好指向上坡"的噪声就原地倒退；Momentum 记住之前的方向，方向一致的分量累积起来、来回震荡的分量互相抵消。**它不是加速器，是稳定器，加速只是副作用。**
**AdaGrad / RMSProp / Adam（自适应步长）**：给每个参数配一双合脚的鞋。AdaGrad 记下每个参数历史上"梯度的平方和"，跳得凶的自动压步子，走得稳的保持大步；毛病是平方和只增不减，后期所有参数的步子都被压没了。RMSProp 把平方和换成"平方的指数加权平均"，让久远的梯度被忘掉。Adam 是 Momentum + RMSProp：既记一阶矩（方向的历史，负责稳），又记二阶矩（幅度的历史，负责定步长），再加一步偏差修正——两个矩初值都是 0，不修正的话训练头几步等于没走。
**学习率调度（step / cosine / warmup）**：前期离得远，步子大点省时间；后期快到谷底，步子大了会在谷底来回蹦、永远落不下来；warmup 则是先用极小的学习率走几百步，等 Adam 的统计量攒起来再放大，免得开局就被噪声带跑偏。
学习率不对的症状很好认：**太大**是 loss 上蹿下跳然后变成 `nan`，**太小**是 loss 几乎不动、曲线像一条水平线。这两种曲线形状要刻在脑子里，调试时第一眼就看它。
最后一条反直觉的：**SGD 的噪声是好事**。噪声让参数不会精确地滑进某个窄而尖的坑里，而是被推出来继续找一个更宽、更稳的谷底。这也是小 batch 常常比大 batch 泛化好的原因。
优先级记一句话：**SGD + Momentum 是基线，Adam 是省事的默认选择。** 前者调好了往往泛化更好，后者几乎不用调就能跑起来。

## 最小数学
> 看不懂就跳过，先去做「落到代码」，回来再读。

**1. 三种变体：同一个梯度的不同估计。** 真实梯度是全数据的平均 $g = \frac{1}{n}\sum_{i=1}^{n}\nabla_\theta \ell_i(\theta)$，区别只在拿多少条来估它：

| 变体 | 用它估 $g$ | 每步代价 | 方差 |
|---|---|---|---|
| 全批量 | 全部 $n$ 条 | $O(n)$ | 0（准，但贵） |
| 单样本 | 1 条 | $O(1)$ | 极大 |
| 小批量 | $B$ 条 | $O(B)$ | $\propto 1/B$ |

方差和 $1/B$ 成正比——这就是"batch 越大越稳"的出处，也是"batch 大到一定程度就白花钱"的出处：方差按 $1/B$ 降，代价按 $B$ 涨。

**2. 学习率的稳定区间能算出来。** 把损失在极小点附近二阶展开 $L \approx L^\star + \frac{1}{2}\lambda\|\theta - \theta^\star\|^2$（$\lambda$ 是曲率），梯度下降稳定当且仅当 $0 < \eta < 2/\lambda$。曲率大的方向（陡）只能配很小的步长——这就是"病态问题"里梯度下降慢得像爬的原因，也是所有自适应方法存在的理由。

**3. Momentum。** $v \leftarrow \beta v + g$，$\theta \leftarrow \theta - \eta v$，$\beta$ 通常 0.9。注意第一步 $v = g$，梯度被放大了，所以加了 Momentum 通常要把学习率调小一点。

**4. AdaGrad / RMSProp。** AdaGrad 累积平方 $s \leftarrow s + g^2$，更新用 $\theta \leftarrow \theta - \eta\, g / (\sqrt{s} + \epsilon)$；RMSProp 只改一处，$s \leftarrow \beta s + (1-\beta) g^2$。$\epsilon$ 是防除零的小常数（1e-8 量级），**放在根号外面**，和 PyTorch 保持一致。

**5. Adam。** $m \leftarrow \beta_1 m + (1-\beta_1) g$，$v \leftarrow \beta_2 v + (1-\beta_2) g^2$。因为初值是 0，第 $t$ 步的期望被缩了 $(1-\beta^t)$ 倍，所以要做偏差修正：

$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}, \qquad \theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

默认 $\beta_1 = 0.9$、$\beta_2 = 0.999$、$\epsilon = 10^{-8}$，**$t$ 从 1 开始数**——$t = 0$ 时 $1-\beta^0 = 0$，第一步就除零。

**6. 学习率调度。** 阶梯 $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$；余弦退火 $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$。

**7. 凸与非凸。** 凸函数里任何局部极小都是全局极小，线性回归的 MSE 就是凸的。MLP 的损失是非凸的，但高维空间里"坏"的局部极小（明显比全局差）其实很少，多的是**鞍点**（有的方向升、有的方向降）和一大片"差不多好"的平坦区域。所以"深度学习会被困在局部极小"这个担心在实践中基本不成立；真正卡住训练的是鞍点附近梯度接近 0，以及病态曲率导致的震荡——Momentum 和自适应方法恰好就是治这两样的。

## 落到代码

### `src/pdl/optim/` —— 基类就三样东西

```python
class Optimizer:
    def __init__(self, params: list[Tensor], lr: float) -> None:
        self.params = list(params)          # 参数的引用，不是拷贝
        self.lr = lr
        self.state: list[dict[str, torch.Tensor]] = [{} for _ in self.params]
        #   state[i] 是第 i 个参数的私有缓存，子类想存什么存什么
    def zero_grad(self) -> None:
        """把每个参数的 .grad 置零。不许碰 state。"""
    def step(self) -> None:
        """读 .grad，在 torch.no_grad() 里原地改参数。子类实现。"""
```
三个设计决定，每个都有理由：
- **`params` 存引用，不存拷贝。** 优化器要能改到模型里那块真内存；`list(params)` 只是防止外面之后改这个 list 影响到优化器。
- **`state` 放在优化器里，不放在参数上。** 这是本章最值得想清楚的一点：Adam 要给每个参数缓存 $m$、$v$、步数 $t$，这些是**训练过程**的状态，不是模型的一部分；塞进 `Tensor` 就得为"我可能被某个优化器用"预留字段，换优化器时还得手工清理，忘了就是脏数据；存检查点时分得清——模型参数一份（推理要用），优化器状态一份（继续训练才要）。`state` 用和 `params` **同序的 list**，第 `i` 个参数的缓存就是 `state[i]`，下标对齐最直白，代价是参数顺序必须稳定（写进文档）。
- **`zero_grad()` 只清 `.grad`，绝不动 `state`。** 见「工程视角」，这是最常见的实现 bug。

### 子类接口

```python
class SGD(Optimizer):
    def __init__(self, params, lr,
                 momentum: float = 0.0, weight_decay: float = 0.0) -> None: ...

class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None: ...
```
`Momentum` **不做成单独的类**，它就是 `SGD(momentum=0.9)`——和 SGD 的差别只有 `state` 里多存一个 `v`。这个决定要写进文档，否则调用方会到处找 `pdl.optim.Momentum`。`AdaGrad` / `RMSProp` 建议按同样接口各实现一遍（十来行）当练习，不作为 Stage 2 的硬性交付。

### 学习率调度

```python
class LRScheduler:
    def __init__(self, optimizer: Optimizer, ...) -> None: ...
    def get_lr(self) -> float: ...   # 现在该用多少
    def step(self) -> None: ...      # 推进一格，写回 optimizer.lr
```
**命名陷阱**：`optimizer.step()` 每个 batch 调一次，`scheduler.step()` 每个 epoch 调一次。两个都叫 `step`，调错地方的后果是学习率以 batch 的速度掉光。文档里各写一行"调用频率"。

### 训练循环里的调用顺序

```
for epoch in range(epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    scheduler.step()          # 看缩进：它在 batch 循环外面
```

## 工程视角

### 1. 优化器测试：用一个有解析答案的函数
别拿线性回归测优化器——变量太多，坏了不知道怪谁。用一个一维二次函数 $f(w) = (w-3)^2$，它的 $f'(w) = 2(w-3)$、$w^\star = 3$ 全是手算的：

```python
def test_sgd_converges_to_known_minimum():
    w = torch.tensor([0.0], requires_grad=True)
    opt = SGD([w], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        ((w - 3.0) ** 2).sum().backward()
        opt.step()
    assert abs(w.item() - 3.0) < 1e-3
```
这一个测试同时覆盖了 `zero_grad`、`backward` 填梯度、`step` 更新三件事。更值钱的是**稳定性边界**的断言：更新式是 $w \leftarrow w - 2\eta(w-3)$，也就是 $w_{k+1} - 3 = (1-2\eta)(w_k - 3)$，于是：

| 学习率 | 理论行为 | 测试断言什么 |
|---|---|---|
| `0.1` | 每步衰减到 0.8 倍，稳 | 300 步后误差 < 1e-3 |
| `0.5` | $1-2\eta = 0$，**一步到位** | 1 步之后就是 3.0 |
| `1.0` | $1-2\eta = -1$，在 0 和 6 之间**永远震荡** | 100 步后仍在震荡、不收敛 |
| `1.1` | $\lvert 1-2\eta \rvert > 1$，**发散** | 50 步内出现 `inf` 或 `nan` |

`lr = 1.0` 那一行特别值得写：它证明"loss 不降"不等于"代码错了"，有时候只是学习率踩在稳定区间的边界上。这种**定性断言**比"跑完不报错"有用得多。

### 2. `zero_grad()` 的时机：三种写错法
正确顺序只有一个：**清梯度 → 前向 → 反向 → 更新**。写错的三种后果各不相同，正好各写一个测试：
- **完全忘了 `zero_grad`**：梯度一直累加，等效学习率随步数线性增长。症状是 loss 先正常降、然后突然炸成 `nan`。测试：不调 `zero_grad` 跑 20 步，断言 `w.grad` 的绝对值在变大。
- **在 `backward()` 之后、`step()` 之前清**：梯度刚填好就被清零，`step()` 拿到全 0，**参数永远不动**，loss 是一条直线。这个 bug 最阴——代码看着"很规范"，每步都清了梯度。
- **在 `step()` 之后、下一轮 `backward()` 之前清**：等价于在循环开头清，**是对的**。所以"多清一次"通常无害，"清错位置"才致命。
建议把顺序写死在 `Trainer.train_step` 里，并断言"跑完一个 batch 参数确实变了"——这条能一次性抓住第二种 bug。

### 3. `state` 不能被 `zero_grad` 清掉
超常见的写法错误：`zero_grad` 和 `__init__` 长得像，复制粘贴就把 `self.state = [{} for _ in ...]` 抄进了 `zero_grad`。结果 Adam 的 $m$、$v$ 每步归零，动量彻底失效，看起来还在训练、loss 也在慢慢降，**只是慢得像没加优化器**。这种 bug 全靠测试抓：

```python
# 先跑两步 Adam，让 state 里攒上东西
snapshot = opt.state[0]["m"].copy()
opt.zero_grad()
assert torch.equal(opt.state[0]["m"], snapshot)   # zero_grad 不该动 state
```
顺手再断言 `zero_grad()` 之后**每个**参数的 `.grad` 都是 0（不是 `None`，也不是只清了第一个参数）。

### 4. 数值陷阱
- **`eps` 不能省**。某个参数从头到尾没吃到梯度（比如 ReLU 死掉），$v = 0$，`g / sqrt(0)` 直接 `inf`；`eps` 放根号外加。
- **Adam 的 `t` 从 1 开始**，$1 - \beta^0 = 0$，从 0 开始第一步就除零。
- **原地更新要在 `torch.no_grad()` 里做，也绝不能重新绑定名字**。参数是叶子张量、`requires_grad=True`，正确写法是 `with torch.no_grad(): p -= lr * p.grad`——不在 `no_grad` 里对需要梯度的叶子张量做原地操作，torch 会直接报错（本仓库禁止用 `.data`，它绕过 autograd）。反过来的写法同样要命：`p = p - lr * p.grad` 只是把局部名字重新绑定到一个新张量，优化器持有的那个参数根本没变，训练会静默地不学习。这两条都能用一个测试守住：`step()` 前后断言参数对象身份不变（`id(p)` 相同）、且值确实变了。这条写进 `optim` 的 docstring。
- **存进 history 的 loss 要 `loss.item()`**。存 Tensor 的话整个 epoch 的计算图都活着，内存随时间线性涨，Stage 2 数据小看不出来，Stage 3 就会 OOM。
- **权重衰减别作用到 bias 上**（第 04 章那张表）。Stage 2 先做整体 `weight_decay`，但接口别堵死——是给 `SGD` 传一个参数列表，还是加个 `no_decay` 集合，动手前想清楚。

### 5. 实验纪律：一次只改一个超参数
超参数之间会互相影响（改 batch size 等于改了梯度方差，也就改了能用的学习率上限）。所以一次只动一个，其余全部固定，包括 seed 和数据；所有实验用**同一份数据、同一个 seed**，只让被研究的那一个参数变；结果落成表存下来，别靠翻终端滚动条。
学习率扫描做成小脚本 `examples/lr_scan.py`，`uv run python examples/lr_scan.py --seed 0 --lrs 1e-4 1e-3 1e-2 1e-1 1.0`，输出一张表：

```
lr       final_loss   diverged
1e-4     2.3100       no        # 几乎没动，学习率太小
1e-3     0.0048       no        # 正常
1e-2     0.0049       no
1e-1     0.0310       no        # 后期在谷底来回蹦，落不下去
1.0      inf          yes       # 直接炸
```
这张表本身就是最好的学习材料：**同一个模型、同一份数据，只换一个数字，结果从"没反应"到"炸掉"全齐了。** 它也该进你的实验记录。

## 常见误解

| 误解 | 实际 |
|---|---|
| "Adam 自适应，所以不用调学习率" | 自适应只解决"**方向之间**的尺度差异"，不解决"**整体**步子太大还是太小"。Adam 的默认 `1e-3` 是个好起点，但换任务、换 batch size 时照调——论文里这个默认值本来就是调出来的 |
| "loss 不降就是优化器写错了" | 优化器只是嫌疑人之一。按这个顺序查：数据（输入和标签配错了没）→ 损失（1/2 和 n 对不对）→ 学习率（是不是太大直接炸、太小看不出动）→ `zero_grad` 顺序 → 最后才轮到优化器本身。绝大多数"不降"是学习率的问题 |
| "batch 越大越稳，所以越大越好" | 稳不等于进步快。batch 翻倍，方差减半（收益有限），每步算力翻倍（代价线性）。而且大步长的均匀梯度容易把你带进**尖锐**的极小值，那里泛化差。常见做法是 batch 尽算力允许，然后**按 batch 大小同步调学习率**，不是盲目往上堆 |
| "momentum 越大越快" | $\beta \to 1$ 时更新量约等于过去所有梯度的平均，惯性极大，快到谷底时**刹不住**，在谷底来回冲。0.9 是常态，0.99 只在特别需要平滑时用，而且必须配更小的学习率 |

## 自测题

> 每题的依据都在**本章**，不需要课外知识。答不上来就回去翻，不用猜。第 3、4 题是手算底线——拿张纸把更新式代几步，别在脑子里过。
>
> 标记：`[查]` 翻本章的表或清单 · `[算]` 手算一个具体数字 · `[跑]` 写一两行代码看结果 · `[判]` 判断一个说法对不对

1. `[查]` 在 $f(w) = (w-3)^2$ 上，本章那张表列了 `0.1`、`0.5`、`1.0`、`1.1` 四个学习率。哪一个"一步到位"？哪一个"永远震荡"？哪一个会"发散"？
   *怎么确认：本章「工程视角」第 1 节，紧接"稳定性边界"那段的那张三列表。*

2. `[查]` 本章「学习率调度」小节末尾有个"命名陷阱"：`optimizer.step()` 和 `scheduler.step()` 各应该多久调一次？把后者写成前者的频率会怎样？
   *怎么确认：本章「落到代码」里「学习率调度」小节的最后一段，再对上「训练循环里的调用顺序」那段代码的缩进（`scheduler.step()` 在 batch 循环外面）。*

3. `[算]` `f(w) = (w-3)^2`，初值 `w = 0`：
   - (a) 先写出 $w_{k+1} - 3$ 与 $w_k - 3$ 的关系式（本章推过）。
   - (b) 取 `lr = 0.25`，算出 `w1`、`w2`、`w3`。
   - (c) 取 `lr = 1.0`，把前四步的 `w` 写出来。
   *怎么确认：本章「工程视角」第 1 节给了 $w_{k+1}-3 = (1-2\eta)(w_k-3)$；(b)(c) 算完，和同一节那张表的"理论行为"对一眼。*

4. `[算]` Momentum：$v \leftarrow \beta v + g$，$w \leftarrow w - \eta v$。取 $\beta = 0.9$，每一步的 $g$ 都等于 `1`：
   - (a) `v1`、`v2`、`v3` 分别是多少？
   - (b) 如果 $g$ 一直等于 `1`，$v$ 会稳定在多少？（提示：让 $v = 0.9v + 1$ 解一下，或者多代几步看趋势。）
   *怎么确认：本章「最小数学」第 3 节的两行更新式，以及「常见误解」表里"momentum 越大越快"那一行。*

5. `[跑]` 跑下面这段（不用你的 `pdl` 代码，更新就手写一行），把 `w` 打印出来：
   ```python
   w = torch.tensor([0.0], requires_grad=True)
   for _ in range(3):
       ((w - 3.0) ** 2).sum().backward()
       w.grad.zero_()                       # 注意这一行的位置
       with torch.no_grad():
           w -= 0.1 * w.grad
   print(w.item())
   ```
   再把 `w.grad.zero_()` 移到 `backward()` **之前**，重跑。两次打印差在哪？
   *怎么确认：本章「工程视角」第 2 节，"在 `backward()` 之后、`step()` 之前清"那一条。*

6. `[判]` 小美说："Momentum 和 Adam 都是加速器，加上去收敛一定更快。" 本章把 Momentum 叫什么？它到底治的是哪种病？
   *怎么确认：本章「直觉」里 Momentum 那一段的最后一句，和「最小数学」第 7 节最后一句（"真正卡住训练的是……"）。*

7. `[判]` 小美写 `zero_grad()` 时，把 `__init__` 里的 `self.state = [{} for _ in params]` 顺手复制了进去。代码不报错、loss 也还在慢慢降。本章怎么形容这个 bug 的表现？
   *怎么确认：本章「工程视角」第 3 节，看那个"超常见的写法错误"下面的描述，以及它给的测试怎么断言。*

## 延伸阅读
- D2L [11.3 梯度下降](https://zh.d2l.ai/chapter_optimization/gd.html) 一路到 [11.10 Adam](https://zh.d2l.ai/chapter_optimization/adam.html)，再加 [11.11 学习率调度器](https://zh.d2l.ai/chapter_optimization/lr-scheduler.html) —— 本章每条公式的完整推导
- [Why Momentum Really Works (Distill)](https://distill.pub/2017/momentum/) —— 带可交互图，把动量的"稳定器"作用看得最清楚
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) —— 原始论文，其实只看 Algorithm 1 就够了
- 下一章：[06 过拟合、欠拟合与模型选择](06-generalization.md) —— 训练 loss 降下去之后，怎么知道它能不能用在没见过的数据上
