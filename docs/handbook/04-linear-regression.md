# 03 · 线性回归：第一个能学习的模型

> **一句话**：模型是"输入怎么变成输出"的一个假设，损失是给它打分的尺子，优化是照着分数调参数的过程——线性回归用最少的零件把这三件事演了一遍，后面所有模型只是换了个 `f`。
>
> **对应 D2L**：3.1 线性回归、3.2 从零实现、3.3 简洁实现（选做） ｜ **对应 Stage**：2 ｜ **对应代码**：`src/pdl/data/`、`src/pdl/loss/`、`src/pdl/optim/`、`src/pdl/trainer/`

## 为什么需要它
先看一个具体问题：**预测房价**。每套房的输入是面积、房龄、到地铁距离、卧室数，输出是成交价。写成张量就是 `X: (n, 4)` 和 `y: (n,)`。
如果允许你用任意函数，这题没法做——函数有无穷多个。所以得先**把函数的形式限制住**，这个限制叫模型（假设空间）。最简单的一种限制是：每个特征贡献一份，加起来，再加个底价：

$$\hat{y} = Xw + b, \qquad X:(n,d),\quad w:(d,),\quad b:(,),\quad \hat{y}:(n,)$$

`w` 是 4 个特征的权重，`b` 是底价。参数一共 5 个，`n` 可以有一万套房——**用 5 个数去解释一万个观测，这就是"学习"的全部含义。** 代码里就是 `X @ w + b`，`b` 靠广播加到最后那个维度上。
为什么第一站是它，而不是直接上神经网络：
- **它有解析解。** 一个公式直接算出最优 `w`，所以你能拿它当标准答案，验证 `DataLoader`、`MSELoss`、`SGD`、`Trainer` 有没有写错。别的模型给不了你这个。
- **零件齐全，而且是唯一你能完全看穿的模型。** 参数、前向、损失、梯度、更新一个不少，只是长得简单；出了事（loss 不降、参数乱飞）你能手算出来该是什么样。Stage 3 换 Softmax、Stage 5 换 MLP，骨架一行不改。

## 直觉
你去洗澡，左冷水阀、右热水阀，各拧多少度你不知道。于是：
1. 凭感觉先拧两下（**初始化参数**）
2. 伸手试水温，和想要的温度比一下，差多少、偏冷还是偏热（**前向 + 损失**）
3. 判断"热水阀往哪边拧、拧多少，冷水阀同理"（**梯度**）
4. 各拧一点点，回去重试（**参数更新**）
几个关键点：
- 你不需要知道整条管道怎么工作，只需要知道**这次拧的方向对不对**。
- 两个阀门对水温的影响不一样大：热水阀灵、冷水阀钝，它俩不该拧一样的幅度——这就是 AdaGrad / Adam 要解决的"每个参数配不同步长"。
- 水温试一次不够准、试一百次又太累，这就是**小批量**的位置；而"闭着眼一步步试"才是能扩展到几百个阀门的做法，一步到位的解析解反而扩展不了。

## 最小数学
> 看不懂就跳过，先去做「落到代码」，回来再读。

### 1. 损失：为什么是平方，为什么带个 1/2

$$L(w, b) = \frac{1}{2n}\left\| y - \hat{y} \right\|^2 = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- **为什么平方而不是绝对值**：平方处处可导（绝对值在 0 点不可导），而且错得越离谱罚得越重。
- **为什么除以 n**：让损失和数据量无关，不然一万套房的 loss 是一套房的 100 倍，两个 batch 没法比。
- **为什么多个 1/2**：纯粹为了求导好看。对 `ŷ` 求导得到 $\partial L/\partial \hat{y} = (\hat{y}-y)/n$，那个 2 被"拉下来"跟 1/2 抵消了。没有它梯度就是 `2(ŷ-y)/n`，本质只是把学习率除以 2，但会让你的梯度和 D2L、PyTorch 对不上，测试期望值也得多带个 2。**照 D2L 的写法来。**

### 2. 梯度

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^\top (\hat{y} - y), \qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i}(\hat{y}_i - y_i)$$

`X^T` 是 `(d, n)`，乘上 `(n,)` 得到 `(d,)`，跟 `w` 同形；`b` 是把所有样本的误差平均成一个标量。这两个式子你不用手写——`torch.autograd` 会替你算。**但你要看得出来它们是"误差反向分配到每个特征"**：某个特征在样本上取值越大，分到的梯度越大。

### 3. 解析解（正规方程）

$$w^\star = (X^\top X)^{-1} X^\top y$$

`b` 怎么办：给 `X` 左边拼一列 1，把 `b` 塞进 `w` 一起解。**实际不用它**：求逆是 $O(d^3)$，`d` 上万就废；`X^T X` 可能不可逆（特征线性相关）；MLP 根本没这公式。但它是**极好的单元测试参照物**——同一份数据上，正规方程解的 `w` 和 SGD 跑几千步的 `w` 应该几乎一样。

真正训练用的是解析解那条路的迭代版，`B` 是一个 batch 的样本数，`η` 是学习率：

$$w \leftarrow w - \eta \cdot \frac{1}{B} X_B^\top (\hat{y}_B - y_B)$$

## 落到代码
四个模块各管一段，边界要划清楚。**Stage 2 里模型本身不进 `src/pdl/`，就写在 `examples/linear_regression.py` 里**——它只有十几行，等 Stage 5 的 `nn.Linear` 出来再收编。

### `src/pdl/data/` —— 谁来喂数据

```python
class Dataset:
    """只回答两个问题：一共几条、第 i 条长什么样。"""
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]: ...

class TensorDataset(Dataset):
    """把已经在内存里的 (X, y) 包成 Dataset。"""
    def __init__(self, X: Tensor, y: Tensor) -> None: ...

class DataLoader:
    """切批、打乱、按批吐出来。"""
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True,
                 seed: int | None = None, drop_last: bool = False) -> None: ...
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def __len__(self) -> int: ...      # 批数，不是样本数
```
动手前先把边界想清楚：
- `Dataset` **不做**切批、不打乱，它就是一张表；`DataLoader` 不碰模型、不算损失，只管"按什么顺序、一次给几个"。
- `shuffle` 用 `DataLoader` 自己的 `torch.Generator`，不是全局随机状态——这样"谁打乱了数据"在代码里看得见。
- `__len__` 返回**批数** `ceil(n / batch_size)`，测试要用它。

### `src/pdl/loss/` —— 怎么打分

```python
class MSELoss:
    """L = (1 / 2n) * sum((y_pred - y_true) ** 2)，返回标量 Tensor。"""
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...
```
两个约定写进 docstring：**返回标量**（只有标量能 `backward()`，见第 02 章）；**1/2 在 loss 内部**，好让测试按 `1/2` 的版本断言。

### `src/pdl/optim/` —— 怎么改参数

```python
class Optimizer:
    def __init__(self, params: list[Tensor], lr: float) -> None: ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float,
                 momentum: float = 0.0, weight_decay: float = 0.0) -> None: ...
```
细节第 05 章展开。这里只要知道：`Trainer` 拿着 `optimizer`，不关心它是 SGD 还是 Adam。

### `src/pdl/trainer/` —— 把上面串起来

```python
class Trainer:
    def __init__(self, model, loss_fn, optimizer) -> None: ...
    def train_step(self, X: Tensor, y: Tensor) -> float:
        """一个 batch：前向 → 损失 → 反向 → 更新。返回 float，不返回 Tensor。"""
    def validate(self, X: Tensor, y: Tensor) -> float:
        """只前向、不算梯度。返回 loss 数值。"""
    def fit(self, train_loader: DataLoader, epochs: int,
            val_loader: DataLoader | None = None) -> list[dict]:
        """跑完 epochs 轮，每轮回 {'epoch', 'train_loss', 'val_loss'}。"""
```
`model` 用**鸭子类型**，不强制继承：有 `__call__(X) -> Tensor` 和 `parameters() -> list[Tensor]` 就能用。Stage 2 的手写线性模型满足它，Stage 5 的 `nn` 模块照样满足。**别提前设计基类。**

### 一个 epoch 里发生了什么（按这个顺序写）
1. `train_loader` 按本轮顺序吐出 `(X_batch, y_batch)`
2. `optimizer.zero_grad()`，清掉上个 batch 留下的梯度
3. `y_pred = model(X_batch)` 前向，`loss = loss_fn(y_pred, y_batch)` 得到一个标量 Tensor
4. `loss.backward()`，梯度填进每个参数的 `.grad`
5. `optimizer.step()`，按梯度改参数，然后记 `loss.item()`（本仓库禁止用 `.data`，它会绕过 autograd）
6. 本轮结束，有 `val_loader` 就跑一遍 `validate`，记 `val_loss`

**2 和 3 不能换，5 和 6 不能换。** 这几行的顺序就是第 05 章「工程视角」里那些 bug 的来源。

## 工程视角
模型谁都会写，会验证的人少。这节是本章最值钱的部分。

### 1. 合成数据：真值已知，才能断言"学到了"
真实房价没有标准答案，你只能说"loss 降了"，说不了"学对了"。所以用合成数据：

```python
g = torch.Generator().manual_seed(seed)
X = torch.randn((n, d), generator=g)
w_true, b_true = torch.randn((d,), generator=g), 2.0
noise = torch.randn((n,), generator=g) * 0.1
y = X @ w_true + b_true + noise
```
`w_true`、`b_true` 是你自己定的，训练完可以直接比：`torch.abs(w_learned - w_true).max() < 0.05`。
这是**整章的关键设计**：把"训练有没有起作用"从主观判断变成可断言的事实。`noise` 让任务不那么平凡（否则解析解一步到位，测不出优化器），同时它给出了 loss 的**下界**——噪声方差 `σ²` 意味着最优 loss 大约在 `σ²/2` 附近，降到这里就该平了，再降不下去不是 bug。脚本要能控制 `n`、`d`、`noise`，你才能造出"数据少得可怜"和"噪声大到学不动"这两种极端。

### 2. 可复现：同 seed 两次跑必须一模一样
DoD 写死了：`uv run python examples/linear_regression.py --seed 0` 跑两次，输出完全一致。要做到，**每处随机都得能追溯到 seed**：造数据的 `Generator`、`DataLoader.shuffle` 内部打乱用的 `Generator`、参数初值，全都从 seed 建，不能拍脑袋写 `torch.zeros`。
`pdl/utils/` 里放一个 `seed_everything(seed)`，一次性固定 torch（以及 Python `random`）的随机源，方便临时调试和写测试。但**库代码里别依赖全局随机状态**——依赖它意味着"这次结果怎么不一样"永远查不出是谁动了随机数。验证方式：跑两次，stdout 各存一个文件，`diff` 空才算过，别用肉眼看日志。

### 3. 测试清单
每条都是"不通过就说明哪块坏了"的定位器：

| 测试 | 断言什么 | 坏了说明什么 |
|---|---|---|
| 解析解对照 | 正规方程解的 `w` 与 SGD 跑几千步的 `w` 差距小于阈值 | 整套流程里有一步不对 |
| 收敛性 | 每 50 步记一次 loss，整体降到 `≈ σ²/2` 附近 | 学习率、梯度符号、zero_grad 顺序 |
| 参数收敛 | `max|w - w_true| < 阈值`，`|b - b_true| < 阈值` | 模型或更新式写错 |
| 批大小整除 | `n=64, bs=32` → `len(loader) == 2`，最后一批也是 32 条 | `__len__` / 切批边界 |
| 批大小不整除 | `n=65, bs=32` → `len(loader) == 3`，最后一批只有 1 条 | 丢数据（最常见）或形状崩掉 |
| 数据量小于一个 batch | `n=5, bs=32` 能正常跑完，`len(loader) == 1` | 除零、切片越界 |
| 可复现 | 同 seed 两次，loss 序列 `torch.equal` | 有随机源没被 seed 管住 |

"loss 单调下降"这句话要小心：小批量 SGD 每一步的 loss 是**抖**着往下走的。所以断言要下在"趋势"上——比较首轮和末轮的平均 loss，或者对序列做滑动平均再看单调性。直接断言"每一步都不大于上一步"必然红。

### 4. 训练脚本的形态
`examples/linear_regression.py`，支持 `--seed`（必须）、`--epochs`、`--lr`、`--batch-size`。跑完打印类似：

```
seed=0  n=512 d=4   epochs=100  bs=32
epoch  10  train_loss 3.2140  val_loss 3.1902
epoch 100  train_loss 0.0051  val_loss 0.0049
w_err=0.0121  b_err=0.0087  (noise floor ~ 0.0050)
```
把 `w_err` 和 noise floor 打出来是刻意的：**让你一眼看出训练是"到位了"还是"还差得远"。** loss 数字本身看不出来，和噪声下界一比才看得出来。

## 常见误解

| 误解 | 实际 |
|---|---|
| "线性模型太简单，学了没用" | 它是**验证工具**。你的 DataLoader / 损失 / 优化器 / Trainer 全靠它来确认没写错，因为只有它有标准答案。跳过去，后面出了问题你连"是不是数据管道坏了"都判断不了 |
| "MSE 只能配线性回归" | MSE 是**回归任务的默认损失**，配 MLP 完全没问题（Stage 5 就是），只要输出的物理含义说得通。它不适合分类，因为对概率分布不敏感，原因见第 07 章 |
| "bias 也要做权重衰减" | 一般**不要**。权重衰减惩罚的是"输入特征的影响强度"，`b` 只是个平移量，惩罚它等于逼模型去别处补回来。实现上要把参数分成"decay"和"不 decay"两组——Stage 2 先做整体 `weight_decay` 就够，别提前分组 |
| "batch 越大越好" | 大 batch 的梯度估计更准，但每次更新的信息量更少、单位算力进步更慢，还更容易落进尖锐的极小值。小 batch 抖动大，常泛化得更好。Stage 2 先用 32 这种小值，权衡在第 05 章讲 |
| "loss 降到 0 才算训练成功" | 有噪声的数据上 loss 到不了 0，`σ²/2` 就是地板。追着它降只会让参数去拟合噪声，这是第 06 章的过拟合 |

## 自测题

> 每题的依据都在**本章**，不需要课外知识。答不上来就回去翻，不用猜。第 1 题问的是整章的主干顺序——答不顺就回去看「落到代码」最后那一小节。
>
> 标记：`[查]` 翻本章的表或清单 · `[算]` 手算一个具体数字 · `[跑]` 写一两行代码看结果 · `[判]` 判断一个说法对不对

1. `[查]` 本章「一个 epoch 里发生了什么（按这个顺序写）」列了 6 步。哪两组顺序**绝对不能换**？本章用哪句加粗的话标出了它们？
   *怎么确认：本章「落到代码」最后那一小节的 6 步，和紧跟其后那行加粗的话。*

2. `[查]` 本章「测试清单」表里，`n = 65, bs = 32` 那一行断言什么？"坏了说明什么"那一栏又写的是什么？
   *怎么确认：本章「工程视角」第 3 节的测试清单表，只对着「批大小不整除」这一行看完两列。*

3. `[算]` `y_pred = [3, 5]`、`y_true = [1, 1]`：
   - (a) `MSELoss` 等于多少？把式子写出来再交卷。
   - (b) 换成 `X` 是 `(2, 4)`，`w` 的梯度形状是什么？
   *怎么确认：本章「最小数学」第 1 节的损失公式（注意分母是 $2n$）和第 2 节那个"`X^T` 是 `(d, n)`，乘上 `(n,)` 得到 `(d,)`"的说明。*

4. `[算]` 都取 `drop_last = False`：
   - (a) `n = 100`、`batch_size = 32`：`len(loader)` 是多少？最后一批几条？
   - (b) `n = 65`、`batch_size = 32`：同样问。
   - (c) `n = 5`、`batch_size = 32`：同样问。
   *怎么确认：本章「落到代码」`src/pdl/data/` 小节写明 `__len__` 返回的是 `ceil(n / batch_size)`；「工程视角」第 3 节测试清单里这三组数字都给了断言。*

5. `[跑]` 跑下面这段，看打印出来是不是 `True`：
   ```python
   def make():
       g = torch.Generator().manual_seed(0)
       return torch.randn(3, generator=g)
   print(torch.equal(make(), make()))
   ```
   再把 `g` 挪到 `make()` 外面只建一次，重跑。两次结果为什么不同？
   *怎么确认：本章「工程视角」第 2 节——"每处随机都得能追溯到 seed"，以及它给的验证方式（跑两次、`diff` 空才算过）。*

6. `[判]` 小美说："正规方程一步就能出最优 `w`，那训练代码直接用它就行，学习率也不用调了。" 假设 `X` 是 `(n, 10000)`，本章会拿哪条理由顶回去？
   *怎么确认：本章「最小数学」第 3 节，注意那几个"实际不用它"的理由，以及最后一句讲它真正用处的地方。*

7. `[判]` 小美写测试时断言"每个 batch 的 loss 都不大于上一个 batch 的 loss"。本章说她这条测试会怎样？原因出在哪句话上？
   *怎么确认：本章「工程视角」第 3 节最后一段，第一句就是关键词。*

## 延伸阅读
- D2L [3.1 线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)、[3.2 从零开始实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html) —— 本章公式与数据构造的来源
- [Ordinary least squares (Wikipedia)](https://en.wikipedia.org/wiki/Ordinary_least_squares) —— 想清正规方程怎么推、为什么数值上危险时读
- [scikit-learn: 普通最小二乘](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) —— 看工业实现怎么处理"矩阵奇异"这个坑
- 下一章：[05 梯度下降与优化器](05-optimization.md) —— 把本章那个"按梯度改参数"的黑盒拆开
