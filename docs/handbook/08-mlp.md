# 07 · 多层感知机 ★

> **一句话**：线性模型只能画直线，而把两个线性层叠起来**仍然是直线**；只有塞进一个非线性激活函数，模型才第一次获得"弯曲"的能力——这就是多层感知机（MLP）。
>
> **对应 D2L**：4.1–4.3、4.7 ｜ **对应 Stage**：5 ★ 里程碑 ｜ **对应代码**：`src/pdl/nn/`

---

## 为什么需要它

这是本课程的里程碑章节（roadmap 的 Stage 5）：前面几章攒下的张量、自动微分、损失、优化器、数据管道，在这一章第一次拼成一个"真正的神经网络"。

`y = Wx + b` 画出来是一条直线（高维里叫超平面）。线性回归、softmax 回归，本质都是这个。

那我需要一条曲线怎么办？很自然的想法是"多叠一层"：

```python
h = W1 @ x + b1;        y = W2 @ h + b2        # 看着是两层，其实还是一层
h = relu(W1 @ x + b1);  y = W2 @ h + b2        # 这才是 MLP
```

把第一行代进第二行：$y = W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2)$。$W_2W_1$ 还是个矩阵，$W_2b_1+b_2$ 还是个向量——**两个线性层叠起来，数学上完全等价于一个线性层**，只是参数换了个写法；堆一百层结论一样。

所以让网络变强的不是"层数"，而是层与层之间那个**非线性**。这是全章的逻辑起点：**没有激活函数，深度毫无意义。**

---

## 直觉：直尺与折页

线性层像一根直尺：不管怎么摆、怎么转，它永远是直的。把两根直尺首尾焊死成一根——还是直的；焊 100 根也一样。

激活函数就是那个"掰"的动作：在直尺的接缝处允许你掰一个角度。一旦允许掰，几段短直尺就能拼出任意弯曲的形状——想拼海岸线就多掰几下，想拼圆弧就均匀地掰。ReLU 掰得很粗暴（一头压平，另一头原样不动），sigmoid 温柔些，是个平滑的 S。隐藏层因此成了自动的特征提取器：每个单元盯住输入的某个方向，第二层再组合它们，层数越深组合出的概念越抽象（边缘 → 部件 → 物体）。

---

## 最小数学

> 看不懂就跳过，先去做「落到代码」，回来再读。

### 1. 前向传播

$$\mathbf{h} = \phi(XW_1 + b_1), \qquad \mathbf{o} = \mathbf{h}W_2 + b_2$$

形状对一遍：$X$ 是 `(batch, in)`，$W_1$ 是 `(in, hidden)`，$b_1$ 是 `(hidden,)`，于是 $\mathbf{h}$ 是 `(batch, hidden)`；$W_2$ 是 `(hidden, out)`，$\mathbf{o}$ 是 `(batch, out)`。$\phi$ 是**逐元素**作用，不改变形状——所以它的反向也必须逐元素。

### 2. 激活函数：三种常见选择

| 函数 | 公式 | 值域 | 优点 | 坑 |
|------|------|------|------|-----|
| ReLU | $\max(0, x)$ | $[0,+\infty)$ | 正区间梯度恒为 1，不饱和；只比大小，快；输出稀疏 | 负区间梯度为 0，输入长期为负的单元会**永久死掉** |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0,1)$ | 平滑可导；输出可当概率用 | 两端**饱和**，梯度趋近 0，深层传不回去；非零中心；含 `exp`，慢 |
| Tanh | $\tanh(x)$ | $(-1,1)$ | 零中心，比 sigmoid 好优化 | 两端照样饱和 |

选型口诀：**隐藏层默认 ReLU；输出层看任务**（回归不加，二分类 sigmoid，多分类交给 softmax）；tanh 基本只在很浅的旧网络或需要零中心输出时用。

### 3. 万能近似定理（只讲直觉，不证明）

一句话：**一个足够宽的单隐藏层 MLP，能在有界闭区间上以任意精度逼近任意连续函数。** 它给你信心——"这个结构理论上装得下答案"。

但它没说三件事，这才是重点：① 只说**存在**这样一组权重，没说怎么找到（那是优化器的事）；② 没说需要多宽，理论上可能宽到不现实；③ **"能表示" ≠ "能学到"**——梯度下降未必找得到，数据也未必够。所以实践中"深而窄"常常比"浅而宽"划算。

### 4. 反向传播：链式法则在一层上的样子

设 $z = XW + b$、$\mathbf{h} = \phi(z)$，已知上游传回的 $\frac{\partial L}{\partial \mathbf{h}}$：

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \mathbf{h}} \odot \phi'(z), \qquad \frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial z}, \qquad \frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial z}$$

$\odot$ 是逐元素乘。$b$ 的梯度要**沿 batch 维求和**，因为前向时它被广播到了每个样本上——广播扩出去的维度，反向要收回来。继续往上游传是 $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial z} W^\top$。

**这里没有新算法。** 第 02、03 章已经讲清了链式法则和梯度是怎么求的：用 `matmul`、`+`、`relu` 把前向写出来，计算图自己搭好，`backward()` 自己把链式法则跑到头。"反向传播"在 MLP 这里被单独起名，只是因为这条链比较长。

**但你要能自己推一遍。** 这一章的「落到代码」里有一道必须过的关：手推这四个式子，再用 `torch.autograd` 对答案。

---

## 落到代码

### Module：为什么需要这个抽象

1. **参数注册**：`Linear` 里的 `W`、`b` 是需要梯度的 `Tensor`，优化器要一次性拿到全部：`model.parameters()`。没有统一抽象，你就得手写 `[l1.W, l1.b, l2.W, l2.b, ...]`，加一层改一次。
2. **统一调用**：所有层都实现 `forward(x)`，外面统一 `model(x)`，`Sequential` 才能把任意层串起来。
3. **可组合**：`Sequential` 自己也是 `Module`，能再塞进别的 `Module`——组合能力来自"大家长得一样"。

```python
# src/pdl/nn/module.py
class Module:
    def forward(self, x: Tensor) -> Tensor: ...      # 子类必须实现：这一层怎么算
    def __call__(self, x: Tensor) -> Tensor: ...     # 内部转到 forward，子类不要覆盖
    def parameters(self) -> list[Tensor]: ...        # 本模块及子模块里所有需梯度的参数
    def num_parameters(self) -> int: ...             # 参数的标量总数，用于结构和测试
    def train(self, mode: bool = True) -> None: ...   # 训练模式（Dropout 生效），递归子模块
    def eval(self) -> None: ...                      # 推理模式（Dropout 不生效）
    def __repr__(self) -> str: ...                   # 打印结构，确认模型和你想的一样
```

### 具体层（只给签名和职责）

```python
# src/pdl/nn/linear.py
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None: ...
    # W: (in, out)、b: (out,)。初始化必须非零，方案见第 09 章
# src/pdl/nn/activation.py
class ReLU(Module):    def forward(self, x: Tensor) -> Tensor: ...  # max(0,x)，无参数
class Sigmoid(Module): def forward(self, x: Tensor) -> Tensor: ...  # 1/(1+exp(-x))
class Tanh(Module):    def forward(self, x: Tensor) -> Tensor: ...  # tanh(x)
# src/pdl/nn/dropout.py
class Dropout(Module):  # __init__(self, p: float = 0.5)
    def forward(self, x: Tensor) -> Tensor: ...      # 训练：置 0 并除以 (1-p)；eval：原样输出
# src/pdl/nn/sequential.py
class Sequential(Module):  # __init__(self, *layers: Module)
    def forward(self, x: Tensor) -> Tensor: ...      # 依次调用每一层
```

用起来长这样（`parameters()` 拿到的就是优化器要吃的东西）：

```python
model = pdl.nn.Sequential(pdl.nn.Linear(784, 256), pdl.nn.ReLU(), pdl.nn.Linear(256, 10))
opt = pdl.optim.SGD(model.parameters(), lr=0.1)
```

### 手推一层的反向（本 Stage 的分水岭）

上面的层你当然可以只靠 `backward()` 就跑通。但这一章有一道**必须过的关**：手推 `Linear` 和 `ReLU` 的反向，然后证明它和 `torch.autograd` 算出的一样。

为什么值得花这个时间？因为"反向传播"这四个字里唯一需要你理解的东西，就是**链式法则在一层上的具体形状**。`backward()` 会替你做，但你**看不出它做错了**——第 03 章那个校验器验的是"数值自不自洽"，手推反向验的是"你到底懂不懂"。

放在 `examples/manual_backward.py`。**只做 `Linear` 和 `ReLU` 两个，不做通用引擎**：

```python
def manual_backward_linear(x, W, b, grad_z) -> tuple[Tensor, Tensor, Tensor]:
    """前向是 z = x @ W + b。返回 (grad_x, grad_W, grad_b)，形状分别同 x、W、b。"""

def manual_backward_relu(z, grad_h) -> Tensor:
    """前向是 h = relu(z)。返回 grad_z，形状同 z。"""
```

推导（先看懂，再自己写）。设 `x` 是 `(B, in)`、`W` 是 `(in, out)`、`b` 是 `(out,)`，上游传回来的 `grad_h` 形状是 `(B, out)`。

**先记一条铁律：任何张量的梯度，形状一定和它自己一样。** 于是四步：

1. **`grad_z = grad_h * (z > 0)`**，形状 `(B, out)`。ReLU 逐元素作用，反向也逐元素：`z > 0` 是一张 1/0 掩码，正的位置放行，负的位置直接掐断。
2. **`grad_b = grad_z.sum(dim=0)`**，形状 `(out,)`。为什么要**求和**？前向时 `b` 被**广播**到了 `B` 个样本上，每个样本用的是同一个 `b`；反向就得把这 `B` 份贡献收回来。这就是"广播的反向要归约"。
3. **`grad_W = x.T @ grad_z`**，形状 `(in, B) @ (B, out) = (in, out)`，正好等于 `W`。为什么 `x.T` 在左边？从形状倒推：结果必须是 `(in, out)`，带 `in` 的那维只能在左，`B` 那维必须被消掉——只有这个写法对得上。**形状是唯一的裁判。**
4. **`grad_x = grad_z @ W.T`**，形状 `(B, out) @ (out, in) = (B, in)`，正好等于 `x`。同理，`W` 带 `out` 的那维要消掉，所以它得转置放到右边。算出的 `grad_x` 继续往上游传，成为下一层手里的 `grad_h`。

最后和 torch 对一遍——**这就是本 Stage 的 DoD**：

```python
# torch 侧：z = x @ W + b；h = relu(z)；h.sum().backward()
# 手动侧：grad_h = torch.ones_like(h)；先 manual_backward_relu，再 manual_backward_linear
# 断言：三个 grad 分别与 x.grad / W.grad / b.grad 对齐（float64，atol=1e-6）
```

三个坑，都会让你"算出来了但是错的"：

- **`ReLU` 在 `z == 0` 那一点不可导**（左导 0、右导 1）。手推时归到哪边都行（`z > 0` 或 `z >= 0`），但**测试输入必须避开 0**：数值梯度在那里会给你一个不存在的"标准答案"（`(relu(h) - relu(-h)) / 2h = 0.5`，真值根本没有）。
- **`grad_b` 忘了求和**是最高频的错误。好在形状会从 `(out,)` 变成 `(B, out)`，形状断言能当场抓住它。
- **`x.T` 和 `W.T` 放反了**，在方阵时形状可能碰巧也对。所以**必须用非方阵测**（比如 `in=3, out=5`）。

---

## 工程视角

### 1. 每一层都要有梯度校验测试

Stage 1 那套 `check_gradient` 从这一章起是**每个层的标配**：

```python
@pytest.mark.parametrize("in_features", [1, 3, 7])
def test_linear_gradient(in_features):
    layer = Linear(in_features, 5)
    x = torch.randn(4, in_features)

    # 一次把 x、W、b 全验了：check_gradient 会对每个传进去的张量分别对照
    check_gradient(lambda: layer(x).sum(), x, layer.W, layer.b)
```

- `Linear` 要**分别**验 `W` 和 `b`，别只验一个；`ReLU` / `Sigmoid` / `Tanh` 各验一遍
- **ReLU 在 0 点不可导**：输入避开 0 附近，否则数值梯度会给出错误的"标准答案"（`(relu(h)-relu(-h))/2h = 0.5`，而真值不存在）
- `Sequential` 串起来再验一遍，确认真的是"端到端一张计算图"

### 2. XOR：最有说服力的测试

必须有的一个测试，因为它**同时证明了"线性不行"和"MLP 行"**：

- 数据只有四个点：`(0,0)->0`、`(0,1)->1`、`(1,0)->1`、`(1,1)->0`
- 线性模型（不加激活的 `Sequential`）训练几百轮，loss 卡在一个下不去的值上，准确率 ≤ 0.75
- 加一个 `ReLU` 隐藏层（宽度 4 就够），同样轮数，loss 掉到接近 0
- 断言写成"最终 loss < 0.05"，别断言参数等于某个值——权重解不唯一

这个测试跑通，说明 `Module`、`Linear`、激活函数、`backward()`、优化器**全都接对了**。

### 3. 参数量统计

`num_parameters()` 的设计：遍历 `parameters()`，把每个张量的**元素个数**加起来（不是张量个数）。它一举两得——打印模型时一眼看出模型多大，给测试提供硬断言：`Sequential(Linear(784,256), ReLU(), Linear(256,10)).num_parameters() == 784*256 + 256 + 256*10 + 10`（203530）。激活函数**没有参数**，别算进去。

### 4. 初始化全 0：一个能复现的失败实验

把 `Linear` 的 `W` 和 `b` 全设成 0，训练 `Sequential(Linear(2,8), ReLU(), Linear(8,1))`，你会看到 loss 几乎不动；再检查隐藏层输出：**8 个隐藏单元的输出一模一样**。

原因很朴素：所有 $W_1$ 的行相同 → 隐藏单元输出相同 → 反向收到的梯度相同 → 更新后仍然相同 → 永远一样。**8 个隐藏单元退化成 1 个。** 把这条写成一个测试（初始化后断言隐藏单元的方差不接近 0），比写在注释里有说服力得多。

### 5. 测试清单

| 测什么 | 怎么断言 |
|--------|----------|
| 参数量 | `num_parameters()` 等于手算值；`Linear(3,5,bias=False)` 比默认少 5 个 |
| 输出形状 | `Linear(3,5)(randn(4,3)).shape == (4,5)`；batch 大小变化不影响形状 |
| `Sequential` 顺序 | 两层不同的 `Linear` 手工算一遍前向，和 `Sequential` 结果 `torch.allclose`——顺序反了算得出数，但不是同一个数 |
| 空 `Sequential` | 行为提前定死：建议直接抛异常，或定义为恒等映射并写进测试。**不要留"看情况"** |
| 模式切换 | `train()` / `eval()` 递归作用到子模块；`Dropout` 的测试细节见第 09 章 |

---

## 常见误解

| 误解 | 实际 |
|------|------|
| "层数越多越好" | 两个线性层等于一个线性层，没有非线性时层数只是浪费算力；有非线性时加深也有代价：更难训、更吃数据 |
| "参数越多越好" | 参数多于数据量就直接过拟合，模型容量要匹配数据规模（第 06 章主线） |
| "激活函数可有可无，加不加差不多" | 不是"差不多"，是**完全等价于线性模型**。这不是调参问题，是数学事实 |
| "用了 MLP 就不用做特征工程" | 理论上可以，小数据上人工特征往往还更强；MLP 擅长的是数据够多时自动提特征 |
| "加深网络一定提升准确率" | 常见结果是**连训练误差都降不下去**（退化、梯度消失）。真管用的加深要配合初始化、残差、归一化 |

---

## 自测题

> 每题的依据都在**本章**，不需要课外知识。答不上来就回去翻，不用猜。
>
> 标记：`[查]` 翻本章的表或清单 · `[算]` 手算一个具体数字 · `[跑]` 写一两行代码看结果 · `[判]` 判断一个说法对不对

1. `[查]` 本章那张激活函数表里，ReLU、Sigmoid、Tanh 各自的"坑"是什么？隐藏层默认选哪个？
   *怎么确认：本章「最小数学」第 2 节那张表，以及表下面那句"选型口诀"。*

2. `[查]` 万能近似定理保证了什么？它**没有**保证的三件事是哪三件？
   *怎么确认：本章「最小数学」第 3 节，那三个带 ①②③ 的点。*

3. `[算]` `model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))`。
   - (a) 写出 `num_parameters()` 的算式和结果（`W` 是 `(2,8)`、`b` 是 `(8,)`，两层都要算）。
   - (b) 把第一个 `Linear` 改成 `bias=False`，总数变成多少？
   - (c) `ReLU()` 贡献几个参数？
   *怎么确认：本章「工程视角」第 3 节"把每个张量的元素个数加起来"；第 5 节测试表第一行还有 `bias=False` 的对照。*

4. `[算]` `x` 是 `(4,2)`、`W1` 是 `(2,8)`、`b1` 是 `(8,)`、`W2` 是 `(8,1)`、`b2` 是 `(1,)`。
   - (a) 不加激活时，`h = x @ W1 + b1` 和 `y = h @ W2 + b2` 的形状分别是什么？
   - (b) 本章说这两步能合并成一层，合并后那一层的权重矩阵是什么形状？偏置是什么形状？
   *怎么确认：本章「为什么需要它」里把第一行代进第二行的那段推导，以及「最小数学」第 1 节的形状对照。*

5. `[跑]` 跑两行，看 ReLU 在 0 点附近的数值梯度：
   ```python
   h = 1e-5
   print((max(0.0, h) - max(0.0, -h)) / (2 * h))
   ```
   打印出来是几？本章为什么说这个数是"错误的标准答案"？做梯度校验时该怎么避开？
   *怎么确认：本章「工程视角」第 1 节第二条"ReLU 在 0 点不可导"那句，它把这个数写出来了。*

6. `[判]` "不加激活也行——多堆几个 `Linear`，参数更多，模型更强。" 本章同意吗？把两层代进去以后，它等价于什么？
   *怎么确认：本章「为什么需要它」那两行代入推导，以及「常见误解」表的前两条。*

7. `[判]` 训练 `Sequential(Linear(2, 8), ReLU(), Linear(8, 1))` 时把 `W` 和 `b` 全初始化成 0，那 8 个隐藏单元的输出会一样吗？最后会退化成几个单元？
   *怎么确认：本章「工程视角」第 4 节，那条"能复现的失败实验"。*

---

## 延伸阅读

- D2L 4.1–4.3 多层感知机（从零实现 + 简洁实现）、4.7 正向/反向传播 —— 本章主线
- [Michael Nielsen, *Neural Networks and Deep Learning* 第 4 章](http://neuralnetworksanddeeplearning.com/chap4.html) —— 用可视化讲万能近似，无证明，直觉极好
- [Karpathy, *Yes you should understand backprop*](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) —— 为什么"有 autograd"不等于"懂反向传播"
- 下一章：[09 训练技巧：初始化、正则化、数值稳定](09-training-tricks.md) —— 网络搭好了，怎么让它真的训得起来
