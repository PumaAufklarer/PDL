# 06 · Softmax 回归：从回归到分类

> **一句话**：分类模型先给每个类别打一个分数，softmax 把分数变成概率，交叉熵衡量这个概率离正确答案有多远——它俩合起来的梯度干净得不可思议：$\hat p - y$。
>
> **对应 D2L**：3.4 softmax 回归、3.5 图像分类数据集、3.6 从零实现、3.7 简洁实现（选做）｜ **对应 Stage**：3 ｜ **对应代码**：`src/pdl/loss/`、`src/pdl/metrics/`、`src/pdl/data/`

## 为什么需要它

Stage 2 的线性回归输出一个数，比如"这套房值 320 万"——要预测的东西本身就是个数，直接比对就行。

现在换任务：给一张衣服图，判断它是 T 恤、裤子还是鞋。你可能会想输出 0/1/2，但不行：**类别之间没有大小关系**（鞋不是"两个 T 恤"），硬让模型逼近 0/1/2 等于塞给它一个不存在的顺序；而模型内部算出来的 $z=Wx+b$ 只是一组可正可负的实数，缺一个把"分数"翻译成"概率"的环节。所以这章的事就三件：给每个类别打分（logits）→ 变成概率（softmax）→ 打分（交叉熵）。名字叫"回归"，干的却是分类，历史遗留叫法，别被带跑。

## 直觉

你在给一道选择题打分。线性层给三个选项分别打了分：A 得 3.2，B 得 1.0，C 得 −0.5。这组原始得分叫 **logits**。

现在要把分数变成"我有多倾向选它"：分数高得多，就该占压倒性优势，所以不能简单归一化；同时**必须都为正、加起来等于 1**，这才叫概率。softmax 做的事情就是：先取指数把差距拉开并变成正的，再除以总和。

交叉熵在做另一件事：**只看正确答案那个位置上的概率**。

- 说是 T 恤的概率 0.9 → 罚 $-\log 0.9 \approx 0.105$；说是 T 恤的概率 0.01 → 罚 $-\log 0.01 \approx 4.6$

它的态度很鲜明：**答错了不怪你，但你在错误答案上还特别自信，就往死里罚。** "我确定这是鞋"结果它是裤子，比"我不太确定"严重得多。

## 最小数学

> 看不懂就跳过，先去做「落到代码」，回来再读。

**1. softmax**

$$\hat p_j = \text{softmax}(z)_j = \frac{\exp(z_j)}{\sum_{k=1}^{C} \exp(z_k)}$$

三个性质全都要进测试：每个 $\hat p_j > 0$ 且 $\sum_j \hat p_j = 1$；保持相对大小（$z_j > z_k \iff \hat p_j > \hat p_k$）；**平移不变**——所有 $z_j$ 加同一个常数，$\hat p$ 不变。最后这条是下一节的关键。

**2. 交叉熵**

$$\ell(z, y) = -\log \hat p_y = -z_y + \log\sum_{j=1}^{C}\exp(z_j) \qquad L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p^{(i)}_{y^{(i)}}$$

$y$ 是正确答案的**整数索引**，$\hat p_y$ 就是从概率向量里取出那一个。和最大似然的关系：模型给的条件概率是 $p(y\mid x) = \hat p_y$，最大化似然 $\prod_i \hat p^{(i)}_{y^{(i)}}$ 就是最大化对数似然，也就是**最小化**上面这个 $L$。所以"用交叉熵"和"用最大似然估计"是同一件事的两种说法。

信息论点到为止：$H(p,q) = H(p) + D_{KL}(p\Vert q)$；标签 $p$ 是 one-hot，所以 $H(p)=0$，交叉熵就等于 KL 散度——最小化它，就是让预测分布靠近真实分布。

**3. log-sum-exp（全章最重要的一步）**

看公式里的 $\log\sum_j \exp(z_j)$：$z_j = 1000$ 时 $\exp(1000)$ 直接溢出成 `inf`，`inf/inf` 得到 `nan`，loss 就废了。其实不用等到 1000：float32 在 $\exp(88)$ 左右就炸，float64 也撑不过 709。

解法是利用平移不变性，把最大值减掉：

$$m = \max_j z_j \qquad \log\sum_j \exp(z_j) = m + \log\sum_j \exp(z_j - m)$$

**为什么结果不变**：括号里每一项都相当于除以 $\exp(m)$，而对数里除掉 $\exp(m)$ 之后，外面正好补一个 $+m$，一进一出抵消。

**为什么不再溢出**：减完之后最大的那一项是 $z_{\max} - m = 0$，于是每项都落在 $(-\infty, 0]$，$\exp$ 的结果在 $(0, 1]$，求和至多 $C$。上溢被彻底消灭。下溢呢？某项比 $m$ 小 800 时 $\exp(-800)$ 变成 0，但它的真实贡献本来就小到可以忽略，累加结果依然是准的。**这就是你要写的实现。**

**4. 梯度就是 $\hat p - y$**

$$\frac{\partial \ell}{\partial z_j} = \hat p_j - y_j$$

把 softmax 和交叉熵**当成一个整体**求导（分开求会冒出一堆中间项），结果就是"预测概率 − 真实标签"：预测得对且自信（$\hat p_y = 1$）梯度为 0；预测 0.9 时梯度 $-0.1$，轻推一下；预测 0.1 时梯度 $-0.9$，狠狠推。

**为什么工程上要合成一个算子**：分开实现的话，你不仅要手写 $C\times C$ 的 softmax 雅可比（极容易在减 $m$ 的地方写错），还要在中间存一次概率、多做一次 $\log$——又慢又不稳。合成之后，前向只算一次 log-sum-exp，反向直接用 $\hat p - y$。

## 落到代码

**交叉熵损失**（注意它收的是 logits，不是概率）：

```python
# src/pdl/loss/cross_entropy.py
class CrossEntropyLoss:
    def __call__(self, logits, target) -> Tensor:
        """logits: (N, C)，未过 softmax 的原始分数；target: (N,)，int 类别索引。
        返回 batch 平均后的标量 Tensor，可以直接 .backward()。
        职责：内部做 log-sum-exp，绝不显式算 exp(z) 再取 log。
        不做：不接受已经 softmax 过的概率；不做 label smoothing / 类别权重（超出本课程范围）。
        """
def softmax(logits) -> Tensor:  # 推理用：分数 → 概率；和 loss 内部共用同一套 log-sum-exp
```

三条接口约定写进 docstring：输入就叫 `logits`（不叫 `probs`、不叫 `pred`）；`target` 是**整数索引**形状 `(N,)`，不是 one-hot；类别数 $C$ 从 `logits.shape[1]` 推出来，不用额外传参。

**指标**（`metrics/` 里一个指标一个函数，签名统一）：

```python
# src/pdl/metrics/accuracy.py
def accuracy(logits, target) -> float:
    """logits: (N, C)；target: (N,)。内部 argmax 再比对，返回 [0, 1]。不做累加状态——
    Trainer 自己维护 correct/total 两个计数器就够了，不必为此造一个类。"""

# src/pdl/metrics/confusion.py
def confusion_matrix(logits, target, num_classes: int) -> torch.Tensor:
    """返回 (C, C) 矩阵：行 = 真实类别，列 = 预测类别，元素是计数。归一化留给调用方：
    想看召回就按行除，想看精确率就按列除。"""
```

**数据集**：

```python
# src/pdl/data/fashion_mnist.py
class FashionMNIST(Dataset):   # 实现第 04 章的 Dataset 接口：__len__ / __getitem__
    def __init__(self, root: str | Path = "~/.cache/pdl", train: bool = True,
                 download: bool = True) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        """返回 (图像, 标签)。图像用 (1,28,28) 还是展平 (784,)，你定，但要写进 docstring。
        标签是 **int64 整数索引**（不是 one-hot），单样本形状 ()，切批后自动拼成 (N,)。"""
```

本地缓存结构（`~/.cache/pdl/fashion-mnist/`，下载一次之后离线可用）：

```
train-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz   # 60000 张
t10k-images-idx3-ubyte.gz    t10k-labels-idx1-ubyte.gz    # 10000 张
```

四个文件都有公开的 MD5，下载后校验；校验失败就删掉重下，别留下半个坏文件让下次更难查。

## 工程视角

**1. 数值稳定性测试（本章的核心测试）**
```python
def test_extreme_logits_stay_finite():
    logits = torch.tensor([[1e4, 0.0, -1e4]], requires_grad=True)
    loss = CrossEntropyLoss()(logits, torch.tensor([0]))
    assert torch.isfinite(loss) and loss.item() < 1e-6   # 不是 inf / nan，且接近 0
    loss.backward()
    assert torch.all(torch.isfinite(logits.grad))        # 梯度里也不能有 nan
```

注意这里没有 `.data`：本仓库**禁止用 `.data`**（它绕过 autograd 的版本计数，容易写出静默错误），要拿数值用 `loss.item()`，要断开计算图用 `detach()`。

再补一个反向的极端例子：logits 是 `[-1e4, 0.0, 1e4]`、正确答案是索引 0，loss 应该是个**有限的大数**而不是 `inf`。关键点：**光断言 loss 有限不够，必须断言梯度也没有 nan**——有 bug 的实现很容易在反向里 `inf - inf`，前向看起来却是好的。

**2. 和参考实现对照**
手写算子必须有大腿可抱。**这里的标准答案就是 `torch.nn.functional`**——注意边界很关键：`src/pdl/` 里禁止用 `torch.nn.functional`（见 `docs/roadmap.md` 第 4 节），但**测试里恰恰应该用它当 oracle**。因为"拿官方实现当标准答案来测自己的实现"才是正确的测试思路，而"拿自己的代码测自己"是无效的。这也是 roadmap Stage 3 的 DoD 里写明的：与 `torch.nn.functional.cross_entropy` 对得上。

```python
import torch.nn.functional as F
z, y = make_random_logits(...)
torch.testing.assert_close(softmax(z), torch.softmax(z, dim=1))
torch.testing.assert_close(CrossEntropyLoss()(z, y), F.cross_entropy(z, y))
# 也可以只对照 log-sum-exp 那一项
torch.testing.assert_close(CrossEntropyLoss()(z, y),
                           (torch.logsumexp(z, dim=1) - z[torch.arange(len(y)), y]).mean())
```

还要做一个"你的稳定版 vs 你的朴素版"的对照：在 $|z| < 10$ 的正常范围里两者对得上，在 $10^4$ 时朴素版溢出、稳定版活着。这个测试把"为什么要减最大值"从口头知识变成可执行的证据。

**3. 测试用例清单**

| 用例 | 断言什么 |
|------|---------|
| 单个样本 `(1, C)` | loss 和手算值对得上 |
| batch size = 1 | 结果和"单个样本"一致（别在 batch 维度上写死假设） |
| 全同类 target | target 全相同时也正常，loss 约等于 $-\log \hat p_y$ |
| 极端 logits $10^4$ | loss 有限、梯度无 nan（见上） |
| 均匀 logits（全 0） | loss 恰好是 $\log C$，$C=10$ 时约 2.3026 |
| target 传成 one-hot | **直接报错**，而不是静默算出一个奇怪的 loss |
| 梯度 = $\hat p - y$ | 用 02 章的数值梯度校验再验一遍，相对误差 < 1e-6 |

倒数第二条尤其重要：这类接口误用会静默产生错误的 loss，是最难查的 bug。宁可让它在第一秒就炸。

**4. 准确率不是万能指标**
Fashion-MNIST 十类大致均衡，accuracy 够用。但换到"999 张正常 + 1 张异常"的数据集，一个永远输出"正常"的废物模型也能拿 99.9%。

所以 `metrics/` 要留好扩展点，规则定死：一个指标一个函数，签名统一是 `f(logits, target, **kwargs)`；不引入 Metric 基类，也不让 Trainer 依赖具体指标（它只调 `accuracy` 打日志）。将来加 `precision` / `recall` / `f1_score` 时只加文件、不改已有文件。类别不均衡时先用 `confusion_matrix` 看清错在哪：是某一类整体认不出来（召回低），还是别的类都被判成它（精确率低）——这两个病的药方完全不同。

**5. 数据下载失败怎么办**
`download=True` 在受限网络下会很干脆地失败，这很正常，不是你的 bug。降级方案**不用改接口**：

```python
# src/pdl/data/synthetic.py
def make_synthetic_classification(n_samples: int = 6000, n_features: int = 20,
                                  n_classes: int = 10, class_sep: float = 1.0,
                                  seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """每类一个高斯簇，返回 (X, y)：X 形状 (n_samples, n_features)，y 是 int 索引。
    约定和真实数据集完全一致，所以训练脚本一行都不用改。"""
```

CI 里跑合成数据（快、稳、离线），你自己的实验跑真实数据；训练脚本**用一个参数选择数据源**，而不是复制两份脚本。这是本章最值钱的工程习惯。

## 常见误解

| 误解 | 实际 |
|------|------|
| "softmax 输出的概率是模型置信度" | 它只是**分数归一化后的相对大小**。模型完全没训练过、甚至输入是噪声时，它照样能输出 0.99。它是排序工具，不是校准过的置信度 |
| "交叉熵只能用于分类" | 它是分布之间的距离，哪儿都能用：语言模型预测下一个词、自编码器的像素分布、知识蒸馏的软标签，全是交叉熵 |
| "先手动 softmax 再传给交叉熵" | 会算两次、数值上更不稳，而且 `CrossEntropyLoss` 应该直接拒绝概率输入。前向多一次 `log`，反向多一层雅可比，纯亏 |
| "准确率到 99% 就一定好" | 先看类别分布。不均衡时准确率会骗人，必须配合混淆矩阵，必要时换 macro-F1 |

## 自测题

> 每题的依据都在**本章**，不需要课外知识。第 4 题是本章的核心——做不出来就回去看「最小数学」第 4 节和「工程视角」第 1 节。
>
> 标记：`[查]` 翻本章的表或清单 · `[算]` 手算一个具体数字 · `[跑]` 写一两行代码看结果 · `[判]` 判断一个说法对不对

1. `[查]` 本章要求把 softmax 的哪几个性质都写进测试？逐条把它们写出来。
   *怎么确认：本章「最小数学」第 1 节，它把"三个性质全都要进测试"列成了三句。*

2. `[查]` `CrossEntropyLoss` 的三条接口约定：输入叫什么名字？`target` 是什么形状、什么类型？类别数 $C$ 从哪里来？
   *怎么确认：本章「落到代码」里 `cross_entropy.py` 那个代码块后面的三条。*

3. `[算]` 某次预测 $\hat p = [0.2, 0.7, 0.1]$，正确答案是索引 1。
   - (a) 交叉熵要用 $\hat p$ 的哪一个分量？
   - (b) 算出这个 loss 的数值（$\ln 0.7 \approx -0.357$）。
   - (c) 如果正确答案换成索引 0，loss 会变大还是变小？
   *怎么确认：本章「最小数学」第 2 节的公式，以及「直觉」里 $-\log 0.9 \approx 0.105$、$-\log 0.01 \approx 4.6$ 那两个例子。*

4. `[算]` 还是 $\hat p = [0.2, 0.7, 0.1]$、正确答案是索引 1。
   - (a) 写出 $\partial \ell / \partial z$（也就是 $\hat p - y$）这三个数。
   - (b) 这三个数加起来是多少？
   - (c) 换任何一组 $\hat p$、任何一个标签，这个和会变吗？为什么？
   *怎么确认：本章「最小数学」第 4 节给的 $\hat p_j - y_j$；第 1 节说过 $\sum_j \hat p_j = 1$，而 $y$ 是 one-hot。*

5. `[跑]` 跑下面三行，看 $10^4$ 这种 logits 会出什么：
   ```python
   import torch
   z = torch.tensor([1e4, 0.0, -1e4])
   print((torch.exp(z) / torch.exp(z).sum())[0])   # 朴素 softmax 的那个分量
   print(torch.logsumexp(z, dim=0) - z[0])         # 本章的 log-sum-exp
   ```
   两个数分别是什么？哪一个不是数（`nan`）？它是在哪一步产生的？
   *怎么确认：本章「最小数学」第 3 节，$z_j = 1000$ 那段，以及"float32 在 $\exp(88)$ 左右就炸"那句。*

6. `[判]` "为了看清楚概率，我先自己 softmax，再把概率传给 `CrossEntropyLoss`，反正结果一样。" 本章同意吗？问题出在哪两处？
   *怎么确认：本章「常见误解」里"先手动 softmax 再传给交叉熵"那一条，以及「最小数学」第 4 节"为什么工程上要合成一个算子"。*

7. `[判]` "模型压根没训练过，喂一张纯噪声图，它给某一类输出 0.99——说明模型对这张图很有把握。" 本章怎么说？
   *怎么确认：本章「常见误解」第一条，关键词是"排序工具"和"校准过的置信度"。*

## 延伸阅读

- D2L 3.4 softmax 回归 / 3.5 图像分类数据集 / 3.6 从零实现 —— 本章知识来源，3.6 的实现路径和这里一致
- [Eli Bendersky, The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) —— 想把那堆偏导自己推一遍时看这个，推导非常清楚
- [torch.logsumexp](https://pytorch.org/docs/stable/generated/torch.logsumexp.html) —— 测试里的参考实现，注意看它为什么自带 max 偏移
- 下一章：[08 多层感知机](08-mlp.md) —— 把线性分类器堆成能画非线性边界的模型
