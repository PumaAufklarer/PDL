# examples

端到端、可以直接跑、跑完能看见数字的脚本。每个 Stage 的交付物会落到这里。

约定：

- 全部支持 `--seed`，同一个 seed 两次运行结果必须完全一致
- 只依赖 `pdl` 与 `numpy`，不引入额外的实验框架
- 打印的内容要能一眼看出"训练有没有在起作用"（loss 在降、准确率在涨）

```
uv run python examples/<name>.py --seed 0
```

当前为空。Stage 1 会在这里落地第一个脚本（`manual_backward.py`）。
