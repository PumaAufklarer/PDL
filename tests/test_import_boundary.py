"""守护「从零」边界的测试。

`docs/roadmap.md` 第 1、4 节规定了 `src/pdl/` 里允许用什么、禁止用什么：
张量运算和 autograd 用 torch 的，但层、损失、优化器、数据管道、训练循环必须自己写。

约定只写在文档里会被忘记，写成测试才会真的被执行。
Stage 7 的「用 torch.nn 重写一遍做对照」放在 `examples/` 下，不在本测试的管辖范围内。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pdl"

# 整个子树都被禁止的模块前缀。
BANNED_PREFIXES = (
    "torch.nn.functional",
    "torch.nn.init",
    "torch.optim",
    "torch.utils.data",
)

# torch.nn 整体禁止，只有这个符号例外：它只是「这个张量要训练」的标记，不是层实现。
NN_ALLOWED_SYMBOLS = frozenset({"Parameter"})


def _source_files() -> list[pathlib.Path]:
    """返回 src/pdl/ 下所有需要检查的 Python 文件。"""
    return sorted(SRC.rglob("*.py"))


def _attribute_chain(node: ast.expr) -> str | None:
    """把 a.b.c 这样的属性链还原成字符串；不是纯名字链就返回 None。"""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _violations(path: pathlib.Path) -> list[str]:
    """检查单个文件，返回违反边界的描述列表；空列表表示通过。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    # 先把 `import torch.nn as nn` 这类别名记下来，
    # 这样后面看到 `nn.Linear` 也能还原成 `torch.nn.Linear`。
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def expand(name: str) -> str:
        head, _, rest = name.partition(".")
        full = aliases.get(head, head)
        return f"{full}.{rest}" if rest else full

    problems: list[str] = []

    def check(name: str) -> None:
        if name == "torch.nn":
            # 只是拿到模块本身没关系；真正的使用会在属性链上被查出来。
            return
        for banned in BANNED_PREFIXES:
            if name == banned or name.startswith(banned + "."):
                problems.append(f"`{name}` 属于禁用模块 `{banned}`，必须自己实现")
                return
        if name.startswith("torch.nn."):
            symbol = name[len("torch.nn.") :].split(".")[0]
            if symbol not in NN_ALLOWED_SYMBOLS:
                problems.append(
                    f"`{name}` 不可用；src/pdl/ 里只允许从 torch.nn 取 "
                    f"{sorted(NN_ALLOWED_SYMBOLS)}，层和模型必须自己写"
                )

    # 属性链只检查最外层，否则 `torch.nn.Parameter` 里的 `torch.nn` 会被误报。
    inner_nodes = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check(expand(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                check(expand(f"{node.module}.{alias.name}"))
        elif isinstance(node, ast.Attribute) and id(node) not in inner_nodes:
            chain = _attribute_chain(node)
            if chain is not None:
                check(expand(chain))

    return problems


def test_source_tree_is_not_empty() -> None:
    # 防呆：如果 SRC 写错了，下面的参数化会变成空集合，守护测试就静默失效了。
    assert _source_files(), f"{SRC} 下没有 .py 文件，检查路径是不是写错了"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_source_respects_the_from_scratch_boundary(path: pathlib.Path) -> None:
    problems = _violations(path)
    assert not problems, (
        f"src/pdl/{path.relative_to(SRC)} 违反「从零」边界：\n"
        + "\n".join(f"  - {p}" for p in problems)
        + "\n  边界定义见 docs/roadmap.md 第 4 节。"
    )


def test_the_guard_actually_catches_violations(tmp_path: pathlib.Path) -> None:
    # 守护测试自己也要被测试，否则它可能只是一个永远通过的空壳。
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import torch\n"
        "from torch.nn import Linear\n"
        "import torch.optim as optim\n"
        "import torch.nn.functional as F\n"
        "x = torch.nn.functional.relu(torch.zeros(3))\n"
        "y = torch.nn.Sequential()\n",
        encoding="utf-8",
    )
    assert len(_violations(sample)) >= 5


def test_the_allowed_forms_pass(tmp_path: pathlib.Path) -> None:
    sample = tmp_path / "allowed.py"
    sample.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.nn import Parameter\n"
        "\n"
        "p = Parameter(torch.zeros(3))\n"
        "q = nn.Parameter(torch.randn(2, 2, requires_grad=True))\n"
        "z = torch.matmul(q, p) + torch.exp(p)\n",
        encoding="utf-8",
    )
    assert _violations(sample) == []
