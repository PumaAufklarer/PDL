"""骨架自检。

这个文件属于「工程地基」：它保证包能导入、版本号存在、roadmap 承诺的子包都已建好。
Stage 1 之后这里应该只剩最上层的冒烟检查，真正的测试放到 tests/<子包>/ 下。
"""

from __future__ import annotations

import importlib

import pytest
import torch

import pdl

# 与 src/pdl/ 下的目录一一对应，新增子包时同步更新。
SUBPACKAGES = [
    "data",
    "loss",
    "metrics",
    "nn",
    "optim",
    "trainer",
    "utils",
]


def test_version_is_declared() -> None:
    assert pdl.__version__ != ""


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable_and_documented(name: str) -> None:
    module = importlib.import_module(f"pdl.{name}")
    assert module.__doc__, f"pdl.{name} 缺少说明用途的 docstring"


def test_torch_is_available_as_the_tensor_backend() -> None:
    # 全课程的张量运算都建立在 torch 上。顺带把默认 dtype 钉死：
    # 本课程统一 float32，不跟着 numpy 的 float64 走。
    assert torch.zeros(2, 3).dtype == torch.float32
