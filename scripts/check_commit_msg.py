"""校验 commit message 是否符合本仓库的提交规范。

规范格式::

    <type>(<scope>): <英文小写祈使句 summary> (#<issue-id>)

其中 ``<scope>`` 可省略。完整规则见 CONTRIBUTING.md 第 3 节。用法::

    # 校验一条写好的 message
    python3 scripts/check_commit_msg.py "feat(tensor): support broadcasting in add (#12)"

    # 校验一个文件（pre-commit 的 commit-msg 钩子用这个）
    python3 scripts/check_commit_msg.py --file .git/COMMIT_EDITMSG

    # 批量校验（CI 里校验 PR 上的所有 commit）
    git log --format=%s origin/main..HEAD | python3 scripts/check_commit_msg.py --stdin
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TYPES = ("build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "style", "test")

PATTERN = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[a-z0-9_]+)\))?"
    r": (?P<summary>.+)"
    r" \(#(?P<issue>\d+)\)$"
)

MAX_SUBJECT_LENGTH = 100
MIN_SUMMARY_LENGTH = 20

# git 和工具自己生成的提交不受本规范约束。
SKIP_PREFIXES = ("Merge ", "Revert ", "fixup!", "squash!", "amend!")


def first_line(message: str) -> str:
    """取出 commit message 的第一行，跳过空行和以 # 开头的注释行。"""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def check(message: str) -> list[str]:
    """校验一条 commit message，返回问题描述列表；空列表表示通过。"""
    subject = first_line(message)
    if not subject:
        return ["commit message 是空的"]
    if subject.startswith(SKIP_PREFIXES):
        return []

    problems: list[str] = []
    if len(subject) > MAX_SUBJECT_LENGTH:
        problems.append(f"标题长度 {len(subject)} 超过上限 {MAX_SUBJECT_LENGTH}")

    match = PATTERN.match(subject)
    if match is None:
        problems.append(
            "不符合格式 <type>(<scope>): <小写祈使句 summary> (#<issue-id>)，"
            "例如 feat(tensor): support broadcasting in add (#12)"
        )
        return problems

    commit_type = match.group("type")
    if commit_type not in TYPES:
        problems.append(f"type `{commit_type}` 不在允许的集合里：{', '.join(TYPES)}")

    summary = match.group("summary")
    if len(summary) < MIN_SUMMARY_LENGTH:
        problems.append(
            f"summary 只有 {len(summary)} 个字符，至少 {MIN_SUMMARY_LENGTH} 个"
            "（写清改了什么，别写 update / fix bug）"
        )
    if summary[0].isupper():
        problems.append("summary 不要以大写字母开头，用英文小写祈使句")
    if summary.endswith("."):
        problems.append("summary 结尾不要加句号")
    return problems


def main(argv: list[str] | None = None) -> int:
    """命令行入口：校验一条或一批 commit message，不合规返回 1。"""
    parser = argparse.ArgumentParser(description="校验 commit message 是否符合本仓库规范")
    parser.add_argument("message", nargs="?", help="直接传入一条 commit message")
    parser.add_argument("--file", type=Path, help="从文件读取（pre-commit commit-msg 钩子）")
    parser.add_argument("--stdin", action="store_true", help="从标准输入按行读取多条")
    args = parser.parse_args(argv)

    if args.file is not None:
        messages = [args.file.read_text(encoding="utf-8")]
    elif args.stdin:
        messages = [line for line in sys.stdin.read().splitlines() if line.strip()]
    elif args.message is not None:
        messages = [args.message]
    else:
        parser.error("需要提供 message、--file 或 --stdin 三者之一")

    if not messages:
        print("没有需要校验的 commit message")
        return 0

    failed = 0
    for message in messages:
        subject = first_line(message) or "(空)"
        problems = check(message)
        if problems:
            failed += 1
            print(f"✗ {subject}", file=sys.stderr)
            for problem in problems:
                print(f"    - {problem}", file=sys.stderr)
        else:
            print(f"✓ {subject}")

    if failed:
        print(
            f"\n{failed}/{len(messages)} 条不合规，规范见 CONTRIBUTING.md 第 3 节。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
