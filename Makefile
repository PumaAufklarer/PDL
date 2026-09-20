# make check 会依次执行下面几步，任何一步失败都应视为"改动不可信"。
.PHONY: help sync fmt lint type test test-cov check msg clean

help:
	@echo "make sync      安装/同步依赖（含 dev 依赖）"
	@echo "make fmt       自动格式化并修复可自动修复的 lint 问题"
	@echo "make lint      只检查、不修改：格式 + lint"
	@echo "make type      运行 mypy 类型检查"
	@echo "make test      运行测试"
	@echo "make test-cov  运行测试并输出覆盖率"
	@echo "make check     提 PR 前必须通过：lint + type + test"
	@echo "make msg       校验一条 commit message，用法：make msg M=\"feat(tensor): ... (#12)\""
	@echo "make clean     清理本地缓存与构建产物"

sync:
	uv sync --all-groups

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run mypy

test:
	uv run pytest

test-cov:
	uv run pytest --cov=pdl --cov-report=term-missing --cov-fail-under=90

check: lint type test

msg:
	@python3 scripts/check_commit_msg.py "$(M)"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
