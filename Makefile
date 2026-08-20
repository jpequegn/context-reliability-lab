.PHONY: sync lint format test build demo check

sync:
	uv sync --locked --all-groups --no-editable --reinstall-package context-reliability-lab

lint:
	uv run --no-editable ruff check .
	uv run --no-editable ruff format --check .

format:
	uv run --no-editable ruff check --fix .
	uv run --no-editable ruff format .

test:
	uv run --no-editable pytest

build:
	uv build

demo:
	uv run --no-editable context-lab demo --output-dir artifacts/demo

check: sync lint test build
