.PHONY: sync lint format test build check

sync:
	uv sync --locked --all-groups --no-editable

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

build:
	uv build

check: sync lint test build
