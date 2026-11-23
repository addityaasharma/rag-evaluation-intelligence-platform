.PHONY: install lint format typecheck test check ingest eval serve

install:
	uv sync

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff check --fix . && uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

ingest:
	uv run rag ingest data/sample_docs --reset

eval:
	uv run rag eval --min-score faithfulness=0.8 --min-score hit_rate=0.8

serve:
	uv run rag serve
