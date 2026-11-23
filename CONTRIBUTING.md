# Contributing

## Setup
```bash
uv sync
cp .env.example .env   # only needed for `ask`, `eval` and `serve`
make check             # lint, type-check, tests
```

Tests never call a real LLM or download models; they use the fakes in `tests/conftest.py`.

## Guidelines
- Keep modules small and single-purpose; see [docs/architecture.md](docs/architecture.md).
- New pluggable pieces (embedder, LLM, store) should satisfy the existing `Protocol` rather than add inheritance.
- Add or update tests with every behaviour change. Add eval examples to `eval/datasets/golden.jsonl` when changing retrieval or prompts, and include before/after `rag eval` numbers in the PR.
- Comment only to explain *why*; names and types should carry the *what*.
- Run `make format` before pushing.

## Pull requests
Keep PRs focused, describe the motivation, and make sure CI is green.
