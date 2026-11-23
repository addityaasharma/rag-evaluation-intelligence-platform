# ragkit

A modular retrieval-augmented generation (RAG) toolkit with a built-in evaluation harness. Hybrid retrieval (dense + BM25), grounded and cited answers, any LLM provider via LiteLLM, and CI-gateable quality scores.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env            # add your provider API key
uv run rag ingest data/sample_docs --reset
uv run rag ask "How long are backups retained?"
```

## Evaluate

```bash
uv run rag eval --min-score faithfulness=0.8 --min-score hit_rate=0.8
```

Prints a metric table, writes `eval/reports/latest.json` with per-question results, and exits non-zero if a gate fails. See [docs/architecture.md](docs/architecture.md) for the metric definitions.

Compare retrieval strategies on the same dataset:

```bash
RAG_RETRIEVAL_MODE=dense  uv run rag eval
RAG_RETRIEVAL_MODE=bm25   uv run rag eval
RAG_RETRIEVAL_MODE=hybrid uv run rag eval
```

To evaluate your own corpus, ingest it and write a JSONL file of `{"question", "reference_answer", "relevant_sources"}` lines, then pass `--dataset`.

## Serve

```bash
uv run rag serve
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which plans support SAML SSO?"}'
```

## Use as a library

```python
from pathlib import Path
from ragkit import build_pipeline

pipeline = build_pipeline()
pipeline.ingest(Path("data/sample_docs"))
print(pipeline.ask("What is the Pro plan rate limit?").text)
```

## Configuration

All settings are `RAG_*` environment variables; see `.env.example` and `ragkit/config.py`. Models are LiteLLM strings, so switching provider is a config change, e.g. `RAG_LLM_MODEL=openai/gpt-4o`.

## Development

```bash
make check   # ruff, mypy, pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT.
