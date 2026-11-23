# RAG Evaluation & Intelligence Platform

A modular retrieval-augmented generation (RAG) platform (Python package `ragkit`) with a built-in evaluation harness.

- **Documents:** PDF, DOCX, TXT and Markdown, via CLI, API or the web app.
- **Chat:** multi-turn Q&A with hybrid retrieval (dense + BM25), and any LLM provider via LiteLLM.
- **Citations:** each answer cites `[n]` passages, shown with file and page number.
- **Hallucination check:** claim-level verification flags statements the context does not support.
- **Evaluation:** faithfulness, relevancy, precision and recall, with CI gates.
- **Comparison and history:** sweep chunk sizes, embedding models and retrieval modes; every run is stored so versions can be compared over time.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env            # add your provider API key
uv run rag ingest data/sample_docs --reset
uv run rag ask "How long are backups retained?"
```

## Web app

```bash
uv sync --all-extras
uv run rag ui
```

Pages: **Chat** (with a hallucination-check toggle), **Documents** (upload, list, delete) and **Evaluation** (metric dashboard, per-question results, run history, configuration comparison).

## Evaluate

```bash
uv run rag eval --min-score faithfulness=0.8 --min-score hit_rate=0.8
```

Prints a metric table, writes `eval/reports/latest.json` with per-question results, and exits non-zero if a gate fails. See [docs/architecture.md](docs/architecture.md) for the metric definitions.

Every run is saved to `.rag/history.db`. List runs with `uv run rag history`; use `--label` to name a pipeline version and `--no-record` to skip saving.

Compare chunk sizes, embedding models and retrieval modes in one table. This scores retrieval only, so it needs no API key; add `--with-judges` to also score answers:

```bash
uv run rag compare data/sample_docs \
  --chunk-size 400,800 \
  --embedding-model BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2 \
  --retrieval-mode dense,bm25,hybrid
```

To evaluate your own corpus, ingest it and write a JSONL file of `{"question", "reference_answer", "relevant_sources"}` lines, then pass `--dataset`.

## Serve

```bash
uv run rag serve
curl -s -F file=@report.pdf localhost:8000/documents
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which plans support SAML SSO?", "verify": true}'
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
make install   # uv sync --all-extras
make check     # ruff, mypy, pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT.
