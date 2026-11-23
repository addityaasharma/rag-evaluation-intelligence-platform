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
cp .env.example .env            # add a free Gemini key: https://aistudio.google.com/apikey
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

## Deploy to Hugging Face Spaces

The app runs as a Docker Space (free tier). Each visitor gets a private index with the sample documents loaded; uploads are capped at 5 MB, chat at 30 questions and evaluation at one run per session, because all visitors share one API key.

1. Create a free Gemini API key at https://aistudio.google.com/apikey.
2. On huggingface.co, create a **Docker** Space (blank template, public).
3. In the Space settings, add a secret `GEMINI_API_KEY`.
4. In this GitHub repo, add a secret `HF_TOKEN` (a Hugging Face token with write access) and a variable `HF_SPACE` (`your-username/space-name`).
5. Run the **Deploy to Hugging Face Space** workflow. It also runs on every push to `main`.

Free Spaces sleep when idle and lose uploaded data on restart. Set `RAG_PUBLIC_DEMO=0` to turn off the demo limits on a private deployment.

## Configuration

All settings are `RAG_*` environment variables; see `.env.example` and `ragkit/config.py`. Models are LiteLLM strings, so switching provider is a config change, e.g. `RAG_LLM_MODEL=groq/openai/gpt-oss-20b` with `GROQ_API_KEY`. The default is Gemini Flash-Lite (the `latest` alias, so it survives model retirements) on Google's free tier, which is rate limited; evaluation runs may take a few minutes.

## Development

```bash
make install   # uv sync --all-extras
make check     # ruff, mypy, pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT.
