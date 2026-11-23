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

## Deploy

### Streamlit Community Cloud (free)

1. Create a free Gemini API key at https://aistudio.google.com/apikey.
2. At https://share.streamlit.io choose **Create app**, pick this repository, branch `main`, and main file `streamlit_app.py`.
3. Under **Advanced settings**, select Python 3.12 and paste these secrets:
   ```toml
   GEMINI_API_KEY = "your-key"
   RAG_PUBLIC_DEMO = "1"
   RAG_CHROMA_PATH = "/tmp/chroma"
   RAG_HISTORY_PATH = "/tmp/history.db"
   ```
4. Deploy. The app redeploys on every push to `main`.

`RAG_PUBLIC_DEMO=1` gives each visitor a private index preloaded with the sample documents, caps uploads at 5 MB, chat at 30 questions and evaluation at one run per session, since all visitors share one API key. Free apps sleep when idle and lose uploaded data on restart.

### Docker

```bash
docker build -t rag-platform .
docker run -p 8501:8501 -e GEMINI_API_KEY=your-key rag-platform
```

## Configuration

All settings are `RAG_*` environment variables; see `.env.example` and `ragkit/config.py`. Models are LiteLLM strings, so switching provider is a config change, e.g. `RAG_LLM_MODEL=groq/openai/gpt-oss-20b` with `GROQ_API_KEY`. The default is Gemini Flash-Lite (the `latest` alias, so it survives model retirements) on Google's free tier, which is rate limited; evaluation runs may take a few minutes.

## Development

```bash
make install   # uv sync --all-extras
make check     # ruff, mypy, pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT.
