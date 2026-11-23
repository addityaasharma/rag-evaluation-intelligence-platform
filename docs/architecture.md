# Architecture

```
            ingest                                   ask
 files ─▶ loaders ─▶ chunking ─▶ ChromaStore    question ─▶ Retriever ─▶ generation ─▶ Answer
                                  (embeddings)              │  dense (Chroma)   ▲
                                        ▲                   │  bm25 (in-memory) │
                                        └───────────────────┴─ RRF fusion ──────┘ LLM (LiteLLM)
```

| Module | Responsibility |
| --- | --- |
| `ingestion/` | Load `.pdf`/`.docx`/`.txt`/`.md` with page numbers; split pages into overlapping chunks |
| `embeddings.py` | `Embedder` protocol; local fastembed implementation |
| `vectorstore.py` | Chroma persistence, cosine search |
| `retrieval/` | Dense, BM25 or hybrid retrieval fused with reciprocal rank fusion |
| `generation.py` | Grounded prompt with numbered, cited context |
| `llm.py` | `LLM` protocol; LiteLLM client (any provider) |
| `verification.py` | Claim-level hallucination detector |
| `pipeline.py` | Wires the above; the public `RagPipeline` API |
| `evaluation/` | Dataset, metrics, LLM judges, runner, SQLite run history, configuration sweeps |
| `ui/app.py` | Streamlit app (optional `ui` extra) |
| `api.py`, `cli.py` | FastAPI service and Typer CLI over `RagPipeline` |

## Evaluation

Each example in the JSONL dataset has a question, a reference answer and the source files that hold the answer.

| Metric | Type | Meaning |
| --- | --- | --- |
| `hit_rate` | retrieval | Any relevant source appears in the top-k |
| `recall` | retrieval | Fraction of relevant sources retrieved |
| `precision` | retrieval | Fraction of retrieved chunks from relevant sources |
| `mrr` | retrieval | Reciprocal rank of the first relevant chunk |
| `faithfulness` | LLM judge | Answer is supported by the retrieved context |
| `answer_relevance` | LLM judge | Answer addresses the question |
| `correctness` | LLM judge | Answer matches the reference |

Judge scores are 1–5, normalised to 0–1. Unanswerable examples (empty `relevant_sources`) skip retrieval metrics and test that the system declines to answer. Judge parse failures are logged and excluded from means rather than counted as zero.

`rag eval --min-score metric=value` exits non-zero when a mean falls below a gate, for use in CI. Use `RAG_RETRIEVAL_MODE` to compare dense, BM25 and hybrid on the same dataset.

## History and comparison

`HistoryStore` (SQLite, `.rag/history.db`) records each run with a snapshot of the settings that affect results, so runs of different pipeline versions are comparable. `run_sweep` evaluates every chunk size × embedding model × retrieval mode combination: each (size, model) pair gets its own Chroma collection, indexed once, and chunk overlap scales with size. Without a judge it scores retrieval only and makes no LLM calls.

## Hallucination detection

Two complementary signals. At query time, `HallucinationDetector` splits an answer into claims and checks each against the retrieved context (`ask(..., verify=True)`), listing the unsupported ones. In evaluation, answers whose normalised faithfulness is below 0.5 are flagged as possible hallucinations.

## Limits
- Scanned PDFs have no text layer and are rejected; OCR is not included.
- Verification is an extra LLM call per answer and is itself fallible.
- Relevance is labelled per source file, not per passage, so retrieval metrics are coarse.
- BM25 is built in memory from the whole collection; replace it with a search backend for large corpora.
- LLM judges are noisy; use a stronger or different model family for judging than for generation when possible.
