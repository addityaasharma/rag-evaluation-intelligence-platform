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
| `ingestion/` | Load `.md`/`.txt`, split into paragraph-aligned overlapping chunks |
| `embeddings.py` | `Embedder` protocol; local fastembed implementation |
| `vectorstore.py` | Chroma persistence, cosine search |
| `retrieval/` | Dense, BM25 or hybrid retrieval fused with reciprocal rank fusion |
| `generation.py` | Grounded prompt with numbered, cited context |
| `llm.py` | `LLM` protocol; LiteLLM client (any provider) |
| `pipeline.py` | Wires the above; the public `RagPipeline` API |
| `evaluation/` | Dataset, retrieval metrics, LLM judges, runner and CI gates |
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

## Limits
- Relevance is labelled per source file, not per passage, so retrieval metrics are coarse.
- BM25 is built in memory from the whole collection; replace it with a search backend for large corpora.
- LLM judges are noisy; use a stronger or different model family for judging than for generation when possible.
