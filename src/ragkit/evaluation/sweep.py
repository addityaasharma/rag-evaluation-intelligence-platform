import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragkit.config import RetrievalMode, Settings
from ragkit.evaluation.dataset import EvalExample
from ragkit.evaluation.history import HistoryStore, config_snapshot
from ragkit.evaluation.judges import LLMJudge
from ragkit.evaluation.runner import run_evaluation
from ragkit.pipeline import RagPipeline, build_pipeline


@dataclass(frozen=True)
class SweepGrid:
    chunk_sizes: list[int]
    embedding_models: list[str]
    retrieval_modes: list[RetrievalMode]


class SweepResult(BaseModel):
    config: dict[str, Any]
    summary: dict[str, float]


def run_sweep(
    base: Settings,
    docs: Path,
    examples: list[EvalExample],
    grid: SweepGrid,
    *,
    judge: LLMJudge | None = None,
    history: HistoryStore | None = None,
    build: Callable[[Settings], RagPipeline] = build_pipeline,
    workers: int = 4,
) -> list[SweepResult]:
    """Evaluate every combination in `grid` against the same documents and dataset.

    Each (chunk size, embedding model) pair gets its own collection and is indexed
    once; retrieval modes reuse it. Chunk overlap scales with chunk size.
    With `judge=None` only retrieval metrics are computed, which costs no LLM calls.
    """
    results = []
    indexed: set[str] = set()
    for size, model, mode in product(grid.chunk_sizes, grid.embedding_models, grid.retrieval_modes):
        index_key = hashlib.sha1(f"{size}|{model}".encode()).hexdigest()[:8]
        settings = Settings(
            **{
                **base.model_dump(),
                "chunk_size": size,
                "chunk_overlap": size // 8,
                "embedding_model": model,
                "retrieval_mode": mode,
                "collection": f"{base.collection}-{index_key}",
            }
        )
        pipeline = build(settings)
        if index_key not in indexed:
            pipeline.ingest(docs, reset=True)
            indexed.add(index_key)

        report = run_evaluation(pipeline, examples, judge, workers=workers)
        config = config_snapshot(settings, dataset="sweep")
        if history:
            label = f"sweep size={size} embedding={model} mode={mode}"
            history.record(label, config, report)
        results.append(SweepResult(config=config, summary=report.summary))
    return results
