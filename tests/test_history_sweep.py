from pathlib import Path

import chromadb
from conftest import FakeLLM, HashEmbedder

from ragkit.config import Settings
from ragkit.evaluation import (
    EvalExample,
    HistoryStore,
    SweepGrid,
    config_snapshot,
    load_dataset,
    run_evaluation,
    run_sweep,
)
from ragkit.pipeline import RagPipeline
from ragkit.retrieval import Retriever
from ragkit.vectorstore import ChromaStore

ROOT = Path(__file__).parent.parent
CLIENT = chromadb.EphemeralClient()


def build(settings: Settings) -> RagPipeline:
    store = ChromaStore(HashEmbedder(), settings.collection, client=CLIENT)
    return RagPipeline(settings, store, Retriever(store, mode=settings.retrieval_mode), FakeLLM())


def test_retrieval_only_evaluation_makes_no_llm_calls(store):
    llm = FakeLLM()
    pipeline = RagPipeline(Settings(), store, Retriever(store), llm)
    pipeline.ingest(ROOT / "data" / "sample_docs")
    example = EvalExample(
        question="backups retained", reference_answer="35 days", relevant_sources=["security.md"]
    )
    report = run_evaluation(pipeline, [example], judge=None)
    assert report.summary["hit_rate"] == 1.0
    assert "faithfulness" not in report.summary
    assert llm.calls == []


def test_history_round_trip(tmp_path):
    history = HistoryStore(tmp_path / "nested" / "history.db")
    report = run_report()
    first = history.record("v1", config_snapshot(Settings(), "golden.jsonl"), report)
    second = history.record("v2", config_snapshot(Settings(chunk_size=400), "golden.jsonl"), report)

    runs = history.runs()
    assert [r.id for r in runs] == [second, first]
    assert runs[0].config["chunk_size"] == 400
    assert history.runs(limit=1)[0].label == "v2"
    assert history.report(first).summary == report.summary


def run_report():
    from ragkit.evaluation.runner import EvalReport

    return EvalReport(summary={"hit_rate": 0.5}, examples=[])


def test_sweep_covers_grid_and_records_history(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    grid = SweepGrid(
        chunk_sizes=[200, 800], embedding_models=["hash"], retrieval_modes=["dense", "bm25"]
    )
    results = run_sweep(
        Settings(collection="sweep-test"),
        ROOT / "data" / "sample_docs",
        load_dataset(ROOT / "eval" / "datasets" / "golden.jsonl"),
        grid,
        history=history,
        build=build,
    )

    assert [(r.config["chunk_size"], r.config["retrieval_mode"]) for r in results] == [
        (200, "dense"),
        (200, "bm25"),
        (800, "dense"),
        (800, "bm25"),
    ]
    assert all("hit_rate" in r.summary for r in results)
    assert len(history.runs()) == 4
