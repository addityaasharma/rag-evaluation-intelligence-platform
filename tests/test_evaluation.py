from pathlib import Path

import pytest
from conftest import FakeLLM

from ragkit.config import Settings
from ragkit.evaluation import EvalExample, LLMJudge, load_dataset, run_evaluation
from ragkit.evaluation.judges import JudgeError, parse_score
from ragkit.evaluation.metrics import retrieval_scores
from ragkit.pipeline import RagPipeline
from ragkit.retrieval import Retriever

ROOT = Path(__file__).parent.parent


def test_retrieval_scores():
    scores = retrieval_scores(["x.md", "a.md", "a.md"], {"a.md", "b.md"})
    assert scores == {"hit_rate": 1.0, "recall": 0.5, "precision": 2 / 3, "mrr": 0.5}


def test_retrieval_scores_miss():
    scores = retrieval_scores(["x.md"], {"a.md"})
    assert scores == {"hit_rate": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"score": 5, "reason": "ok"}', 1.0),
        ('Sure!\n```json\n{"score": 1, "reason": "bad"}\n```', 0.0),
        ('{"score": 3}', 0.5),
    ],
)
def test_parse_score(raw, expected):
    assert parse_score(raw) == expected


@pytest.mark.parametrize("raw", ["no json", '{"score": 9}', '{"score": "high"}', '{"reason": "x"}'])
def test_parse_score_rejects_bad_output(raw):
    with pytest.raises(JudgeError):
        parse_score(raw)


def test_golden_dataset_is_valid():
    examples = load_dataset(ROOT / "eval" / "datasets" / "golden.jsonl")
    assert len(examples) >= 10
    sources = {p.name for p in (ROOT / "data" / "sample_docs").iterdir()}
    for example in examples:
        assert set(example.relevant_sources) <= sources


def test_run_evaluation_end_to_end(store):
    pipeline = RagPipeline(Settings(), store, Retriever(store), FakeLLM("The limit is 600 [1]."))
    pipeline.ingest(ROOT / "data" / "sample_docs")

    judge = LLMJudge(FakeLLM('{"score": 5, "reason": "good"}'))
    examples = [
        EvalExample(
            question="How many requests per minute does the Pro plan allow?",
            reference_answer="600",
            relevant_sources=["rate_limits.md"],
        ),
        EvalExample(question="Does it offer GPUs?", reference_answer="unknown"),
    ]
    report = run_evaluation(pipeline, examples, judge, workers=2)

    assert report.summary["faithfulness"] == 1.0
    assert report.summary["hit_rate"] == 1.0
    assert "hit_rate" not in report.examples[1].scores
    assert report.failing({"correctness": 0.9}) == {}
    assert report.failing({"mrr": 1.1}) == {"mrr": report.summary["mrr"]}


def test_judge_failure_is_recorded_not_fatal(store):
    pipeline = RagPipeline(Settings(), store, Retriever(store), FakeLLM())
    pipeline.ingest(ROOT / "data" / "sample_docs")
    judge = LLMJudge(FakeLLM("not json"))
    report = run_evaluation(pipeline, [EvalExample(question="q", reference_answer="a")], judge)
    assert report.examples[0].scores["faithfulness"] is None
    assert "faithfulness" not in report.summary
