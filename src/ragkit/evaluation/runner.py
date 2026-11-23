import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean

from pydantic import BaseModel

from ragkit.evaluation.dataset import EvalExample
from ragkit.evaluation.judges import JudgeError, LLMJudge
from ragkit.evaluation.metrics import retrieval_scores
from ragkit.pipeline import RagPipeline

logger = logging.getLogger(__name__)


class ExampleResult(BaseModel):
    question: str
    answer: str
    reference_answer: str
    retrieved_sources: list[str]
    scores: dict[str, float | None]


class EvalReport(BaseModel):
    summary: dict[str, float]
    examples: list[ExampleResult]

    def failing(self, thresholds: dict[str, float]) -> dict[str, float]:
        """Metrics whose mean is below its threshold (missing metrics count as 0)."""
        return {
            name: self.summary.get(name, 0.0)
            for name, minimum in thresholds.items()
            if self.summary.get(name, 0.0) < minimum
        }


def run_evaluation(
    pipeline: RagPipeline,
    examples: list[EvalExample],
    judge: LLMJudge,
    *,
    workers: int = 4,
    top_k: int | None = None,
) -> EvalReport:
    def evaluate(example: EvalExample) -> ExampleResult:
        return _evaluate_example(pipeline, judge, example, top_k)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, examples))
    return EvalReport(summary=_summarise(results), examples=results)


def _evaluate_example(
    pipeline: RagPipeline, judge: LLMJudge, example: EvalExample, top_k: int | None
) -> ExampleResult:
    answer = pipeline.ask(example.question, top_k)
    sources = [c.chunk.source for c in answer.contexts]

    scores: dict[str, float | None] = {}
    if example.relevant_sources:
        scores.update(retrieval_scores(sources, set(example.relevant_sources)))

    judgements: dict[str, Callable[[], float]] = {
        "faithfulness": lambda: judge.faithfulness(answer.text, answer.contexts),
        "answer_relevance": lambda: judge.answer_relevance(example.question, answer.text),
        "correctness": lambda: judge.correctness(
            example.question, answer.text, example.reference_answer
        ),
    }
    for name, grade in judgements.items():
        try:
            scores[name] = grade()
        except JudgeError:
            logger.warning("Judge failed for %s on %r", name, example.question, exc_info=True)
            scores[name] = None

    return ExampleResult(
        question=example.question,
        answer=answer.text,
        reference_answer=example.reference_answer,
        retrieved_sources=sources,
        scores=scores,
    )


def _summarise(results: list[ExampleResult]) -> dict[str, float]:
    names = sorted({name for r in results for name in r.scores})
    summary = {}
    for name in names:
        values = [v for r in results if (v := r.scores.get(name)) is not None]
        if values:
            summary[name] = round(fmean(values), 4)
    return summary
