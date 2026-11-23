import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean

from pydantic import BaseModel

from ragkit.evaluation.dataset import EvalExample
from ragkit.evaluation.judges import JudgeError, LLMJudge
from ragkit.evaluation.metrics import retrieval_scores
from ragkit.models import RetrievedChunk
from ragkit.pipeline import RagPipeline

logger = logging.getLogger(__name__)

# Normalised faithfulness below this marks an answer as a likely hallucination.
HALLUCINATION_THRESHOLD = 0.5


class ExampleResult(BaseModel):
    question: str
    answer: str
    reference_answer: str
    retrieved_sources: list[str]
    scores: dict[str, float | None]

    @property
    def possible_hallucination(self) -> bool:
        faithfulness = self.scores.get("faithfulness")
        return faithfulness is not None and faithfulness < HALLUCINATION_THRESHOLD


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
    judge: LLMJudge | None,
    *,
    workers: int = 4,
    top_k: int | None = None,
) -> EvalReport:
    """Evaluate a pipeline. With `judge=None` only retrieval is scored (no LLM calls)."""

    def evaluate(example: EvalExample) -> ExampleResult:
        return _evaluate_example(pipeline, judge, example, top_k)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, examples))
    return EvalReport(summary=_summarise(results), examples=results)


def _evaluate_example(
    pipeline: RagPipeline, judge: LLMJudge | None, example: EvalExample, top_k: int | None
) -> ExampleResult:
    contexts: list[RetrievedChunk]
    if judge is None:
        contexts, text = pipeline.retrieve(example.question, top_k), ""
    else:
        answer = pipeline.ask(example.question, top_k)
        contexts, text = answer.contexts, answer.text
    sources = [c.chunk.source for c in contexts]

    scores: dict[str, float | None] = {}
    if example.relevant_sources:
        scores.update(retrieval_scores(sources, set(example.relevant_sources)))
    if judge is not None:
        scores.update(_judge_scores(judge, example, text, contexts))

    return ExampleResult(
        question=example.question,
        answer=text,
        reference_answer=example.reference_answer,
        retrieved_sources=sources,
        scores=scores,
    )


def _judge_scores(
    judge: LLMJudge, example: EvalExample, text: str, contexts: list[RetrievedChunk]
) -> dict[str, float | None]:
    judgements: dict[str, Callable[[], float]] = {
        "faithfulness": lambda: judge.faithfulness(text, contexts),
        "answer_relevance": lambda: judge.answer_relevance(example.question, text),
        "correctness": lambda: judge.correctness(example.question, text, example.reference_answer),
    }
    scores: dict[str, float | None] = {}
    for name, grade in judgements.items():
        try:
            scores[name] = grade()
        except JudgeError:
            logger.warning("Judge failed for %s on %r", name, example.question, exc_info=True)
            scores[name] = None
    return scores


def _summarise(results: list[ExampleResult]) -> dict[str, float]:
    names = sorted({name for r in results for name in r.scores})
    summary = {}
    for name in names:
        values = [v for r in results if (v := r.scores.get(name)) is not None]
        if values:
            summary[name] = round(fmean(values), 4)
    return summary
