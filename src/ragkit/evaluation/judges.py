import json
import re

from ragkit.generation import format_context
from ragkit.llm import LLM
from ragkit.models import RetrievedChunk

JUDGE_SYSTEM = """\
You are a strict evaluator of question-answering systems.
Grade the answer on a 1-5 integer scale using the criterion given.
1 is the worst and 5 is the best.
Respond with only a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""

_CRITERIA = {
    "faithfulness": (
        "Every claim in the answer is supported by the context. "
        "Penalise any statement the context does not back up. "
        "An answer that says the information is unavailable is faithful if the context lacks it."
    ),
    "answer_relevance": (
        "The answer directly and completely addresses the question, without irrelevant content."
    ),
    "correctness": (
        "The answer agrees with the reference answer in substance. "
        "Differences in wording are fine; missing or contradicting facts are not."
    ),
}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeError(RuntimeError):
    pass


class LLMJudge:
    """LLM-as-judge scoring, normalised to [0, 1]."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def faithfulness(self, answer: str, contexts: list[RetrievedChunk]) -> float:
        return self._grade(
            "faithfulness", f"Context:\n{format_context(contexts)}\n\nAnswer: {answer}"
        )

    def answer_relevance(self, question: str, answer: str) -> float:
        return self._grade("answer_relevance", f"Question: {question}\n\nAnswer: {answer}")

    def correctness(self, question: str, answer: str, reference: str) -> float:
        return self._grade(
            "correctness",
            f"Question: {question}\n\nReference answer: {reference}\n\nAnswer: {answer}",
        )

    def _grade(self, criterion: str, payload: str) -> float:
        prompt = f"Criterion: {_CRITERIA[criterion]}\n\n{payload}"
        return parse_score(self._llm.complete(JUDGE_SYSTEM, prompt))


def parse_score(raw: str) -> float:
    match = _JSON_OBJECT.search(raw)
    if not match:
        raise JudgeError(f"No JSON in judge output: {raw!r}")
    try:
        score = int(json.loads(match.group())["score"])
    except (ValueError, KeyError, TypeError) as exc:
        raise JudgeError(f"Malformed judge output: {raw!r}") from exc
    if not 1 <= score <= 5:
        raise JudgeError(f"Score out of range: {score}")
    return (score - 1) / 4
