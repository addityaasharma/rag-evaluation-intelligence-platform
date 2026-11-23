import logging

from ragkit.generation import format_context
from ragkit.llm import LLM, extract_json_object
from ragkit.models import Claim, RetrievedChunk, Verification

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You check whether an answer is supported by the given context.
Split the answer into atomic factual claims and judge each one against the context only,
ignoring your own knowledge. Skip citation markers and statements that information is unavailable.
Respond with only a JSON object:
{"claims": [{"text": "<claim>", "supported": true or false}]}"""


class VerificationError(RuntimeError):
    pass


class HallucinationDetector:
    """Claim-level check that an answer is grounded in the retrieved context."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def check(self, answer: str, contexts: list[RetrievedChunk]) -> Verification:
        raw = self._llm.complete(
            SYSTEM_PROMPT, f"Context:\n{format_context(contexts)}\n\nAnswer: {answer}"
        )
        try:
            claims = extract_json_object(raw)["claims"]
            return Verification(
                claims=[Claim(text=str(c["text"]), supported=bool(c["supported"])) for c in claims]
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise VerificationError(f"Malformed verifier output: {raw!r}") from exc
