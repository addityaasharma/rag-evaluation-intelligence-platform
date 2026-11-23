from ragkit.llm import LLM
from ragkit.models import ChatMessage, RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions using only the numbered context passages provided.
- Cite the passages you rely on as [1], [2], etc.
- If the context does not contain the answer, reply exactly:
  I don't know based on the provided documents.
- Do not use outside knowledge. Be concise."""

CONDENSE_PROMPT = """\
Rewrite the final user question as a standalone question, resolving pronouns and references
using the conversation. Reply with only the rewritten question."""

_MAX_HISTORY = 6


def format_context(contexts: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({_location(c)})\n{c.chunk.text}" for i, c in enumerate(contexts, start=1)
    )


def _location(context: RetrievedChunk) -> str:
    chunk = context.chunk
    return f"source: {chunk.source}" + (f", page {chunk.page}" if chunk.page else "")


def _transcript(history: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in history[-_MAX_HISTORY:])


def condense_question(llm: LLM, history: list[ChatMessage], question: str) -> str:
    """Make a follow-up question retrievable on its own."""
    prompt = f"Conversation:\n{_transcript(history)}\n\nFinal user question: {question}"
    return llm.complete(CONDENSE_PROMPT, prompt) or question


def generate_answer(
    llm: LLM,
    question: str,
    contexts: list[RetrievedChunk],
    history: list[ChatMessage] | None = None,
) -> str:
    parts = [f"Context:\n{format_context(contexts)}"]
    if history:
        parts.append(f"Conversation so far:\n{_transcript(history)}")
    parts.append(f"Question: {question}")
    return llm.complete(SYSTEM_PROMPT, "\n\n".join(parts))
