from ragkit.llm import LLM
from ragkit.models import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions using only the numbered context passages provided.
- Cite the passages you rely on as [1], [2], etc.
- If the context does not contain the answer, reply exactly:
  I don't know based on the provided documents.
- Do not use outside knowledge. Be concise."""


def format_context(contexts: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.chunk.source})\n{c.chunk.text}" for i, c in enumerate(contexts, start=1)
    )


def generate_answer(llm: LLM, question: str, contexts: list[RetrievedChunk]) -> str:
    user = f"Context:\n{format_context(contexts)}\n\nQuestion: {question}"
    return llm.complete(SYSTEM_PROMPT, user)
