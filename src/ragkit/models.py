import re
from typing import Literal

from pydantic import BaseModel


class Page(BaseModel):
    # None for formats without a stable page concept (txt, md, docx).
    number: int | None = None
    text: str


class Document(BaseModel):
    id: str
    source: str
    pages: list[Page]


class Chunk(BaseModel):
    id: str
    document_id: str
    source: str
    index: int
    text: str
    page: int | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Claim(BaseModel):
    text: str
    supported: bool


class Verification(BaseModel):
    claims: list[Claim]

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]

    @property
    def has_hallucination(self) -> bool:
        return bool(self.unsupported)


class Answer(BaseModel):
    question: str
    text: str
    contexts: list[RetrievedChunk]
    verification: Verification | None = None

    @property
    def cited_indices(self) -> list[int]:
        """1-based positions in `contexts` that the answer cites as [n]."""
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]", self.text)}
        return sorted(n for n in numbers if 1 <= n <= len(self.contexts))
