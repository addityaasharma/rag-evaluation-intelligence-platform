import hashlib
import math
import re
import uuid
from collections.abc import Callable

import chromadb
import pytest

from ragkit.models import Chunk, RetrievedChunk
from ragkit.vectorstore import ChromaStore

DIM = 256


class HashEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeLLM:
    def __init__(self, reply: str | Callable[[str, str], str] = "stub answer [1]") -> None:
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._reply(system, user) if callable(self._reply) else self._reply


@pytest.fixture
def store() -> ChromaStore:
    return ChromaStore(
        HashEmbedder(), collection=f"test-{uuid.uuid4().hex}", client=chromadb.EphemeralClient()
    )


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    def _make(id_: str, text: str = "text", source: str = "a.md") -> Chunk:
        return Chunk(id=id_, document_id="d", source=source, index=0, text=text)

    return _make


@pytest.fixture
def ranked(make_chunk: Callable[..., Chunk]) -> Callable[..., list[RetrievedChunk]]:
    def _ranked(*ids: str) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=make_chunk(i), score=0.0) for i in ids]

    return _ranked
