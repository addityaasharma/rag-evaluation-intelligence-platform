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


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal text PDF, one string per page, so tests need no PDF library to write."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    for i, text in enumerate(pages):
        page_id, content_id = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    objects.insert(2, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)
