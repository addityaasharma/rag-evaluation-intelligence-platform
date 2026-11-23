import pytest

from ragkit.ingestion.chunking import chunk_document
from ragkit.models import Document


def doc(text: str) -> Document:
    return Document(id="doc", source="doc.md", text=text)


def test_short_document_is_single_chunk():
    chunks = chunk_document(doc("Hello world."), size=100, overlap=10)
    assert [c.text for c in chunks] == ["Hello world."]
    assert chunks[0].id == "doc:0"


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph number {i} with some words." for i in range(30))
    chunks = chunk_document(doc(text), size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_oversized_paragraph_is_split():
    chunks = chunk_document(doc("x" * 250), size=100, overlap=0)
    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_overlap_repeats_trailing_words():
    text = "\n\n".join(["alpha beta gamma delta", "epsilon zeta eta theta"])
    chunks = chunk_document(doc(text), size=30, overlap=12)
    assert len(chunks) == 2
    assert "delta" in chunks[1].text


def test_no_content_is_lost():
    paragraphs = [f"unique-token-{i} filler words here" for i in range(20)]
    chunks = chunk_document(doc("\n\n".join(paragraphs)), size=90, overlap=15)
    joined = " ".join(c.text for c in chunks)
    assert all(f"unique-token-{i}" in joined for i in range(20))


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(doc("text"), size=10, overlap=10)
