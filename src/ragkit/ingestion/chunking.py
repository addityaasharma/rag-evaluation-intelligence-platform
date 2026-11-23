from collections.abc import Iterator

from ragkit.models import Chunk, Document


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into paragraph-aligned chunks of at most `size` characters.

    Consecutive chunks share roughly `overlap` trailing characters (snapped to a
    word boundary) so that facts straddling a boundary remain retrievable.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    texts: list[str] = []
    current = ""
    for unit in _units(document.text, size):
        if current and len(current) + len(unit) + 2 > size:
            texts.append(current)
            current = _tail(current, overlap)
        current = f"{current}\n\n{unit}" if current else unit
    if current and (not texts or current != _tail(texts[-1], overlap)):
        texts.append(current)

    return [
        Chunk(
            id=f"{document.id}:{i}",
            document_id=document.id,
            source=document.source,
            index=i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def _units(text: str, size: int) -> Iterator[str]:
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        for start in range(0, len(paragraph), size):
            piece = paragraph[start : start + size].strip()
            if piece:
                yield piece


def _tail(text: str, overlap: int) -> str:
    if overlap == 0:
        return ""
    tail = text[-overlap:]
    _, sep, rest = tail.partition(" ")
    return rest if sep else tail
