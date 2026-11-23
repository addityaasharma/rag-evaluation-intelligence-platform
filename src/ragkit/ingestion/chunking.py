from collections.abc import Iterator

from ragkit.models import Chunk, Document


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into paragraph-aligned chunks of at most `size` characters.

    Pages are chunked independently so every chunk maps to exactly one page.
    Consecutive chunks within a page share roughly `overlap` trailing characters
    (snapped to a word boundary) so facts straddling a boundary stay retrievable.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    for page in document.pages:
        for text in _split(page.text, size, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document.id}:{index}",
                    document_id=document.id,
                    source=document.source,
                    index=index,
                    text=text,
                    page=page.number,
                )
            )
    return chunks


def _split(text: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for unit in _units(text, size):
        if current and len(current) + len(unit) + 2 > size:
            pieces.append(current)
            current = _tail(current, overlap)
        current = f"{current}\n\n{unit}" if current else unit
    if current and (not pieces or current != _tail(pieces[-1], overlap)):
        pieces.append(current)
    return pieces


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
