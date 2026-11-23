import hashlib
from collections.abc import Callable
from pathlib import Path

from ragkit.models import Document, Page


class DocumentLoadError(RuntimeError):
    pass


def _read_text(path: Path) -> list[Page]:
    return [Page(text=path.read_text(encoding="utf-8"))]


def _read_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return [
        Page(number=n, text=page.extract_text() or "") for n, page in enumerate(reader.pages, 1)
    ]


def _read_docx(path: Path) -> list[Page]:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    blocks = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        blocks.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    return [Page(text="\n\n".join(blocks))]


_READERS: dict[str, Callable[[Path], list[Page]]] = {
    ".md": _read_text,
    ".txt": _read_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}
SUPPORTED_SUFFIXES = frozenset(_READERS)


def load_documents(path: Path) -> list[Document]:
    """Load supported files from a file or directory tree.

    `Document.source` is relative to `path` so it stays stable across machines,
    which the evaluation dataset relies on.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    root = path if path.is_dir() else path.parent
    files = [path] if path.is_file() else sorted(path.rglob("*"))

    documents = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES or not file.is_file():
            continue
        try:
            pages = _READERS[suffix](file)
        except Exception as exc:
            raise DocumentLoadError(f"Could not read {file.name}: {exc}") from exc
        pages = [p.model_copy(update={"text": p.text.strip()}) for p in pages if p.text.strip()]
        if not pages:
            continue
        source = file.relative_to(root).as_posix()
        documents.append(
            Document(id=hashlib.sha1(source.encode()).hexdigest()[:16], source=source, pages=pages)
        )
    return documents
