import hashlib
from pathlib import Path

from ragkit.models import Document

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})


def load_documents(path: Path) -> list[Document]:
    """Load supported text files from a file or directory tree.

    `Document.source` is relative to `path` so it stays stable across machines,
    which the evaluation dataset relies on.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    root = path if path.is_dir() else path.parent
    files = [path] if path.is_file() else sorted(path.rglob("*"))

    documents = []
    for file in files:
        if file.suffix.lower() not in SUPPORTED_SUFFIXES or not file.is_file():
            continue
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        source = file.relative_to(root).as_posix()
        documents.append(
            Document(id=hashlib.sha1(source.encode()).hexdigest()[:16], source=source, text=text)
        )
    return documents
