import pytest
from conftest import make_pdf
from docx import Document as DocxDocument

from ragkit.ingestion import load_documents
from ragkit.ingestion.loaders import DocumentLoadError


def test_pdf_pages_are_numbered(tmp_path):
    (tmp_path / "report.pdf").write_bytes(make_pdf(["Alpha findings", "Beta findings"]))
    (doc,) = load_documents(tmp_path)
    assert doc.source == "report.pdf"
    assert [(p.number, p.text) for p in doc.pages] == [(1, "Alpha findings"), (2, "Beta findings")]


def test_docx_includes_paragraphs_and_tables(tmp_path):
    docx = DocxDocument()
    docx.add_paragraph("Intro paragraph")
    table = docx.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Plan"
    table.rows[0].cells[1].text = "Pro"
    docx.save(tmp_path / "notes.docx")

    (doc,) = load_documents(tmp_path)
    assert doc.pages[0].number is None
    assert "Intro paragraph" in doc.pages[0].text
    assert "Plan | Pro" in doc.pages[0].text


def test_text_and_markdown_load_and_blank_files_are_skipped(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.md").write_text("# title")
    (tmp_path / "blank.txt").write_text("  \n")
    (tmp_path / "ignored.csv").write_text("x,y")
    assert [d.source for d in load_documents(tmp_path)] == ["a.txt", "b.md"]


def test_corrupt_file_raises_load_error(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    with pytest.raises(DocumentLoadError, match=r"broken\.pdf"):
        load_documents(tmp_path)


def test_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
