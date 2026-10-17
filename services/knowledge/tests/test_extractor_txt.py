"""Test extracteur texte brut."""
from pathlib import Path

from app.ingestion.extractors import extract, extract_text


def test_extract_text_simple(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("Contenu texte simple.", encoding="utf-8")
    doc = extract_text(p)
    assert doc.filename == "doc.txt"
    assert doc.mime_type == "text/plain"
    assert len(doc.pages) == 1
    assert "Contenu texte simple" in doc.pages[0].text


def test_extract_text_empty(tmp_path: Path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    doc = extract_text(p)
    assert doc.pages == []


def test_extract_dispatch_txt(tmp_path: Path):
    p = tmp_path / "d.txt"
    p.write_text("abc", encoding="utf-8")
    doc = extract(d)
