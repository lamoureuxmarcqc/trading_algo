"""Tests des extracteurs."""
from pathlib import Path

import pytest

from app.ingestion.extractors import (
    ExtractionError,
    extract,
    extract_html,
    extract_markdown,
)


def test_extract_unknown_format(tmp_path: Path):
    bad = tmp_path / "file.xyz"
    bad.write_text("hello")
    with pytest.raises(ExtractionError, match="non supporte"):
        extract(bad)


def test_extract_missing_file(tmp_path: Path):
    with pytest.raises(ExtractionError, match="introuvable"):
        extract(tmp_path / "missing.pdf")


def test_extract_markdown_with_headings(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Introduction\n\nTexte 1.\n\n## Section A\n\nTexte 2.\n\n## Section B\n\nTexte 3."
    )

    doc = extract_markdown(md)
    assert doc.filename == "doc.md"
    assert doc.mime_type == "text/markdown"
    assert len(doc.pages) == 3
    assert doc.pages[0].section == "Introduction"
    assert doc.pages[1].section == "Section A"
    assert "Texte 2" in doc.pages[1].text


def test_extract_markdown_no_headings(tmp_path: Path):
    md = tmp_path / "plain.md"
    md.write_text("Juste du texte.\n\nUn autre paragraphe.")

    doc = extract_markdown(md)
    assert len(doc.pages) == 1
    assert doc.pages[0].section is None


def test_extract_html_strips_scripts(tmp_path: Path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><script>alert(1)</script></head>"
        "<body><h1>Titre</h1><p>Contenu utile.</p>"
        "<footer>Footer</footer></body></html>"
    )

    doc = extract_html(html)
    text = doc.full_text
    assert "Titre" in text
    assert "Contenu utile" in text
    assert "alert(1)" not in text
    assert "Footer" not in text