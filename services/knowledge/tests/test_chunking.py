"""Tests du chunking semantique."""
from app.chunking.semantic import Chunk, chunk_pages
from app.ingestion.extractors import ExtractedPage


def test_chunk_small_text():
    pages = [ExtractedPage(text="Court texte.", page=1)]
    chunks = chunk_pages(pages, target_tokens=400)

    assert len(chunks) == 1
    assert chunks[0].text == "Court texte."
    assert chunks[0].page == 1


def test_chunk_respects_target_size():
    # Genere un long texte
    paragraph = "Mot " * 500  # ~500 tokens par paragraphe
    pages = [ExtractedPage(text=paragraph, page=1)]

    chunks = chunk_pages(pages, target_tokens=200)
    assert len(chunks) > 1
    # Aucun chunk ne doit depasser ~1.5x la cible
    for c in chunks:
        assert c.token_count < 400


def test_chunk_preserves_page():
    pages = [
        ExtractedPage(text="Page 1 contenu", page=1),
        ExtractedPage(text="Page 2 contenu", page=2),
    ]
    chunks = chunk_pages(pages, target_tokens=400)

    assert any(c.page == 1 for c in chunks)
    assert any(c.page == 2 for c in chunks)


def test_chunk_preserves_section():
    pages = [
        ExtractedPage(text="Contenu A", page=1, section="Intro"),
        ExtractedPage(text="Contenu B", page=2, section="Body"),
    ]
    chunks = chunk_pages(pages, target_tokens=400)

    sections = {c.section for c in chunks}
    assert "Intro" in sections
    assert "Body" in sections


def test_chunk_empty_page_skipped():
    pages = [
        ExtractedPage(text="", page=1),
        ExtractedPage(text="Contenu", page=2),
    ]
    chunks = chunk_pages(pages, target_tokens=400)
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_chunk_returns_chunk_objects():
    pages = [ExtractedPage(text="Texte simple.", page=1)]
    chunks = chunk_pages(pages)
    assert all(isinstance(c, Chunk) for c in chunks)