"""Extracteurs de texte pour PDF, DOCX, HTML, MD."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


class ExtractionError(Exception):
    """Erreur generique d'extraction."""


@dataclass
class ExtractedPage:
    """Une unite de contenu avec tracabilite."""
    text: str
    page: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Document complet extrait."""
    filename: str
    mime_type: str
    pages: list[ExtractedPage]
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


# ─────────────────────────────── PDF ───────────────────────────────

def extract_pdf(path: Path) -> ExtractedDocument:
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ExtractionError(f"PDF illisible : {e}") from e

    pages: list[ExtractedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(ExtractedPage(text=text, page=i))

    return ExtractedDocument(
        filename=path.name,
        mime_type="application/pdf",
        pages=pages,
        metadata={"page_count": len(reader.pages)},
    )


# ─────────────────────────────── DOCX ──────────────────────────────

def extract_docx(path: Path) -> ExtractedDocument:
    try:
        doc = DocxDocument(str(path))
    except Exception as e:
        raise ExtractionError(f"DOCX illisible : {e}") from e

    paragraphs: list[str] = []
    current_section: str | None = None
    pages: list[ExtractedPage] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Heading = section marker
        if para.style and para.style.name.startswith("Heading"):
            if paragraphs:
                pages.append(
                    ExtractedPage(
                        text="\n".join(paragraphs),
                        section=current_section,
                    )
                )
                paragraphs = []
            current_section = text
        else:
            paragraphs.append(text)

    if paragraphs:
        pages.append(ExtractedPage(text="\n".join(paragraphs), section=current_section))

    return ExtractedDocument(
        filename=path.name,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        pages=pages,
    )


# ─────────────────────────────── HTML ──────────────────────────────

def extract_html(path: Path) -> ExtractedDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # Supprimer scripts/styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n".join(lines)

    return ExtractedDocument(
        filename=path.name,
        mime_type="text/html",
        pages=[ExtractedPage(text=content)] if content else [],
    )


# ─────────────────────────────── Markdown ──────────────────────────

def extract_markdown(path: Path) -> ExtractedDocument:
    content = path.read_text(encoding="utf-8", errors="ignore")

    # Split sur les headings
    lines = content.splitlines()
    sections: list[ExtractedPage] = []
    current_title: str | None = None
    buffer: list[str] = []

    for line in lines:
        if line.startswith("#"):
            if buffer:
                sections.append(
                    ExtractedPage(
                        text="\n".join(buffer).strip(),
                        section=current_title,
                    )
                )
                buffer = []
            current_title = line.lstrip("#").strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append(
            ExtractedPage(text="\n".join(buffer).strip(), section=current_title)
        )

    # Si aucun heading, tout en une seule page
    if not sections and content.strip():
        sections = [ExtractedPage(text=content.strip())]

    return ExtractedDocument(
        filename=path.name,
        mime_type="text/markdown",
        pages=[s for s in sections if s.text],
    )


# ─────────────────────────────── Texte brut ────────────────────────

def extract_text(path: Path) -> ExtractedDocument:
    """Fichier texte brut (UTF-8, fallback ignore errors)."""
    content = path.read_text(encoding="utf-8", errors="ignore")
    if not content.strip():
        return ExtractedDocument(
            filename=path.name,
            mime_type="text/plain",
            pages=[],
        )
    return ExtractedDocument(
        filename=path.name,
        mime_type="text/plain",
        pages=[ExtractedPage(text=content.strip())],
    )


# ─────────────────────────────── Dispatcher ────────────────────────

EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".html": extract_html,
    ".htm": extract_html,
    ".txt": extract_text,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
}


def extract(path: Path) -> ExtractedDocument:
    """Dispatch selon l'extension."""
    if not path.exists():
        raise ExtractionError(f"Fichier introuvable : {path}")

    suffix = path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ExtractionError(f"Format non supporte : {suffix}")

    return extractor(path)