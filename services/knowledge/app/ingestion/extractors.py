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

def _fix_double_encoding(text: str) -> str:
    """
    Corrige l'artefact pdfplumber ou chaque lettre est doublee.
    Ex : 'XXMMLL eett ll' -> 'XML et il'.

    Heuristique : sur les caracteres alphanumeriques UNIQUEMENT
    (les espaces/pontuation ne sont pas doubles par pdfplumber),
    si > 40% de paires consecutives sont identiques, on desDoublonne.
    """
    if len(text) < 20:
        return text

    # Ne considerer que les alphanumeriques
    letters = [c for c in text if c.isalnum()]
    if len(letters) < 10:
        return text

    doubles = sum(1 for i in range(len(letters) - 1) if letters[i] == letters[i + 1])
    ratio = doubles / (len(letters) - 1)

    if ratio > 0.40:
        return text[::2]
    return text


def _clean_cid_artifacts(text: str) -> str:
    """
    Supprime les (cid:N) produits par pdfplumber sur les fonts sans CMap.
    Preserve le texte autour et normalise les espaces.
    """
    import re
    # Retirer les (cid:N)
    text = re.sub(r"\(cid:\d+\)", " ", text)
    # Normaliser les espaces multiples
    text = re.sub(r"[ \t]+", " ", text)
    # Retirer les lignes qui ne contiennent que des espaces
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()

def _clean_control_chars(text: str) -> str:
    """
    Supprime les caracteres de controle non-imprimables que PostgreSQL
    refuse dans les champs TEXT.

    PostgreSQL interdit :
        - 0x00 (NUL byte) - strictement interdit
        - 0x01-0x08 (controle)
        - 0x0B-0x0C (controle)
        - 0x0E-0x1F (controle)
        - 0x7F (DEL)

    On preserve :
        - 0x09 (tab) -> espace
        - 0x0A (LF)  -> newline
        - 0x0D (CR)  -> supprime (Windows-style)
    """
    import unicodedata

    # 1. Remplacer les caracteres de controle courants par des equivalents
    text = text.replace("\x0b", "\n").replace("\x0c", "\n")
    text = text.replace("\x0d", "")   # CR Windows
    text = text.replace("\x00", "")   # NUL byte critique

    # 2. Filtrer tout le reste des Cc (control chars) sauf tab/newline
    cleaned = []
    for c in text:
        cat = unicodedata.category(c)
        if cat == "Cc" and c not in ("\t", "\n"):
            continue
        # Remplacer tab par espace (PostgreSQL tolere mais le RAG prefere)
        if c == "\t":
            cleaned.append(" ")
            continue
        cleaned.append(c)

    return "".join(cleaned)

# ─────────────────────────────── PDF ───────────────────────────────

def extract_pdf(path: Path) -> ExtractedDocument:
    """Extraction PDF via pymupdf (robuste sur CMap, fonts, layouts)."""
    # Verifier la signature PDF avant tout
    try:
        with open(path, "rb") as f:
            header = f.read(5)
    except OSError as e:
        raise ExtractionError(f"Lecture impossible : {e}") from e

    if header != b"%PDF-":
        raise ExtractionError(
            f"Fichier non-PDF (header={header!r}, attendu b'%PDF-')"
        )

    try:
        import pymupdf
    except ImportError as e:
        raise ExtractionError(
            "pymupdf non installe. pip install pymupdf"
        ) from e

    pages: list[ExtractedPage] = []
    try:
        doc = pymupdf.open(str(path))
        try:
            for i, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text") or ""
                except Exception:
                    text = ""
                text = _clean_cid_artifacts(text)
                text = _clean_control_chars(text)   # ← AJOUT
                text = _fix_double_encoding(text)   # ← remettre si tu l'as garde
                text = text.strip()
                if text:
                    pages.append(ExtractedPage(text=text, page=i))
            page_count = doc.page_count
        finally:
            doc.close()
    except Exception as e:
        raise ExtractionError(f"PDF illisible : {e}") from e

    return ExtractedDocument(
        filename=path.name,
        mime_type="application/pdf",
        pages=pages,
        metadata={"page_count": page_count, "extractor": "pymupdf"},
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
                text = _clean_control_chars(text)
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
    content = _clean_control_chars(content)
    return ExtractedDocument(
        filename=path.name,
        mime_type="text/html",
        pages=[ExtractedPage(text=content)] if content else [],
    )


# ─────────────────────────────── Markdown ──────────────────────────

def extract_markdown(path: Path) -> ExtractedDocument:
    content = path.read_text(encoding="utf-8", errors="ignore")
    content = _clean_control_chars(content)

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
    content = path.read_text(encoding="utf-8", errors="ignore")
    content = _clean_control_chars(content)   # ← AJOUT
    if not content.strip():
        return ExtractedDocument(
            filename=path.name, mime_type="text/plain", pages=[]
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