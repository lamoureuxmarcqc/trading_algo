"""Pipeline d'ingestion — de fichier à chunks prêts pour embedding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..chunking.semantic import chunk_pages
from .extractors import extract


@dataclass
class IngestedChunk:
    content: str
    page: int | None
    section: str | None
    chunk_index: int
    metadata: dict


@dataclass
class IngestedDocument:
    filename: str
    mime_type: str
    chunks: list[IngestedChunk]
    metadata: dict


def ingest_file(
    path: Path,
    chunk_target_tokens: int = 400,
    overlap_ratio: float = 0.10,
) -> IngestedDocument:
    """
    Ingere un fichier et produit des chunks prets pour embedding.

    Etapes :
        1. Extraction texte + pagination
        2. Chunking semantique
        3. Retour des chunks avec tracabilite
    """
    extracted = extract(path)

    chunks = chunk_pages(
        extracted.pages,
        target_tokens=chunk_target_tokens,
        overlap_ratio=overlap_ratio,
    )

    return IngestedDocument(
        filename=extracted.filename,
        mime_type=extracted.mime_type,
        chunks=[
            IngestedChunk(
                content=c.text,
                page=c.page,
                section=c.section,
                chunk_index=i,
                metadata={"source_file": extracted.filename},
            )
            for i, c in enumerate(chunks)
        ],
        metadata=extracted.metadata,
    )