"""Chunking semantique avec overlap.

Objectif :
    - Decouper un texte long en chunks de taille cible (~400 tokens)
    - Preserver les frontieres de paragraphes
    - Overlap de ~10% entre chunks consecutifs
    - Conserver la tracabilite (page, section)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..ingestion.extractors import ExtractedPage


@dataclass
class Chunk:
    text: str
    page: int | None
    section: str | None
    token_count: int


def _count_tokens(text: str) -> int:
    """Approximation : 1 token ~ 4 caracteres pour FR/EN."""
    return max(1, len(text) // 4)


def _split_paragraphs(text: str) -> list[str]:
    """Split sur lignes vides."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split sur ponctuation forte."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _hard_split(text: str, target_tokens: int) -> list[str]:
    """Decoupage mot-a-mot — dernier recours quand rien d'autre ne marche."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for word in words:
        word_tokens = _count_tokens(word + " ")
        if buffer_tokens + word_tokens > target_tokens and buffer:
            chunks.append(" ".join(buffer))
            buffer = []
            buffer_tokens = 0
        buffer.append(word)
        buffer_tokens += word_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def _to_units(text: str, max_tokens: int) -> list[str]:
    """
    Reduit un texte en unites atomiques <= max_tokens chacune.

    Cascade :
        1. Paragraphes
        2. Phrases (si paragraphe trop gros)
        3. Mots (si phrase trop grosse)
    """
    units: list[str] = []

    for para in _split_paragraphs(text):
        if _count_tokens(para) <= max_tokens:
            units.append(para)
            continue

        for sent in _split_sentences(para):
            if _count_tokens(sent) <= max_tokens:
                units.append(sent)
            else:
                units.extend(_hard_split(sent, max_tokens))

    return units


def _merge_units(units: list[str], target_tokens: int) -> list[str]:
    """Fusionne les unites en chunks <= target_tokens."""
    if not units:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit)
        if buffer_tokens + unit_tokens > target_tokens and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_tokens = 0
        buffer.append(unit)
        buffer_tokens += unit_tokens

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


def _apply_overlap(chunks: list[str], overlap_ratio: float) -> list[str]:
    """Ajoute un chevauchement entre chunks consecutifs."""
    if overlap_ratio <= 0 or len(chunks) < 2:
        return chunks

    result: list[str] = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        overlap_size = int(len(prev) * overlap_ratio)
        if overlap_size > 0:
            tail = prev[-overlap_size:]
            result.append(f"{tail}\n\n{curr}")
        else:
            result.append(curr)

    return result


def chunk_pages(
    pages: list[ExtractedPage],
    target_tokens: int = 400,
    overlap_ratio: float = 0.10,
) -> list[Chunk]:
    """
    Transforme une liste de pages en chunks de taille cible.

    Regles :
        - Chaque page est traitee separement (preserve `page`)
        - Cascade paragraph -> phrase -> mot pour ne jamais depasser la cible
        - Overlap entre chunks consecutifs
        - Tracabilite page + section conservee
    """
    result: list[Chunk] = []

    for page in pages:
        if not page.text.strip():
            continue

        units = _to_units(page.text, target_tokens)
        merged = _merge_units(units, target_tokens)
        with_overlap = _apply_overlap(merged, overlap_ratio)

        for text in with_overlap:
            result.append(
                Chunk(
                    text=text,
                    page=page.page,
                    section=page.section,
                    token_count=_count_tokens(text),
                )
            )

    return result