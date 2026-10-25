"""Construction de citations verifiables pour les resultats de retrieval.

Ce module ne genere aucune reponse de modele. Il convertit uniquement les
chunks deja filtres par tenant et classification en sources citees.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..models import Source

DEFAULT_EXCERPT_CHARS = 500


def exact_excerpt(content: str, *, max_chars: int = DEFAULT_EXCERPT_CHARS) -> str:
    """Retourne un prefixe literal du chunk, sans reecriture ni normalisation.

    Limiter l'extrait protege le contrat API tout en conservant un fragment
    directement verifiable dans le document source.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    return content[:max_chars]


def to_source(result: dict[str, Any], *, max_excerpt_chars: int = DEFAULT_EXCERPT_CHARS) -> Source:
    """Convertit un resultat hybride en citation validee par Pydantic."""
    content = result.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("retrieval result must contain non-empty content for citation")

    try:
        score = float(result["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("retrieval result must contain a numeric confidence") from exc

    return Source(
        document_id=str(result["document_id"]),
        chunk_id=str(result["chunk_id"]),
        page=result.get("page"),
        section=result.get("section"),
        excerpt=exact_excerpt(content, max_chars=max_excerpt_chars),
        score=max(0.0, min(score, 1.0)),
    )


def build_sources(
    results: Iterable[dict[str, Any]],
    *,
    max_excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> list[Source]:
    """Produit une citation par resultat, dans le meme ordre de pertinence."""
    return [
        to_source(result, max_excerpt_chars=max_excerpt_chars)
        for result in results
    ]
