"""Recherche par mots-clés — BM25 (Support Synchrone et Asynchrone)."""
from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from ..config import settings
from ..storage.db import get_session
from ..storage.models import Chunk


def _tokenize(text: str) -> list[str]:
    """Tokenise une chaîne de caractères en mots minuscules."""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=32)
def _build_index(tenant_id: str, classification_key: str) -> tuple[Any, list[list[str]], BM25Okapi]:
    """
    Construit l'index BM25 en mémoire pour un locataire et des classifications spécifiques.
    Mise en cache pour éviter les lectures disques répétitives sur les segments inchangés.
    """
    classifications = classification_key.split("|")

    session = get_session()
    try:
        stmt = (
            select(Chunk.id, Chunk.document_id, Chunk.content,
                   Chunk.page, Chunk.section, Chunk.classification)
            .where(Chunk.tenant_id == tenant_id)
            .where(Chunk.classification.in_(classifications))
        )
        rows = session.execute(stmt).all()
    finally:
        session.close()

    if not rows:
        return (), (), BM25Okapi([["empty"]])

    corpus = [_tokenize(r.content) for r in rows]
    bm25 = BM25Okapi(corpus)

    return rows, corpus, bm25


def invalidate_cache() -> None:
    """Vide le cache de l'index BM25 (Utile après une nouvelle ingestion)."""
    _build_index.cache_clear()


def _score_and_rank(
    query_tokens: list[str], 
    rows: Any, 
    bm25: BM25Okapi, 
    k: int
) -> list[dict]:
    """Logique pure CPU d'évaluation et de tri des scores BM25."""
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results: list[dict] = []
    for idx in ranked:
        if scores[idx] <= 0:
            continue
        r = rows[idx]
        results.append(
            {
                "chunk_id": str(r.id),
                "document_id": str(r.document_id),
                "content": r.content,
                "page": r.page,
                "section": r.section,
                "classification": r.classification,
                "score": float(scores[idx]),
            }
        )
    return results


def search(
    *,
    query: str,
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """Recherche lexicale synchrone originale."""
    k = top_k or settings.retrieval_top_k
    classification_key = "|".join(sorted(allowed_classifications))

    rows, _, bm25 = _build_index(tenant_id, classification_key)
    if not rows:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    return _score_and_rank(query_tokens, rows, bm25, k)


async def search_async(
    *,
    query: str,
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """
    Recherche lexicale asynchrone non-bloquante pour la boucle d'événements.
    Délègue les calculs CPU intensifs de scoring de rank_bm25 à un exécuteur.
    """
    k = top_k or settings.retrieval_top_k
    classification_key = "|".join(sorted(allowed_classifications))

    # Récupération de l'index (La base utilise un pool synchrone standard hébergé dans get_session)
    loop = asyncio.get_running_loop()
    rows, _, bm25 = await loop.run_in_executor(
        None, _build_index, tenant_id, classification_key
    )
    if not rows:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Exécution du tri et scoring mathématique BM25 hors du thread principal
    return await loop.run_in_executor(
        None, _score_and_rank, query_tokens, rows, bm25, k
    )
