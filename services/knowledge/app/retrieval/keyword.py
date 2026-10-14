"""Recherche par mots-cles — BM25."""
from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from ..config import settings
from ..storage.db import get_session
from ..storage.models import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=32)
def _build_index(tenant_id: str, classification_key: str) -> tuple:
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
    _build_index.cache_clear()


def search(
    *,
    query: str,
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    k = top_k or settings.retrieval_top_k
    classification_key = "|".join(sorted(allowed_classifications))

    rows, _, bm25 = _build_index(tenant_id, classification_key)
    if not rows:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

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
