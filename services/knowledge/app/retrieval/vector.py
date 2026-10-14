"""Recherche vectorielle — wrapper autour de pgvector."""
from __future__ import annotations

from ..config import settings
from ..storage.pgvector import vector_search


def search(
    *,
    query_embedding: list[float],
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """Recherche semantique par similarite cosinus."""
    return vector_search(
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        allowed_classifications=allowed_classifications,
        top_k=top_k or settings.retrieval_top_k,
    )
