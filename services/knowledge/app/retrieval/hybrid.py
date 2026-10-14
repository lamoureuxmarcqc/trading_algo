"""Fusion hybride vector + BM25 via Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from ..config import settings
from ..embeddings.local import embed_query
from . import keyword as keyword_search
from . import vector as vector_search


RRF_K = 60


def _rrf_fuse(
    *,
    vector_results: list[dict],
    keyword_results: list[dict],
    top_k: int,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunks[cid] = {**r, "vector_rank": rank + 1}

    for rank, r in enumerate(keyword_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if cid in chunks:
            chunks[cid]["keyword_rank"] = rank + 1
        else:
            chunks[cid] = {**r, "keyword_rank": rank + 1}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {**chunks[cid], "rrf_score": score}
        for cid, score in ranked
    ]


def search(
    *,
    query: str,
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    k = top_k or settings.rerank_top_k
    expanded = settings.retrieval_top_k

    query_vec = embed_query(query, settings.embedding_model).tolist()

    vec_results = vector_search.search(
        query_embedding=query_vec,
        tenant_id=tenant_id,
        allowed_classifications=allowed_classifications,
        top_k=expanded,
    )

    kw_results = keyword_search.search(
        query=query,
        tenant_id=tenant_id,
        allowed_classifications=allowed_classifications,
        top_k=expanded,
    )

    return _rrf_fuse(
        vector_results=vec_results,
        keyword_results=kw_results,
        top_k=k,
    )
