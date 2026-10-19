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

    # Filtrer les chunks de mauvaise qualite
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    filtered: list[tuple[str, float]] = []
    for cid, score in ranked:
        content = chunks[cid].get("content", "")
        # Rejeter si :
        # - moins de 50 caracteres
        # - moins de 40% de lettres alphanumeriques (trop de ponctuation/symboles)
        # Detecter le double-encoding dans le chunk
        alpha = [c for c in content if c.isalpha()]
        if len(alpha) > 20:
            doubles = sum(1 for i in range(len(alpha) - 1) if alpha[i] == alpha[i + 1])
            if doubles / (len(alpha) - 1) > 0.3:
                continue
        if len(content) < 50:
            continue
        alnum = sum(1 for c in content if c.isalnum())
        if alnum / len(content) < 0.4:
            continue
        filtered.append((cid, score))
        if len(filtered) >= top_k:
            break

    return [
        {**chunks[cid], "rrf_score": score}
        for cid, score in filtered
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
