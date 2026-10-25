"""Fusion hybride vector + BM25 via Reciprocal Rank Fusion (RRF), puis reranking optionnel."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import settings
from ..embeddings.local import embed_query
from . import keyword as keyword_search
from . import reranker
from . import vector as vector_search
from .citations import to_source

logger = logging.getLogger("lamoureux_ai.knowledge.retrieval.hybrid")

RRF_K = 60
DEFAULT_RERANK_CANDIDATES = 30


def _cfg(name: str, default: Any) -> Any:
    """Lit un réglage optionnel : les nouveaux champs rerank_* peuvent ne pas exister
    dans ``Settings`` (rétrocompatibilité avec la configuration actuelle)."""
    return getattr(settings, name, default)


def _rerank_enabled() -> bool:
    return bool(_cfg("rerank_enabled", False))


def _build_scorer() -> reranker.Scorer:
    return reranker.build_scorer(
        str(_cfg("rerank_backend", "cross_encoder")),
        model_name=str(_cfg("rerank_model", reranker.DEFAULT_RERANK_MODEL)),
        batch_size=int(_cfg("rerank_batch_size", reranker.DEFAULT_BATCH_SIZE)),
        device=_cfg("rerank_device", None),
    )


def _candidate_pool(k: int) -> int:
    """Taille du pool RRF transmis au reranker (≥ k) ; k si le reranking est désactivé."""
    if not _rerank_enabled():
        return k
    return max(k, int(_cfg("rerank_candidates", DEFAULT_RERANK_CANDIDATES)))


def _normalize_rrf_score(score: float, max_possible_score: float = 0.0333) -> float:
    """
    Normalise le score RRF brut en un indice de confiance strict entre 0.0 et 1.0.
    Un score RRF max théorique pour un chunk top 1 dans les deux listes est ~0.0328.
    """
    if max_possible_score <= 0:
        return 0.0
    normalized = score / max_possible_score
    return min(max(normalized, 0.0), 1.0)


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

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    filtered: list[tuple[str, float]] = []
    for cid, score in ranked:
        content = chunks[cid].get("content", "")
        
        # Filtre 1 : Longueur minimale
        if len(content) < 50:
            continue
            
        # Filtre 2 : Densité de caractères alphanumériques
        alnum = sum(1 for c in content if c.isalnum())
        if alnum / len(content) < 0.4:
            continue

        # Filtre 3 : Détection de double-encodage (lettres répétées anormalement)
        alpha = [c for c in content if c.isalpha()]
        if len(alpha) > 20:
            doubles = sum(1 for i in range(len(alpha) - 1) if alpha[i] == alpha[i + 1])
            if doubles / (len(alpha) - 1) > 0.3:
                continue

        filtered.append((cid, score))
        if len(filtered) >= top_k:
            break

    return [
        {
            "chunk_id": cid,
            "document_id": chunks[cid]["document_id"],
            "content": chunks[cid]["content"],
            "page": chunks[cid].get("page"),
            "section": chunks[cid].get("section"),
            "classification": chunks[cid]["classification"],
            "vector_rank": chunks[cid].get("vector_rank"),
            "keyword_rank": chunks[cid].get("keyword_rank"),
            "confidence": _normalize_rrf_score(score),
            "rrf_score": score,  # Rétrocompatibilité avec vos tests existants
        }
        for cid, score in filtered
    ]


def _attach_citations(results: list[dict]) -> list[dict]:
    """Ajoute une citation structuree a chaque resultat, sans perdre les champs RRF.

    Cette etape est volontairement executee apres le reranking optionnel afin
    que ``citation.score`` corresponde toujours a la confiance finale exposee.
    """
    return [
        {
            **result,
            "citation": to_source(result).model_dump(),
        }
        for result in results
    ]


def search(
    *,
    query: str,
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """Version synchrone originale (Conservée pour ne rien briser)."""
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

    fused = _rrf_fuse(
        vector_results=vec_results,
        keyword_results=kw_results,
        top_k=_candidate_pool(k),
    )
    if not _rerank_enabled():
        return _attach_citations(fused)
    reranked = reranker.rerank(
        query,
        fused,
        top_k=k,
        scorer=_build_scorer(),
        min_confidence=float(_cfg("rerank_min_confidence", 0.20)),
    )
    return _attach_citations(reranked)

async def search_async(
    *,
    query: str,
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """Nouvelle version asynchrone non-bloquante pour l'API Gateway."""
    k = top_k or settings.rerank_top_k
    expanded = settings.retrieval_top_k

    loop = asyncio.get_running_loop()
    query_vec_embedding = await loop.run_in_executor(
        None, embed_query, query, settings.embedding_model
    )
    query_vec = query_vec_embedding.tolist()

    vec_results, kw_results = await asyncio.gather(
        vector_search.search_async(
            query_embedding=query_vec,
            tenant_id=tenant_id,
            allowed_classifications=allowed_classifications,
            top_k=expanded,
        ),
        keyword_search.search_async(
            query=query,
            tenant_id=tenant_id,
            allowed_classifications=allowed_classifications,
            top_k=expanded,
        ),
    )

    fused = _rrf_fuse(
        vector_results=vec_results,
        keyword_results=kw_results,
        top_k=_candidate_pool(k),
    )
    if not _rerank_enabled():
        return _attach_citations(fused)
    reranked = await reranker.rerank_async(
        query,
        fused,
        top_k=k,
        scorer=_build_scorer(),
        min_confidence=float(_cfg("rerank_min_confidence", 0.20)),
    )
    return _attach_citations(reranked)
