"""Intégration hybrid + reranker (Phase 2c) — recherche mockée, sans DB ni modèle."""
from __future__ import annotations

import numpy as np
import pytest

from app.retrieval import hybrid


def _chunk(cid: str, text: str) -> dict:
    return {
        "chunk_id": cid,
        "document_id": f"doc-{cid}",
        "content": text,
        "page": 2,
        "section": None,
        "classification": "CONFIDENTIAL",
    }


# Contenus > 50 caractères, texte « propre » (passent les filtres qualité de _rrf_fuse).
_TEXTS = {
    "c1": "Le rapport décrit la saisonnalité des ventes et les cycles de stocks du secteur.",
    "c2": "Les risques principaux de NVDA incluent la valorisation et la concurrence des puces.",
    "c3": "Procédure interne de classification des documents et gestion des accès aux fichiers.",
}


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {"overrides": {}}

    monkeypatch.setattr(hybrid, "embed_query", lambda q, m: np.zeros(3))
    # RRF placera c1 en tête (rang vectoriel 1 + rang BM25 1), c2 ensuite, c3 dernier.
    vec = [_chunk("c1", _TEXTS["c1"]), _chunk("c2", _TEXTS["c2"]), _chunk("c3", _TEXTS["c3"])]
    kw = [_chunk("c1", _TEXTS["c1"]), _chunk("c2", _TEXTS["c2"])]
    monkeypatch.setattr(hybrid.vector_search, "search", lambda **kw_: list(vec))
    monkeypatch.setattr(hybrid.keyword_search, "search", lambda **kw_: list(kw))

    async def _vec_async(**kw_):  # noqa: ANN003
        return list(vec)

    async def _kw_async(**kw_):  # noqa: ANN003
        return list(kw)

    monkeypatch.setattr(hybrid.vector_search, "search_async", _vec_async)
    monkeypatch.setattr(hybrid.keyword_search, "search_async", _kw_async)

    real_cfg = hybrid._cfg
    monkeypatch.setattr(
        hybrid, "_cfg", lambda name, default: calls["overrides"].get(name, real_cfg(name, default))
    )
    return calls


def _args() -> dict:
    return {"query": "risques NVDA valorisation", "tenant_id": "t", "allowed_classifications": ["CONFIDENTIAL"], "top_k": 2}


def test_disabled_by_default_keeps_rrf_behaviour(patched: dict) -> None:
    out = hybrid.search(**_args())
    assert [r["chunk_id"] for r in out] == ["c1", "c2"]
    assert "rerank_source" not in out[0]
    assert "rrf_score" in out[0]


def test_enabled_reorders_with_stub_and_labels_source(patched: dict) -> None:
    patched["overrides"].update(rerank_enabled=True, rerank_backend="stub")
    out = hybrid.search(**_args())
    assert out[0]["chunk_id"] == "c2"  # recouvrement lexical avec la requête
    assert out[0]["rerank_source"] == "stub_lexical"
    assert len(out) == 2
    assert out[0]["document_id"] == "doc-c2" and out[0]["page"] == 2


def test_pool_is_larger_than_top_k_when_enabled(patched: dict) -> None:
    patched["overrides"].update(rerank_enabled=True, rerank_backend="stub", rerank_candidates=3)
    assert hybrid._candidate_pool(2) == 3
    patched["overrides"]["rerank_enabled"] = False
    assert hybrid._candidate_pool(2) == 2


@pytest.mark.asyncio
async def test_async_path_reranks(patched: dict) -> None:
    patched["overrides"].update(rerank_enabled=True, rerank_backend="stub")
    out = await hybrid.search_async(**_args())
    assert out[0]["chunk_id"] == "c2"
    assert out[0]["rerank_source"] == "stub_lexical"


@pytest.mark.asyncio
async def test_async_disabled_unchanged(patched: dict) -> None:
    out = await hybrid.search_async(**_args())
    assert [r["chunk_id"] for r in out] == ["c1", "c2"]
