"""Tests unitaires du reranker (Phase 2c) — sans réseau, sans modèle."""
from __future__ import annotations

import math

import pytest

from app.retrieval import reranker
from app.retrieval.reranker import StubLexicalScorer, build_scorer, rerank, rerank_async


def _cand(cid: str, content: str, confidence: float = 0.5) -> dict:
    return {
        "chunk_id": cid,
        "document_id": f"doc-{cid}",
        "content": content,
        "page": 1,
        "section": None,
        "classification": "CONFIDENTIAL",
        "confidence": confidence,
        "rrf_score": 0.01,
    }


class _FixedScorer:
    name = "cross_encoder"

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, query: str, passages: list[str]) -> list[float]:
        return self._scores


class _BoomScorer:
    name = "cross_encoder"

    def score(self, query: str, passages: list[str]) -> list[float]:
        raise RuntimeError("modèle indisponible")


def test_rerank_reorders_by_score_and_truncates() -> None:
    cands = [_cand("a", "x"), _cand("b", "y"), _cand("c", "z")]
    out = rerank("q", cands, top_k=2, scorer=_FixedScorer([0.1, 0.9, 0.5]))
    assert [r["chunk_id"] for r in out] == ["b", "c"]


def test_rerank_preserves_fields_and_adds_provenance() -> None:
    cands = [_cand("a", "x", confidence=0.4), _cand("b", "y", confidence=0.8)]
    out = rerank("q", cands, top_k=2, scorer=_FixedScorer([0.2, 0.7]))
    top = out[0]
    assert top["chunk_id"] == "b"
    assert top["document_id"] == "doc-b" and top["page"] == 1
    assert top["rrf_score"] == 0.01  # rétrocompatibilité
    assert top["rrf_confidence"] == 0.8
    assert top["rerank_score"] == 0.7
    assert top["confidence"] == pytest.approx(1.0 / (1.0 + math.exp(-0.7)))
    assert top["pre_rerank_rank"] == 2
    assert top["rerank_source"] == "cross_encoder"
    assert top["confidence_source"] == "cross_encoder"


def test_confidence_is_sigmoid_bounded_in_unit_interval() -> None:
    out = rerank("q", [_cand("a", "x"), _cand("b", "y")], top_k=2, scorer=_FixedScorer([5.0, -3.0]))
    assert [r["confidence"] for r in out] == pytest.approx([
        1.0 / (1.0 + math.exp(-5.0)),
        1.0 / (1.0 + math.exp(3.0)),
    ])
    assert out[0]["rerank_score"] == 5.0  # score brut conservé


def test_failure_falls_back_to_rrf_order_and_is_labelled() -> None:
    cands = [_cand("a", "x"), _cand("b", "y"), _cand("c", "z")]
    out = rerank("q", cands, top_k=2, scorer=_BoomScorer())
    assert [r["chunk_id"] for r in out] == ["a", "b"]
    assert all(r["rerank_source"] == "rrf_fallback" for r in out)
    assert all(r["confidence_source"] == "rrf" for r in out)
    assert out[0]["confidence"] == 0.5  # confiance RRF intacte


def test_score_count_mismatch_falls_back() -> None:
    out = rerank("q", [_cand("a", "x"), _cand("b", "y")], top_k=2, scorer=_FixedScorer([0.9]))
    assert all(r["rerank_source"] == "rrf_fallback" for r in out)


def test_empty_or_invalid_top_k() -> None:
    assert rerank("q", [], top_k=3, scorer=StubLexicalScorer()) == []
    assert rerank("q", [_cand("a", "x")], top_k=0, scorer=StubLexicalScorer()) == []


def test_input_candidates_are_not_mutated() -> None:
    cands = [_cand("a", "x")]
    rerank("q", cands, top_k=1, scorer=_FixedScorer([0.9]))
    assert "rerank_score" not in cands[0]
    assert cands[0]["confidence"] == 0.5


def test_stub_scorer_prefers_lexical_overlap_and_is_labelled() -> None:
    cands = [
        _cand("a", "météo du jour et recette de cuisine"),
        _cand("b", "risques principaux de NVDA et valorisation"),
    ]
    out = rerank("risques NVDA valorisation", cands, top_k=2, scorer=StubLexicalScorer())
    assert out[0]["chunk_id"] == "b"
    assert out[0]["rerank_source"] == "stub_lexical"


def test_build_scorer_backends() -> None:
    assert isinstance(build_scorer("stub"), StubLexicalScorer)
    assert build_scorer("cross_encoder").name == "cross_encoder"
    with pytest.raises(ValueError):
        build_scorer("inconnu")


def test_cross_encoder_without_dependency_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-encoder demandé mais modèle non chargeable → repli RRF étiqueté (pas de stub silencieux)."""

    def _boom(model_name: str, device: str | None) -> object:
        raise ImportError("sentence_transformers absent")

    monkeypatch.setattr(reranker, "_load_cross_encoder", _boom)
    out = rerank("q", [_cand("a", "x")], top_k=1, scorer=build_scorer("cross_encoder"))
    assert out[0]["rerank_source"] == "rrf_fallback"


def test_cross_encoder_scorer_truncates_and_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class _FakeModel:
        def predict(self, pairs, batch_size, show_progress_bar):  # noqa: ANN001
            seen["pairs"], seen["batch"] = pairs, batch_size
            return [0.3 for _ in pairs]

    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda m, d: _FakeModel())
    scorer = build_scorer("cross_encoder", batch_size=4)
    scores = scorer.score("q", ["a" * 5000])
    assert scores == [0.3]
    assert len(seen["pairs"][0][1]) == reranker.MAX_PASSAGE_CHARS
    assert seen["batch"] == 4


@pytest.mark.asyncio
async def test_rerank_async_matches_sync() -> None:
    cands = [_cand("a", "x"), _cand("b", "y")]
    out = await rerank_async("q", cands, top_k=1, scorer=_FixedScorer([0.1, 0.9]))
    assert [r["chunk_id"] for r in out] == ["b"]
