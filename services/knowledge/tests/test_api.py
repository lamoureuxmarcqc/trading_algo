"""Tests HTTP de l'API knowledge.

Utilise stub embeddings (cf. conftest.py) et mock hybrid.search_async.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import _allowed_classifications, app
from app.models import DataClassification


client = TestClient(app)


# ─────────────────────────── Health ───────────────────────────

def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "rerank_enabled" in body


def test_correlation_id_propagated():
    r = client.get("/health", headers={"X-Correlation-ID": "test-123"})
    assert r.headers["X-Correlation-ID"] == "test-123"


# ─────────────────────────── Classification policy ───────────────────────────

def test_classification_none_excludes_restricted():
    allowed = _allowed_classifications(None)
    assert "RESTRICTED" not in allowed
    assert "HIGHLY_CONFIDENTIAL" in allowed
    assert "PUBLIC" in allowed


def test_classification_public_only():
    allowed = _allowed_classifications(DataClassification.PUBLIC)
    assert allowed == ["PUBLIC"]


def test_classification_restricted_includes_all():
    allowed = _allowed_classifications(DataClassification.RESTRICTED)
    assert len(allowed) == 5
    assert "RESTRICTED" in allowed


def test_classification_confidential():
    allowed = _allowed_classifications(DataClassification.CONFIDENTIAL)
    assert allowed == ["PUBLIC", "INTERNAL", "CONFIDENTIAL"]


# ─────────────────────────── Query ───────────────────────────

def test_query_rejects_empty_query():
    r = client.post(
        "/api/v1/knowledge/query",
        json={"query": "", "tenant_id": "lamoureux"},
    )
    assert r.status_code == 422


def test_query_rejects_missing_tenant():
    r = client.post(
        "/api/v1/knowledge/query",
        json={"query": "test"},
    )
    assert r.status_code == 422


def test_query_returns_empty_when_no_results():
    with patch(
        "app.retrieval.hybrid.search_async",
        new=AsyncMock(return_value=[]),
    ):
        r = client.post(
            "/api/v1/knowledge/query",
            json={"query": "test query", "tenant_id": "lamoureux"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] == []
    assert body["confidence"] is None
    assert body["answer"] is None


def test_query_returns_sources_with_citations():
    fake_result = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": "Facture Sears lave-vaisselle 2017 avec details complets du produit.",
        "page": 1,
        "section": None,
        "classification": "HIGHLY_CONFIDENTIAL",
        "confidence": 0.945,
    }
    with patch(
        "app.retrieval.hybrid.search_async",
        new=AsyncMock(return_value=[fake_result]),
    ):
        r = client.post(
            "/api/v1/knowledge/query",
            json={"query": "facture Sears 2017", "tenant_id": "lamoureux"},
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["sources"]) == 1
    src = body["sources"][0]
    assert src["document_id"] == "doc-1"
    assert src["chunk_id"] == "chunk-1"
    assert src["page"] == 1
    assert src["score"] == pytest.approx(0.945, abs=0.001)
    assert "Facture Sears" in src["excerpt"]
    assert body["confidence"] == pytest.approx(0.945, abs=0.001)


def test_query_top_k_passed_to_hybrid():
    mock = AsyncMock(return_value=[])
    with patch("app.retrieval.hybrid.search_async", new=mock):
        client.post(
            "/api/v1/knowledge/query",
            json={"query": "test", "tenant_id": "lamoureux", "top_k": 3},
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["top_k"] == 3


# ─────────────────────────── Document 404 ───────────────────────────

def test_document_404():
    r = client.get("/api/v1/knowledge/documents/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404