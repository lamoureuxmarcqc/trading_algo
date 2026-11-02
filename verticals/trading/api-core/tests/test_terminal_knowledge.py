"""Tests for the terminal's constrained local Knowledge bridge."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api.routes import terminal  # noqa: E402
from app.schemas.terminal import (  # noqa: E402
    TerminalKnowledgeQueryRequest,
    TerminalKnowledgeQueryResponse,
)
from app.services.trading_algo_terminal import (  # noqa: E402
    KnowledgeQueryError,
    TradingAlgoTerminalService,
)


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "query": "Quels risques sont cites ?",
            "answer": None,
            "sources": [
                {
                    "document_id": "document-1",
                    "chunk_id": "chunk-1",
                    "page": 4,
                    "section": "Risques",
                    "excerpt": "Extrait cite.",
                    "score": 0.81,
                }
            ],
            "confidence": 0.81,
        }


class FakeHttpClient:
    def __init__(self) -> None:
        self.url: str | None = None
        self.payload: dict | None = None

    def post(self, url: str, *, json: dict) -> FakeResponse:
        self.url = url
        self.payload = json
        return FakeResponse()


def test_query_knowledge_enforces_confidential_ceiling() -> None:
    client = FakeHttpClient()
    service = TradingAlgoTerminalService(
        extractor=object(),
        risk_manager=object(),
        http_client=client,  # type: ignore[arg-type]
    )

    result = service.query_knowledge(
        query="Quels risques sont cites ?",
        tenant_id="tenant-alpha",
        top_k=3,
    )

    assert result.sources[0].chunk_id == "chunk-1"
    assert client.url == TradingAlgoTerminalService.KNOWLEDGE_QUERY_URL
    assert client.payload == {
        "query": "Quels risques sont cites ?",
        "tenant_id": "tenant-alpha",
        "top_k": 3,
        "classification_max": "CONFIDENTIAL",
    }


def test_terminal_knowledge_rejects_client_controlled_classification() -> None:
    with pytest.raises(ValidationError):
        TerminalKnowledgeQueryRequest.model_validate(
            {"query": "Lire un document", "classification_max": "RESTRICTED"}
        )


def test_terminal_knowledge_route_uses_request_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeKnowledgeService:
        def query_knowledge(self, **kwargs) -> TerminalKnowledgeQueryResponse:
            captured.update(kwargs)
            return TerminalKnowledgeQueryResponse(
                query=kwargs["query"],
                sources=[],
                confidence=None,
            )

    monkeypatch.setattr(terminal, "TradingAlgoTerminalService", FakeKnowledgeService)
    request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant-alpha"))

    response = terminal.query_terminal_knowledge(
        TerminalKnowledgeQueryRequest(query="Question", top_k=2),
        request,
    )

    assert response.query == "Question"
    assert captured == {"query": "Question", "tenant_id": "tenant-alpha", "top_k": 2}


def test_query_knowledge_maps_http_failure_to_service_error() -> None:
    class FailingHttpClient:
        def post(self, url: str, *, json: dict) -> None:
            raise httpx.ConnectError("offline")

    service = TradingAlgoTerminalService(
        extractor=object(),
        risk_manager=object(),
        http_client=FailingHttpClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(KnowledgeQueryError, match="unavailable"):
        service.query_knowledge(query="Question", tenant_id="tenant-alpha")
