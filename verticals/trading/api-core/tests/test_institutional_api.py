from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.api.routes import health  # noqa: E402
from app.middleware import institutional_http_middleware  # noqa: E402
from app.observability import metrics_registry  # noqa: E402


def _build_request(
    path: str,
    method: str = "GET",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "headers": headers or [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "route": type("Route", (), {"path_format": path})(),
    }
    return Request(scope)


def test_healthcheck_payload_shape() -> None:
    payload = health.healthcheck()

    assert payload["status"] == "ok"
    assert payload["service"] == "api-core"
    assert "timestamp" in payload
    assert payload["database"] in {"ok", "unavailable"}


def test_readiness_returns_503_when_database_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(health, "_database_status", lambda: "unavailable")
    response = Response()
    payload = health.readiness(response)

    assert response.status_code == 503
    assert payload["status"] == "degraded"


def test_metrics_endpoint_exposes_request_counters() -> None:
    metrics_registry.record_request("GET", "/api/v1/health", 200, 0.015)
    output = health.metrics()

    assert "institutional_api_requests_total" in output
    assert 'path="/api/v1/health"' in output


def test_middleware_propagates_institutional_headers() -> None:
    request = _build_request(
        "/api/v1/health",
        headers=[
            (b"x-correlation-id", b"corr-123"),
            (b"x-tenant-id", b"tenant-alpha"),
            (b"idempotency-key", b"idem-001"),
        ],
    )

    async def call_next(_: Request) -> Response:
        return Response(status_code=200)

    response = asyncio.run(institutional_http_middleware(request, call_next))

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "corr-123"
    assert response.headers["X-Tenant-ID"] == "tenant-alpha"
    assert response.headers["X-API-Version"] == "v1"
    assert response.headers["X-Standards-Mode"] == "transitional"
    assert response.headers["Idempotency-Key"] == "idem-001"
