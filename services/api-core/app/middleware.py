from __future__ import annotations

import time
from uuid import uuid4

from fastapi import Request
from starlette.responses import Response

from app.observability import metrics_registry


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None)
    if path_format:
        return str(path_format)
    return request.url.path


async def institutional_http_middleware(request: Request, call_next) -> Response:
    started_at = time.perf_counter()

    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    tenant_id = request.headers.get("X-Tenant-ID") or "public"
    idempotency_key = request.headers.get("Idempotency-Key")

    request.state.correlation_id = correlation_id
    request.state.tenant_id = tenant_id
    request.state.idempotency_key = idempotency_key

    response = await call_next(request)
    duration_seconds = time.perf_counter() - started_at

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Tenant-ID"] = tenant_id
    response.headers["X-API-Version"] = "v1"
    response.headers["X-Standards-Mode"] = "transitional"
    if idempotency_key:
        response.headers["Idempotency-Key"] = idempotency_key

    metrics_registry.record_request(
        method=request.method,
        path=_route_path(request),
        status_code=response.status_code,
        duration_seconds=duration_seconds,
    )
    return response
