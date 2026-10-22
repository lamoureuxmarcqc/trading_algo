"""Recherche vectorielle — wrapper autour de pgvector (Support Synchrone et Asynchrone)."""
from __future__ import annotations

import asyncio
import logging
from typing import TypedDict, List

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..config import settings
from ..storage import pgvector

logger = logging.getLogger("lamoureux_ai.knowledge.retrieval.vector")
tracer = trace.get_tracer("lamoureux_ai.vector_search")


class SearchResult(TypedDict):
    """Contrat exact de ce que `pgvector.vector_search` retourne.

    Doit rester synchronise avec `services/knowledge/app/storage/pgvector.py`.
    """

    chunk_id: str
    document_id: str
    content: str
    page: int | None
    section: str | None
    classification: str
    score: float


def _resolve_top_k(top_k: int | None) -> int:
    k = top_k or settings.retrieval_top_k
    if k <= 0:
        raise ValueError(f"top_k must be > 0, got {k}")
    return k


def search(
    *,
    query_embedding: List[float],
    tenant_id: str,
    allowed_classifications: List[str],
    top_k: int | None = None,
) -> List[SearchResult]:
    """Recherche sémantique synchrone par similarité cosinus."""
    k = _resolve_top_k(top_k)

    with tracer.start_as_current_span("vector.search.sync") as span:
        span.set_attribute("tenant.id", tenant_id)
        span.set_attribute("retrieval.top_k", k)
        span.set_attribute(
            "retrieval.allowed_classifications", ",".join(allowed_classifications)
        )

        try:
            results = pgvector.vector_search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                allowed_classifications=allowed_classifications,
                top_k=k,
            )
            span.set_attribute("retrieval.result_count", len(results))
            return results

        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.exception("Vector search (sync) failed", extra={"tenant_id": tenant_id})
            raise


async def search_async(
    *,
    query_embedding: List[float],
    tenant_id: str,
    allowed_classifications: List[str],
    top_k: int | None = None,
    timeout: float | None = None,
) -> List[SearchResult]:
    """Recherche vectorielle asynchrone avec instrumentation OpenTelemetry."""
    k = _resolve_top_k(top_k)

    with tracer.start_as_current_span("vector.search.async") as span:
        span.set_attribute("tenant.id", tenant_id)
        span.set_attribute("retrieval.top_k", k)
        span.set_attribute(
            "retrieval.allowed_classifications", ",".join(allowed_classifications)
        )
        span.set_attribute("retrieval.timeout", timeout or 0)

        async def _run() -> List[SearchResult]:
            if hasattr(pgvector, "vector_search_async"):
                return await pgvector.vector_search_async(
                    query_embedding=query_embedding,
                    tenant_id=tenant_id,
                    allowed_classifications=allowed_classifications,
                    top_k=k,
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: pgvector.vector_search(
                    query_embedding=query_embedding,
                    tenant_id=tenant_id,
                    allowed_classifications=allowed_classifications,
                    top_k=k,
                ),
            )

        try:
            if timeout and timeout > 0:
                results = await asyncio.wait_for(_run(), timeout=timeout)
            else:
                results = await _run()

            span.set_attribute("retrieval.result_count", len(results))
            return results

        except asyncio.TimeoutError as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, "timeout"))
            logger.error(
                "Vector search (async) timed out", extra={"tenant_id": tenant_id}
            )
            raise

        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.exception(
                "Vector search (async) failed", extra={"tenant_id": tenant_id}
            )
            raise