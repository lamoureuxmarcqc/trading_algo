"""FastAPI app — Knowledge Layer HTTP API.

Expose le RAG hybride (vector + BM25 + RRF + cross-encoder) via HTTP.
Ne genere AUCUNE reponse LLM : retourne uniquement les chunks sources cites.
La synthese eventuelle est faite par AI Gateway (separation Research/Generation).
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .models import (
    DataClassification,
    DocumentInfo,
    QueryRequest,
    QueryResponse,
)
from .retrieval import citations
from .retrieval import hybrid

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
)
logger = structlog.get_logger()


# ─────────────────────────── Classification policy ───────────────────────────

_CLASSIFICATION_ORDER = [
    DataClassification.PUBLIC,
    DataClassification.INTERNAL,
    DataClassification.CONFIDENTIAL,
    DataClassification.HIGHLY_CONFIDENTIAL,
    DataClassification.RESTRICTED,
]

# Par defaut on n'expose PAS RESTRICTED (medical) sauf demande explicite
_DEFAULT_MAX = DataClassification.HIGHLY_CONFIDENTIAL


def _allowed_classifications(
    classification_max: DataClassification | None,
) -> list[str]:
    """Retourne toutes les classifications <= classification_max.

    None = HIGHLY_CONFIDENTIAL par defaut (exclut RESTRICTED).
    Une demande explicite de RESTRICTED renvoie les 5 niveaux.
    """
    ceiling = classification_max or _DEFAULT_MAX
    ceiling_idx = _CLASSIFICATION_ORDER.index(ceiling)
    return [c.value for c in _CLASSIFICATION_ORDER[: ceiling_idx + 1]]


# ─────────────────────────── Lifespan ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "knowledge_startup",
        version=__version__,
        rerank_enabled=settings.rerank_enabled,
        rerank_min_confidence=settings.rerank_min_confidence,
        embedding_model=settings.embedding_model,
    )
    yield
    logger.info("knowledge_shutdown")


app = FastAPI(
    title="Lamoureux Knowledge Layer",
    version=__version__,
    lifespan=lifespan,
)


# ─────────────────────────── Middleware ───────────────────────────

@app.middleware("http")
async def correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = cid
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
        correlation_id=cid,
    )
    return response


# ─────────────────────────── Health / Stats ───────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe — verifie que le process repond (pas la DB)."""
    return {
        "status": "ok",
        "version": __version__,
        "rerank_enabled": settings.rerank_enabled,
        "embedding_model": settings.embedding_model,
    }


@app.get("/ready")
async def ready() -> dict:
    """Readiness probe — verifie la connexion PostgreSQL."""
    from sqlalchemy import text

    from .storage.db import get_session

    try:
        session = get_session()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()
        return {"status": "ready"}
    except Exception as exc:
        logger.error("readiness_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="database unavailable") from exc


@app.get("/api/v1/knowledge/stats")
async def stats(tenant_id: str = "lamoureux") -> dict:
    """Statistiques globales pour le tenant."""
    from sqlalchemy import text

    from .storage.db import get_session

    session = get_session()
    try:
        documents = session.execute(
            text("SELECT COUNT(*) FROM documents WHERE tenant_id = :t"),
            {"t": tenant_id},
        ).scalar_one()
        chunks = session.execute(
            text(
                "SELECT COUNT(*) FROM chunks c "
                "JOIN documents d ON d.id = c.document_id "
                "WHERE d.tenant_id = :t"
            ),
            {"t": tenant_id},
        ).scalar_one()
        return {"tenant_id": tenant_id, "documents": documents, "chunks": chunks}
    finally:
        session.close()


# ─────────────────────────── Query ───────────────────────────

@app.post("/api/v1/knowledge/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Recherche hybride + reranking. Retourne les sources citees.

    Ne genere pas de synthese. Utiliser AI Gateway pour la generation.
    """
    cid = getattr(request.state, "correlation_id", None)
    allowed = _allowed_classifications(req.classification_max)

    try:
        results = await hybrid.search_async(
            query=req.query,
            tenant_id=req.tenant_id,
            allowed_classifications=allowed,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.error(
            "query_failed",
            error=str(exc),
            tenant_id=req.tenant_id,
            correlation_id=cid,
        )
        raise HTTPException(status_code=502, detail=f"retrieval failed: {exc}") from exc

    sources = citations.build_sources(results)

    # Confidence = max des scores de sources (indicateur global de qualite)
    confidence = max((s.score for s in sources), default=None)

    logger.info(
        "query_ok",
        tenant_id=req.tenant_id,
        n_results=len(sources),
        confidence=confidence,
        classification_max=(req.classification_max.value if req.classification_max else None),
        correlation_id=cid,
    )

    return QueryResponse(
        query=req.query,
        answer=None,
        sources=sources,
        confidence=confidence,
    )


# ─────────────────────────── Documents ───────────────────────────

@app.get(
    "/api/v1/knowledge/documents/{document_id}",
    response_model=DocumentInfo,
)
async def get_document(document_id: str) -> DocumentInfo:
    """Metadonnees d'un document (sans le contenu)."""
    from sqlalchemy import text

    from .storage.db import get_session

    session = get_session()
    try:
        row = session.execute(
            text("""
                SELECT d.id, d.filename, d.classification, d.ingested_at,
                       COUNT(c.id) AS chunks_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = :id
                GROUP BY d.id, d.filename, d.classification, d.ingested_at
            """),
            {"id": document_id},
        ).first()
    finally:
        session.close()

    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    return DocumentInfo(
        id=str(row.id),
        filename=row.filename,
        classification=DataClassification(row.classification),
        ingested_at=row.ingested_at.isoformat(),
        chunks_count=int(row.chunks_count),
    )


# ─────────────────────────── Error handler ───────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    cid = getattr(request.state, "correlation_id", None)
    logger.error(
        "unhandled_exception",
        error=str(exc),
        error_type=type(exc).__name__,
        correlation_id=cid,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "correlation_id": cid},
    )