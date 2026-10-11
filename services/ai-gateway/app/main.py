"""FastAPI app — AI Gateway."""
import logging
import time
import uuid

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .models import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelTarget,
)
from .privacy import PrivacyViolation, enforce
from .providers import ProviderError
from .router import router

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

app = FastAPI(title="Lamoureux AI Gateway", version=__version__)


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


@app.exception_handler(PrivacyViolation)
async def privacy_handler(request: Request, exc: PrivacyViolation):
    logger.warning(
        "privacy_violation",
        classification=exc.classification.value,
        requested=exc.requested.value,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return JSONResponse(
        status_code=403,
        content={
            "error": "privacy_violation",
            "detail": str(exc),
            "classification": exc.classification.value,
            "requested_target": exc.requested.value,
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    providers = await router.health()
    return HealthResponse(
        status="ok" if any(providers.values()) else "degraded",
        version=__version__,
        providers=providers,
    )


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(req: InferenceRequest, request: Request) -> InferenceResponse:
    # 1. Résolution de la cible
    target = router.resolve(req.classification, req.target)

    # 2. Enforcement privacy AVANT tout appel réseau
    enforce(req.classification, target)

    # 3. Dispatch
    provider = router.get(target)
    started = time.perf_counter()
    try:
        result = await provider.generate(
            prompt=req.prompt,
            model=req.model or "",
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=req.system,
        )
    except ProviderError as e:
        logger.error(
            "provider_error",
            provider=provider.name,
            error=str(e),
            correlation_id=getattr(request.state, "correlation_id", None),
        )
        raise HTTPException(status_code=502, detail=f"provider error: {e}")

    latency_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "inference",
        tenant_id=req.tenant_id,
        classification=req.classification.value,
        target=target.value,
        model=result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=latency_ms,
        correlation_id=getattr(request.state, "correlation_id", None),
    )

    return InferenceResponse(
        content=result.content,
        model=result.model,
        target=target,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=latency_ms,
    )


@app.get("/v1/policy")
async def policy() -> dict:
    """Expose la matrice de routage pour audit."""
    from .privacy import ALLOWED_TARGETS
    return {
        cls.value: sorted(t.value for t in targets)
        for cls, targets in ALLOWED_TARGETS.items()
    }
