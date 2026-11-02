from datetime import datetime, timezone

from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from app.db.session import SessionLocal
from app.observability import metrics_registry

router = APIRouter()


def _database_status() -> str:
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        return "unavailable"
    return "ok"


@router.get("/health")
def healthcheck() -> dict[str, str]:
    database = _database_status()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "api-core",
        "database": database,
    }


@router.get("/ready")
def readiness(response: Response) -> dict[str, str]:
    database = _database_status()
    readiness_status = "ready" if database == "ok" else "degraded"
    if database != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": readiness_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "api-core",
        "database": database,
    }


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_registry.render_prometheus()
