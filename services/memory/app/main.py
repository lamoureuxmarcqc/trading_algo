"""FastAPI app — Memory Layer."""
import logging

import structlog
from fastapi import FastAPI

from . import __version__
from .config import settings
from .models import DecisionEvaluate, DecisionRecord, LongTermValidate, ShortTermAppend

logging.basicConfig(level=settings.log_level)
logger = structlog.get_logger()

app = FastAPI(title="Lamoureux Memory Layer", version=__version__)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/short-term/append")
async def short_term_append(payload: ShortTermAppend) -> dict:
    from .short_term import append
    size = append(payload)
    return {"ok": True, "buffer_size": size}


@app.post("/v1/long-term/validate")
async def long_term_validate(payload: LongTermValidate) -> dict:
    if not payload.human_validation:
        return {"ok": False, "error": "human_validation required"}
    return {"ok": True}


@app.post("/v1/decision/record")
async def decision_record(payload: DecisionRecord) -> dict:
    from .decision import record
    decision_id = record(payload)
    return {"ok": True, "decision_id": decision_id}


@app.post("/v1/decision/{decision_id}/evaluate")
async def decision_evaluate(decision_id: str, payload: DecisionEvaluate) -> dict:
    return {"ok": True, "decision_id": decision_id}
