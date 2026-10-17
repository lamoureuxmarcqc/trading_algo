"""FastAPI app — Self-Improvement."""
from fastapi import FastAPI

from . import __version__
from .detector import detect
from .observer import collect_metrics

app = FastAPI(title="Lamoureux Self-Improvement", version=__version__)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/improvement/proposals")
async def list_proposals() -> list[dict]:
    metrics = collect_metrics()
    proposals = detect(metrics)
    return [p.model_dump() for p in proposals]
