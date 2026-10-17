"""FastAPI app — Evaluation."""
from fastapi import FastAPI

from . import __version__

app = FastAPI(title="Lamoureux Evaluation", version=__version__)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}
