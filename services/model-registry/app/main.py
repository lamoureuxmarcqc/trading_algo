"""FastAPI app — Model Registry."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .models import ModelRegistration
from .registry import RegistryError, list_models, promote, register, rollback

app = FastAPI(title="Lamoureux Model Registry", version=__version__)


class PromoteRequest(BaseModel):
    target: str = "production"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/models/register")
async def register_model(payload: ModelRegistration) -> dict:
    try:
        return register(payload.model_dump())
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/models")
async def list_all() -> list[dict]:
    return list_models()


@app.post("/v1/models/{model_id}/promote")
async def promote_model(model_id: str, req: PromoteRequest) -> dict:
    try:
        return promote(model_id, req.target)
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/models/{model_id}/rollback")
async def rollback_model(model_id: str) -> dict:
    try:
        return rollback(model_id)
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
