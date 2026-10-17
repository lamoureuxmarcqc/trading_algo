"""FastAPI app — Orchestrator."""
from fastapi import FastAPI
from pydantic import BaseModel

from . import __version__
from .supervisor import route

app = FastAPI(title="Lamoureux Orchestrator", version=__version__)


class RouteRequest(BaseModel):
    task: str


class RouteResponse(BaseModel):
    agent: str
    task: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/route", response_model=RouteResponse)
async def route_task(req: RouteRequest) -> RouteResponse:
    agent = route(req.task)
    return RouteResponse(agent=agent, task=req.task)
