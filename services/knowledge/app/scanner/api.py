"""Endpoints FastAPI pour le scanner."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .filters import ScanFilters
from .walker import count_files, walk_source

router = APIRouter(prefix="/api/v1/knowledge/scan", tags=["scanner"])


class PreviewRequest(BaseModel):
    root_path: str
    include_extensions: list[str] | None = None
    max_files: int = 100


class PreviewResponse(BaseModel):
    root_path: str
    files_found: int
    files_listed: int
    sample: list[str]


@router.post("/preview", response_model=PreviewResponse)
async def preview_scan(req: PreviewRequest) -> PreviewResponse:
    root = Path(req.root_path)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Chemin introuvable: {req.root_path}")

    filters = ScanFilters()
    if req.include_extensions:
        filters.include_extensions = req.include_extensions

    total = count_files(root, filters)
    sample: list[str] = []
    for i, result in enumerate(walk_source(root, filters, max_files=req.max_files)):
        sample.append(str(result.path))
        if i >= req.max_files - 1:
            break

    return PreviewResponse(
        root_path=req.root_path,
        files_found=total,
        files_listed=len(sample),
        sample=sample,
    )
