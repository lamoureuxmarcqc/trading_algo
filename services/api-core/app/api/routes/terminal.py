from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@router.get("/terminal", include_in_schema=False)
def terminal() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")

