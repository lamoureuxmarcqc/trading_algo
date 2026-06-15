from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.deps import get_platform_service
from app.schemas.terminal import (
    TerminalSnapshotResponse,
    TradingAlgoCommandRequest,
    TradingAlgoCommandResponse,
)
from app.services.platform_service import PlatformService
from app.services.trading_algo_terminal import TradingAlgoTerminalService

router = APIRouter()

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@router.get("/terminal", include_in_schema=False)
def terminal() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@router.get("/terminal/snapshot", response_model=TerminalSnapshotResponse)
def terminal_snapshot(
    platform_service: PlatformService = Depends(get_platform_service),
) -> TerminalSnapshotResponse:
    return platform_service.get_terminal_snapshot()


@router.post("/terminal/trading-algo", response_model=TradingAlgoCommandResponse)
def run_trading_algo_command(payload: TradingAlgoCommandRequest) -> TradingAlgoCommandResponse:
    return TradingAlgoTerminalService().run(payload)
