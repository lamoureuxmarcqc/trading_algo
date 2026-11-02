from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from app.api.deps import get_platform_service
from app.schemas.terminal import (
    MonteCarloSimulationRequest,
    MonteCarloSimulationResponse,
    TerminalSnapshotResponse,
    TerminalKnowledgeQueryRequest,
    TerminalKnowledgeQueryResponse,
    TradingAlgoCommandRequest,
    TradingAlgoCommandResponse,
)
from app.services.platform_service import PlatformService
from app.services.trading_algo_terminal import KnowledgeQueryError, TradingAlgoTerminalService

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


@router.post("/terminal/knowledge", response_model=TerminalKnowledgeQueryResponse)
def query_terminal_knowledge(
    payload: TerminalKnowledgeQueryRequest,
    request: Request,
) -> TerminalKnowledgeQueryResponse:
    """Expose local RAG evidence to the terminal without accepting RESTRICTED input."""
    tenant_id = getattr(request.state, "tenant_id", "public")
    try:
        return TradingAlgoTerminalService().query_knowledge(
            query=payload.query,
            tenant_id=tenant_id,
            top_k=payload.top_k,
        )
    except KnowledgeQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge service is unavailable",
        ) from exc


@router.post("/simulate/monte-carlo", response_model=MonteCarloSimulationResponse)
def run_monte_carlo_simulation(
    payload: MonteCarloSimulationRequest,
    platform_service: PlatformService = Depends(get_platform_service),
) -> MonteCarloSimulationResponse:
    return platform_service.run_monte_carlo_simulation(payload)
