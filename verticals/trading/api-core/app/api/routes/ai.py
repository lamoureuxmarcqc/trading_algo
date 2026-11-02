from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status

from app.api.deps import get_platform_service
from app.core.config import settings
from app.events.dispatcher import outbox_dispatcher
from app.schemas.ai import (
    AiBacktestRequest,
    AiBacktestResponse,
    AiDiagnosticsResponse,
    AiExplanationResponse,
    AiPaperStageRequest,
    AiPaperStageResponse,
    AiPredictionBandsResponse,
    AiSizingRequest,
    AiSizingResponse,
    ForecastResponse,
    RegimeResponse,
    SignalResponse,
)
from app.services.ai_trading_service import (
    AiTradingService,
    PaperTradingDisabledError,
    PaperTradingGuardrailError,
)
from app.services.platform_service import IdempotencyConflictError, PlatformService

router = APIRouter()


def get_ai_trading_service() -> AiTradingService:
    return AiTradingService()


@router.get("/signals/{symbol}", response_model=SignalResponse)
def get_signal(
    symbol: str,
    platform_service: PlatformService = Depends(get_platform_service),
) -> SignalResponse:
    return platform_service.get_signal(symbol.upper())


@router.get("/forecast/{symbol}", response_model=ForecastResponse)
def get_forecast(
    symbol: str,
    platform_service: PlatformService = Depends(get_platform_service),
) -> ForecastResponse:
    return platform_service.get_forecast(symbol.upper())


@router.get("/regime", response_model=RegimeResponse)
def get_regime(platform_service: PlatformService = Depends(get_platform_service)) -> RegimeResponse:
    return platform_service.get_regime()


@router.get("/ai/models/{symbol}/diagnostics", response_model=AiDiagnosticsResponse)
def get_model_diagnostics(
    symbol: str,
    period: str = "1y",
    ai_trading_service: AiTradingService = Depends(get_ai_trading_service),
) -> AiDiagnosticsResponse:
    try:
        return ai_trading_service.get_diagnostics(symbol.upper(), period)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/ai/models/{symbol}/explanation", response_model=AiExplanationResponse)
def get_model_explanation(
    symbol: str,
    period: str = "1y",
    ai_trading_service: AiTradingService = Depends(get_ai_trading_service),
) -> AiExplanationResponse:
    try:
        return ai_trading_service.get_explanation(symbol.upper(), period)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/ai/predictions/{symbol}/bands", response_model=AiPredictionBandsResponse)
def get_prediction_bands(
    symbol: str,
    period: str = "1y",
    ai_trading_service: AiTradingService = Depends(get_ai_trading_service),
) -> AiPredictionBandsResponse:
    try:
        return ai_trading_service.get_prediction_bands(symbol.upper(), period)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/ai/sizing", response_model=AiSizingResponse)
def size_ai_position(
    payload: AiSizingRequest,
    ai_trading_service: AiTradingService = Depends(get_ai_trading_service),
) -> AiSizingResponse:
    try:
        return ai_trading_service.size_position(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/ai/backtests", response_model=AiBacktestResponse)
def run_ai_backtest(
    payload: AiBacktestRequest,
    ai_trading_service: AiTradingService = Depends(get_ai_trading_service),
) -> AiBacktestResponse:
    return ai_trading_service.run_backtest(payload)


@router.post("/ai/paper-trading/stage", response_model=AiPaperStageResponse)
def stage_ai_paper_orders(
    payload: AiPaperStageRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    platform_service: PlatformService = Depends(get_platform_service),
    ai_trading_service: AiTradingService = Depends(get_ai_trading_service),
) -> AiPaperStageResponse:
    idempotency_key = getattr(request.state, "idempotency_key", None)
    if not idempotency_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key header is required for AI paper-trading staging.",
        )

    try:
        candidates, skipped, warnings = ai_trading_service.plan_paper_stage(payload)
        result = platform_service.stage_paper_orders(
            payload,
            candidates,
            skipped,
            warnings,
            idempotency_key=idempotency_key,
            tenant_id=getattr(request.state, "tenant_id", "public"),
            correlation_id=getattr(request.state, "correlation_id", None),
        )
    except PaperTradingDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except PaperTradingGuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    response.status_code = (
        status.HTTP_200_OK if result.replayed or not result.response.orders else status.HTTP_201_CREATED
    )
    response.headers["X-Idempotency-Status"] = "replayed" if result.replayed else "created"
    if settings.outbox_dispatch_after_write and not result.replayed and result.response.orders:
        background_tasks.add_task(outbox_dispatcher.dispatch_pending, len(result.response.orders))
    return result.response
