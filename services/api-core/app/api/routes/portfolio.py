from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.api.deps import get_platform_service
from app.core.config import settings
from app.events.dispatcher import outbox_dispatcher
from app.schemas.portfolio import (
    AllocationOptimizationResponse,
    BarbellAllocationResponse,
    BarbellStrategyConfig,
    PortfolioHistoryPoint,
    PortfolioOverview,
    PortfolioPerformance,
    PortfolioPosition,
    PortfolioRefreshResponse,
    RebalanceRequest,
    RebalanceResponse,
)
from app.services.platform_service import PlatformService

router = APIRouter()


@router.get("", response_model=PortfolioOverview)
def get_portfolio(platform_service: PlatformService = Depends(get_platform_service)) -> PortfolioOverview:
    return platform_service.get_portfolio_overview()


@router.get("/performance", response_model=PortfolioPerformance)
def get_performance(
    platform_service: PlatformService = Depends(get_platform_service),
) -> PortfolioPerformance:
    return platform_service.get_portfolio_performance()


@router.get("/positions", response_model=list[PortfolioPosition])
def get_positions(platform_service: PlatformService = Depends(get_platform_service)) -> list[PortfolioPosition]:
    return platform_service.get_positions()


@router.get("/history", response_model=list[PortfolioHistoryPoint])
def get_history(
    limit: int = 30,
    platform_service: PlatformService = Depends(get_platform_service),
) -> list[PortfolioHistoryPoint]:
    return platform_service.get_portfolio_history(limit=limit)


@router.get("/barbell", response_model=BarbellAllocationResponse)
def get_barbell_allocation(
    platform_service: PlatformService = Depends(get_platform_service),
) -> BarbellAllocationResponse:
    return platform_service.get_barbell_allocation()


@router.post("/barbell", response_model=BarbellAllocationResponse)
def build_barbell_allocation(
    payload: BarbellStrategyConfig,
    platform_service: PlatformService = Depends(get_platform_service),
) -> BarbellAllocationResponse:
    return platform_service.get_barbell_allocation(payload)


@router.post("/rebalance", response_model=RebalanceResponse)
def rebalance_portfolio(
    request: Request,
    payload: RebalanceRequest,
    platform_service: PlatformService = Depends(get_platform_service),
) -> RebalanceResponse:
    return platform_service.rebalance_portfolio(
        payload,
        tenant_id=getattr(request.state, "tenant_id", "public"),
        correlation_id=getattr(request.state, "correlation_id", None),
    )


@router.post("/barbell/rebalance", response_model=RebalanceResponse)
def rebalance_barbell_portfolio(
    request: Request,
    platform_service: PlatformService = Depends(get_platform_service),
) -> RebalanceResponse:
    return platform_service.rebalance_barbell_portfolio(
        tenant_id=getattr(request.state, "tenant_id", "public"),
        correlation_id=getattr(request.state, "correlation_id", None),
    )


@router.post("/optimize-allocation", response_model=AllocationOptimizationResponse)
def optimize_allocation(
    platform_service: PlatformService = Depends(get_platform_service),
) -> AllocationOptimizationResponse:
    return platform_service.optimize_barbell_allocation()


@router.post("/optimize-allocation/apply", response_model=AllocationOptimizationResponse)
def apply_optimized_allocation(
    request: Request,
    platform_service: PlatformService = Depends(get_platform_service),
) -> AllocationOptimizationResponse:
    return platform_service.optimize_barbell_allocation(
        apply_to_barbell=True,
        tenant_id=getattr(request.state, "tenant_id", "public"),
    )


@router.post("/refresh", response_model=PortfolioRefreshResponse)
def refresh_portfolio_market_data(
    request: Request,
    background_tasks: BackgroundTasks,
    platform_service: PlatformService = Depends(get_platform_service),
) -> PortfolioRefreshResponse:
    result = platform_service.refresh_portfolio_market_data(
        force=True,
        tenant_id=getattr(request.state, "tenant_id", "public"),
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    if settings.outbox_dispatch_after_write:
        background_tasks.add_task(outbox_dispatcher.dispatch_pending, 1)
    return result
