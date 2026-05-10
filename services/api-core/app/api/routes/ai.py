from fastapi import APIRouter, Depends

from app.api.deps import get_platform_service
from app.schemas.ai import ForecastResponse, RegimeResponse, SignalResponse
from app.services.platform_service import PlatformService

router = APIRouter()


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
