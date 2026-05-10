from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_platform_service
from app.schemas.risk import PortfolioRiskSnapshot, PositionRisk, ScenarioResult
from app.services.platform_service import PlatformService

router = APIRouter()


@router.get("/portfolio", response_model=PortfolioRiskSnapshot)
def get_portfolio_risk(
    platform_service: PlatformService = Depends(get_platform_service),
) -> PortfolioRiskSnapshot:
    return platform_service.get_portfolio_risk()


@router.get("/positions", response_model=list[PositionRisk])
def get_position_risk(
    platform_service: PlatformService = Depends(get_platform_service),
) -> list[PositionRisk]:
    return platform_service.get_position_risk()


@router.get("/scenario/{scenario_id}", response_model=ScenarioResult)
def get_scenario(
    scenario_id: str,
    platform_service: PlatformService = Depends(get_platform_service),
) -> ScenarioResult:
    result = platform_service.get_scenario_result(scenario_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")
    return result
