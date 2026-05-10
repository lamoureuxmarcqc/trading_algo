from fastapi import APIRouter, Depends

from app.api.deps import get_platform_service
from app.schemas.research import FactorRank, ResearchIdea, SectorRotation
from app.services.platform_service import PlatformService

router = APIRouter()


@router.get("/screener", response_model=list[ResearchIdea])
def get_screener(platform_service: PlatformService = Depends(get_platform_service)) -> list[ResearchIdea]:
    return platform_service.get_research_screener()


@router.get("/factors", response_model=list[FactorRank])
def get_factor_ranking(platform_service: PlatformService = Depends(get_platform_service)) -> list[FactorRank]:
    return platform_service.get_factor_ranking()


@router.get("/sectors", response_model=list[SectorRotation])
def get_sector_rotation(platform_service: PlatformService = Depends(get_platform_service)) -> list[SectorRotation]:
    return platform_service.get_sector_rotation()
