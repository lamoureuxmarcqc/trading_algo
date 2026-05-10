from fastapi import APIRouter, Depends

from app.api.deps import get_platform_service
from app.schemas.auth import UserProfile
from app.services.platform_service import PlatformService

router = APIRouter()


@router.get("/me", response_model=UserProfile)
def read_me(platform_service: PlatformService = Depends(get_platform_service)) -> UserProfile:
    return platform_service.get_current_user()
