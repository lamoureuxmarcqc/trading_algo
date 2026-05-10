from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_platform_service
from app.core.security import create_access_token
from app.schemas.auth import LoginRequest, LoginResponse, MFAChallengeRequest
from app.services.platform_service import PlatformService

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    platform_service: PlatformService = Depends(get_platform_service),
) -> LoginResponse:
    user = platform_service.authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(subject=user.email)
    refresh_token = f"refresh-{user.id}"
    platform_service.create_user_session(user.email, refresh_token)
    return LoginResponse(
        access_token=token,
        refresh_token=refresh_token,
        expires_in=1800,
        user=user,
    )


@router.post("/logout")
def logout() -> dict[str, str]:
    return {"status": "logged_out"}


@router.post("/mfa")
def verify_mfa(payload: MFAChallengeRequest) -> dict[str, str]:
    if payload.code != "123456":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid MFA code")
    return {"status": "verified"}
