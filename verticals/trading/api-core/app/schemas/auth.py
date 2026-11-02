from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class MFAChallengeRequest(BaseModel):
    code: str


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    mfa_enabled: bool


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    user: UserProfile
