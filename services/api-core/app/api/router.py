from fastapi import APIRouter

from app.api.routes import admin, ai, auth, health, portfolio, research, risk, terminal, trading, users

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(terminal.router, tags=["terminal"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(trading.router, prefix="/orders", tags=["trading"])
api_router.include_router(risk.router, prefix="/risk", tags=["risk"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(ai.router, tags=["ai"])
