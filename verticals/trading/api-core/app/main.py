"""FastAPI entrypoint for the trading vertical."""
import logging
import sys
from pathlib import Path

# ─── Bootstrap: make `trading_algo` importable from verticals/trading/ ───
_VERTICAL_ROOT = Path(__file__).resolve().parents[2]  # → verticals/trading/
if str(_VERTICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERTICAL_ROOT))
# ─────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import settings
from app.db.init_db import initialize_database
from app.events.dispatcher import outbox_dispatcher
from app.middleware import institutional_http_middleware

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(institutional_http_middleware)
    app.include_router(api_router, prefix=settings.api_prefix)
    app.mount("/terminal-assets", StaticFiles(directory=WEB_DIR), name="terminal-assets")

    if settings.auto_create_tables:
        @app.on_event("startup")
        def startup_init() -> None:
            try:
                initialize_database(seed_demo_data=settings.auto_seed_demo_data)
            except Exception as exc:
                logger.warning("Database initialization skipped: %s", exc)
            if settings.outbox_dispatch_on_startup:
                try:
                    outbox_dispatcher.dispatch_pending()
                except Exception as exc:
                    logger.warning("Outbox dispatch on startup skipped: %s", exc)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/api/v1/terminal")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> RedirectResponse:
        return RedirectResponse(url="/terminal-assets/favicon.svg")

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    return app


app = create_app()