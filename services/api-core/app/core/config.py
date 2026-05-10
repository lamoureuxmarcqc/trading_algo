from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Hedge Fund Platform API"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"

    jwt_secret: str = Field(default="change-me-in-production", alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    postgres_dsn: str = Field(
        default="postgresql+psycopg://hedgefund:hedgefund@postgres:5432/hedgefund",
        alias="POSTGRES_DSN",
    )
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    db_echo: bool = False
    auto_create_tables: bool = True
    auto_seed_demo_data: bool = True
    default_user_email: str = "cio@hedgefund.local"

    default_base_currency: str = "USD"
    default_benchmark: str = "SPY"
    enable_demo_data: bool = True
    market_price_refresh_seconds: int = 300
    outbox_dispatch_batch_size: int = 100
    outbox_dispatch_on_startup: bool = True
    outbox_dispatch_after_write: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
