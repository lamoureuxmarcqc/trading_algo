"""Configuration du service memory."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEM_", env_file=".env")

    database_url: str = "postgresql+psycopg://lamoureux:changeme@localhost:5433/lamoureux"
    redis_url: str = "redis://localhost:6379/1"
    short_term_ttl_seconds: int = 3600
    log_level: str = "INFO"


settings = Settings()
