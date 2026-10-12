"""Configuration du service knowledge."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KNOW_", env_file=".env")

    database_url: str = "postgresql+psycopg://lamoureux:changeme@localhost:5433/lamoureux"

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    chunk_target_tokens: int = 400
    chunk_overlap_ratio: float = 0.10
    chunk_min_tokens: int = 50

    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    log_level: str = "INFO"


settings = Settings()
