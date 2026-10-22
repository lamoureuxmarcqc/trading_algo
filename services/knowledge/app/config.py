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

    log_level: str = "INFO"

    # Retrieval + reranking
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    rerank_enabled: bool = True                                # ← AJOUT
    rerank_backend: str = "cross_encoder"                      # ← AJOUT
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # ← AJOUT
    rerank_candidates: int = 30                                # ← AJOUT
    rerank_batch_size: int = 16                                # ← AJOUT
    rerank_device: str | None = None                           # ← AJOUT
    rerank_min_confidence: float = 0.05        # ← 0.15 → 0.05


settings = Settings()
