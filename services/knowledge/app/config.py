"""Configuration du service knowledge."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Le depot possede un .env partage avec les autres services. Les variables
    # hors prefixe KNOW_ ne sont pas des erreurs de configuration du service.
    model_config = SettingsConfigDict(
        env_prefix="KNOW_",
        env_file=".env",
        extra="ignore",
    )

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
    # Le reranking est opt-in : sans configuration explicite, la recherche
    # conserve le comportement RRF et ne charge aucun modele supplementaire.
    rerank_enabled: bool = True
    rerank_backend: str = "cross_encoder"
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidates: int = 30
    rerank_batch_size: int = 16
    rerank_device: str | None = None     
    rerank_min_confidence: float =  0.05 # 0.20

settings = Settings()
