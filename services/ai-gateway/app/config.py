"""Configuration du gateway."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIGW_", env_file=".env")

    # Providers
    ollama_url: str = "http://ollama:11434"
    ollama_default_model: str = "llama3.1:8b"

    vllm_url: str = "http://vllm:8000"
    vllm_default_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    remote_api_url: str = ""
    remote_api_key: str = ""
    remote_default_model: str = ""

    # Politique
    default_target: str = "LOCAL"
    allow_remote_for_public: bool = True
    request_timeout_seconds: float = 60.0

    # Observabilité
    log_level: str = "INFO"


settings = Settings()
