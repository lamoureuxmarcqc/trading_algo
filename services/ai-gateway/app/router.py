"""Routage : classification → provider, avec fallback."""
import structlog
from .config import settings
from .models import DataClassification, ModelTarget
from .privacy import resolve_auto
from .providers import OllamaProvider, VLLMProvider, RemoteProvider, Provider, ProviderError

logger = structlog.get_logger()


class Router:
    def __init__(self) -> None:
        self._providers: dict[ModelTarget, Provider] = {
            ModelTarget.LOCAL: OllamaProvider(
                settings.ollama_url, settings.ollama_default_model, settings.request_timeout_seconds
            ),
            ModelTarget.LOCAL_GPU: VLLMProvider(
                settings.vllm_url, settings.vllm_default_model, settings.request_timeout_seconds * 2
            ),
            ModelTarget.REMOTE: RemoteProvider(
                settings.remote_api_url, settings.remote_api_key, settings.remote_default_model
            ),
        }

    def providers(self) -> dict[ModelTarget, Provider]:
        return self._providers

    def resolve(self, classification: DataClassification, requested: ModelTarget) -> ModelTarget:
        if requested == ModelTarget.AUTO:
            return resolve_auto(classification)
        return requested

    def get(self, target: ModelTarget) -> Provider:
        provider = self._providers.get(target)
        if provider is None:
            raise ProviderError(f"no provider for target {target.value}")
        return provider

    async def health(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for target, provider in self._providers.items():
            try:
                result[target.value] = await provider.health()
            except Exception:  # pragma: no cover - defensive
                result[target.value] = False
        return result


router = Router()
