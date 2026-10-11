"""Provider Ollama — inférence locale."""
import httpx
from .base import Provider, ProviderError, ProviderResult


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str, default_model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None = None,
    ) -> ProviderResult:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama error: {e}") from e

        return ProviderResult(
            content=data.get("response", ""),
            model=data.get("model", payload["model"]),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
