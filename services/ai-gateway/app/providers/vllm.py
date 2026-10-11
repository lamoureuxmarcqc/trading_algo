"""Provider vLLM — inférence GPU locale (API OpenAI-compatible)."""
import httpx
from .base import Provider, ProviderError, ProviderResult


class VLLMProvider(Provider):
    name = "vllm"

    def __init__(self, base_url: str, default_model: str, timeout: float = 120.0):
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"vllm error: {e}") from e

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ProviderResult(
            content=choice["message"]["content"],
            model=data.get("model", payload["model"]),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
