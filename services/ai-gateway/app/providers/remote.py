"""Provider distant — OpenAI-compatible."""
import httpx
from .base import Provider, ProviderError, ProviderResult


class RemoteProvider(Provider):
    name = "remote"

    def __init__(self, base_url: str, api_key: str, default_model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
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
        if not self.base_url or not self.api_key:
            raise ProviderError("remote provider not configured")

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
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"remote error: {e}") from e

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ProviderResult(
            content=choice["message"]["content"],
            model=data.get("model", payload["model"]),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    async def health(self) -> bool:
        return bool(self.base_url and self.api_key)
