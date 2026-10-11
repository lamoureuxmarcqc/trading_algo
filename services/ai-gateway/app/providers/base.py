"""Interface provider."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(Exception):
    """Erreur générique d'un provider."""


@dataclass
class ProviderResult:
    content: str
    model: str
    tokens_in: int
    tokens_out: int


class Provider(ABC):
    name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str | None = None,
    ) -> ProviderResult:
        ...

    @abstractmethod
    async def health(self) -> bool:
        ...
