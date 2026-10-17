"""Agent abstrait."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    name: str
    allowed_tools: list[str] = field(default_factory=list)
    classification_max: str = "INTERNAL"
    budget_tokens: int = 8000
    timeout_seconds: int = 60


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def run(self, task: str, context: dict) -> dict:
        ...
