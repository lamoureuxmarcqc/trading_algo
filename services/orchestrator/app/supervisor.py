"""Superviseur : routing de taches vers agents."""
from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping

from .agents.base import Agent

logger = logging.getLogger(__name__)


@dataclass
class AgentManifest:
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    classification_max: str = "INTERNAL"
    targets: list[str] = field(default_factory=lambda: ["LOCAL"])


class AgentNotRegisteredError(ValueError):
    """Raised when task classification has no configured agent."""


class AgentBudgetExceededError(ValueError):
    """Raised before execution when a task exceeds an agent's token budget."""


class AgentTimeoutError(TimeoutError):
    """Raised when an agent exceeds its configured execution timeout."""


def estimate_task_tokens(task_description: str) -> int:
    """Return a deterministic, conservative whitespace-token estimate.

    This guardrail intentionally estimates rather than claiming tokenizer-level
    precision. Providers can later supply measured token usage to the audit
    trail without weakening the pre-execution budget check.
    """
    return len(task_description.split())


class Supervisor:
    """Routes work to registered agents and enforces their runtime guardrails."""

    def __init__(self, agents: Mapping[str, Agent]) -> None:
        self._agents = dict(agents)

    async def run(self, task_description: str, context: dict[str, Any] | None = None) -> dict:
        """Classify and execute a task within the selected agent's constraints."""
        agent_name = route(task_description)
        try:
            agent = self._agents[agent_name]
        except KeyError as exc:
            raise AgentNotRegisteredError(f"No registered agent for {agent_name!r}") from exc

        config = agent.config
        estimated_tokens = estimate_task_tokens(task_description)
        if estimated_tokens > config.budget_tokens:
            logger.warning(
                "agent_budget_exceeded agent=%s estimated_tokens=%d budget_tokens=%d",
                config.name,
                estimated_tokens,
                config.budget_tokens,
            )
            raise AgentBudgetExceededError(
                f"Task estimate ({estimated_tokens} tokens) exceeds "
                f"agent {config.name!r} budget ({config.budget_tokens} tokens)"
            )

        if config.timeout_seconds <= 0:
            raise ValueError(f"Agent {config.name!r} must have a positive timeout")

        try:
            return await asyncio.wait_for(
                agent.run(task_description, context or {}),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "agent_timeout agent=%s timeout_seconds=%d",
                config.name,
                config.timeout_seconds,
            )
            raise AgentTimeoutError(
                f"Agent {config.name!r} exceeded timeout of {config.timeout_seconds} seconds"
            ) from exc

def classify_task(task_description: str) -> str:
    """Classifie une tache et retourne le nom de l'agent."""
    desc = task_description.lower()

    finance_keywords = [
        "action", "actions", "portefeuille", "investissement", "investir",
        "bourse", "titre", "titres", "dividende", "rendement", "boursier",
        "valuation", "buffett", "aapl", "msft", "nvda", "tsx", "nasdaq",
    ]
    health_keywords = [
        "sante", "santé", "medical", "médical", "symptome", "symptôme",
        "traitement", "medicament", "médicament", "docteur", "medecin", "médecin",
    ]

    if any(k in desc for k in finance_keywords):
        return "finance"
    if any(k in desc for k in health_keywords):
        return "health"
    return "research"

def route(task_description: str) -> str:
    """Retourne l'agent a utiliser."""
    agent = classify_task(task_description)
    logger.info("Task routed to agent=%s", agent)
    return agent
