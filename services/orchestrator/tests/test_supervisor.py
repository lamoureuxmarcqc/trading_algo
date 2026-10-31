"""Tests du superviseur."""
import asyncio

import pytest

from app.agents.base import Agent, AgentConfig
from app.supervisor import AgentBudgetExceededError, Supervisor, classify_task, route


class RecordingFinanceAgent(Agent):
    def __init__(self, budget_tokens: int) -> None:
        super().__init__(
            AgentConfig(
                name="finance",
                budget_tokens=budget_tokens,
                timeout_seconds=5,
            )
        )
        self.was_called = False

    async def run(self, task: str, context: dict) -> dict:
        self.was_called = True
        return {"status": "ok", "task": task}


def test_classify_finance():
    assert classify_task("Analyse mon portefeuille d'actions") == "finance"


def test_classify_health():
    assert classify_task("J'ai des symptomes medicaux") == "health"


def test_classify_research_default():
    assert classify_task("Trouve moi des articles") == "research"


def test_route_returns_agent():
    assert route("Investir dans AAPL") == "finance"


def test_supervisor_rejects_task_that_exceeds_agent_token_budget():
    agent = RecordingFinanceAgent(budget_tokens=10)
    supervisor = Supervisor({"finance": agent})
    task = "action un deux trois quatre cinq six sept huit neuf dix"

    with pytest.raises(AgentBudgetExceededError, match="budget"):
        asyncio.run(supervisor.run(task))

    assert agent.was_called is False
