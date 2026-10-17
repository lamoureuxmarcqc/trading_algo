"""Agent finance (squelette)."""
from .base import Agent, AgentConfig


class FinanceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(AgentConfig(
            name="finance",
            allowed_tools=["knowledge.query", "memory.decision.record"],
            classification_max="HIGHLY_CONFIDENTIAL",
            budget_tokens=8000,
            timeout_seconds=60,
        ))

    async def run(self, task: str, context: dict) -> dict:
        return {
            "agent": "finance",
            "task": task,
            "status": "not_implemented",
            "note": "Phase 10 — Investment Agent",
        }
