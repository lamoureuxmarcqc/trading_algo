"""Contrats Pydantic du service memory."""
from enum import Enum
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM = "LONG_TERM"
    KNOWLEDGE = "KNOWLEDGE"
    DECISION = "DECISION"


class ShortTermAppend(BaseModel):
    tenant_id: str
    session_id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class LongTermValidate(BaseModel):
    tenant_id: str
    fact: str
    source: str | None = None
    human_validation: bool = Field(..., description="Doit etre true")


class DecisionRecord(BaseModel):
    tenant_id: str
    decision: str
    reason: str
    evidence: dict | None = None
    action: str | None = None
    model_version: str | None = None
    confidence: float | None = None
    correlation_id: str | None = None


class DecisionEvaluate(BaseModel):
    outcome: dict
    evaluation: dict | None = None
