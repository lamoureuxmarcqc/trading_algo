"""Contrats Pydantic."""
from enum import Enum
from pydantic import BaseModel, Field


class ProposalType(str, Enum):
    CODE = "CODE"
    AI = "AI"
    KNOWLEDGE = "KNOWLEDGE"
    FEATURE = "FEATURE"
    AGENT = "AGENT"


class ProposalStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


class Proposal(BaseModel):
    id: str
    type: ProposalType
    target: str
    issue: str
    proposal: str
    risk: str = "LOW"
    status: ProposalStatus = ProposalStatus.PENDING
    requires_human_review: bool = True
    auto_rollback_on: list[str] = Field(default_factory=list)
