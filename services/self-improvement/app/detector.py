"""Detector : identifie les opportunites d'amelioration."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .models import Proposal, ProposalStatus, ProposalType


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def detect_code_issues(metrics: dict) -> list[Proposal]:
    """Detecte des opportunites d'amelioration du code."""
    proposals: list[Proposal] = []
    if metrics.get("test_coverage", 1.0) < 0.7:
        proposals.append(Proposal(
            id=str(uuid.uuid4()),
            type=ProposalType.CODE,
            target="coverage",
            issue=f"Couverture de tests : {metrics['test_coverage']:.2%}",
            proposal="Ajouter des tests manquants",
            risk="LOW",
        ))
    return proposals


def detect_ai_issues(metrics: dict) -> list[Proposal]:
    """Detecte des opportunites d'amelioration de l'IA."""
    proposals: list[Proposal] = []
    if metrics.get("hallucination_rate", 0.0) > 0.15:
        proposals.append(Proposal(
            id=str(uuid.uuid4()),
            type=ProposalType.AI,
            target="factuality",
            issue=f"Hallucination rate : {metrics['hallucination_rate']:.2%}",
            proposal="Renforcer les instructions de citation dans les prompts",
            risk="MEDIUM",
            auto_rollback_on=["hallucination_rate > 0.20"],
        ))
    return proposals


def detect(metrics: dict) -> list[Proposal]:
    """Detecte toutes les opportunites selon les 5 dimensions."""
    return (
        detect_code_issues(metrics)
        + detect_ai_issues(metrics)
    )
