"""Superviseur : routing de taches vers agents."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentManifest:
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    classification_max: str = "INTERNAL"
    targets: list[str] = field(default_factory=lambda: ["LOCAL"])

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
