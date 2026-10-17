"""Observer : collecte des metriques multi-sources."""
from __future__ import annotations


def collect_metrics() -> dict:
    """
    Collecte les metriques depuis les autres services.

    V1 : stub. A connecter a evaluation, registry, api-core.
    """
    return {
        "test_coverage": 1.0,
        "hallucination_rate": 0.0,
        "factuality": 1.0,
    }
