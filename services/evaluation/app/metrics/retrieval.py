"""Metriques de retrieval."""
from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k : proportion de pertinents dans le top-k."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k : proportion de pertinents retrouves."""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """MRR : 1/rank du premier pertinent."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0
