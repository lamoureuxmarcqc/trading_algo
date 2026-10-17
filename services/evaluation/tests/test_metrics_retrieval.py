"""Tests des metriques retrieval."""
from app.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_half():
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5


def test_recall_full():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5


def test_mrr_first_hit():
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_mrr_second_hit():
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5


def test_mrr_no_hit():
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0
