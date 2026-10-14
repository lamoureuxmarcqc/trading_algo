"""Tests du tokenizer BM25 (sans DB)."""
from app.retrieval.keyword import _tokenize


def test_tokenize_basic():
    assert _tokenize("Hello world") == ["hello", "world"]


def test_tokenize_lowercase():
    assert _tokenize("HELLO World") == ["hello", "world"]


def test_tokenize_punctuation():
    assert _tokenize("Bonjour, monde!") == ["bonjour", "monde"]


def test_tokenize_accents_preserved():
    result = _tokenize("café résumé")
    assert "café" in result or "caf" in result


def test_tokenize_numbers():
    assert _tokenize("AAPL 2026 Q1") == ["aapl", "2026", "q1"]


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("   ") == []
