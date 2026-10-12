"""Tests des embeddings (mode stub)."""
import numpy as np

from app.embeddings.local import embed_passages, embed_query, embed_texts


def test_embed_texts_returns_correct_shape():
    vectors = embed_texts(["hello", "world"], "stub")
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32


def test_embed_texts_normalized():
    vectors = embed_texts(["a", "b", "c"], "stub")
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_deterministic():
    v1 = embed_texts(["same text"], "stub")
    v2 = embed_texts(["same text"], "stub")
    assert np.array_equal(v1, v2)


def test_embed_different_for_different_texts():
    v = embed_texts(["alpha", "beta"], "stub")
    # Les deux vecteurs doivent etre differents
    assert not np.allclose(v[0], v[1])


def test_embed_empty_list():
    vectors = embed_texts([], "stub")
    assert vectors.shape == (0, 384)


def test_embed_query_single_vector():
    v = embed_query("test query", "stub")
    assert v.shape == (384,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_embed_passages_batch():
    v = embed_passages(["p1", "p2", "p3"], "stub")
    assert v.shape == (3, 384)