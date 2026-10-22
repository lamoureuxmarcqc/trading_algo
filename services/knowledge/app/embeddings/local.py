"""Embeddings locaux via sentence-transformers.

Mode lazy : le modele n'est charge qu'au premier appel.
Mode test : utilise un stub deterministe quand KNOW_EMBEDDING_MODE=stub.
"""
from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Erreur generique d'embedding."""


def _is_stub_mode() -> bool:
    return os.getenv("KNOW_EMBEDDING_MODE", "").lower() == "stub"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Charge le modele une seule fois (lazy)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingError(
            "sentence-transformers non installe. "
            "pip install sentence-transformers"
        ) from e

    logger.info("Chargement du modele d'embedding : %s", model_name)
    model = SentenceTransformer(model_name)
    # Sentence-transformers >3.0 renomme get_sentence_embedding_dimension
    # en get_embedding_dimension. On supporte les deux pour compatibilite.
    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()

    logger.info("Modele charge (dim=%s)", dim)
    return model


def _stub_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """
    Stub deterministe pour les tests.

    Chaque texte -> vecteur normalise derive d'un hash SHA256.
    Reproductible, pas de telechargement, pas de dependance.
    """
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repete le hash pour remplir `dim` valeurs
        seed = int.from_bytes(h[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        result[i] = vec
    return result


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """
    Encode une liste de textes en vecteurs denses.

    Args:
        texts: liste de textes
        model_name: nom HuggingFace ou stub

    Returns:
        array shape (len(texts), dim)
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    if _is_stub_mode():
        return _stub_embed(texts)

    model = _load_model(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Encode une requete unique (prefixe E5 recommande pour queries)."""
    if _is_stub_mode():
        return _stub_embed([query])[0]
    # e5 recommande "query: " pour les requetes, "passage: " pour les docs
    if "e5" in model_name.lower():
        query = f"query: {query}"
    return embed_texts([query], model_name)[0]


def embed_passages(passages: list[str], model_name: str) -> np.ndarray:
    """Encode des passages (documents) avec prefixe E5 si applicable."""
    if not passages:
        return np.zeros((0, 384), dtype=np.float32)
    if _is_stub_mode():
        return _stub_embed(passages)
    if "e5" in model_name.lower():
        passages = [f"passage: {p}" for p in passages]
    return embed_texts(passages, model_name)