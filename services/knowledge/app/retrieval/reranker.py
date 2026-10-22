"""Reranking local post-RRF (cross-encoder) avec repli explicite.

Principes (AGENTS.md / ADR-001) :
- 100 % local : aucun appel réseau à l'inférence (le modèle est chargé depuis le cache local).
- Pas de fausse précision : chaque résultat expose ``rerank_source`` et ``confidence_source``.
  Si le cross-encoder est indisponible, on retombe sur l'ordre RRF et on l'indique ;
  on ne substitue JAMAIS silencieusement un autre scoreur.
- Le score RRF d'origine est conservé (``rrf_score``, ``rrf_confidence``) pour l'audit.
"""
from __future__ import annotations

import asyncio
import logging
import re
import math
from functools import lru_cache
from typing import Any, Literal, Protocol

logger = logging.getLogger("lamoureux_ai.knowledge.retrieval.reranker")

RerankSource = Literal["cross_encoder", "stub_lexical", "rrf_fallback"]

# Multilingue (corpus FR/EN). Surchargeable via settings.rerank_model.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_PASSAGE_CHARS = 2000
DEFAULT_BATCH_SIZE = 16


class Scorer(Protocol):
    """Interface d'un scoreur (query, passages) -> scores, un par passage."""

    name: RerankSource

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """Cross-encoder local (sentence-transformers). Chargement paresseux et mis en cache."""

    name: RerankSource = "cross_encoder"

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    def score(self, query: str, passages: list[str]) -> list[float]:
        model = _load_cross_encoder(self.model_name, self.device)
        pairs = [(query, p[:MAX_PASSAGE_CHARS]) for p in passages]
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in raw]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str | None) -> Any:
    from sentence_transformers import CrossEncoder  # import tardif : dépendance lourde

    logger.info("reranker_model_loading model=%s device=%s", model_name, device or "auto")
    return CrossEncoder(model_name, device=device)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class StubLexicalScorer:
    """Scoreur déterministe (recouvrement lexical) pour tests / CPU sans modèle.

    À activer explicitement (``rerank_backend="stub"``). Ne reflète PAS une pertinence
    sémantique ; ses résultats sont étiquetés ``stub_lexical``.
    """

    name: RerankSource = "stub_lexical"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_TOKEN_RE.findall(query.lower()))
        if not q:
            return [0.0 for _ in passages]
        out: list[float] = []
        for p in passages:
            tokens = set(_TOKEN_RE.findall(p.lower()))
            out.append(len(q & tokens) / len(q))
        return out




def _sigmoid(x: float) -> float:
    """Convertit un logit cross-encoder en probabilite [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

_COMMON_WORDS = frozenset({
    # Francais
    "le", "la", "les", "de", "du", "et", "un", "une", "des", "en",
    "au", "aux", "ce", "ces", "que", "qui", "pour", "par", "sur",
    "dans", "avec", "est", "sont", "sont", "pas", "ne", "se",
    # Anglais
    "the", "and", "to", "of", "in", "is", "for", "with", "this",
    "that", "on", "at", "by", "from", "as", "an", "or", "not",
})

_ASCII_SHIFT_WORD_RE = re.compile(r"\b[a-z]{2,}\b", re.UNICODE)


def _looks_like_ascii_shift(text: str) -> bool:
    """
    Detecte un decalage ASCII typique des PDF avec ToUnicode casse.

    Heuristique : un texte valide FR/EN contient au moins ~5% de mots
    courts courants (le, la, de, the, and, ...). Un texte decale n'en
    contient AUCUN car chaque lettre est shiftée.

    Ex : '8QLFDVW5HPRWH2EMHFW' -> 'UnicastRemoteObject' (shift -3)
    """
    if len(text) < 100:
        return False

    words = _ASCII_SHIFT_WORD_RE.findall(text.lower())
    if len(words) < 10:
        return False

    common = sum(1 for w in words if w in _COMMON_WORDS)
    ratio = common / len(words)

    # Un texte FR/EN valide a typiquement 10-20% de mots communs.
    # Si < 2%, c'est presque certainement du texte decale.
    return ratio < 0.02

def _is_low_alpha_density(text: str, min_ratio: float = 0.30) -> bool:
    """
    Detecte les chunks sans contenu textuel utile (matrices, tableaux binaires,
    listes de chiffres).

    Un texte naturel FR/EN contient au moins ~30% de lettres (a-z, A-Z, accents).
    Les matrices type "1 1 0 1 2 1..." ont 0% de lettres.
    """
    if not text:
        return True
    letters = sum(1 for c in text if c.isalpha())
    return letters / len(text) < min_ratio

def build_scorer(
    backend: str = "cross_encoder",
    *,
    model_name: str = DEFAULT_RERANK_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> Scorer:
    """Fabrique le scoreur. ``backend`` : ``cross_encoder`` (défaut) ou ``stub``."""
    if backend == "stub":
        return StubLexicalScorer()
    if backend == "cross_encoder":
        return CrossEncoderScorer(model_name, batch_size=batch_size, device=device)
    raise ValueError(f"rerank backend inconnu: {backend!r}")


def rerank(
    query: str,
    candidates: list[dict],
    *,
    top_k: int,
    scorer: Scorer | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Reclasse ``candidates`` (sortie de ``_rrf_fuse``) et retourne les ``top_k`` meilleurs.

    Champs ajoutés à chaque résultat :
    - ``rerank_score``      : score brut du scoreur
    - ``confidence``        : sigmoïde du score brut, dans [0, 1]
    - ``rrf_confidence``    : confiance RRF d'origine
    - ``pre_rerank_rank``   : rang avant reranking (1 = meilleur)
    - ``rerank_source`` / ``confidence_source`` : provenance du score

    Args:
        min_confidence : seuil sous lequel un chunk est écarté (0.0 = pas de filtre).
    """
    if top_k <= 0 or not candidates:
        return []

    # 1) Pré-filtre qualité : écarter les chunks inexploitables avant le scoring
    pre_filtered = []
    skipped_shifted = 0
    skipped_low_alpha = 0
    for c in candidates:
        content = c.get("content", "")
        if _looks_like_ascii_shift(content):
            skipped_shifted += 1
            continue
        if _is_low_alpha_density(content):
            skipped_low_alpha += 1
            continue
        pre_filtered.append(c)

    if skipped_shifted or skipped_low_alpha:
        logger.info(
            "rerank_prefilter_shifted skipped_shifted=%d skipped_low_alpha=%d total=%d",
            skipped_shifted, skipped_low_alpha, len(candidates),
        )
    candidates = pre_filtered
    if not candidates:
        return []

    # 2) Définir le scorer actif
    active = scorer or build_scorer()

    # 3) Scorer les candidats restants
    try:
        scores = active.score(query, [c["content"] for c in candidates])
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"scoreur a retourné {len(scores)} scores pour {len(candidates)} candidats"
            )
    except Exception as exc:  # noqa: BLE001 — repli explicite, jamais silencieux
        logger.warning("rerank_failed_fallback_rrf error=%s", exc, exc_info=True)
        return [
            {
                **c,
                "rerank_source": "rrf_fallback",
                "confidence_source": "rrf",
                "pre_rerank_rank": i + 1,
            }
            for i, c in enumerate(candidates[:top_k])
        ]

    # 4) Trier et filtrer par confiance
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

    results: list[dict] = []
    filtered_count = 0
    for i in order:
        conf = _sigmoid(scores[i])
        if conf < min_confidence:
            filtered_count += 1
            continue
        c = candidates[i]
        results.append(
            {
                **c,
                "rrf_confidence": c.get("confidence"),
                "rerank_score": scores[i],
                "confidence": conf,
                "pre_rerank_rank": i + 1,
                "rerank_source": active.name,
                "confidence_source": active.name,
            }
        )
        if len(results) >= top_k:
            break

    logger.info(
        "rerank_done source=%s candidates=%d returned=%d filtered_low_conf=%d min_conf=%.2f",
        active.name,
        len(candidates),
        len(results),
        filtered_count,
        min_confidence,
    )
    return results


async def rerank_async(
    query: str,
    candidates: list[dict],
    *,
    top_k: int,
    scorer: Scorer | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Version non bloquante : l'inférence CPU/GPU s'exécute dans un thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: rerank(
            query,
            candidates,
            top_k=top_k,
            scorer=scorer,
            min_confidence=min_confidence,
        ),
    )
