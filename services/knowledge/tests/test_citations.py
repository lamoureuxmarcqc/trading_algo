"""Tests des citations exactes exposees par le Knowledge Layer."""
from __future__ import annotations

import pytest

from app.models import QueryResponse
from app.retrieval.citations import build_sources, exact_excerpt, to_source


def _result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "content": "Le risque de marche demeure eleve dans ce scenario.",
        "page": 7,
        "section": "Risques",
        "confidence": 0.82,
    }
    result.update(overrides)
    return result


def test_to_source_preserves_traceability_and_exact_excerpt() -> None:
    content = "A" * 501
    source = to_source(_result(content=content), max_excerpt_chars=500)

    assert source.document_id == "document-1"
    assert source.chunk_id == "chunk-1"
    assert source.page == 7
    assert source.section == "Risques"
    assert source.excerpt == content[:500]
    assert source.score == pytest.approx(0.82)


def test_citation_score_is_bounded_without_claiming_calibration() -> None:
    source = to_source(_result(confidence=4.2))

    assert source.score == 1.0


def test_build_sources_preserves_retrieval_order() -> None:
    sources = build_sources([_result(chunk_id="first"), _result(chunk_id="second")])

    assert [source.chunk_id for source in sources] == ["first", "second"]


def test_invalid_citation_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty content"):
        to_source(_result(content=""))

    with pytest.raises(ValueError, match="greater than zero"):
        exact_excerpt("content", max_chars=0)


def test_query_response_exposes_answer_sources_and_confidence() -> None:
    response = QueryResponse(
        query="Quels risques sont cites ?",
        answer=None,
        sources=build_sources([_result()]),
        confidence=0.82,
    )

    assert response.model_dump()["answer"] is None
    assert response.sources[0].excerpt.startswith("Le risque")
