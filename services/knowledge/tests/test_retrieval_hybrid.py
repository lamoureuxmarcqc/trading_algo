"""Tests de la fusion RRF (sans DB)."""
from app.retrieval.hybrid import _rrf_fuse


def _chunk(cid: str, content: str = "") -> dict:
    return {
        "chunk_id": cid,
        "document_id": f"doc-{cid}",
        "content": content or f"content {cid}",
        "page": 1,
        "section": None,
        "classification": "INTERNAL",
        "score": 0.5,
    }


def test_rrf_empty_inputs():
    result = _rrf_fuse(vector_results=[], keyword_results=[], top_k=5)
    assert result == []


def test_rrf_only_vector():
    v = [_chunk("a"), _chunk("b"), _chunk("c")]
    result = _rrf_fuse(vector_results=v, keyword_results=[], top_k=3)

    assert len(result) == 3
    assert result[0]["chunk_id"] == "a"
    assert result[1]["chunk_id"] == "b"
    assert result[2]["chunk_id"] == "c"
    assert all("rrf_score" in r for r in result)


def test_rrf_only_keyword():
    k = [_chunk("x"), _chunk("y")]
    result = _rrf_fuse(vector_results=[], keyword_results=k, top_k=2)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "x"


def test_rrf_boost_overlap():
    v = [_chunk("common"), _chunk("v1"), _chunk("v2")]
    k = [_chunk("k1"), _chunk("common"), _chunk("k2")]

    result = _rrf_fuse(vector_results=v, keyword_results=k, top_k=5)
    by_id = {r["chunk_id"]: r for r in result}

    assert result[0]["chunk_id"] == "common"
    assert by_id["common"]["rrf_score"] > by_id["v1"]["rrf_score"]


def test_rrf_top_k_limits():
    v = [_chunk(f"v{i}") for i in range(20)]
    result = _rrf_fuse(vector_results=v, keyword_results=[], top_k=5)
    assert len(result) == 5


def test_rrf_tracks_source_ranks():
    v = [_chunk("a"), _chunk("b")]
    k = [_chunk("b"), _chunk("c")]

    result = _rrf_fuse(vector_results=v, keyword_results=k, top_k=5)
    by_id = {r["chunk_id"]: r for r in result}

    assert by_id["a"]["vector_rank"] == 1
    assert by_id["b"]["vector_rank"] == 2
    assert by_id["b"]["keyword_rank"] == 1
    assert by_id["c"]["keyword_rank"] == 2


def test_rrf_score_decreases():
    v = [_chunk(f"c{i}") for i in range(5)]
    result = _rrf_fuse(vector_results=v, keyword_results=[], top_k=5)

    scores = [r["rrf_score"] for r in result]
    assert scores == sorted(scores, reverse=True)
