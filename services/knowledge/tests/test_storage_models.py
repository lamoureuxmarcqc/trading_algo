"""Tests des modeles SQLAlchemy (sans DB)."""
from app.storage.models import Chunk, Document


def test_document_table_name():
    assert Document.__tablename__ == "documents"


def test_chunk_table_name():
    assert Chunk.__tablename__ == "chunks"


def test_document_columns():
    cols = {c.name for c in Document.__table__.columns}
    assert "id" in cols
    assert "tenant_id" in cols
    assert "classification" in cols
    assert "filename" in cols
    assert "ingested_at" in cols


def test_chunk_columns():
    cols = {c.name for c in Chunk.__table__.columns}
    assert "id" in cols
    assert "document_id" in cols
    assert "content" in cols
    assert "embedding" in cols
    assert "page" in cols
    assert "section" in cols