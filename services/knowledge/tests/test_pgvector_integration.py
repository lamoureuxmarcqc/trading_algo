"""Optional integration coverage for the PostgreSQL + pgvector storage path."""
from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.models import DataClassification
from app.storage.db import get_session
from app.storage.models import Chunk, Document
from app.storage.pgvector import insert_document, vector_search


def _pgvector_database_available() -> bool:
    """Check that the configured test database has migrated pgvector tables."""
    try:
        engine = create_engine(
            settings.database_url,
            connect_args={"connect_timeout": 1},
            pool_pre_ping=True,
        )
        with engine.connect() as connection:
            has_vector = connection.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar()
            documents_table = connection.execute(
                text("SELECT to_regclass('public.documents')")
            ).scalar()
            chunks_table = connection.execute(text("SELECT to_regclass('public.chunks')")).scalar()
        engine.dispose()
        return bool(has_vector and documents_table and chunks_table)
    except SQLAlchemyError:
        return False


pytestmark = pytest.mark.skipif(
    not _pgvector_database_available(),
    reason="PostgreSQL with migrated pgvector tables is not available",
)


@pytest.fixture
def pgvector_document() -> Iterator[tuple[str, str]]:
    """Insert isolated test data and remove it after the test, even on failure."""
    tenant_id = f"pytest-pgvector-{uuid4()}"
    matching_content = "Le chunk cible traite de la recherche vectorielle pgvector."
    non_matching_content = "Le second chunk traite des controles de risque." 
    matching_embedding = [1.0] + [0.0] * (settings.embedding_dim - 1)
    non_matching_embedding = [0.0, 1.0] + [0.0] * (settings.embedding_dim - 2)

    try:
        insert_document(
            tenant_id=tenant_id,
            classification=DataClassification.INTERNAL,
            filename="pytest-pgvector.txt",
            mime_type="text/plain",
            metadata={"integration_test": True},
            chunks=[
                {
                    "content": matching_content,
                    "embedding": matching_embedding,
                    "chunk_index": 0,
                    "page": 1,
                },
                {
                    "content": non_matching_content,
                    "embedding": non_matching_embedding,
                    "chunk_index": 1,
                    "page": 1,
                },
            ],
        )
        yield tenant_id, matching_content
    finally:
        session = get_session()
        try:
            session.execute(delete(Chunk).where(Chunk.tenant_id == tenant_id))
            session.execute(delete(Document).where(Document.tenant_id == tenant_id))
            session.commit()
        finally:
            session.close()


def test_vector_search_returns_the_closest_inserted_chunk(
    pgvector_document: tuple[str, str],
) -> None:
    tenant_id, matching_content = pgvector_document
    query_embedding = [1.0] + [0.0] * (settings.embedding_dim - 1)

    results = vector_search(
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        allowed_classifications=[DataClassification.INTERNAL.value],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["content"] == matching_content
    assert results[0]["score"] == pytest.approx(1.0)
