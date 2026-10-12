"""Operations pgvector : insertion, recherche vectorielle."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from ..models import DataClassification
from .db import get_session
from .models import Chunk, Document


def insert_document(
    *,
    tenant_id: str,
    classification: DataClassification,
    filename: str,
    mime_type: str | None,
    metadata: dict | None,
    chunks: list[dict],
) -> str:
    session = get_session()
    try:
        doc = Document(
            tenant_id=tenant_id,
            classification=classification.value,
            filename=filename,
            mime_type=mime_type,
            doc_metadata=metadata or {},
        )
        session.add(doc)
        session.flush()

        for c in chunks:
            session.add(
                Chunk(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    classification=classification.value,
                    content=c["content"],
                    embedding=c["embedding"],
                    page=c.get("page"),
                    section=c.get("section"),
                    chunk_index=c["chunk_index"],
                    chunk_metadata=c.get("metadata") or {},
                )
            )

        session.commit()
        return str(doc.id)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_document(document_id: str) -> Document | None:
    session = get_session()
    try:
        return session.get(Document, uuid.UUID(document_id))
    finally:
        session.close()


def vector_search(
    *,
    query_embedding: list[float],
    tenant_id: str,
    allowed_classifications: list[str],
    top_k: int,
) -> list[dict]:
    session = get_session()
    try:
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(Chunk, distance)
            .where(Chunk.tenant_id == tenant_id)
            .where(Chunk.classification.in_(allowed_classifications))
            .order_by(distance)
            .limit(top_k)
        )
        rows = session.execute(stmt).all()
        return [
            {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "page": chunk.page,
                "section": chunk.section,
                "classification": chunk.classification,
                "score": 1.0 - float(dist),
            }
            for chunk, dist in rows
        ]
    finally:
        session.close()
