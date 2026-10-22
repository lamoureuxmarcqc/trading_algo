"""Worker : traite 1 fichier (extract -> chunk -> embed -> store)."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings
from ..embeddings.local import embed_passages
from ..ingestion.pipeline import ingest_file
from ..models import DataClassification
from ..retrieval.keyword import invalidate_cache
from ..storage.pgvector import insert_document

logger = logging.getLogger(__name__)


def process_file(
    *,
    file_path: Path,
    tenant_id: str,
    classification: DataClassification,
) -> dict:
    logger.info("Traitement : %s", file_path)

    doc = ingest_file(
        file_path,
        chunk_target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
    )

    if not doc.chunks:
        raise ValueError(f"Aucun contenu extrait de {file_path}")

    contents = [c.content for c in doc.chunks]
    embeddings = embed_passages(contents, settings.embedding_model)

    chunks_payload = [
        {
            "content": c.content,
            "embedding": embeddings[i].tolist(),
            "page": c.page,
            "section": c.section,
            "chunk_index": c.chunk_index,
            "metadata": c.metadata,
        }
        for i, c in enumerate(doc.chunks)
    ]

    document_id = insert_document(
        tenant_id=tenant_id,
        classification=classification,
        filename=doc.filename,
        mime_type=doc.mime_type,
        metadata={"source_path": str(file_path)},
        chunks=chunks_payload,
    )

    # Invalider le cache BM25 : sinon les nouveaux chunks ne sont jamais
    # indexes par keyword_search jusqu'au redemarrage du process.
    invalidate_cache()

    return {
        "document_id": document_id,
        "chunks_count": len(doc.chunks),
        "filename": doc.filename,
    }