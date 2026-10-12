"""Initial : documents + chunks avec pgvector.

Revision ID: 001_initial
Revises:
Create Date: 2026-10-11
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    # Extension pgvector (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column("classification", sa.String(32), nullable=False, index=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("metadata", JSONB, nullable=True),
    )

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column("classification", sa.String(32), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("page", sa.Integer, nullable=True),
        sa.Column("section", sa.String(512), nullable=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("metadata", JSONB, nullable=True),
    )

    # Index IVFFlat pour la recherche vectorielle
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    op.drop_table("chunks")
    op.drop_table("documents")
    # Note : on ne supprime pas l'extension vector (partagee)