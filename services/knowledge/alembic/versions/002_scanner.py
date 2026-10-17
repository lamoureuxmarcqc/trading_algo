"""Scanner : scan_sources, scan_runs, scanned_files.

Revision ID: 002_scanner
Revises: 001_initial
Create Date: 2026-10-16
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision: str = "002_scanner"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── Sources de scan ───
    op.create_table(
        "scan_sources",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("root_path", sa.Text, nullable=False),
        sa.Column("classification", sa.String(32), nullable=False),
        sa.Column("include_extensions", ARRAY(sa.String), nullable=False),
        sa.Column("exclude_patterns", ARRAY(sa.String), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("max_file_size_mb", sa.Integer, nullable=False, server_default="50"),
        sa.Column("max_files", sa.Integer, nullable=True),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ─── Runs de scan ───
    op.create_table(
        "scan_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            UUID(as_uuid=True),
            sa.ForeignKey("scan_sources.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("files_seen", sa.Integer, nullable=False, server_default="0"),
        sa.Column("files_ingested", sa.Integer, nullable=False, server_default="0"),
        sa.Column("files_skipped", sa.Integer, nullable=False, server_default="0"),
        sa.Column("files_errored", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    # ─── Fichiers scannes (dedup par SHA256) ───
    op.create_table(
        "scanned_files",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            UUID(as_uuid=True),
            sa.ForeignKey("scan_sources.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("path_hash", sa.String(64), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("size_bytes", sa.BigInteger, nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "last_seen",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("file_metadata", JSONB, nullable=True),
    )

    # Index pour dedup rapide
    op.create_index(
        "ix_scanned_files_path_hash",
        "scanned_files",
        ["path_hash"],
    )
    op.create_index(
        "ix_scanned_files_sha256",
        "scanned_files",
        ["sha256"],
        postgresql_where=sa.text("sha256 IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_scanned_files_sha256", table_name="scanned_files")
    op.drop_index("ix_scanned_files_path_hash", table_name="scanned_files")
    op.drop_table("scanned_files")
    op.drop_table("scan_runs")
    op.drop_table("scan_sources")
