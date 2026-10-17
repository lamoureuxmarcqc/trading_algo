#!/usr/bin/env python3
"""
setup_migrations.py — Configure Alembic + crée les migrations pour :
    - services/memory (decisions, facts)
    - services/knowledge (scan_sources, scan_runs, scanned_files)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SERVICES = ROOT / "services"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  + {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════
# MEMORY — Alembic setup
# ═══════════════════════════════════════════════════════════════════

def setup_memory_alembic() -> None:
    print("\n[Memory] Alembic setup")
    base = SERVICES / "memory"

    write(base / "alembic.ini", '''
[alembic]
script_location = alembic
prepend_sys_path = .

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
''')

    write(base / "alembic" / "env.py", '''
"""Alembic environment pour services/memory."""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.decision import Base

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
''')

    write(base / "alembic" / "script.py.mako", '''
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
''')

    write(base / "alembic" / "versions" / "001_decisions_and_facts.py", '''
"""Memory : decisions (immuable) + facts (long-term).

Revision ID: mem_001
Revises:
Create Date: 2026-10-16
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "mem_001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── Table decisions (APPEND-ONLY) ───
    op.create_table(
        "decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("evidence", JSONB, nullable=True),
        sa.Column("action", sa.Text, nullable=True),
        sa.Column("outcome", JSONB, nullable=True),
        sa.Column("evaluation", JSONB, nullable=True),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("correlation_id", sa.String(128), nullable=True),
    )

    # Trigger : interdit UPDATE et DELETE
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_decision_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'decisions is append-only: DELETE interdit';
            END IF;

            IF TG_OP = 'UPDATE' THEN
                -- Autoriser uniquement si les colonnes immuables sont inchangees
                IF OLD.decision IS DISTINCT FROM NEW.decision
                   OR OLD.reason IS DISTINCT FROM NEW.reason
                   OR OLD.evidence IS DISTINCT FROM NEW.evidence
                   OR OLD.action IS DISTINCT FROM NEW.action
                   OR OLD.tenant_id IS DISTINCT FROM NEW.tenant_id
                   OR OLD.created_at IS DISTINCT FROM NEW.created_at
                   OR OLD.model_version IS DISTINCT FROM NEW.model_version
                   OR OLD.confidence IS DISTINCT FROM NEW.confidence
                   OR OLD.correlation_id IS DISTINCT FROM NEW.correlation_id
                THEN
                    RAISE EXCEPTION 'decisions est append-only: seuls outcome et evaluation peuvent etre ajoutes';
                END IF;

                -- Interdire de vider outcome/evaluation une fois remplis
                IF OLD.outcome IS NOT NULL AND NEW.outcome IS NULL THEN
                    RAISE EXCEPTION 'outcome deja enregistre, ne peut etre efface';
                END IF;
                IF OLD.evaluation IS NOT NULL AND NEW.evaluation IS NULL THEN
                    RAISE EXCEPTION 'evaluation deja enregistree, ne peut etre effacee';
                END IF;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER decisions_immutable
        BEFORE UPDATE OR DELETE ON decisions
        FOR EACH ROW EXECUTE FUNCTION prevent_decision_mutation();
    """)

    # ─── Table facts (long-term memory, validée humainement) ───
    op.create_table(
        "facts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column("fact", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=True),
        sa.Column(
            "validated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("validated_by", sa.String(128), nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("metadata", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS decisions_immutable ON decisions")
    op.execute("DROP FUNCTION IF EXISTS prevent_decision_mutation()")
    op.drop_table("facts")
    op.drop_table("decisions")
''')

    print("  Memory Alembic OK")


# ═══════════════════════════════════════════════════════════════════
# KNOWLEDGE — Migration 002 (scanner)
# ═══════════════════════════════════════════════════════════════════

def setup_knowledge_migration_002() -> None:
    print("\n[Knowledge] Migration 002 — scanner tables")
    base = SERVICES / "knowledge"

    write(base / "alembic" / "versions" / "002_scanner.py", '''
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
        sa.Column("path_hash", sa.String(64), nullable=False, index=True),
        sa.Column("sha256", sa.String(64), nullable=True, index=True),
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
''')

    print("  Knowledge migration 002 OK")


# ═══════════════════════════════════════════════════════════════════
# FIX pyproject.toml (warning asyncio)
# ═══════════════════════════════════════════════════════════════════

def fix_pyproject_warnings() -> None:
    print("\n[Fix] pyproject.toml — warning asyncio")
    services = ["model-registry", "evaluation", "self-improvement"]
    for s in services:
        p = SERVICES / s / "pyproject.toml"
        if not p.exists():
            print(f"  ! {s} : fichier introuvable")
            continue
        content = p.read_text(encoding="utf-8")
        if "asyncio_mode" in content:
            print(f"  = {s} deja OK")
            continue
        content = content.replace(
            'addopts = "--basetemp=.pytest_tmp"',
            'asyncio_mode = "auto"\nasyncio_default_fixture_loop_scope = "function"\naddopts = "--basetemp=.pytest_tmp"',
        )
        p.write_text(content, encoding="utf-8")
        print(f"  + {s}/pyproject.toml")


# ═══════════════════════════════════════════════════════════════════
# APPLIQUER LES MIGRATIONS
# ═══════════════════════════════════════════════════════════════════

def run_migration(service: str) -> bool:
    service_dir = SERVICES / service
    print(f"\n[Migration] {service}")
    try:
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=service_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.strip().splitlines():
            if "Running upgrade" in line or "INFO" in line:
                print(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERREUR : {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("LAMOUREUX AI — Setup migrations PostgreSQL")
    print("=" * 60)

    print("\n[1/3] Configuration Alembic")
    setup_memory_alembic()
    setup_knowledge_migration_002()

    print("\n[2/3] Fix warnings pytest")
    fix_pyproject_warnings()

    print("\n[3/3] Application des migrations")
    ok_memory = run_migration("memory")
    ok_knowledge = run_migration("knowledge")

    print()
    print("=" * 60)
    if ok_memory and ok_knowledge:
        print("SUCCES — Toutes les migrations appliquees")
        return 0
    print("ECHEC — Voir les erreurs ci-dessus")
    return 1


if __name__ == "__main__":
    sys.exit(main())