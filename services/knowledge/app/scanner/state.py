"""Operations PostgreSQL pour scanner : sources, runs, fichiers scannes."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from hashlib import sha256 as _sha256

from sqlalchemy import select

from ..storage.db import get_session
from ..storage.models import Document


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def path_hash(path: str) -> str:
    """Hash deterministe du chemin (pour dedup rapide)."""
    return _sha256(path.encode("utf-8")).hexdigest()


# ─────────────────────── Sources ───────────────────────

def create_source(
    *,
    tenant_id: str,
    name: str,
    root_path: str,
    classification: str,
    include_extensions: list[str],
    exclude_patterns: list[str] | None = None,
    max_file_size_mb: int = 50,
    max_files: int | None = None,
) -> str:
    from sqlalchemy import text

    session = get_session()
    try:
        source_id = uuid.uuid4()
        session.execute(
            text("""
                INSERT INTO scan_sources
                    (id, tenant_id, name, root_path, classification,
                     include_extensions, exclude_patterns, max_file_size_mb, max_files, enabled)
                VALUES
                    (:id, :tenant_id, :name, :root_path, :classification,
                     :include_extensions, :exclude_patterns, :max_file_size_mb, :max_files, true)
            """),
            {
                "id": source_id,
                "tenant_id": tenant_id,
                "name": name,
                "root_path": root_path,
                "classification": classification,
                "include_extensions": include_extensions,
                "exclude_patterns": exclude_patterns or [],
                "max_file_size_mb": max_file_size_mb,
                "max_files": max_files,
            },
        )
        session.commit()
        return str(source_id)
    finally:
        session.close()


def list_sources(tenant_id: str | None = None) -> list[dict]:
    from sqlalchemy import text

    session = get_session()
    try:
        q = "SELECT id, tenant_id, name, root_path, classification, enabled, created_at FROM scan_sources"
        params = {}
        if tenant_id:
            q += " WHERE tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id
        q += " ORDER BY created_at DESC"
        rows = session.execute(text(q), params).all()
        return [
            {
                "id": str(r.id),
                "tenant_id": r.tenant_id,
                "name": r.name,
                "root_path": r.root_path,
                "classification": r.classification,
                "enabled": r.enabled,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        session.close()


def get_source(source_id: str) -> dict | None:
    from sqlalchemy import text

    session = get_session()
    try:
        row = session.execute(
            text("SELECT * FROM scan_sources WHERE id = :id"),
            {"id": uuid.UUID(source_id)},
        ).first()
        if not row:
            return None
        return {
            "id": str(row.id),
            "tenant_id": row.tenant_id,
            "name": row.name,
            "root_path": row.root_path,
            "classification": row.classification,
            "include_extensions": list(row.include_extensions),
            "exclude_patterns": list(row.exclude_patterns or []),
            "max_file_size_mb": row.max_file_size_mb,
            "max_files": row.max_files,
            "enabled": row.enabled,
        }
    finally:
        session.close()


# ─────────────────────── Runs ───────────────────────

def create_run(source_id: str) -> str:
    from sqlalchemy import text

    session = get_session()
    try:
        run_id = uuid.uuid4()
        session.execute(
            text("""
                INSERT INTO scan_runs (id, source_id, status)
                VALUES (:id, :source_id, 'running')
            """),
            {"id": run_id, "source_id": uuid.UUID(source_id)},
        )
        session.commit()
        return str(run_id)
    finally:
        session.close()


def update_run_progress(run_id: str, **counters) -> None:
    from sqlalchemy import text

    if not counters:
        return
    session = get_session()
    try:
        sets = ", ".join(f"{k} = :{k}" for k in counters)
        params = dict(counters)
        params["id"] = uuid.UUID(run_id)
        session.execute(
            text(f"UPDATE scan_runs SET {sets} WHERE id = :id"),
            params,
        )
        session.commit()
    finally:
        session.close()


def finish_run(run_id: str, status: str, error: str | None = None) -> None:
    from sqlalchemy import text

    session = get_session()
    try:
        session.execute(
            text("""
                UPDATE scan_runs
                SET status = :status, finished_at = :finished_at, error_message = :error
                WHERE id = :id
            """),
            {
                "id": uuid.UUID(run_id),
                "status": status,
                "finished_at": _utcnow(),
                "error": error,
            },
        )
        session.commit()
    finally:
        session.close()


def get_run(run_id: str) -> dict | None:
    from sqlalchemy import text

    session = get_session()
    try:
        row = session.execute(
            text("SELECT * FROM scan_runs WHERE id = :id"),
            {"id": uuid.UUID(run_id)},
        ).first()
        if not row:
            return None
        return {
            "id": str(row.id),
            "source_id": str(row.source_id),
            "status": row.status,
            "started_at": row.started_at.isoformat(),
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "files_seen": row.files_seen,
            "files_ingested": row.files_ingested,
            "files_skipped": row.files_skipped,
            "files_errored": row.files_errored,
            "error_message": row.error_message,
        }
    finally:
        session.close()


# ─────────────────────── Fichiers ───────────────────────

def get_scanned_file(source_id: str, path: str) -> dict | None:
    from sqlalchemy import text

    session = get_session()
    try:
        row = session.execute(
            text("""
                SELECT id, sha256, document_id, status
                FROM scanned_files
                WHERE source_id = :source_id AND path_hash = :path_hash
            """),
            {
                "source_id": uuid.UUID(source_id),
                "path_hash": path_hash(path),
            },
        ).first()
        if not row:
            return None
        return {
            "id": str(row.id),
            "sha256": row.sha256,
            "document_id": str(row.document_id) if row.document_id else None,
            "status": row.status,
        }
    finally:
        session.close()


def record_file(
    *,
    source_id: str,
    path: str,
    sha256: str | None,
    size_bytes: int | None,
    modified_at: float | None,
    document_id: str | None,
    status: str,
    error_message: str | None = None,
) -> None:
    from sqlalchemy import text

    session = get_session()
    try:
        existing = session.execute(
            text("""
                SELECT id FROM scanned_files
                WHERE source_id = :source_id AND path_hash = :path_hash
            """),
            {"source_id": uuid.UUID(source_id), "path_hash": path_hash(path)},
        ).first()

        modified_dt = None
        if modified_at is not None:
            modified_dt = datetime.fromtimestamp(modified_at, tz=timezone.utc)

        if existing:
            session.execute(
                text("""
                    UPDATE scanned_files
                    SET sha256 = :sha256, size_bytes = :size_bytes,
                        modified_at = :modified_at, document_id = :document_id,
                        status = :status, error_message = :error,
                        last_seen = :last_seen
                    WHERE id = :id
                """),
                {
                    "id": existing.id,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "modified_at": modified_dt,
                    "document_id": uuid.UUID(document_id) if document_id else None,
                    "status": status,
                    "error": error_message,
                    "last_seen": _utcnow(),
                },
            )
        else:
            session.execute(
                text("""
                    INSERT INTO scanned_files
                        (id, source_id, path, path_hash, sha256, size_bytes,
                         modified_at, document_id, status, error_message, last_seen)
                    VALUES
                        (:id, :source_id, :path, :path_hash, :sha256, :size_bytes,
                         :modified_at, :document_id, :status, :error, :last_seen)
                """),
                {
                    "id": uuid.uuid4(),
                    "source_id": uuid.UUID(source_id),
                    "path": path,
                    "path_hash": path_hash(path),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "modified_at": modified_dt,
                    "document_id": uuid.UUID(document_id) if document_id else None,
                    "status": status,
                    "error": error_message,
                    "last_seen": _utcnow(),
                },
            )
        session.commit()
    finally:
        session.close()


def count_scanned(source_id: str) -> dict:
    from sqlalchemy import text

    session = get_session()
    try:
        rows = session.execute(
            text("""
                SELECT status, COUNT(*) as cnt
                FROM scanned_files
                WHERE source_id = :id
                GROUP BY status
            """),
            {"id": uuid.UUID(source_id)},
        ).all()
        return {r.status: r.cnt for r in rows}
    finally:
        session.close()