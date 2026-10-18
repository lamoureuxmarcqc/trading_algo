"""Runner : orchestre un scan complet d'une source."""
from __future__ import annotations

import logging
from pathlib import Path

from ..ingestion.extractors import ExtractionError
from ..models import DataClassification
from .filters import ScanFilters
from .hashing import hash_file
from .state import (
    create_run,
    finish_run,
    get_scanned_file,
    get_source,
    record_file,
    update_run_progress,
)
from .walker import walk_source
from .worker import process_file

logger = logging.getLogger(__name__)


def _is_unchanged(existing: dict, result) -> bool:
    """Verifie si un fichier deja scanne n'a pas change (size + mtime)."""
    size_same = existing.get("size_bytes") == result.size_bytes
    if not size_same:
        return False
    existing_mtime = existing.get("modified_at")
    if existing_mtime is None:
        return False
    if hasattr(existing_mtime, "timestamp"):
        existing_ts = existing_mtime.timestamp()
    else:
        existing_ts = float(existing_mtime)
    return abs(existing_ts - result.modified_at) < 1.0


def run_scan(
    *,
    source_id: str,
    max_files: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute un scan complet d'une source.

    Args:
        source_id: ID de la source (scan_sources)
        max_files: limite optionnelle (utile pour pilote)
        dry_run: si True, ne fait que lister, pas d'ingestion

    Returns:
        dict avec run_id, counters, errors
    """
    source = get_source(source_id)
    if not source:
        raise ValueError(f"Source introuvable: {source_id}")

    if not source["enabled"]:
        raise ValueError(f"Source desactivee: {source_id}")

    root = Path(source["root_path"])
    if not root.exists():
        raise ValueError(f"Chemin introuvable: {root}")

    filters = ScanFilters(
        include_extensions=source["include_extensions"],
        exclude_patterns=source["exclude_patterns"],
        max_file_size_mb=source["max_file_size_mb"],
    )

    classification = DataClassification(source["classification"])
    tenant_id = source["tenant_id"]

    limit = max_files if max_files is not None else source.get("max_files")

    run_id = create_run(source_id)

    seen = 0
    ingested = 0
    skipped = 0
    errored = 0
    errors: list[dict] = []

    try:
        for result in walk_source(root, filters, max_files=limit):
            seen += 1
            path_str = str(result.path)

            try:
                existing = get_scanned_file(source_id, path_str)
                current_sha = None

                # ─── Skip definitif si corrupted ET fichier inchange ───
                if existing and existing["status"] == "corrupted":
                    if _is_unchanged(existing, result):
                        skipped += 1
                        update_run_progress(
                            run_id,
                            files_seen=seen,
                            files_ingested=ingested,
                            files_skipped=skipped,
                            files_errored=errored,
                        )
                        logger.info("SKIP (corrupted, unchanged): %s", path_str)
                        continue
                    logger.info("RETRY corrupted (file changed): %s", path_str)

                # ─── Dedup normal pour fichiers ingeres ───
                if existing and existing["status"] == "ingested":
                    current_sha = hash_file(result.path)
                    if current_sha == existing["sha256"]:
                        skipped += 1
                        update_run_progress(
                            run_id,
                            files_seen=seen,
                            files_ingested=ingested,
                            files_skipped=skipped,
                            files_errored=errored,
                        )
                        logger.info("SKIP (unchanged): %s", path_str)
                        continue

                if dry_run:
                    logger.info("DRY-RUN: %s (%d bytes)", path_str, result.size_bytes)
                    record_file(
                        source_id=source_id,
                        path=path_str,
                        sha256=None,
                        size_bytes=result.size_bytes,
                        modified_at=result.modified_at,
                        document_id=None,
                        status="dry_run",
                    )
                    skipped += 1
                    continue

                # ─── Ingestion ───
                output = process_file(
                    file_path=result.path,
                    tenant_id=tenant_id,
                    classification=classification,
                )

                sha = current_sha or hash_file(result.path)
                record_file(
                    source_id=source_id,
                    path=path_str,
                    sha256=sha,
                    size_bytes=result.size_bytes,
                    modified_at=result.modified_at,
                    document_id=output["document_id"],
                    status="ingested",
                )
                ingested += 1
                logger.info(
                    "INGESTED: %s (%d chunks)", path_str, output["chunks_count"]
                )

            except ExtractionError as e:
                # Erreur permanente → fichier corrompu
                errored += 1
                error_msg = f"{type(e).__name__}: {e}"
                logger.error("CORRUPTED: %s - %s", path_str, error_msg)
                errors.append({"path": path_str, "error": error_msg})
                try:
                    record_file(
                        source_id=source_id,
                        path=path_str,
                        sha256=None,
                        size_bytes=result.size_bytes,
                        modified_at=result.modified_at,
                        document_id=None,
                        status="corrupted",
                        error_message=error_msg,
                    )
                except Exception:
                    pass

            except Exception as e:
                # Erreur transitoire (reseau, disque, timeout)
                errored += 1
                error_msg = f"{type(e).__name__}: {e}"
                logger.error("ERROR: %s - %s", path_str, error_msg)
                errors.append({"path": path_str, "error": error_msg})
                try:
                    record_file(
                        source_id=source_id,
                        path=path_str,
                        sha256=None,
                        size_bytes=result.size_bytes,
                        modified_at=result.modified_at,
                        document_id=None,
                        status="error",
                        error_message=error_msg,
                    )
                except Exception:
                    pass

            update_run_progress(
                run_id,
                files_seen=seen,
                files_ingested=ingested,
                files_skipped=skipped,
                files_errored=errored,
            )

        finish_run(run_id, "done")

    except Exception as e:
        logger.exception("Scan crashed")
        finish_run(run_id, "failed", error=str(e))
        raise

    return {
        "run_id": run_id,
        "source_id": source_id,
        "files_seen": seen,
        "files_ingested": ingested,
        "files_skipped": skipped,
        "files_errored": errored,
        "errors": errors,
    }
