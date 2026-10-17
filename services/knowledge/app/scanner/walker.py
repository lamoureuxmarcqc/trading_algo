"""Parcours recursif d'une source avec filtres."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .filters import ScanFilters, is_excluded, should_scan

logger = logging.getLogger(__name__)


@dataclass
class WalkResult:
    path: Path
    size_bytes: int
    modified_at: float


def walk_source(
    root: Path,
    filters: ScanFilters,
    max_files: int | None = None,
) -> Iterator[WalkResult]:
    if not root.exists():
        logger.warning("Racine introuvable : %s", root)
        return

    count = 0
    stack: list[Path] = [root]

    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError) as e:
            logger.warning("Acces refuse sur %s : %s", current, e)
            continue

        for entry in entries:
            try:
                if entry.is_dir():
                    if not is_excluded(entry, filters.exclude_patterns):
                        stack.append(entry)
                elif should_scan(entry, filters):
                    stat = entry.stat()
                    yield WalkResult(
                        path=entry,
                        size_bytes=stat.st_size,
                        modified_at=stat.st_mtime,
                    )
                    count += 1
                    if max_files and count >= max_files:
                        return
            except (PermissionError, OSError) as e:
                logger.warning("Erreur sur %s : %s", entry, e)
                continue


def count_files(root: Path, filters: ScanFilters) -> int:
    return sum(1 for _ in walk_source(root, filters))
