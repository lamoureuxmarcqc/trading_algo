"""Filtres pour le scanner : extensions, taille, exclusions."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScanFilters:
    include_extensions: list[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".md", ".txt", ".html", ".htm"
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "~$*", "*.tmp", "*.bak", ".git", "node_modules",
        "__pycache__", ".DS_Store", "Thumbs.db", "@eaDir"
    ])
    max_file_size_mb: int = 50
    min_file_size_bytes: int = 16


def normalize_ext(ext: str) -> str:
    ext = ext.lower()
    return ext if ext.startswith(".") else f".{ext}"


def is_extension_allowed(path: Path, include_extensions: list[str]) -> bool:
    allowed = {normalize_ext(e) for e in include_extensions}
    return path.suffix.lower() in allowed


def is_excluded(path: Path, exclude_patterns: list[str]) -> bool:
    name = path.name
    parts = path.parts
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def is_size_allowed(path: Path, max_mb: int, min_bytes: int) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return min_bytes <= size <= max_mb * 1024 * 1024


def should_scan(path: Path, filters: ScanFilters) -> bool:
    if not path.is_file():
        return False
    if is_excluded(path, filters.exclude_patterns):
        return False
    if not is_extension_allowed(path, filters.include_extensions):
        return False
    if not is_size_allowed(path, filters.max_file_size_mb, filters.min_file_size_bytes):
        return False
    return True
