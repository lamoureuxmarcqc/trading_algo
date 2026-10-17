"""Tests des filtres du scanner."""
from pathlib import Path

from app.scanner.filters import (
    ScanFilters,
    is_excluded,
    is_extension_allowed,
    should_scan,
)


def test_extension_allowed_pdf():
    assert is_extension_allowed(Path("doc.pdf"), [".pdf", ".docx"])


def test_extension_rejected_mp4():
    assert not is_extension_allowed(Path("video.mp4"), [".pdf", ".docx"])


def test_exclude_pattern_temp():
    assert is_excluded(Path("~$doc.docx"), ["~$*"])


def test_exclude_pattern_dir():
    assert is_excluded(Path("proj/node_modules/foo.js"), ["node_modules"])


def test_should_scan_valid(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"x" * 1000)
    assert should_scan(p, ScanFilters())


def test_should_scan_too_small(tmp_path: Path):
    p = tmp_path / "tiny.pdf"
    p.write_bytes(b"x")
    filters = ScanFilters(min_file_size_bytes=100)
    assert not should_scan(p, filters)


def test_should_scan_wrong_ext(tmp_path: Path):
    p = tmp_path / "video.mp4"
    p.write_bytes(b"x" * 1000)
    assert not should_scan(p, ScanFilters())


def test_should_scan_excluded(tmp_path: Path):
    p = tmp_path / "~$doc.pdf"
    p.write_bytes(b"x" * 1000)
    assert not should_scan(p, ScanFilters())
