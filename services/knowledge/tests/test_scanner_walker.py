"""Tests du walker."""
from pathlib import Path

from app.scanner.filters import ScanFilters
from app.scanner.walker import count_files, walk_source


def _make_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.pdf").write_bytes(b"x" * 1000)
    (tmp_path / "b.docx").write_bytes(b"x" * 1000)
    (tmp_path / "c.mp4").write_bytes(b"x" * 1000)
    (tmp_path / "~$skip.docx").write_bytes(b"x" * 1000)

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.pdf").write_bytes(b"x" * 1000)

    node = tmp_path / "node_modules"
    node.mkdir()
    (node / "e.pdf").write_bytes(b"x" * 1000)

    return tmp_path


def test_walk_finds_valid_files(tmp_path: Path):
    root = _make_tree(tmp_path)
    names = sorted(r.path.name for r in walk_source(root, ScanFilters()))
    assert "a.pdf" in names
    assert "b.docx" in names
    assert "d.pdf" in names


def test_walk_skips_invalid(tmp_path: Path):
    root = _make_tree(tmp_path)
    names = [r.path.name for r in walk_source(root, ScanFilters())]
    assert "c.mp4" not in names
    assert "~$skip.docx" not in names
    assert "e.pdf" not in names


def test_walk_max_files(tmp_path: Path):
    root = _make_tree(tmp_path)
    assert len(list(walk_source(root, ScanFilters(), max_files=1))) == 1


def test_count_files(tmp_path: Path):
    root = _make_tree(tmp_path)
    assert count_files(root, ScanFilters()) == 3


def test_walk_missing_root(tmp_path: Path):
    assert list(walk_source(tmp_path / "nope", ScanFilters())) == []
