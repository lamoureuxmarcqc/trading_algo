#!/usr/bin/env python3
"""
separate_phase2.py — Déplace les fichiers du trading vers verticals/trading/.

Phase 2 : Déplacement physique. Utilise git mv pour préserver l'historique.
Aucun fichier supprimé. Réversible via git reset.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def git_mv(src: Path, dst: Path) -> None:
    """Déplace via git mv pour préserver l'historique."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  🚚 {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
    subprocess.run(
        ["git", "mv", str(src), str(dst)],
        cwd=ROOT,
        check=True,
    )


def phase2_move():
    print("\n=== Phase 2 : Déplacement des fichiers trading ===\n")

    trading = ROOT / "verticals" / "trading"

    # 1. trading_algo
    print("→ trading_algo")
    if (ROOT / "trading_algo").exists():
        git_mv(ROOT / "trading_algo", trading / "trading_algo")

    # 2. api-core
    print("\n→ api-core")
    if (ROOT / "services" / "api-core").exists():
        git_mv(ROOT / "services" / "api-core", trading / "api-core")

    # 3. web (anciennement apps/web)
    print("\n→ web")
    if (ROOT / "apps" / "web").exists():
        git_mv(ROOT / "apps" / "web", trading / "web")

    print("\n✅ Phase 2 terminée : fichiers déplacés.")
    print("\n⚠️  Les imports sont cassés. Phase 3 va les réparer.")


if __name__ == "__main__":
    phase2_move()