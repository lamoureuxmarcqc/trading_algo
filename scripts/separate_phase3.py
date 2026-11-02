#!/usr/bin/env python3
"""
separate_phase3.py — Crée les fichiers de configuration par vertical.

Phase 3 : pyproject.toml + conftest.py + README pour chaque vertical.
Ne touche PAS aux imports Python (qui doivent être corrigés à la main).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def write(path: Path, content: str) -> None:
    if path.exists():
        print(f"  ⏭️  {path.relative_to(ROOT)} (existe déjà)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  📄 {path.relative_to(ROOT)}")


def phase3_configs():
    print("\n=== Phase 3 : Configuration par vertical ===\n")

    # ─── trading/api-core ───
    print("→ trading/api-core")
    api_core = ROOT / "verticals" / "trading" / "api-core"

    write(
        api_core / "pyproject.toml",
        """[project]
name = "lamoureux-vertical-trading-api"
version = "0.1.0"
description = "Vertical trading — institutional API"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "--basetemp=.pytest_tmp"
""",
    )

    write(
        api_core / "conftest.py",
        '''"""Rend `app` importable pour pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
''',
    )

    # ─── trading/web ───
    print("\n→ trading/web")
    web = ROOT / "verticals" / "trading" / "web"

    write(
        web / "README.md",
        '''# Trading Web Cockpit

Frontend Next.js du vertical trading.

## Démarrage

    npm install
    npm run dev
    # http://localhost:3000

## Dépend de

- `api-core` sur `http://localhost:8000`
- `knowledge` sur `http://localhost:8002`
''',
    )

    # ─── Structure agents ───
    print("\n→ trading/agents")
    agents = ROOT / "verticals" / "trading" / "agents"
    write(
        agents / "__init__.py",
        '"""Agents spécialisés du vertical trading."""',
    )

    print("\n✅ Phase 3 terminée : configurations créées.")
    print("\n⚠️  Actions manuelles à faire :")
    print("   - Vérifier les imports dans verticals/trading/api-core/app/")
    print("   - Vérifier les imports dans verticals/trading/trading_algo/")


if __name__ == "__main__":
    phase3_configs()