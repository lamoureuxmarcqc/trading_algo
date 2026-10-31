#!/usr/bin/env python3
"""
separate_platform.py — Sépare la plateforme des verticales métiers.

Phase 1 : Crée la structure cible sans déplacer aucun fichier.
Aucun risque. Vérifiable. Réversible via git.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"  📁 {path.relative_to(ROOT)}")


def write_readme(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"  📄 {path.relative_to(ROOT)}")
    else:
        print(f"  ⏭️  {path.relative_to(ROOT)} (existe déjà)")


def phase1_structure():
    print("\n=== Phase 1 : Structure cible ===\n")

    # ─── Plateforme (services restent où ils sont) ───
    print("→ Plateforme (services/ inchangés)")

    # ─── Verticales ───
    print("\n→ Verticales")

    trading = ROOT / "verticals" / "trading"
    ensure_dir(trading)
    ensure_dir(trading / "agents")

    write_readme(
        trading / "README.md",
        """# Vertical TRADING

Premier vertical de la Lamoureux AI Platform.

## Composants

- `trading_algo/` — moteur quant (à déplacer)
- `api-core/` — API institutionnelle (à déplacer)
- `web/` — cockpit Next.js (à déplacer)
- `agents/` — agents spécialisés

## Dépend du socle pour

- Knowledge (rapports, filings, analyses)
- Memory (decision memory)
- AI Gateway (explications)
- Orchestrator (Investment Agent)

## Tenant

`lamoureux` — classification max `HIGHLY_CONFIDENTIAL`
""",
    )

    for vertical in ["lawyer", "doctor", "crm", "cnesst"]:
        path = ROOT / "verticals" / vertical
        ensure_dir(path)
        write_readme(
            path / "README.md",
            f"""# Vertical {vertical.upper()}

Statut : **planifié** (pas encore implémenté).

Voir `docs/verticals/{vertical}.md` pour les spécifications.
""",
        )

    # ─── App plateforme ───
    print("\n→ App plateforme")
    web_platform = ROOT / "apps" / "web-platform"
    ensure_dir(web_platform)
    write_readme(
        web_platform / "README.md",
        """# Web Platform

Portail de la Lamoureux AI Platform.

## Rôle

- Sélectionner une verticale (trading, lawyer, doctor, crm, cnesst)
- Admin de la plateforme (models, agents, evaluation)
- Chat universel avec agents
- Recherche dans toute la connaissance (selon permissions)

## Pages prévues

- `/` — Landing + sélection verticale
- `/search` — RAG générique
- `/agents` — gestion des agents
- `/models` — model registry
- `/admin` — admin plateforme
""",
    )

    # ─── Docs verticales ───
    print("\n→ Docs verticales")
    docs_vert = ROOT / "docs" / "verticals"
    ensure_dir(docs_vert)

    for vertical in ["trading", "lawyer", "doctor", "crm", "cnesst"]:
        write_readme(
            docs_vert / f"{vertical}.md",
            f"""# Vertical {vertical.upper()}

Voir `verticals/{vertical}/README.md` pour l'implémentation.
""",
        )

    # ─── Docs plateforme ───
    print("\n→ Docs plateforme")
    docs_platform = ROOT / "docs" / "platform"
    ensure_dir(docs_platform)

    print("\n✅ Phase 1 terminée : structure créée, aucun fichier déplacé.")


if __name__ == "__main__":
    phase1_structure()