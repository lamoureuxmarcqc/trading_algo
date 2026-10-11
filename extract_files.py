#!/usr/bin/env python3
"""
extract_files.py — Extrait les fichiers depuis un document de livraison Markdown.

Lit un document contenant des sections `## `chemin/fichier.ext`` suivies
d'un bloc de code, et crée les fichiers aux chemins correspondants.

Usage :
    python extract_files.py AI_PLATFORM.md
    python extract_files.py AI_PLATFORM.md --dry-run
    python extract_files.py AI_PLATFORM.md --delete
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECTION_RE = re.compile(r"^##\s+`([^`]+)`\s*$")
BOUNDARY_RE = re.compile(r"^(?:##\s+`[^`]+`\s*$|---\s*$|#\s+[^#].*)$")


def is_fence(line: str) -> bool:
    return line.lstrip().startswith("```")


def is_safe_path(path: str) -> bool:
    p = Path(path)
    if p.is_absolute():
        return False
    if ".." in p.parts:
        return False
    return True


def find_sections(lines):
    headers = []
    for i, line in enumerate(lines):
        m = SECTION_RE.match(line)
        if m:
            headers.append((m.group(1), i))
    sections = []
    for idx, (path, start) in enumerate(headers):
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(lines)
        sections.append((path, start, end))
    return sections


def find_outer_fence(lines, start, end):
    opening = None
    for i in range(start, end):
        if is_fence(lines[i]):
            opening = i
            break
    if opening is None:
        return None

    candidates = []
    for i in range(opening, end):
        if not is_fence(lines[i]):
            continue
        j = i + 1
        while j < end and not lines[j].strip():
            j += 1
        if j >= end or BOUNDARY_RE.match(lines[j]):
            candidates.append(i)

    if not candidates:
        return None
    closing = candidates[-1]
    if closing <= opening:
        return None
    return opening, closing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true",
                    help="Affiche sans créer les fichiers")
    ap.add_argument("--delete", action="store_true",
                    help="Supprime le source après extraction")
    args = ap.parse_args()

    if not args.source.is_file():
        print(f"❌ Fichier source introuvable : {args.source}", file=sys.stderr)
        return 1

    lines = args.source.read_text(encoding="utf-8").splitlines()
    sections = find_sections(lines)
    if not sections:
        print("❌ Aucune section `## `chemin`` détectée.", file=sys.stderr)
        return 1

    print(f"📖 {len(sections)} section(s) détectée(s) dans {args.source}\n")

    created = skipped = 0
    for path, start, end in sections:
        if not is_safe_path(path):
            print(f"⚠️  Chemin non sécurisé ignoré : {path}")
            skipped += 1
            continue

        result = find_outer_fence(lines, start + 1, end)
        if result is None:
            print(f"⚠️  {path}  → aucun bloc de code, ignoré")
            skipped += 1
            continue

        opening, closing = result
        content_lines = lines[opening + 1:closing]
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()
        content = "\n".join(content_lines) + "\n"

        out = args.output / path
        if args.dry_run:
            print(f"→ {out}  ({len(content)} octets)")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
            print(f"✅ {out}  ({len(content)} octets)")
        created += 1

    print()
    print(f"Résumé : {created} fichier(s) créé(s), {skipped} ignoré(s).")

    if args.delete and not args.dry_run:
        args.source.unlink()
        print(f"🗑️  Source supprimée : {args.source}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())