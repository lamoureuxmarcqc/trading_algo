"""CLI pour le scanner — usage : python -m app.scanner.cli <command>"""
from __future__ import annotations

import argparse
import json
import sys

from .runner import run_scan
from .state import (
    count_scanned,
    create_source,
    get_run,
    get_source,
    list_sources,
)


def cmd_add_source(args) -> int:
    exts = [e.strip() for e in args.extensions.split(",")]
    exclude = [e.strip() for e in args.exclude.split(",")] if args.exclude else []

    source_id = create_source(
        tenant_id=args.tenant,
        name=args.name,
        root_path=args.path,
        classification=args.classification,
        include_extensions=exts,
        exclude_patterns=exclude,
        max_file_size_mb=args.max_size,
        max_files=args.max_files,
    )
    print(f"Source creee : {source_id}")
    return 0


def cmd_list_sources(args) -> int:
    sources = list_sources(args.tenant)
    if not sources:
        print("Aucune source")
        return 0
    print(f"{'ID':<40} {'Nom':<25} {'Chemin':<50} {'Class':<20}")
    print("-" * 135)
    for s in sources:
        print(f"{s['id']:<40} {s['name']:<25} {s['root_path']:<50} {s['classification']:<20}")
    return 0


def cmd_scan(args) -> int:
    print(f"Scan de la source {args.source_id}...")
    print(f"  max_files : {args.max_files or 'illimite'}")
    print(f"  dry_run   : {args.dry_run}")
    print()

    result = run_scan(
        source_id=args.source_id,
        max_files=args.max_files,
        dry_run=args.dry_run,
    )

    print()
    print(f"Run ID      : {result['run_id']}")
    print(f"Vus         : {result['files_seen']}")
    print(f"Ingeres     : {result['files_ingested']}")
    print(f"Skippes     : {result['files_skipped']}")
    print(f"Erreurs     : {result['files_errored']}")

    if result["errors"]:
        print()
        print("Erreurs detaillees :")
        for e in result["errors"][:10]:
            print(f"  - {e['path']}")
            print(f"    {e['error']}")

    return 0


def cmd_status(args) -> int:
    if args.run_id:
        run = get_run(args.run_id)
        if not run:
            print(f"Run introuvable: {args.run_id}")
            return 1
        print(json.dumps(run, indent=2))
    elif args.source_id:
        counts = count_scanned(args.source_id)
        print(f"Source {args.source_id} :")
        for status, cnt in counts.items():
            print(f"  {status:<15} : {cnt}")
    else:
        print("Specifier --run-id ou --source-id")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scanner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # add-source
    p = sub.add_parser("add-source", help="Ajouter une source de scan")
    p.add_argument("--tenant", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--classification", required=True,
                   choices=["PUBLIC", "INTERNAL", "CONFIDENTIAL",
                            "HIGHLY_CONFIDENTIAL", "RESTRICTED"])
    p.add_argument("--extensions", default=".pdf,.docx,.md,.txt,.html,.htm")
    p.add_argument("--exclude", default="")
    p.add_argument("--max-size", type=int, default=50, dest="max_size")
    p.add_argument("--max-files", type=int, default=None, dest="max_files")
    p.set_defaults(func=cmd_add_source)

    # list-sources
    p = sub.add_parser("list-sources", help="Lister les sources")
    p.add_argument("--tenant", default=None)
    p.set_defaults(func=cmd_list_sources)

    # scan
    p = sub.add_parser("scan", help="Lancer un scan")
    p.add_argument("--source-id", required=True)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_scan)

    # status
    p = sub.add_parser("status", help="Statut d'un run ou d'une source")
    p.add_argument("--run-id", default=None)
    p.add_argument("--source-id", default=None)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())