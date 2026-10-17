#!/usr/bin/env python3
"""Corrige les env.py Alembic pour utiliser des version_tables séparées."""
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def fix_memory() -> None:
    p = ROOT / "services" / "memory" / "alembic" / "env.py"
    content = p.read_text(encoding="utf-8")

    # Remplacer le bloc online
    old_online = (
        "    with connectable.connect() as connection:\n"
        "        context.configure(connection=connection, target_metadata=target_metadata)\n"
    )
    new_online = (
        "    with connectable.connect() as connection:\n"
        "        context.configure(\n"
        "            connection=connection,\n"
        "            target_metadata=target_metadata,\n"
        '            version_table="alembic_version_memory",\n'
        "        )\n"
    )
    if old_online in content:
        content = content.replace(old_online, new_online)
        print("  + memory online configure")
    elif "alembic_version_memory" in content:
        print("  = memory deja OK")
    else:
        print("  ! memory online pattern NON trouve")

    # Remplacer le bloc offline
    old_offline = (
        "    context.configure(\n"
        "        url=settings.database_url,\n"
        "        target_metadata=target_metadata,\n"
        "        literal_binds=True,\n"
        "    )\n"
    )
    new_offline = (
        "    context.configure(\n"
        "        url=settings.database_url,\n"
        "        target_metadata=target_metadata,\n"
        "        literal_binds=True,\n"
        '        version_table="alembic_version_memory",\n'
        "    )\n"
    )
    if old_offline in content:
        content = content.replace(old_offline, new_offline)
        print("  + memory offline configure")

    p.write_text(content, encoding="utf-8")


def fix_knowledge() -> None:
    p = ROOT / "services" / "knowledge" / "alembic" / "env.py"
    content = p.read_text(encoding="utf-8")

    # Online
    old_online = (
        "    with connectable.connect() as connection:\n"
        "        context.configure(connection=connection, target_metadata=target_metadata)\n"
    )
    new_online = (
        "    with connectable.connect() as connection:\n"
        "        context.configure(\n"
        "            connection=connection,\n"
        "            target_metadata=target_metadata,\n"
        '            version_table="alembic_version_knowledge",\n'
        "        )\n"
    )
    if old_online in content:
        content = content.replace(old_online, new_online)
        print("  + knowledge online configure")
    elif "alembic_version_knowledge" in content:
        print("  = knowledge deja OK")
    else:
        print("  ! knowledge online pattern NON trouve")

    # Offline
    old_offline = (
        "    context.configure(\n"
        "        url=settings.database_url,\n"
        "        target_metadata=target_metadata,\n"
        "        literal_binds=True,\n"
        '        dialect_opts={"paramstyle": "named"},\n'
        "    )\n"
    )
    new_offline = (
        "    context.configure(\n"
        "        url=settings.database_url,\n"
        "        target_metadata=target_metadata,\n"
        "        literal_binds=True,\n"
        '        dialect_opts={"paramstyle": "named"},\n'
        '        version_table="alembic_version_knowledge",\n'
        "    )\n"
    )
    if old_offline in content:
        content = content.replace(old_offline, new_offline)
        print("  + knowledge offline configure")

    p.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    print("[Fix] memory/alembic/env.py")
    fix_memory()
    print("\n[Fix] knowledge/alembic/env.py")
    fix_knowledge()
    print("\nOK")