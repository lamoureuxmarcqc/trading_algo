@'
from pathlib import Path

p = Path("app/scanner/state.py")
c = p.read_text(encoding="utf-8")

if "SELECT id, sha256, document_id, status, size_bytes, modified_at" in c:
    print("Fix B deja present")
else:
    c = c.replace(
        "SELECT id, sha256, document_id, status\n",
        "SELECT id, sha256, document_id, status, size_bytes, modified_at\n",
        1,
    )
    c = c.replace(
        '"status": row.status,\n        }\n    finally:',
        '"status": row.status,\n            "size_bytes": row.size_bytes,\n            "modified_at": row.modified_at,\n        }\n    finally:',
        1,
    )
    p.write_text(c, encoding="utf-8")
    print("Fix B applique")
'@ | Set-Content -Path "patch_b.py" -Encoding UTF8

python patch_b.py
Remove-Item patch_b.py