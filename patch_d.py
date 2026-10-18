@'
from pathlib import Path

p = Path("app/ingestion/extractors.py")
c = p.read_text(encoding="utf-8")

if 'header != b"%PDF-"' in c:
    print("Fix D deja present")
else:
    old = """def extract_pdf(path: Path) -> ExtractedDocument:
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ExtractionError(f"PDF illisible : {e}") from e
"""
    new = """def extract_pdf(path: Path) -> ExtractedDocument:
    # Verifier la signature PDF avant tout
    try:
        with open(path, "rb") as f:
            header = f.read(5)
    except OSError as e:
        raise ExtractionError(f"Lecture impossible : {e}") from e

    if header != b"%PDF-":
        raise ExtractionError(
            f"Fichier non-PDF (header={header!r}, attendu b'%PDF-')"
        )

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ExtractionError(f"PDF illisible : {e}") from e
"""
    if old in c:
        c = c.replace(old, new, 1)
        p.write_text(c, encoding="utf-8")
        print("Fix D applique")
    else:
        print("ERREUR: pattern original introuvable, verifier extractors.py")
'@ | Set-Content -Path "patch_d.py" -Encoding UTF8

python patch_d.py
Remove-Item patch_d.py