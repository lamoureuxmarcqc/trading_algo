cd C:\Users\marc\source\projet_trading\services\knowledge
@'
from app.retrieval.hybrid import search

queries = [
    "Quel est le montant de l'avis de cotisation ?",
    "Quel est le bilan financier de 2012 pour Louise ?",
    "Combien ai-je payé chez le dentiste en 2021 ?",
    "Quelle est la date de la déclaration de revenus 2024 ?",
    "Quel est le numéro du reçu REEE ?",
]

for q in queries:
    print(f"\n{'=' * 70}")
    print(f"Q: {q}")
    print("=" * 70)
    results = search(
        query=q,
        tenant_id="lamoureux",
        allowed_classifications=["HIGHLY_CONFIDENTIAL"],
        top_k=3,
    )
    if not results:
        print("  Aucun resultat")
        continue
    for r in results:
        print(f"  [{r.get('confidence', 0):.3f}] p{r['page']} {r['content'][:120]}")
'@ | Set-Content -Path "test_query.py" -Encoding UTF8

python test_query.py