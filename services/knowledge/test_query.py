"""Test manuel du RAG hybride."""
from app.retrieval.hybrid import search

QUERIES = [
    "Quelle est la démarche d'ingénierie des besoins proposée ?",
    "Quel est le montant de l'avis de cotisation ?",
    "facture Sears 2017 lave-vaisselle",
    "garantie Mitsubishi thermopompe",
]

for q in QUERIES:
    print(f"\n{'=' * 70}")
    print(f"Query: {q}")
    print("=" * 70)
    results = search(
        query=q,
        tenant_id="lamoureux",
        allowed_classifications=["HIGHLY_CONFIDENTIAL", "CONFIDENTIAL"],
        top_k=3,
    )
    if not results:
        print("  Aucun resultat")
        continue
    for r in results:
        print(f"  [{r.get('confidence', 0):.3f}] p{r['page']} {r['content'][:120]}")
