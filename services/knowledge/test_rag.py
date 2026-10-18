@'
from app.retrieval.hybrid import search

queries = ["mandat", "signature", "date", "contrat"]

for q in queries:
    print(f"\n=== Query: {q} ===")
    results = search(
        query=q,
        tenant_id="lamoureux",
        allowed_classifications=["HIGHLY_CONFIDENTIAL", "CONFIDENTIAL"],
        top_k=2,
    )
    if not results:
        print("  Aucun resultat")
        continue
    for r in results:
        print(f"  Score: {r['rrf_score']:.4f} | Doc: {r['document_id'][:8]}... | Page: {r.get('page')}")
        print(f"  {r['content'][:150]}...")
'@ | Set-Content -Path "test_rag.py" -Encoding UTF8

python test_rag.py