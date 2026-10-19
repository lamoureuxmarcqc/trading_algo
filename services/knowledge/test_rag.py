from app.retrieval.hybrid import search

queries = ["chapitre", "présentation", "rapport", "PSA"]

for q in queries:
    print(f"\n=== {q} ===")
    results = search(
        query=q,
        tenant_id="lamoureux",
        allowed_classifications=["HIGHLY_CONFIDENTIAL", "CONFIDENTIAL"],
        top_k=2,
    )
    for r in results:
        print(f"  Score: {r['rrf_score']:.4f} | Page: {r.get('page')}")
        print(f"  {r['content'][:200]}...")
