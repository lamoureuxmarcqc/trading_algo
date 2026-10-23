"""Test manuel du RAG hybride (vector + BM25 + cross-encoder).

Affiche :
- Rerank confidence (0-1, plus lisible que RRF brut)
- RRF score (pour comparaison)
- Source du rerank (cross_encoder / rrf_fallback / stub_lexical)
- Page et extrait

Usage :
    python test_rag.py
"""
from app.retrieval.hybrid import search

# Requetes naturelles (pas des mots-cles) pour tester le cross-encoder
# QUERIES = [
#    "Quelle est la démarche d'ingénierie des besoins proposée ?",
#    "Que contient le rapport de réingénierie de Jean-Pierre Vickoff ?",
#    "Comment fonctionne le service de workflow ?",
#    "Quels sont les chapitres de la thèse sur l'ingénierie des besoins ?",
#]
QUERIES = [
    "Quel est le montant de l'avis de cotisation ?",
    "Quelles sont les garanties d'électroménagers ?",
    "Quel est le bilan financier ?",
]

TENANT = "lamoureux"
CLASSIFICATIONS = ["HIGHLY_CONFIDENTIAL", "CONFIDENTIAL"]
TOP_K = 3


def main() -> None:
    for q in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"Query : {q}")
        print("=" * 70)

        results = search(
            query=q,
            tenant_id=TENANT,
            allowed_classifications=CLASSIFICATIONS,
            top_k=TOP_K,
        )

        if not results:
            print("  Aucun resultat pertinent (filtre qualite ou min_confidence)")
            continue

        for i, r in enumerate(results, start=1):
            conf = r.get("confidence", 0.0)
            rrf = r.get("rrf_score", 0.0)
            source = r.get("rerank_source", "?")
            page = r.get("page", "?")
            content = r.get("content", "").replace("\n", " ")[:180]

            print(f"\n  [{i}] confidence={conf:.3f}  rrf={rrf:.4f}  page={page}  source={source}")
            print(f"      {content}...")


if __name__ == "__main__":
    main()