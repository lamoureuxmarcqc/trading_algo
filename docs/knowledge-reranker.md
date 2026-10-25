# Knowledge Layer — Phases 2c–2d : reranking et citations

**Date :** 2026-10-20
**Statut :** Phase 2c livrée (validation stack reelle restante), Phase 2d livree et couverte par tests unitaires
**Périmètre :** `services/knowledge/app/retrieval/`, `app/models.py`

---

## 1. Objectif

Après la fusion RRF (vector + BM25), reclasser les meilleurs candidats avec un cross-encoder **local** afin d'écarter les chunks lexicalement proches mais non pertinents (ex. chunks décalés `SODQ`, `EUHDN` observés en Phase 2b).

```text
vector ─┐
        ├─► RRF ─► filtres qualité ─► pool (N=30) ─► RERANKER ─► top_k ─► citations (2d)
BM25  ──┘
```

## 2. Fichiers

| Fichier | Rôle |
|---|---|
| `app/retrieval/reranker.py` | Scoreurs (`CrossEncoderScorer`, `StubLexicalScorer`), `rerank()`, `rerank_async()` |
| `app/retrieval/hybrid.py` | Intégration post-RRF dans `search()` et `search_async()` (opt-in) |
| `tests/test_reranker.py` | 12 tests unitaires du reranker |
| `tests/test_hybrid_rerank.py` | 5 tests d'intégration hybrid + reranker (recherches mockées) |

## 3. Configuration

Tous les réglages sont **optionnels** (lus via `getattr` avec valeur par défaut) : la configuration actuelle continue de fonctionner sans modification.

| Réglage | Défaut | Description |
|---|---|---|
| `rerank_enabled` | `False` | Active le reranking. **Désactivé par défaut** : comportement RRF inchangé. |
| `rerank_backend` | `cross_encoder` | `cross_encoder` ou `stub` (tests / CPU sans modèle). |
| `rerank_model` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Modèle multilingue (corpus FR/EN). |
| `rerank_candidates` | `30` | Taille du pool RRF transmis au reranker (≥ `top_k`). |
| `rerank_batch_size` | `16` | Taille de lot d'inférence. |
| `rerank_device` | `None` (auto) | `cpu` / `cuda`. |

Pour activer, ajouter ces champs à `Settings` (`services/knowledge/app/config.py`) ou les fournir par variables d'environnement selon le préfixe en place, puis `rerank_enabled=True`.

## 4. Gouvernance

- **Local-first (ADR-001)** : aucun appel réseau à l'inférence. Le modèle est téléchargé **une seule fois** depuis le hub au premier chargement ; ensuite, positionner `HF_HUB_OFFLINE=1` pour garantir l'absence de tout accès réseau. Aucun contenu de document ne quitte la machine.
- **Pas de fausse précision** : chaque résultat porte `rerank_source` (`cross_encoder` | `stub_lexical` | `rrf_fallback`) et `confidence_source`. Le score RRF d'origine reste disponible (`rrf_score`, `rrf_confidence`, `pre_rerank_rank`).
- **Repli explicite** : si le cross-encoder est indisponible (dépendance, modèle, erreur d'inférence), `rerank()` retourne l'ordre RRF tronqué, étiqueté `rrf_fallback`, avec un warning journalisé. Aucun scoreur de substitution n'est activé silencieusement.
- **Confiance** : `confidence` = score du cross-encoder borné à [0, 1]. **L'ordre est fiable ; la valeur absolue n'est pas calibrée.** Elle ne doit pas être présentée comme une probabilité (calibration prévue avec le framework Evaluation, ECE).
- **Classification** : le reranker ne reçoit que des candidats déjà filtrés par tenant et classification en amont (vector/BM25). Il n'élargit jamais le périmètre.

## 5. Compatibilité

- Reranking désactivé : `search()` / `search_async()` retournent exactement la même sortie qu'avant (mêmes champs, mêmes valeurs).
- Reranking activé : les champs existants (`chunk_id`, `document_id`, `content`, `page`, `section`, `classification`, `vector_rank`, `keyword_rank`, `rrf_score`) sont conservés ; `confidence` est remplacé par la confiance du reranker (confiance RRF dans `rrf_confidence`) ; champs ajoutés : `rerank_score`, `rerank_source`, `confidence_source`, `pre_rerank_rank`.
- Aucune migration de base de données.

## 6. Validation

**Fait (suite standard, 2026-10-25)** : 57 tests verts depuis la racine du depot :

```powershell
.\venv\Scripts\python.exe -m pytest services\knowledge\tests -q
```

Cette suite comprend les tests de citations et verifie que la configuration
isole les variables d'environnement des autres services. Les 17 tests de
reranking historiques restent dans `app/test/` et ne font pas partie du
`testpaths` standard ; ils doivent etre executes sur une machine ou
l'executueur de threads Python est disponible pour valider le chemin async.

**Non vérifié — à exécuter dans le dépôt :**

```powershell
cd services\knowledge
pytest                       # les 48 tests existants + 17 nouveaux
```

Test end-to-end sur les 34 documents ingérés :

1. Activer `rerank_enabled=True`, `rerank_backend=cross_encoder`.
2. Interroger avec 3–5 requêtes de référence ; vérifier que les chunks parasites ont disparu du top 5 et que `rerank_source == "cross_encoder"`.
3. Mesurer la latence ajoutée (30 paires, CPU) et la consigner ici.

Le choix du modèle multilingue par défaut est une hypothèse (corpus FR/EN) : à confirmer par la comparaison avant/après ci-dessus.

## 7. Phase 2d — Citations automatiques

Chaque resultat de retrieval peut maintenant etre converti par
`app.retrieval.citations.build_sources()` en une liste de `Source` Pydantic
ordonnee par pertinence. Une citation expose :

- `document_id` et `chunk_id` pour retrouver l'evidence ;
- `page` et `section` lorsqu'elles sont disponibles ;
- `excerpt`, prefixe literal du chunk (500 caracteres par defaut, sans
  reecriture) ;
- `score`, confiance de retrieval bornee dans `[0, 1]`.

Ce score est un signal de retrieval, pas une probabilite calibree. La
calibration reste du ressort du framework Evaluation. Le Knowledge Layer ne
genere pas de synthese via un LLM : le champ `answer` du contrat
`QueryResponse` est explicitement nul tant qu'une synthese n'a pas ete creee
par un flux distinct passant par AI Gateway.

Les citations sont construites seulement apres les filtres tenant et
classification executes par le retrieval. Elles ne peuvent donc pas elargir
le perimetre des documents accessibles.

## 8. Suite

- **2e** — API FastAPI `/api/v1/knowledge/*` avec authentification et
  propagation du tenant avant exposition du service.
- **2f** — Docker + integration.
