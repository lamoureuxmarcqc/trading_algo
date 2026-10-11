# Validation Report — Phase 1

**Date :** 2026-10-11
**Statut :** ✅ Validé
**Validateur :** Marc Lamoureux

---

## 1. Périmètre validé

Phase 1 de la LAMOUREUX AI PLATFORM : AI Gateway + stack d'infrastructure locale.

| Composant | Statut | Preuve |
|---|---|---|
| Tests unitaires ai-gateway | ✅ | 8 / 8 |
| Tests unitaires api-core | ✅ | 33 / 33 |
| Tests racine | ✅ | 3 / 3 |
| **Total tests** | ✅ | **44 / 44** |
| Stack Docker | ✅ | 5 conteneurs Up |
| Gateway → Ollama | ✅ | `LOCAL:true` |
| End-to-end LLM | ✅ | "Hello." en 2.4s |
| Privacy enforcement | ✅ | RESTRICTED → REMOTE = 403 |

---

## 2. Preuves techniques

### 2.1 Tests

```
services/ai-gateway : 8 passed in 0.34s
services/api-core   : 33 passed in 7.06s
tests/ (racine)     : 3 passed
```

### 2.2 Stack Docker

```
lamoureux-postgres     Up (healthy)    5432
lamoureux-redis        Up (healthy)    6379
lamoureux-qdrant       Up (unhealthy*) 6333-6334
lamoureux-ollama       Up              11434
lamoureux-ai-gateway   Up (healthy)    8001
```

\* Qdrant répond correctement (`healthz check passed`) — le statut `unhealthy` vient d'un healthcheck mal configuré dans le compose, non bloquant.

### 2.3 Gateway health

```json
{"status":"ok","version":"0.1.0","providers":{"LOCAL":true,"LOCAL_GPU":false,"REMOTE":false}}
```

### 2.4 Matrice de privacy (`/v1/policy`)

```json
{
  "PUBLIC": ["LOCAL","LOCAL_GPU","REMOTE"],
  "INTERNAL": ["LOCAL","LOCAL_GPU","REMOTE"],
  "CONFIDENTIAL": ["LOCAL","LOCAL_GPU"],
  "HIGHLY_CONFIDENTIAL": ["LOCAL","LOCAL_GPU"],
  "RESTRICTED": ["LOCAL"]
}
```

### 2.5 Refus RESTRICTED → REMOTE

```http
HTTP/1.1 403 Forbidden
x-correlation-id: aaca0444-baeb-4919-af2e-36f31b326366

{"error":"privacy_violation","detail":"Classification RESTRICTED interdit la cible REMOTE","classification":"RESTRICTED","requested_target":"REMOTE"}
```

La règle est appliquée **avant tout appel réseau** (ADR-003).

### 2.6 Test end-to-end

Requête :
```json
{"prompt":"Say hello in one word.","classification":"PUBLIC","tenant_id":"smoke","target":"LOCAL"}
```

Réponse :
```json
{
  "content": "Hello.",
  "model": "llama3.2:3b",
  "target": "LOCAL",
  "tokens_in": 31,
  "tokens_out": 3,
  "latency_ms": 2362
}
```

**Chaîne validée :** Client → Gateway → Ollama → LLM → réponse.

---

## 3. Modèle LLM utilisé

| Attribut | Valeur |
|---|---|
| Nom | `llama3.2:3b` |
| Taille | 2.0 GB |
| Latence cold start | ~38 s |
| Latence warm | ~2.4 s |

Modèle léger choisi pour la validation. Modèles plus gros (`llama3.1:8b`, `qwen2.5`) téléchargeables à la demande.

---

## 4. Points cosmétiques identifiés

| # | Point | Impact | Priorité |
|---|---|---|---|
| 1 | Qdrant `unhealthy` dans `docker ps` | Aucun (Qdrant fonctionne) | Basse |
| 2 | `latency_ms` 38s au 1er appel (cold start) | Normal, peut être mitigé par warm-up | Basse |
| 3 | Pas de `test_main.py` dans ai-gateway | Bug `structlog` a échappé aux tests unitaires | Moyenne |
| 4 | Warning `datetime.utcnow()` (résolu) | Aucun | Résolu |

---

## 5. Corrections appliquées pendant la validation

| # | Fichier | Correction |
|---|---|---|
| 1 | `services/ai-gateway/app/models.py` | Séparation `InferenceRequest` / `InferenceResponse` + suppression `model_used` |
| 2 | `services/ai-gateway/app/main.py` | Conversion `log_level` string → entier (`getattr(logging, ...)`) |
| 3 | `services/ai-gateway/pyproject.toml` | Ajout `pythonpath`, `asyncio_default_fixture_loop_scope` |
| 4 | `services/ai-gateway/conftest.py` | Ajout `sys.path.insert` pour import `app` |
| 5 | `services/api-core/pyproject.toml` | Création avec config pytest locale |
| 6 | `services/api-core/tests/test_ai_trading_service.py` | Dates relatives (`datetime.now(tz=timezone.utc)`) au lieu de hardcodées |
| 7 | `infra/docker/.env` | `OLLAMA_MODEL=llama3.2:3b` (au lieu de `llama3.1:8b`) |
| 8 | `pyproject.toml` (racine) | `norecursedirs` pour exclure `services/` |
| 9 | `.gitignore` | Ajout `.pytest_tmp/`, `.pytest_cache/` |

---

## 6. Conformité aux ADRs

| ADR | Règle | Statut |
|---|---|---|
| ADR-001 | Local-first par défaut | ✅ |
| ADR-002 | AI Gateway point d'entrée unique | ✅ |
| ADR-003 | Classification avant routage | ✅ (preuve 2.5) |
| ADR-004 | Mémoire multi-niveaux | ⏳ (phase 3) |
| ADR-005 | Permissions agents | ⏳ (phase 4) |
| ADR-006 | Frontière infra/vertical | ✅ (structure respectée) |

---

## 7. Ce qui est débloqué

Phase 1 terminée ouvre :

1. **Phase 2 — Knowledge Layer** : ingestion PDF, embeddings, RAG, citations
2. **Phase 3 — Memory Layer** : 4 niveaux de mémoire
3. **Phase 4 — Orchestrator** : superviseur + agents

L'infrastructure est prête à recevoir les services suivants.

---

## 8. Commandes de reproduction

```powershell
# Stack
cd infra\docker
Copy-Item .env.example .env
# Éditer .env : OLLAMA_MODEL=llama3.2:3b
docker compose up -d
docker exec lamoureux-ollama ollama pull llama3.2:3b

# Tests
cd services\ai-gateway ; pytest
cd ..\api-core        ; pytest
cd ..\..              ; pytest

# Smoke test
$body = '{"prompt":"Say hello.","classification":"PUBLIC","tenant_id":"smoke","target":"LOCAL"}'
Invoke-RestMethod -Uri http://localhost:8001/v1/inference -Method POST -ContentType "application/json" -Body $body
```

---

## 9. Signature

**Validé par :** Marc Lamoureux
**Date :** 2026-10-11
**Prochaine étape :** Phase 2 — Knowledge Layer