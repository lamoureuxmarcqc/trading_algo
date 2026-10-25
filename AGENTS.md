# AGENTS.md — LAMOUREUX AI PLATFORM

> Lis ce fichier en entier avant toute modification.
> Ce document est la source de vérité pour Codex sur ce repository.

---

## 1. Contexte

Tu travailles sur **LAMOUREUX AI PLATFORM**, une infrastructure IA privée, modulaire, local-first et model-agnostic, capable d'auto-amélioration contrôlée.

Le repository contient deux couches complémentaires :

1. **AI Infrastructure Layer** (`services/ai-gateway`, `knowledge`, `memory`, `orchestrator`, `model-registry`, `evaluation`, `self-improvement`)
2. **Institutional Trading Layer** (`trading_algo`, `services/api-core`, `apps/web`)

Le trading est le **premier vertical**. La santé, la recherche et les PME viendront ensuite.

Principe directeur :

> **Build the intelligence infrastructure, not just another AI application.**

---

## 2. Règles non négociables

1. **Local-first** — les données sensibles restent locales par défaut.
2. **Classification avant routage** — aucune requête ne part vers un provider distant sans classification explicite.
3. **AI Gateway obligatoire** — tout appel LLM passe par `services/ai-gateway`.
4. **Research ≠ Execution** — aucun modèle de recherche n'obtient l'autorité d'exécution live.
5. **Risk controls server-side** — les contrôles critiques sont dans l'API, jamais seulement dans le frontend.
6. **Idempotence** — toute opération critique (ordre, staging) exige `Idempotency-Key`.
7. **Audit trail** — toute décision matérielle est traçable.
8. **Pas de fausse précision** — confiance, fraîcheur et source toujours exposées.
9. **Human in the loop** — les décisions `HIGHLY_CONFIDENTIAL`, `RESTRICTED` et toute auto-amélioration exigent validation.
10. **ADRs** — toute décision structurante est documentée dans `docs/architecture/adr/`.
11. **Aucun auto-déploiement** — toute amélioration passe par le pipeline `SELF_IMPROVEMENT.md` (7 étages).
12. **Aucune modification** de sécurité, risques, ordres, audit sans double validation.
13. **Tout changement est réversible** — rollback obligatoire.
14. **Tout changement est testé** avant déploiement.

---

## 3. Classification des données

| Niveau | Cible autorisée |
|---|---|
| `PUBLIC` | LOCAL, LOCAL_GPU, REMOTE |
| `INTERNAL` | LOCAL, LOCAL_GPU, REMOTE |
| `CONFIDENTIAL` | LOCAL, LOCAL_GPU |
| `HIGHLY_CONFIDENTIAL` | LOCAL, LOCAL_GPU |
| `RESTRICTED` | LOCAL uniquement |

`RESTRICTED` ne quitte **jamais** le périmètre local.

---

## 4. Stack technique

### Backend
- Python 3.10+
- FastAPI
- Pydantic v2
- SQLAlchemy
- Alembic
- PostgreSQL + pgvector
- Redis
- Qdrant

### IA
- Ollama (local)
- vLLM (GPU)
- Modèles open-weight
- Embeddings locaux
- Reranking local

### Frontend
- Next.js
- React
- TypeScript strict

### Infra
- Docker Compose
- OpenTelemetry
- Prometheus / Grafana

---

## 5. Structure cible du repository

```text
.
├── AGENTS.md
├── SELF_IMPROVEMENT.md
├── LAMOUREUX_AI_PLATFORM.md
├── README.md
├── trading_algo/
├── services/
│   ├── api-core/
│   ├── ai-gateway/
│   ├── knowledge/
│   ├── memory/
│   ├── orchestrator/
│   ├── model-registry/
│   ├── evaluation/
│   └── self-improvement/
├── apps/
│   └── web/
├── docs/
│   ├── architecture/
│   │   └── adr/
│   ├── hedgefund-platform.md
│   └── ai-trading-model-integration-roadmap.md
├── infra/
│   ├── docker/
│   ├── ollama/
│   ├── vllm/
│   └── monitoring/
└── tests/
```

---

## 6. Conventions de code

### Python
- PEP 8
- Type hints partout
- `async` quand pertinent
- Pydantic pour les contrats
- Pas de `print` — logging structuré
- Tests unitaires obligatoires pour toute logique métier

### API
- Versioning `/api/v1`
- `X-Correlation-ID`
- `X-Tenant-ID`
- `Idempotency-Key` pour les opérations critiques
- Erreurs normalisées
- Health / readiness / metrics exposés

### Base de données
- Migrations Alembic uniquement
- Jamais de modification manuelle du schéma
- Chaque migration est réversible si possible

### Frontend
- TypeScript strict
- Aucune logique de risque côté client
- Le frontend ne contourne jamais un guardrail API

### Git
- Conventional Commits
- Une PR par phase
- Ne jamais committer de secrets

---

## 7. Contrats clés à respecter

### AI Gateway

```http
POST /v1/inference
```

Champs obligatoires : `prompt`, `classification`, `tenant_id`.
Champs optionnels : `model`, `target`, `max_tokens`, `temperature`.

Réponse : `content`, `model_used`, `target_used`, `tokens_in/out`, `latency_ms`, `confidence`, `sources`.

### Knowledge Layer

```http
POST /api/v1/knowledge/ingest
POST /api/v1/knowledge/query
```

Toute réponse RAG expose `answer`, `sources`, `confidence`.

### Memory Layer

Quatre types : `SHORT_TERM` (Redis), `LONG_TERM` (PostgreSQL), `KNOWLEDGE` (pgvector/Qdrant), `DECISION` (PostgreSQL, append-only).

Une décision enregistrée ne peut **jamais** être modifiée.

### Orchestrator

```http
POST /api/v1/agents/route
```

Le superviseur : classifie, autorise, exécute avec guardrails, demande validation si requis, écrit l'audit.

### Model Registry

```http
POST /api/v1/models/register
POST /api/v1/models/{id}/promote
POST /api/v1/models/{id}/rollback
```

### Self-Improvement

```http
GET  /api/v1/improvement/proposals
POST /api/v1/improvement/proposals/{id}/simulate
POST /api/v1/improvement/proposals/{id}/approve
POST /api/v1/improvement/proposals/{id}/deploy
POST /api/v1/improvement/proposals/{id}/rollback
```

---

## 8. Tâches prioritaires

### Phase 0 — Fondations
- Créer les 6 ADRs fondateurs
- Mettre en place le docker-compose avec PostgreSQL+pgvector, Redis, Qdrant, Ollama
- Créer la structure des services vides avec tests

### Phase 1 — AI Gateway
- Providers `ollama`, `vllm`, `remote`
- `privacy.py`, `router.py`, `fallback.py`
- Tests unitaires + intégration

### Phase 2 — Knowledge Layer
- Ingestion PDF/DOCX/HTML/MD
- Chunking sémantique, embeddings locaux
- pgvector + BM25 + RRF + reranking
- Citations obligatoires

### Phase 3 — Memory Layer
- `short_term.py`, `long_term.py`, `knowledge.py`, `decision.py`, `consolidation.py`

### Phase 4 — Orchestrator
- Supervisor, agents (research, finance, health)
- Guardrails : budget, timeout, sandbox, permissions
- Audit trail

### Phase 5 — Model Registry
- Enregistrement, promotion, rollback, comparaison

### Phase 6 — Evaluation
- Factuality, hallucination rate, retrieval precision, calibration

### Phase 7 — Self-Improvement (Observer + Detector)
- Collecte métriques multi-sources
- Détection d'opportunités (code, AI, knowledge, feature, agent)
- Génération de propositions

### Phase 8 — Self-Improvement (Simulator + Reviewer)
- Sandbox de simulation
- Interface de validation humaine
- Circuit breakers + rate limiting

### Phase 9 — Self-Improvement (Deployer + Evaluator)
- Staged rollout
- Monitoring post-déploiement
- Rollback automatique

### Phase 10 — Investment Agent
- Intégration `AchatVenteAction`
- Decision Memory pour chaque ordre
- Pre-trade checks institutionnels

### Phase 11 — Health Workspace
- Workspace isolé, ingéré médical, evidence classification

### Phase 12 — Distributed GPU
- vLLM, workers distants, model routing, monitoring

---

## 9. Definition of Done

Une tâche n'est complète que si :

- Les tests passent
- Le build frontend passe
- Les migrations Alembic sont fournies
- Le fallback / degraded mode est explicite
- L'audit trail est émis
- La classification est respectée
- Aucun secret n'est commité
- La documentation est mise à jour
- Aucun chemin d'exécution live n'est implicitement activé
- Pour les améliorations : pipeline `SELF_IMPROVEMENT.md` respecté (7 étages)

---

## 10. Ce que Codex ne doit PAS faire

- ❌ Activer le live trading
- ❌ Envoyer une donnée `RESTRICTED` vers un provider distant
- ❌ Court-circuiter l'AI Gateway
- ❌ Mettre de la logique de risque côté frontend
- ❌ Committer des secrets
- ❌ Introduire une dépendance à un fournisseur unique
- ❌ Présenter un fallback heuristique comme SHAP/LIME
- ❌ Modifier une décision enregistrée en Decision Memory
- ❌ Créer un ordre sans `Idempotency-Key`
- ❌ Modifier le schéma sans migration Alembic
- ❌ Auto-déployer une amélioration sans validation humaine
- ❌ Modifier la sécurité, les risques, les ordres ou l'audit sans double validation
- ❌ Supprimer automatiquement de la connaissance ou du code
- ❌ Sauter un étage du pipeline d'auto-amélioration

## Sécurité — Règles pour les secrets
## Sécurité — Credentials
## ** JAMAIS dans le NAS** :
- Mots de passe en clair
- Fichiers `.key`, `.pem`, `.env` avec secrets
- Tokens API
- Clés privées

## ** TOUJOURS dans Bitwarden** (ou KeePassXC) :
- Mots de passe de comptes
- Clés API
- Réponses aux questions de sécurité

**Patterns d'exclusion obligatoires** dans les scan_sources :

**Avant tout scan d'un nouveau dossier** :

1. Vérifier manuellement le contenu des `.txt` courts (< 2 KB) :
   ```powershell
   Get-ChildItem $path -Recurse -Filter "*.txt" -File |
       Where-Object { $_.Length -lt 2000 } |
       Select-String -Pattern "(?i)(password|mot de passe|login|username)\s*[:=]" |
       Select-Object Path, Line
---
SELECT d.filename FROM documents d
JOIN chunks c ON c.document_id = d.id
WHERE c.content ~* '(password\s*[:=]|mot\s+de\s+passe\s*[:=])'
  AND LENGTH(c.content) < 500;

## 11. Commandes utiles

```bash
# Installation
python -m venv .venv
source .venv/bin/activate  # ou .\.venv\Scripts\Activate.ps1
pip install -e .

# Infra locale
docker compose -f infra/docker/docker-compose.yml up -d

# API
uvicorn app.main:app --reload --app-dir services/api-core

# Tests
python -m unittest discover tests
pytest services/api-core/tests -q

# Migrations
alembic upgrade head
```

---

## 12. Prompt d'amorçage pour Codex

> Lis `AGENTS.md` et `SELF_IMPROVEMENT.md` en entier.
> Commence par la **Phase 0** : crée les ADRs 001 à 006 et la structure vide des services.
> Ensuite implémente la **Phase 1 — AI Gateway** avec tests.
> Ne touche pas au live trading.
> N'implémente **aucune** auto-amélioration avant que Evaluation + Model Registry + Audit trail soient complets.
> Ouvre une PR par phase.
> Pour chaque PR : tests, documentation, migration si nécessaire, audit trail.

---

## 13. Références

- `LAMOUREUX_AI_PLATFORM.md`
- `SELF_IMPROVEMENT.md`
- `README.md`
- `docs/hedgefund-platform.md`
- `docs/ai-trading-model-integration-roadmap.md`
- `docs/architecture/README.md`
- `docs/architecture/adr/README.md`