# Codex — Prompts par phase

Chaque prompt est autonome. Copier-coller dans Codex après avoir lu `AGENTS.md`.

---

## Phase 0 — Fondations

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 0 — Fondations.

1. Vérifier que les 6 ADRs existent dans docs/architecture/adr/ (001 à 006).
2. Vérifier la structure infra/docker/docker-compose.yml.
3. Créer les dossiers vides des services avec __init__.py et README.md :
   - services/ai-gateway
   - services/knowledge
   - services/memory
   - services/orchestrator
   - services/model-registry
   - services/evaluation
   - services/self-improvement
4. Créer un Makefile à la racine avec : up, down, logs, test, lint.
5. Créer .env.example à la racine.

Contraintes :
- Aucun secret commité.
- Chaque service a un pyproject.toml minimal.
- Pas de code métier à ce stade.

Ouvre une PR intitulée "phase-0: foundations".
```

---

## Phase 1 — AI Gateway

```
Lis AGENTS.md.

Tâche : Phase 1 — AI Gateway.

Le scaffold existe dans services/ai-gateway/. Complète-le :

1. Ajouter un provider "openai" (remote) dans app/providers/openai.py.
2. Ajouter un fichier app/fallback.py qui, sur échec d'un provider, tente la cible suivante autorisée par classification.
3. Ajouter un endpoint GET /v1/metrics exposant les Prometheus counters :
   - aigw_requests_total{target,classification}
   - aigw_tokens_in_total{target}
   - aigw_tokens_out_total{target}
   - aigw_latency_seconds{target}
4. Ajouter tests/test_inference.py avec :
   - test LOCAL classification PUBLIC → 200
   - test RESTRICTED + REMOTE → 403
   - test CONFIDENTIAL + REMOTE → 403
   - test fallback quand ollama indisponible
5. Mettre à jour services/ai-gateway/README.md.

Contraintes :
- Classification appliquée AVANT tout appel réseau.
- Tests passent sans réseau (mock providers).
- Type hints partout.

Ouvre une PR intitulée "phase-1: ai-gateway".
```

---

## Phase 2 — Knowledge Layer

```
Lis AGENTS.md.

Tâche : Phase 2 — Knowledge Layer.

Créer services/knowledge/ avec :

1. app/ingestion/extractors.py : PDF, DOCX, HTML, MD
2. app/chunking/semantic.py : chunking sémantique avec overlap 10%
3. app/embeddings/local.py : embeddings via sentence-transformers (BAAI/bge-small-en-v1.5)
4. app/storage/pgvector.py : schéma chunks + metadata + classification
5. app/retrieval/hybrid.py : vector + BM25 + RRF
6. app/retrieval/reranker.py : cross-encoder local
7. app/citations.py : attache document_id, page, excerpt à chaque réponse
8. app/main.py : endpoints POST /ingest, POST /query, GET /documents/{id}

Schéma DB (Alembic) :
- documents(id, tenant_id, classification, filename, ingested_at)
- chunks(id, document_id, content, embedding vector(384), page, section, metadata jsonb)

Contraintes :
- Classification héritée du document, jamais devinée.
- Toute réponse expose sources + confidence.
- RESTRICTED refusé si tenant mismatch.

Tests :
- ingestion PDF → chunks
- query → top_k avec citations
- classification respectée

Ouvre une PR intitulée "phase-2: knowledge-layer".
```

---

## Phase 3 — Memory Layer

```
Lis AGENTS.md.

Tâche : Phase 3 — Memory Layer.

Créer services/memory/ avec les 4 mémoires :

1. app/short_term.py — Redis, TTL 1h, conversation buffer
2. app/long_term.py — PostgreSQL, faits validés, table facts
3. app/knowledge.py — proxy vers services/knowledge (pas de stockage local)
4. app/decision.py — PostgreSQL, table decisions, APPEND-ONLY

Table decisions (Alembic) :
- id, tenant_id, created_at, decision, reason, evidence jsonb,
  action, outcome jsonb NULL, evaluation jsonb NULL,
  model_version, confidence, correlation_id

Règles :
- UPDATE sur decisions est interdit par trigger PostgreSQL.
- Seuls outcome et evaluation peuvent être ajoutés (colonnes NULL → valeur).
- Un trigger protège l'immuabilité.

Endpoints :
- POST /short-term/append
- POST /long-term/validate  (nécessite human_validation=true)
- POST /decision/record
- POST /decision/{id}/evaluate

Tests :
- tentative UPDATE sur decision → erreur
- evaluate ajoute outcome sans modifier decision

Ouvre une PR intitulée "phase-3: memory-layer".
```

---

## Phase 4 — Orchestrator

```
Lis AGENTS.md.

Tâche : Phase 4 — Orchestrator.

Créer services/orchestrator/ avec :

1. app/supervisor.py : classification de tâche → agent
2. app/agents/base.py : Agent abstrait (permissions, budget, timeout)
3. app/agents/research.py
4. app/agents/finance.py
5. app/agents/health.py
6. app/guardrails.py : budget tokens, timeout, sandbox, permissions tools
7. app/audit.py : émission d'événements vers outbox
8. app/main.py : POST /route, GET /agents, GET /audit/{id}

Chaque agent a un manifest YAML :
services/orchestrator/agents/finance.yaml
```yaml
name: finance
tools: [knowledge.query, memory.decision.record]
classification_max: HIGHLY_CONFIDENTIAL
targets: [LOCAL, LOCAL_GPU]
budget_tokens: 8000
timeout_seconds: 60
sandbox: true
```

Contraintes :
- Aucun agent ne peut élever sa classification.
- Aucun agent n'a accès au live trading.
- Toute invocation est auditée.

Ouvre une PR intitulée "phase-4: orchestrator".
```

---

## Phase 5 — Model Registry

```
Lis AGENTS.md.

Tâche : Phase 5 — Model Registry.

Créer services/model-registry/ avec :

1. Table models (Alembic) :
   - id, name, version, base_model, dataset_id, parameters jsonb,
     metrics jsonb, status, created_at, provenance
2. Endpoints :
   - POST /models/register
   - GET /models
   - GET /models/{id}
   - POST /models/{id}/promote
   - POST /models/{id}/rollback
   - GET /models/compare?a=&b=
3. app/provenance.py : trace dataset, code commit, params
4. app/comparison.py : compare deux versions sur métriques

Règles :
- Une version promue en production doit avoir metrics non vides.
- Rollback restaure la version précédente.
- Toute promotion est auditée.

Tests :
- register → promote → rollback

Ouvre une PR intitulée "phase-5: model-registry".
```

---

## Phase 6 — Evaluation

```
Lis AGENTS.md.

Tâche : Phase 6 — Evaluation.

Créer services/evaluation/ avec :

1. app/metrics/factuality.py — vérifie claims vs sources
2. app/metrics/hallucination.py — claims non supportés / total
3. app/metrics/retrieval.py — precision@k, recall@k
4. app/metrics/calibration.py — confidence calibration (ECE)
5. app/datasets/loader.py — datasets versionnés
6. app/runner.py — exécute une évaluation complète
7. Endpoints :
   - POST /evaluate
   - GET /evaluations/{id}
   - GET /metrics/summary

Dataset de référence :
services/evaluation/datasets/finance-eval-v1.jsonl
Format : {"query": "...", "expected_sources": [...], "reference_answer": "..."}

Règles :
- Aucun appel externe pour évaluer RESTRICTED.
- Les métriques sont stockées et immuables.
- Une évaluation référence toujours un dataset versionné.

Ouvre une PR intitulée "phase-6: evaluation".
```

---

## Phase 7 — Self-Improvement (Observer + Detector)

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 7 — Self-Improvement Observer + Detector.

PRÉREQUIS : phases 5 et 6 complètes.

Créer services/self-improvement/ avec :

1. app/observer.py : collecte métriques depuis
   - evaluation (factuality, hallucination, calibration)
   - model-registry (drift)
   - api-core (latence, erreurs)
   - audit (agents, budgets)
2. app/detector.py : détecte opportunités selon 5 dimensions
   - CODE, AI, KNOWLEDGE, FEATURE, AGENT
3. app/proposer.py : génère propositions au format YAML (cf. SELF_IMPROVEMENT.md)
4. app/storage.py : table proposals (Alembic)

Endpoint :
- GET /improvement/proposals
- GET /improvement/proposals/{id}

Règles :
- Rate limiting strict (5 CODE/j, 3 AI/j, 10 KNOWLEDGE/j, etc.)
- Aucune proposition n'est auto-déployée.
- Toute proposition est stockée immuablement.

Tests :
- detect → propose → proposition stockée avec status="pending_review"

Ouvre une PR intitulée "phase-7: self-improvement-observer".
```

---

## Phase 8 — Self-Improvement (Simulator + Reviewer)

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 8 — Simulator + Reviewer.

1. app/simulator.py : sandbox par dimension
   - CODE → run tests sur branche isolée
   - AI → A/B sur dataset de référence
   - KNOWLEDGE → dry-run ingestion
   - FEATURE → analytics simulation
   - AGENT → replay d'invocations passées
2. app/reviewer.py : workflow validation humaine
3. app/main.py : endpoints simulate / approve / reject / modify
4. Circuit breakers (app/breakers.py) :
   - 2 rollbacks consécutifs → pause dimension
   - factuality -15% → halt
   - hallucination > 15% → halt

Endpoints :
- POST /improvement/proposals/{id}/simulate
- POST /improvement/proposals/{id}/approve
- POST /improvement/proposals/{id}/reject

Règles :
- Aucune approbation sans simulation préalable.
- Double validation pour propositions touchant sécurité/risques/ordres/audit.
- Toute décision humaine est auditée.

Ouvre une PR intitulée "phase-8: self-improvement-simulator".
```

---

## Phase 9 — Self-Improvement (Deployer + Evaluator)

```
Lis AGENTS.md et SELF_IMPROVEMENT.md.

Tâche : Phase 9 — Deployer + Evaluator.

1. app/deployer.py : staged rollout
   - apply change sur branche/cible
   - monitoring 24h
   - promotion automatique si métriques stables
2. app/evaluator.py : métriques post-déploiement
   - factuality, hallucination, latence, erreurs
3. app/rollback.py : rollback automatique si critère violé
4. Endpoints :
   - POST /improvement/proposals/{id}/deploy
   - POST /improvement/proposals/{id}/rollback
   - GET /improvement/metrics
   - GET /improvement/audit

Règles :
- Un seul déploiement à la fois.
- Rollback window 24h.
- Toute décision finale (KEEP/ROLLBACK) est auditée.
- Circuit breaker déclenche rollback automatique.

Ouvre une PR intitulée "phase-9: self-improvement-deployer".
```

---

## Phase 10 — Investment Agent

```
Lis AGENTS.md.

Tâche : Phase 10 — Investment Agent.

1. Créer services/orchestrator/app/agents/investment.py
2. Intégrer AchatVenteAction via services/api-core
3. Chaque ordre stagé crée un record Decision Memory :
   - decision (BUY/SELL/HOLD)
   - reason
   - evidence (RAG sources)
   - action
   - outcome (NULL au départ)
4. Pre-trade checks institutionnels :
   - restricted list
   - liquidité
   - turnover max
   - concentration
5. Pro-forma blotter avant staging

Contraintes :
- Aucun ordre live.
- Idempotency-Key obligatoire.
- Classification HIGHLY_CONFIDENTIAL minimum.
- Decision Memory append-only.

Ouvre une PR intitulée "phase-10: investment-agent".
```

---

## Phase 11 — Health Workspace

```
Lis AGENTS.md.

Tâche : Phase 11 — Health Workspace.

Workspace strictement isolé : tenant dédié, DB isolée logiquement.

1. services/knowledge/workspaces/health/ : config isolation
2. Ingestion documents médicaux → classification RESTRICTED obligatoire
3. Classification des niveaux de preuve :
   FACT / SOURCE / INTERPRETATION / HYPOTHESIS / UNCERTAINTY
4. Timeline des événements médicaux
5. Détection de contradictions (signalement, pas résolution)

Règles :
- Aucune donnée santé ne sort du workspace.
- RESTRICTED → LOCAL only, sans exception.
- Aucune donnée santé ne peut être utilisée pour fine-tuning sans validation.

Ouvre une PR intitulée "phase-11: health-workspace".
```

---

## Phase 12 — Distributed GPU

```
Lis AGENTS.md.

Tâche : Phase 12 — Distributed GPU.

1. infra/vllm/ : config vLLM pour GPU server
2. infra/docker/docker-compose.gpu.yml : override GPU
3. services/ai-gateway/app/providers/vllm.py : support multi-endpoints
4. services/orchestrator/app/scheduler.py : routage par charge
5. Monitoring : Prometheus + Grafana dashboards
6. Fault tolerance : fallback GPU → LOCAL si GPU indisponible

Règles :
- RESTRICTED ne va JAMAIS sur GPU distant si GPU est hors du périmètre local.
- Le fallback respecte la politique de classification.
- Le scheduler ne peut pas router vers une cible non autorisée.

Ouvre une PR intitulée "phase-12: distributed-gpu".
```
```

---

# ✅ Résumé de la livraison

| # | Artefact | Statut | Utilité immédiate |
|---|---|---|---|
| 1 | `services/ai-gateway/` complet | ✅ Fonctionnel | Débloque tout le reste |
| 2 | `infra/docker/docker-compose.yml` | ✅ Fonctionnel | Stack locale en 1 commande |
| 3 | 6 ADRs fondateurs | ✅ Complets | Gouvernance posée |
| 4 | Prompts Codex 13 phases | ✅ Prêts | Développement piloté |

## Démarrage en 3 commandes

```bash
# 1. Stack
cd infra/docker && cp .env.example .env && make up && make pull-model

# 2. Smoke test du gateway
make smoke

# 3. Tests
cd ../../services/ai-gateway && pip install -r requirements.txt && pytest
```

## Ce qui est débloqué

- **AI Gateway opérationnel** avec enforcement privacy testé
- **Stack Docker** PostgreSQL+pgvector, Redis, Qdrant, Ollama
- **6 ADRs** qui figent les décisions structurantes
- **13 prompts Codex** pour enchaîner phases 0 → 12 sans réfléchir

## Prochaine action concrète

```bash
git checkout -b phase-0/foundations
# copier les fichiers livrés
git add . && git commit -m "phase-0: foundations"
# puis prompt Codex Phase 0
