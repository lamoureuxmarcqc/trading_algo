# SELF_IMPROVEMENT.md — Auto-amélioration contrôlée

> **Principe :** le système peut proposer, tester et déployer des améliorations sur lui-même — mais jamais sans validation, jamais sans rollback, jamais sans audit.
>
> L'auto-amélioration est un **pipeline gouverné**, pas un comportement émergent.

---

## 1. Pourquoi

Un AI Operating System privé doit apprendre de son usage :

- les prompts qui fonctionnent mieux que d'autres ;
- les documents qui manquent à la base de connaissance ;
- les agents qui échouent silencieusement ;
- les features inutilisées ou mal conçues ;
- les modèles qui dérivent ;
- le code qui devient dette technique.

Sans auto-amélioration, la plateforme stagne.
Avec auto-amélioration **non contrôlée**, la plateforme devient dangereuse.

La réponse est un **pipeline d'amélioration à 7 étages**, avec validation humaine et rollback obligatoires.

---

## 2. Les 5 dimensions d'auto-amélioration

```text
┌──────────────────────────────────────────────────────────┐
│              SELF-IMPROVEMENT ENGINE                     │
│                                                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│   │  CODE    │  │   AI     │  │ KNOWLEDGE│              │
│   │ Improver │  │ Improver │  │ Improver │              │
│   └──────────┘  └──────────┘  └──────────┘              │
│                                                          │
│   ┌──────────┐  ┌──────────┐                            │
│   │ FEATURE  │  │  AGENT   │                            │
│   │ Improver │  │ Improver │                            │
│   └──────────┘  └──────────┘                            │
└──────────────────────────────────────────────────────────┘
```

| Dimension | Cible | Exemples |
|---|---|---|
| **CODE** | Repository | refactoring, tests, docs, perf, bugs |
| **AI** | Prompts, modèles, RAG | prompt tuning, fine-tuning, reranking |
| **KNOWLEDGE** | Base documentaire | découverte, contradictions, gaps |
| **FEATURE** | Produit | UX, workflows, suggestions |
| **AGENT** | Comportement | stratégies, outils, permissions |

---

## 3. Pipeline d'amélioration (7 étages)

```text
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. OBSERVE    → métriques, logs, feedback, évaluations │
│         ↓                                               │
│  2. DETECT     → anomalies, gaps, dérives, opportunités │
│         ↓                                               │
│  3. PROPOSE    → hypothèse d'amélioration + plan        │
│         ↓                                               │
│  4. SIMULATE   → sandbox, tests, évaluation offline     │
│         ↓                                               │
│  5. REVIEW     → validation humaine obligatoire         │
│         ↓                                               │
│  6. DEPLOY     → staged rollout + monitoring            │
│         ↓                                               │
│  7. EVALUATE   → métriques post-déploiement             │
│         ↓                                               │
│     ┌───┴───┐                                           │
│     │ KEEP  │  → promotion en production                │
│     │ ROLL  │  → retour arrière automatique             │
│     └───────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Règle absolue :** aucun étage ne peut être sauté. Aucun déploiement sans `REVIEW`.

---

## 4. Service `services/self-improvement/`

```text
services/self-improvement/
├── app/
│   ├── main.py
│   ├── observer.py          # collecte métriques multi-sources
│   ├── detector.py          # détection d'opportunités
│   ├── proposer.py          # génération de propositions
│   ├── simulator.py         # sandbox + tests offline
│   ├── reviewer.py          # workflow de validation humaine
│   ├── deployer.py          # staged rollout
│   ├── evaluator.py         # métriques post-déploiement
│   ├── rollback.py          # retour arrière automatique
│   └── improvers/
│       ├── code_improver.py
│       ├── ai_improver.py
│       ├── knowledge_improver.py
│       ├── feature_improver.py
│       └── agent_improver.py
├── policies/
│   ├── code_policy.yaml
│   ├── ai_policy.yaml
│   ├── knowledge_policy.yaml
│   ├── feature_policy.yaml
│   └── agent_policy.yaml
└── tests/
```

---

## 5. Improver 1 — CODE

### Objectif
Améliorer le code du repository : qualité, tests, performance, documentation.

### Détections
- couverture de tests insuffisante
- complexité cyclomatique élevée
- duplication de code
- dépendances obsolètes
- fonctions trop longues
- absence de type hints
- TODO/FIXME non résolus
- dead code
- goulots de performance (profiling)
- incohérences de style

### Actions proposées
- ajout de tests manquants
- refactoring ciblé (jamais massif)
- mise à jour de dépendances (avec tests)
- ajout de type hints
- extraction de fonctions
- remplacement de patterns obsolètes
- correction de bugs détectés par tests
- génération/mise à jour de docstrings

### Garde-fous
- **Jamais** de modification de code lié à la sécurité, aux risques, aux ordres ou à l'audit sans double validation.
- **Jamais** de modification du schéma DB sans migration Alembic.
- **Toujours** accompagné de tests.
- **Toujours** en PR séparée, reviewable.
- **Budget** : maximum N fichiers par proposition.

### Exemple de proposition

```yaml
id: prop-2026-10-10-001
type: CODE
target: services/knowledge/app/retrieval.py
issue: "Fonction `hybrid_search` dépasse 120 lignes, complexité 18"
proposal: |
  Extraire trois fonctions :
  - `_vector_search()`
  - `_keyword_search()`
  - `_fuse_results()`
impact_estimate: "Lisibilité +40%, tests +6"
test_plan: "Tests existants + 6 nouveaux tests unitaires"
risk: LOW
requires_human_review: true
auto_rollback_on: [test_failure, coverage_drop]
```

---

## 6. Improver 2 — AI

### Objectif
Améliorer les prompts, les modèles, le RAG et les agents.

### Détections
- prompts avec faible factuality
- prompts avec hallucination rate élevé
- retrieval precision faible
- reranking inefficace
- drift modèle
- confidence mal calibrée
- agents avec taux d'échec élevé
- workflows longs

### Actions proposées
- optimisation de prompts (test A/B contre baseline)
- ajustement de paramètres RAG (top_k, chunk size, overlap)
- changement de modèle d'embedding
- activation/désactivation de reranker
- fine-tuning sur decision memory (dataset versionné)
- recalibration de confidence
- modification de stratégie d'agent

### Garde-fous
- **Jamais** de promotion d'un nouveau modèle sans évaluation comparative chiffrée.
- **Toujours** contre un dataset de référence.
- **Toujours** avec versioning (Model Registry).
- **Toujours** avec rollback possible.
- **Jamais** de fine-tuning sur données non validées humainement.
- **Jamais** de modification de la politique de routage IA Gateway sans ADR.

### Exemple

```yaml
id: prop-2026-10-10-002
type: AI
target: prompt.finance.analysis.v3
issue: "Factuality 0.71 sur dataset finance-eval-v2"
proposal: |
  Remplacer prompt v3 par v4 avec :
  - instructions de citation obligatoires
  - structure FACT/SOURCE/INTERPRETATION
  - refus explicite si sources insuffisantes
baseline_metrics:
  factuality: 0.71
  hallucination_rate: 0.18
expected_metrics:
  factuality: 0.85
  hallucination_rate: 0.06
evaluation_dataset: finance-eval-v2
risk: MEDIUM
requires_human_review: true
auto_rollback_on: [factuality < 0.78, hallucination_rate > 0.12]
```

---

## 7. Improver 3 — KNOWLEDGE

### Objectif
Améliorer la base de connaissance : couverture, qualité, cohérence.

### Détections
- questions fréquentes sans réponse
- documents contradictoires
- sources de faible qualité
- entités manquantes dans le knowledge graph
- concepts orphelins
- lacunes thématiques
- documents obsolètes

### Actions proposées
- suggérer des documents à ingérer
- signaler des contradictions (sans les résoudre)
- marquer des sources comme à revoir
- proposer des liens manquants dans le graph
- suggérer des questions à poser à un expert
- signaler des documents périmés

### Garde-fous
- **Jamais** de suppression automatique de connaissance.
- **Jamais** de résolution automatique de contradiction.
- Les contradictions sont **signalées**, pas résolues.
- Toute ingestion respecte la classification.

### Exemple

```yaml
id: prop-2026-10-10-003
type: KNOWLEDGE
target: tenant:lamoureux/health
issue: "12 questions sans réponse sur le thème 'thyroïde' (30 derniers jours)"
proposal: |
  Suggérer l'ingestion de :
  - ATA Guidelines 2024
  - ETA Guidelines 2024
  - 3 revues systématiques identifiées
requires_human_review: true
classification_required: RESTRICTED
```

---

## 8. Improver 4 — FEATURE

### Objectif
Améliorer le produit : UX, workflows, fonctionnalités.

### Détections
- features jamais utilisées
- workflows abandonnés
- temps de complétion élevé
- erreurs utilisateur récurrentes
- suggestions utilisateur (feedback)
- features à forte demande non implémentées

### Actions proposées
- suggestions d'amélioration UX
- propositions de nouveaux endpoints
- propositions de nouveaux workflows
- propositions de dépréciation de features mortes
- A/B test de variantes UI

### Garde-fous
- **Jamais** de suppression de feature sans validation.
- **Jamais** de modification de l'UI qui contourne un guardrail API.
- **Toujours** avec analytics d'usage.

---

## 9. Improver 5 — AGENT

### Objectif
Améliorer les agents : stratégies, outils, permissions.

### Détections
- agents avec taux d'échec élevé
- outils sous-utilisés ou mal utilisés
- agents qui dépassent leur budget
- agents qui demandent des permissions excessives
- agents lents
- agents qui hallucinent

### Actions proposées
- ajuster les prompts d'agent
- proposer de nouveaux outils
- réduire les permissions (principe du moindre privilège)
- proposer de nouveaux agents spécialisés
- fusionner des agents redondants

### Garde-fous
- **Jamais** d'augmentation de permissions sans ADR.
- **Jamais** d'agent avec accès direct au live trading.
- **Toujours** avec sandbox et audit.

---

## 10. Workflow de validation humaine

Chaque proposition traverse :

```text
PROPOSAL
   ↓
AUTO-VALIDATION (règles)
   ↓
HUMAN REVIEW (dashboard)
   ↓
APPROVE / REJECT / MODIFY
   ↓
STAGED DEPLOYMENT
   ↓
MONITORING
   ↓
KEEP / ROLLBACK
```

### Interface de review

Nouvelle surface `/improvement` dans le cockpit Next.js :

- File d'attente des propositions
- Détails (issue, proposal, tests, risk)
- Diff prévisualisé
- Métriques avant/après
- Boutons : `Approve` / `Reject` / `Modify` / `Simulate`

---

## 11. Règles de sécurité (non négociables)

1. **Aucun auto-déploiement** sans validation humaine.
2. **Aucune modification** de sécurité, risques, ordres, audit sans double validation.
3. **Aucune suppression** de donnée ou de connaissance automatique.
4. **Aucun accès** au live trading par un improver.
5. **Aucune modification** du schéma DB sans migration Alembic.
6. **Tout** changement est réversible.
7. **Tout** changement est auditable.
8. **Tout** changement est testé avant déploiement.
9. **Tout** changement a un impact estimé.
10. **Tout** changement a un critère de rollback automatique.

---

## 12. Rate limiting

Pour éviter l'instabilité :

| Dimension | Limite |
|---|---|
| Propositions CODE / jour | 5 |
| Propositions AI / jour | 3 |
| Propositions KNOWLEDGE / jour | 10 |
| Propositions FEATURE / jour | 5 |
| Propositions AGENT / jour | 3 |
| Déploiements simultanés | 1 |
| Rollback window | 24h |

---

## 13. Circuit breakers

Le système s'arrête automatiquement si :

- 2 rollbacks consécutifs sur la même dimension
- factuality chute de plus de 15% après déploiement
- hallucination rate dépasse 15%
- latence augmente de plus de 50%
- erreurs API augmentent de plus de 20%
- audit trail échoue

---

## 14. Audit trail d'auto-amélioration

Chaque proposition génère un enregistrement immuable :

```json
{
  "id": "prop-2026-10-10-002",
  "type": "AI",
  "detected_at": "2026-10-10T08:00:00Z",
  "detected_by": "ai_improver",
  "evidence": {},
  "proposal": {},
  "simulated_at": "2026-10-10T08:15:00Z",
  "simulation_results": {},
  "reviewed_by": "marc.lamoureux",
  "reviewed_at": "2026-10-10T09:30:00Z",
  "decision": "APPROVED",
  "deployed_at": "2026-10-10T10:00:00Z",
  "rollback_criteria": {},
  "evaluation_window": "7d",
  "final_decision": "KEEP",
  "evaluated_at": "2026-10-17T10:00:00Z"
}
```

---

## 15. API contracts

```http
GET  /api/v1/improvement/proposals
GET  /api/v1/improvement/proposals/{id}
POST /api/v1/improvement/proposals/{id}/simulate
POST /api/v1/improvement/proposals/{id}/approve
POST /api/v1/improvement/proposals/{id}/reject
POST /api/v1/improvement/proposals/{id}/modify
POST /api/v1/improvement/proposals/{id}/deploy
POST /api/v1/improvement/proposals/{id}/rollback
GET  /api/v1/improvement/metrics
GET  /api/v1/improvement/audit
```

---

## 16. Intégration avec les autres services

```text
EVALUATION ─────► détecte les faiblesses
     │
     ▼
SELF-IMPROVEMENT ──► propose
     │
     ▼
MODEL REGISTRY ──► versionne
     │
     ▼
AI GATEWAY ──────► route le nouveau modèle
     │
     ▼
MEMORY ──────────► enregistre la décision d'amélioration
     │
     ▼
ORCHESTRATOR ────► coordonne les agents d'amélioration
```

---

## 17. Roadmap d'auto-amélioration

| Phase | Livrable | Prérequis |
|---|---|---|
| A | Observer + Detector | Métriques existantes |
| B | Knowledge Improver | Knowledge Layer |
| C | AI Improver (prompts) | Evaluation Framework |
| D | Code Improver (tests, refactoring) | CI/CD robuste |
| E | Feature Improver | Analytics d'usage |
| F | Agent Improver | Orchestrator stable |
| G | Model fine-tuning pipeline | Decision Memory |
| H | Circuit breakers + rate limiting | Tous les précédents |

**Ne pas commencer avant :** Evaluation Framework + Model Registry + Audit trail complets.

---

## 18. Principe directeur final

> **Un système qui s'améliore doit d'abord être capable de se comprendre.**
>
> L'auto-amélioration est un **processus gouverné**, pas une propriété émergente.
>
> Chaque amélioration est : **observée, proposée, simulée, validée, déployée, évaluée, réversible, auditée.**
>
> Sans ces sept garanties, ce n'est pas de l'auto-amélioration. C'est de la dérive.