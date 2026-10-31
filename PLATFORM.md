# PLATFORM.md — Lamoureux AI Platform

> **Ce document est un contrat d'architecture.**
> Toute décision structurante doit être cohérente avec ce qui suit.
> Toute violation doit être documentée par un ADR.

**Version :** 1.0
**Date :** 2026-10-31
**Auteur :** Marc Lamoureux

---

## 1. Principe directeur

> **La plateforme est le socle. Les verticales métiers sont des applications qui l'utilisent.**

La valeur durable n'est pas dans une application métier. Elle est dans :

1. **la connaissance indexée** (documents, mémoire, décisions)
2. **la mémoire persistante** (4 niveaux)
3. **la privacy par classification**
4. **les agents et l'orchestration**
5. **les évaluations et la traçabilité**
6. **l'auto-amélioration gouvernée**

Un vertical métier peut être remplacé, ajouté, retiré. Le socle reste.

---

## 2. Architecture en deux couches

```text
┌──────────────────────────────────────────────────────────────────┐
│                    LAMOUREUX AI PLATFORM                          │
│                    (Infrastructure — le socle)                    │
│                                                                   │
│   AI Gateway      │  Knowledge     │  Memory      │  Orchestrator │
│   Model Registry  │  Evaluation    │  Self-Improvement            │
│   Audit Trail     │  Privacy       │  Data Classification         │
│                                                                   │
│   ← Aucune logique métier. Aucune connaissance du trading,        │
│     du droit, de la santé ou de l'administration.                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────────┐
        │                  │                  │                  │
        ▼                  ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
   │ TRADING │        │ LAWYER  │        │ DOCTOR  │        │  CRM    │
   └─────────┘        └─────────┘        └─────────┘        └─────────┘
        │                  │                  │                  │
        └──────────────────┼──────────────────┴──────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   CNESST    │
                    │   Autres…   │
                    └─────────────┘
```

---

## 3. Frontière stricte

### 3.1 Ce que fait la plateforme (socle)

| Service | Rôle |
|---|---|
| `services/ai-gateway` | Routage modèle, privacy enforcement, fallback |
| `services/knowledge` | Ingestion, chunking, embeddings, RAG, citations |
| `services/memory` | 4 niveaux : short, long, knowledge, decision |
| `services/orchestrator` | Superviseur, agents, guardrails |
| `services/model-registry` | Versioning, promotion, rollback |
| `services/evaluation` | Factualité, hallucination, calibration |
| `services/self-improvement` | Observer, proposer, simuler, déployer |
| `services/audit` (à venir) | Trail unifié, immuable |

**Règle absolue** : aucun service de la plateforme ne contient le mot « AAPL », « ordre », « portefeuille », « patient », « contrat », « dossier CNESST », etc.

### 3.2 Ce que fait un vertical

Un vertical est une **application** qui :
- a sa propre UI (Next.js)
- a sa propre API (si nécessaire)
- a ses propres agents spécialisés
- a ses propres documents indexés (avec un `tenant_id` dédié)
- utilise les services de la plateforme **sans les modifier**

### 3.3 Interdiction de contamination

- **Aucun** vertical ne peut modifier le code de `services/*`.
- **Aucun** `if vertical == "trading"` ne peut apparaître dans un service de la plateforme.
- Si un besoin métier remonte vers la plateforme, il faut **soit** :
  - le résoudre dans le vertical (préféré)
  - créer un ADR + une abstraction générique dans la plateforme

---

## 4. Verticaux cibles

### 4.1 TRADING (premier vertical — mature)

**Statut** : en production personnelle
**Tenant** : `lamoureux` (classification HIGHLY_CONFIDENTIAL pour le portefeuille)

**Composants** :
- `verticals/trading/trading_algo/` — moteur quant
- `verticals/trading/api-core/` — API institutionnelle
- `verticals/trading/web/` — cockpit Next.js
- `verticals/trading/agents/` — Investment Agent (à venir)

**Dépend du socle pour** :
- Knowledge (analyses, rapports, filings)
- Memory (decision memory pour chaque ordre)
- AI Gateway (explications, synthèses)
- Orchestrator (Investment Agent)

### 4.2 LAWYER (deuxième vertical — planifié)

**Statut** : planifié
**Tenant** : `lawyer-{cabinet}` (classification RESTRICTED pour dossiers clients)

**Fonctions** :
- Indexation de contrats, jurisprudence, dossiers clients
- Recherche avec citations
- Rédaction assistée (avec validation humaine obligatoire)
- Suivi des échéances
- Chronologie des dossiers
- Détection de contradictions

**Dépend du socle pour** :
- Knowledge (documents juridiques)
- Memory (long_term pour les dossiers, decision pour les avis)
- Orchestrator (agent de recherche juridique)
- AI Gateway (synthèses avec sources)

**Classification obligatoire** : `RESTRICTED` → local uniquement.

### 4.3 DOCTOR (troisième vertical — planifié)

**Statut** : planifié
**Tenant** : `doctor-{praticien}` (classification RESTRICTED)

**Fonctions** :
- Indexation de documents médicaux
- Recherche scientifique (PubMed, guidelines)
- Chronologie patient (isolée)
- Comparaison de résultats
- Synthèse d'articles
- Génération de questions pour le médecin

**Interdiction absolue** : jamais de diagnostic autonome. Assistant de recherche uniquement.

**Dépend du socle pour** :
- Knowledge (workspace isolé)
- Memory (RESTRICTED, jamais distant)
- Orchestrator (agent de recherche médicale)

### 4.4 CRM (quatrième vertical — planifié)

**Statut** : planifié
**Tenant** : `crm-{entreprise}` (classification CONFIDENTIAL)

**Fonctions** :
- Indexation des échanges clients (emails, notes, appels)
- Recherche dans l'historique client
- Suggestions de prochaines actions
- Résumés de compte
- Détection d'opportunités

**Dépend du socle pour** :
- Knowledge (emails, documents)
- Memory (short_term pour les conversations)
- Orchestrator (Business Intelligence Agent)

### 4.5 CNESST (cinquième vertical — pilote recommandé)

**Statut** : pilote recommandé (le plus simple pour valider l'architecture)
**Tenant** : `cnesst-{dossier}` (classification CONFIDENTIAL)

**Fonctions** :
- Indexation des documents de dossier
- Suivi des échéances
- Recherche dans les formulaires et règlements
- Génération de correspondance
- Chronologie des événements

**Pourquoi c'est le bon pilote** :
- Pas de données RESTRICTED (pas de médical)
- Pas de données financières (pas de pre-trade checks)
- RAG basique suffit
- Volume de documents gérable
- Permet de valider le modèle en 1-2 semaines

---

## 5. Structure cible du repository

```text
.
├── PLATFORM.md                       ← ce document
├── AGENTS.md                         ← instructions Codex
├── SELF_IMPROVEMENT.md               ← architecture auto-amélioration
├── README.md
│
├── services/                         ← PLATEFORME (générique)
│   ├── ai-gateway/
│   ├── knowledge/
│   ├── memory/
│   ├── orchestrator/
│   ├── model-registry/
│   ├── evaluation/
│   ├── self-improvement/
│   └── audit/                        (à venir)
│
├── verticals/                        ← VERTICALES MÉTIERS
│   ├── trading/
│   │   ├── trading_algo/
│   │   ├── api-core/
│   │   ├── web/
│   │   └── agents/
│   ├── lawyer/                       (planifié)
│   ├── doctor/                       (planifié)
│   ├── crm/                          (planifié)
│   └── cnesst/                       (pilote)
│
├── apps/                             ← UI transversales
│   └── web-platform/                 ← portail de la plateforme
│                                         (sélection de verticale, admin, agents)
│
├── docs/
│   ├── architecture/
│   │   └── adr/
│   ├── verticals/
│   │   ├── trading.md
│   │   ├── lawyer.md
│   │   ├── doctor.md
│   │   ├── crm.md
│   │   └── cnesst.md
│   ├── platform/
│   │   ├── ai-gateway.md
│   │   ├── knowledge.md
│   │   ├── memory.md
│   │   └── orchestrator.md
│   └── hedgefund-platform.md
│
├── infra/
│   └── docker/
│
└── tests/
```

---

## 6. Contrats entre plateforme et vertical

### 6.1 Ce qu'un vertical peut faire

| Action | Service | Méthode |
|---|---|---|
| Envoyer une requête LLM | AI Gateway | `POST /v1/inference` |
| Indexer un document | Knowledge | `POST /api/v1/knowledge/ingest` |
| Chercher dans la base | Knowledge | `POST /api/v1/knowledge/query` |
| Enregistrer une décision | Memory | `POST /v1/decision/record` |
| Enregistrer un fait validé | Memory | `POST /v1/long-term/validate` |
| Router vers un agent | Orchestrator | `POST /api/v1/agents/route` |
| Versionner un modèle | Model Registry | `POST /api/v1/models/register` |

### 6.2 Ce qu'un vertical ne peut pas faire

- ❌ Appeler Ollama/vLLM directement (doit passer par AI Gateway)
- ❌ Accéder à la base `knowledge` sans `tenant_id` et `classification`
- ❌ Modifier une décision enregistrée (append-only)
- ❌ Ignorer les guardrails de l'Orchestrator
- ❌ Traiter du `RESTRICTED` sans workspace isolé

---

## 7. Tenants et classifications

Chaque vertical a son propre `tenant_id` et ses classifications autorisées :

| Vertical | Tenant | Classifications |
|---|---|---|
| Trading | `lamoureux` | PUBLIC, INTERNAL, CONFIDENTIAL, HIGHLY_CONFIDENTIAL |
| Lawyer | `lawyer-{cabinet}` | INTERNAL, CONFIDENTIAL, HIGHLY_CONFIDENTIAL, RESTRICTED |
| Doctor | `doctor-{praticien}` | CONFIDENTIAL, HIGHLY_CONFIDENTIAL, RESTRICTED |
| CRM | `crm-{entreprise}` | INTERNAL, CONFIDENTIAL |
| CNESST | `cnesst-{dossier}` | INTERNAL, CONFIDENTIAL |

**Isolation stricte** : le tenant `lamoureux` (trading) ne peut **jamais** voir les documents de `doctor-{praticien}`.

---

## 8. Ce qui ne change pas entre les verticales

| Composant | Partagé |
|---|---|
| Privacy enforcement | ✅ (5 classifications, même matrice) |
| Memory layers | ✅ (4 niveaux, même schéma) |
| RAG + citations | ✅ (même pipeline) |
| AI Gateway | ✅ (même abstraction) |
| Audit trail | ✅ (même format) |
| Model Registry | ✅ (même versioning) |
| Evaluation | ✅ (mêmes métriques) |
| Self-Improvement | ✅ (même pipeline 7 étages) |

**Ce qui change** : uniquement les documents indexés, les agents spécialisés, et la logique métier.

---

## 9. Roadmap de séparation

### Phase 1 — Documentation (cette semaine)

- [x] Créer `PLATFORM.md`
- [ ] Créer `docs/verticals/{trading,lawyer,doctor,crm,cnesst}.md`
- [ ] Mettre à jour `README.md` avec la section « Architecture plateforme »

### Phase 2 — Séparation physique (1-2 semaines)

- [ ] Renommer `apps/web` → `verticals/trading/web`
- [ ] Déplacer `services/api-core` → `verticals/trading/api-core`
- [ ] Déplacer `trading_algo` → `verticals/trading/trading_algo`
- [ ] Créer `apps/web-platform` (portail de sélection)

### Phase 3 — Pilote CNESST (2-4 semaines)

- [ ] Créer `verticals/cnesst/` (structure complète)
- [ ] Indexer 100 documents de test
- [ ] Créer 3 agents spécialisés
- [ ] Valider le modèle de bout en bout

### Phase 4 — Verticaux suivants (3-6 mois)

- [ ] Lawyer (données RESTRICTED)
- [ ] CRM (données CONFIDENTIAL)
- [ ] Doctor (le plus réglementé)

---

## 10. Principe final

> **La plateforme est l'actif. Les verticales sont les revenus.**

Un vertical peut être vendu ou arrêté sans impacter la plateforme.
La plateforme peut servir un vertical sans être modifiée.

C'est cette séparation qui transforme un projet personnel en **produit technologique**.

---

## Références

- `AGENTS.md` — instructions Codex
- `SELF_IMPROVEMENT.md` — auto-amélioration gouvernée
- `docs/architecture/adr/ADR-006-trading-vertical-boundary.md` — frontière infra/vertical
- `LAMOUREUX_AI_PLATFORM.md` — vision stratégique