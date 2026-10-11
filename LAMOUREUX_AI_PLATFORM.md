# LAMOUREUX AI PLATFORM
## Architecture stratégique, feuille de route et spécifications initiales

**Version : 2.0**
**Date : 10 octobre 2026**
**Auteur : Marc Lamoureux**

---

## 1. Vision

Construire une plateforme d'intelligence artificielle privée, modulaire et installable localement, capable de fonctionner sur un PC, un serveur local ou une infrastructure GPU distante.

La plateforme doit permettre :

- l'utilisation de modèles d'IA locaux ou distants;
- l'indexation privée de documents et données;
- une mémoire persistante et contrôlée;
- des agents spécialisés;
- un routage intelligent des tâches;
- une boucle d'apprentissage contrôlée;
- une **auto-amélioration gouvernée**;
- une architecture distribuée;
- une traçabilité complète;
- une séparation stricte des données sensibles;
- l'intégration progressive de la finance, de la recherche et de la santé.

L'objectif n'est pas de créer un simple chatbot, mais un **AI Operating System privé** pouvant devenir une plateforme technologique et commerciale.

---

# 2. Positionnement stratégique

## Nom de travail

**LAMOUREUX AI PLATFORM**

## Principe directeur

> **Build the intelligence infrastructure, not just another AI application.**

La plateforme doit être indépendante d'un fournisseur ou d'un modèle particulier.

Le modèle d'IA est interchangeable.

La valeur stratégique se trouve dans :

1. les données;
2. la mémoire;
3. les connaissances indexées;
4. l'orchestration;
5. les agents;
6. les règles métier;
7. les évaluations;
8. les workflows;
9. la sécurité;
10. l'historique des décisions.

---

# 3. Architecture cible

```text
                         ┌──────────────────────────────┐
                         │        USER INTERFACE        │
                         │ Chat / Dashboard / Reports   │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │       AI ORCHESTRATOR        │
                         │ Task Router / Agent Manager  │
                         └──────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
      ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
      │ LOCAL LLM     │         │ REMOTE LLM    │         │ SPECIALIZED   │
      │ Ollama/vLLM   │         │ Optional API  │         │ ML MODELS     │
      └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │       KNOWLEDGE LAYER        │
                         │ PostgreSQL / Vector DB / RAG │
                         │ Documents / Knowledge Graph  │
                         └──────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
      ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
      │ INVESTMENT    │         │ HEALTH        │         │ RESEARCH      │
      │ INTELLIGENCE  │         │ RESEARCH      │         │ INTELLIGENCE  │
      └───────────────┘         └───────────────┘         └───────────────┘
```

---

# 4. Principes d'architecture

## 4.1 Local First

Les données sensibles doivent rester locales par défaut.

Le système doit décider explicitement si une donnée peut être envoyée vers une infrastructure distante.

## 4.2 Model Agnostic

L'application ne doit pas dépendre d'un seul fournisseur.

Exemples :

- Llama;
- Qwen;
- Gemma;
- autres modèles open-weight;
- modèles propriétaires accessibles par API.

## 4.3 Privacy by Design

La confidentialité doit être intégrée dans l'architecture et non ajoutée après coup.

## 4.4 Human in the Loop

Les décisions importantes doivent pouvoir être validées par l'utilisateur.

## 4.5 Reproducibility

Chaque réponse importante doit pouvoir être reconstruite à partir :

- du modèle;
- de la version du prompt;
- des sources;
- des données;
- des paramètres;
- de la date.

## 4.6 Fail Safe

Une incertitude doit être affichée comme une incertitude.

Le système ne doit jamais transformer automatiquement une hypothèse en fait.

## 4.7 Governed Self-Improvement

Le système peut s'améliorer, mais uniquement via un pipeline gouverné à 7 étages, avec validation humaine, tests, rollback et audit.

---

# 5. Les cinq piliers

## PILIER 1 — Private AI

Assistant IA personnel installé localement.

Fonctions :

- chat;
- recherche documentaire;
- résumé;
- analyse;
- extraction;
- classification;
- génération de rapports;
- mémoire;
- recherche sémantique.

---

## PILIER 2 — Investment Intelligence

Intégration progressive avec **AchatVenteAction**.

Objectif :

Construire un moteur d'intelligence pour l'analyse des investissements.

### Fonctionnalités

- données de marché;
- données financières;
- analyse fondamentale;
- analyse quantitative;
- valorisation;
- scoring Buffett;
- qualité financière;
- moat;
- risque;
- croissance;
- dividendes;
- analyse des nouvelles;
- génération de thèses;
- suivi des catalyseurs;
- alertes;
- portefeuille;
- scénarios.

### Exemple de sortie

```text
NVDA

Investment Score       87/100
Quality                9/10
Moat                   9/10
Balance Sheet          9/10
FCF                    8/10
Growth                 10/10
Valuation              6/10
Risk                   6/10

Recommendation:
BUY / HOLD / SELL

Confidence:
87%

Key risks:
- Valuation
- AI capex cycle
- Competition
- Semiconductor cyclicality
```

Le système doit toujours expliquer les facteurs ayant conduit à une décision.

---

## PILIER 3 — Health Research Assistant

Le module santé doit être conçu comme un **assistant de recherche et d'organisation de l'information**, et non comme un médecin autonome.

### Fonctions

- indexation de documents médicaux;
- extraction de données;
- chronologie;
- comparaison de résultats;
- recherche scientifique;
- synthèse d'articles;
- identification de contradictions;
- génération de questions pour un professionnel;
- suivi des hypothèses;
- classification des niveaux de preuve.

### Règle fondamentale

```text
FACT
↓
SOURCE
↓
INTERPRETATION
↓
HYPOTHESIS
↓
UNCERTAINTY
```

Le système doit clairement distinguer :

- fait établi;
- observation;
- corrélation;
- hypothèse;
- opinion;
- information insuffisante.

---

## PILIER 4 — Distributed AI Infrastructure

Une fois la plateforme locale stabilisée, ajouter :

- serveurs GPU;
- workers distants;
- model routing;
- task routing;
- load balancing;
- monitoring;
- files de tâches;
- gestion des versions;
- déploiement automatisé.

Architecture :

```text
                 MASTER ORCHESTRATOR
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          PC LOCAL    GPU SERVER   CLOUD GPU
             │           │           │
             └───────────┼───────────┘
                         ▼
                    MODEL ROUTER
```

---

## PILIER 5 — Self-Improvement Layer

> **Un système qui s'améliore doit d'abord être capable de se comprendre.**

La plateforme doit pouvoir **proposer** des améliorations sur elle-même — code, IA, connaissance, features, agents — mais **jamais** les déployer sans validation humaine, tests, audit et rollback.

### Les 5 dimensions

```text
┌──────────────────────────────────────────────────────────┐
│              SELF-IMPROVEMENT ENGINE                     │
│                                                          │
│   CODE    │   AI      │ KNOWLEDGE │ FEATURE │  AGENT    │
│ Improver  │ Improver  │ Improver  │ Improver│ Improver  │
└──────────────────────────────────────────────────────────┘
```

| Dimension | Cible | Exemples |
|---|---|---|
| **CODE** | Repository | refactoring, tests, docs, perf, bugs |
| **AI** | Prompts, modèles, RAG | prompt tuning, fine-tuning, reranking |
| **KNOWLEDGE** | Base documentaire | découverte, contradictions, gaps |
| **FEATURE** | Produit | UX, workflows, suggestions |
| **AGENT** | Comportement | stratégies, outils, permissions |

### Pipeline à 7 étages

```text
1. OBSERVE   → métriques, logs, feedback
2. DETECT    → anomalies, gaps, dérives
3. PROPOSE   → hypothèse + plan
4. SIMULATE  → sandbox + tests offline
5. REVIEW    → validation humaine obligatoire
6. DEPLOY    → staged rollout + monitoring
7. EVALUATE  → KEEP ou ROLLBACK automatique
```

### Règles absolues

- Aucun auto-déploiement sans validation humaine.
- Aucune modification de sécurité, risques, ordres, audit sans double validation.
- Aucune suppression automatique de connaissance.
- Aucun accès au live trading par un improver.
- Tout changement est réversible, testé, auditable.
- Rate limiting strict.
- Circuit breakers automatiques sur dégradation.

### Intégration

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
AI GATEWAY ──────► route
     │
     ▼
MEMORY ──────────► enregistre la décision
```

Voir [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) pour le détail complet.

### Séquence d'activation

**Ne pas commencer l'auto-amélioration avant :**
1. Evaluation Framework complet
2. Model Registry opérationnel
3. Audit trail unifié
4. Circuit breakers en place

Sinon, c'est de la dérive, pas de l'amélioration.

---

# 6. Architecture logicielle

## Backend

- Python
- FastAPI
- Pydantic
- PostgreSQL + pgvector
- Redis
- Qdrant

## IA

- Ollama pour le développement initial;
- vLLM pour les déploiements GPU;
- modèles open-weight;
- embeddings locaux;
- reranking.

## RAG

- pgvector ou Qdrant;
- chunking;
- embeddings;
- retrieval;
- reranking;
- citations des sources.

## Frontend

- React;
- Next.js;
- dashboard;
- interface conversationnelle.

## Observabilité

- logs structurés;
- OpenTelemetry;
- métriques;
- traces;
- audit trail.

---

# 7. Couche de connaissance

La plateforme doit posséder plusieurs niveaux de mémoire.

```text
                    KNOWLEDGE SYSTEM
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    SQL DATA          VECTOR MEMORY       KNOWLEDGE GRAPH
        │                  │                  │
  Structured          Semantic Search      Relations
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                         RAG
                           │
                           ▼
                         LLM
```

## Types de données

### Structured

- portefeuille;
- transactions;
- entreprises;
- indicateurs;
- résultats financiers;
- paramètres.

### Unstructured

- PDF;
- rapports;
- articles;
- notes;
- documents;
- présentations.

### Semantic

- embeddings;
- concepts;
- relations;
- historique des conversations.

---

# 8. Mémoire

La mémoire doit être séparée en plusieurs catégories.

## Short-Term Memory

Contexte de la conversation courante.

## Long-Term Memory

Informations explicitement validées et utiles dans le temps.

## Knowledge Memory

Documents et connaissances indexées.

## Decision Memory

Historique des décisions et de leurs résultats.

Exemple :

```text
Decision
    ↓
Reason
    ↓
Evidence
    ↓
Action
    ↓
Outcome
    ↓
Evaluation
```

Cette dernière couche est particulièrement importante pour le système d'investissement.

---

# 9. Apprentissage contrôlé

## Ne pas commencer par l'auto-apprentissage autonome.

Architecture recommandée :

```text
DATA
  ↓
INGESTION
  ↓
RAG
  ↓
RESPONSE
  ↓
EVALUATION
  ↓
HUMAN VALIDATION
  ↓
TRAINING DATASET
  ↓
FINE-TUNING
  ↓
MODEL EVALUATION
  ↓
VERSION
  ↓
DEPLOYMENT
```

Chaque modèle doit posséder :

- version;
- dataset;
- paramètres;
- métriques;
- tests;
- date;
- provenance.

Un rollback doit être possible.

---

# 10. Agent Architecture

La plateforme peut utiliser plusieurs agents spécialisés.

```text
                     SUPERVISOR
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
   RESEARCH           FINANCE           HEALTH
    AGENT              AGENT             AGENT
       │                 │                 │
       ▼                 ▼                 ▼
   Web/Papers        Markets/Data      Medical Docs
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
                     SYNTHESIS
                         │
                         ▼
                       REPORT
```

## Agents futurs

- Coding Agent;
- Document Agent;
- Investment Agent;
- Research Agent;
- Health Research Agent;
- Business Intelligence Agent;
- Portfolio Risk Agent.

---

# 11. AI Gateway

Toutes les demandes de modèles doivent passer par une couche d'abstraction.

```text
APPLICATION
     │
     ▼
 AI GATEWAY
     │
 ┌───┼────────────┐
 ▼   ▼            ▼
LOCAL API      REMOTE API    SPECIALIZED MODEL
```

Avantages :

- changement de modèle sans modification de l'application;
- contrôle des coûts;
- contrôle de confidentialité;
- monitoring;
- fallback;
- routage intelligent.

---

# 12. Data Classification

Chaque donnée doit recevoir une classification.

```text
PUBLIC
INTERNAL
CONFIDENTIAL
HIGHLY_CONFIDENTIAL
RESTRICTED
```

Exemple :

| Donnée | Niveau |
|---|---|
| Article public | PUBLIC |
| Analyse personnelle | CONFIDENTIAL |
| Données financières personnelles | HIGHLY_CONFIDENTIAL |
| Documents médicaux | RESTRICTED |
| Secrets / credentials | RESTRICTED |

Les données RESTRICTED ne doivent jamais être transmises automatiquement à un service externe.

---

# 13. Sécurité

Principes :

- authentification;
- autorisation;
- RBAC;
- chiffrement au repos;
- chiffrement en transit;
- secrets hors du code;
- audit logs;
- segmentation réseau;
- contrôle des outils;
- validation des entrées;
- sandbox pour agents.

Chaque agent doit avoir uniquement les permissions nécessaires.

---

# 14. Audit Trail

Chaque opération importante doit être journalisée.

```text
USER
 ↓
REQUEST
 ↓
AGENT
 ↓
TOOLS
 ↓
DATA
 ↓
MODEL
 ↓
RESPONSE
```

Informations à conserver :

- utilisateur;
- timestamp;
- agent;
- modèle;
- version;
- outils utilisés;
- sources;
- résultat;
- erreurs;
- validation humaine.

---

# 15. Roadmap

## PHASE 0 — Architecture

Durée cible : 1–2 semaines.

Livrables :

- architecture;
- repository;
- Docker;
- configuration;
- API skeleton;
- PostgreSQL;
- logging;
- tests.

---

## PHASE 1 — Local AI

Durée cible : 2–4 semaines.

Construire :

- Ollama;
- modèle local;
- FastAPI;
- interface Web;
- chat;
- historique;
- authentification.

### Résultat

Un assistant IA fonctionnel entièrement local.

---

## PHASE 2 — Knowledge Engine

Durée cible : 3–5 semaines.

Construire :

- document ingestion;
- PDF extraction;
- embeddings;
- vector database;
- RAG;
- citations;
- recherche sémantique.

### Résultat

L'IA connaît les documents autorisés.

---

## PHASE 3 — Investment Intelligence

Durée cible : 4–8 semaines.

Intégrer AchatVenteAction :

- portfolio;
- financial data;
- valuation;
- quality score;
- Buffett score;
- risk engine;
- AI investment agent.

### Résultat

Une première application à forte valeur personnelle.

---

## PHASE 4 — Controlled Learning

Durée cible : 4–6 semaines.

Construire :

- evaluation framework;
- datasets;
- feedback;
- model registry;
- fine-tuning pipeline;
- regression tests;
- model comparison.

---

## PHASE 5 — Health Research

Durée cible : 4–8 semaines.

Construire :

- isolated health workspace;
- medical document ingestion;
- research search;
- evidence classification;
- timeline;
- reporting.

Le module santé doit rester strictement séparé des autres espaces de données.

---

## PHASE 6 — Distributed AI

Durée cible : 6–12 semaines.

Ajouter :

- GPU workers;
- remote inference;
- model routing;
- queues;
- scheduling;
- monitoring;
- fault tolerance.

---

## PHASE 7 — Self-Improvement

Durée cible : 6–10 semaines.

Prérequis : Phases 4, 5, 6 complètes.

Construire :

- observer + detector;
- proposer + simulator;
- reviewer (interface humaine);
- deployer + evaluator;
- rollback automatique;
- circuit breakers + rate limiting.

### Résultat

La plateforme propose ses propres améliorations, sous contrôle humain strict.

---

# 16. MVP

Le premier MVP ne doit PAS contenir tout le projet.

### MVP recommandé

```text
FastAPI
+
PostgreSQL
+
Ollama
+
Local LLM
+
Qdrant/pgvector
+
RAG
+
Web UI
+
Authentication
+
Audit Log
```

### Première démonstration

Importer :

- 10 PDF;
- quelques notes;
- quelques rapports;
- quelques documents d'investissement.

Puis poser :

> « Analyse l'ensemble de ces documents et construis une synthèse avec les sources. »

Si cette expérience fonctionne parfaitement, l'architecture possède une base solide.

---

# 17. Intégration avec AchatVenteAction

AchatVenteAction devient le premier client interne de la plateforme.

```text
LAMOUREUX AI PLATFORM
          │
          ▼
   INVESTMENT AGENT
          │
          ▼
    AchatVenteAction
          │
 ┌────────┼───────────┐
 ▼        ▼           ▼
Portfolio Valuation  Risk
```

Le système doit pouvoir produire :

- analyse d'une entreprise;
- analyse de portefeuille;
- classement des titres;
- alertes;
- scénarios;
- recommandations;
- rapports.

---

# 18. KPI du projet

## Technique

- latence;
- disponibilité;
- précision RAG;
- taux d'erreur;
- coût par requête;
- utilisation GPU;
- temps d'ingestion.

## IA

- factuality;
- hallucination rate;
- retrieval precision;
- answer relevance;
- confidence calibration.

## Investment

- performance des décisions;
- hit rate;
- drawdown;
- Sharpe;
- attribution;
- erreur de valorisation.

## Self-Improvement

- propositions générées / semaine;
- taux d'approbation humaine;
- taux de rollback;
- gain moyen par amélioration;
- stabilité post-déploiement;
- temps de cycle (detect → deploy).

## Business

- utilisateurs;
- rétention;
- coût d'infrastructure;
- revenu par utilisateur;
- marge;
- valeur du produit.

---

# 19. Principaux risques

## Hallucination

Solution :

- RAG;
- citations;
- évaluation;
- validation.

## Data leakage

Solution :

- classification;
- local-first;
- permissions;
- chiffrement.

## Model dependency

Solution :

- AI Gateway;
- model abstraction.

## Auto-learning incontrôlé

Solution :

- human-in-the-loop;
- datasets versionnés;
- tests;
- rollback.

## Auto-amélioration incontrôlée

Solution :

- pipeline gouverné à 7 étages;
- validation humaine obligatoire;
- circuit breakers;
- rate limiting.

## Complexité excessive

Solution :

> Build the smallest useful system first.

---

# 20. Stratégie commerciale potentielle

Une fois la plateforme stable, plusieurs marchés peuvent être envisagés.

### 1. Gestion de patrimoine

IA privée pour professionnels de l'investissement.

### 2. PME

Knowledge assistant privé.

### 3. Cabinets professionnels

Recherche documentaire privée.

### 4. Santé

Assistant de recherche et documentation dans des environnements contrôlés.

### 5. Entreprises

AI infrastructure privée.

### 6. Family Offices

Private AI Investment Intelligence.

---

# 21. Vision à long terme

Le système pourrait évoluer vers :

```text
                 LAMOUREUX AI
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     PERSONAL       BUSINESS       INVESTMENT
        AI             AI              AI
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                AI INFRASTRUCTURE
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           LOCAL      GPU       CLOUD
```

La plateforme devient alors une **couche d'intelligence indépendante des modèles**.

---

# 22. Décision stratégique

La séquence recommandée est :

### 1.
**Local AI**

### 2.
**Private Knowledge / RAG**

### 3.
**Investment Intelligence**

### 4.
**Controlled Learning**

### 5.
**Health Research**

### 6.
**Distributed Infrastructure**

### 7.
**Self-Improvement**

Cette séquence minimise le risque et maximise les actifs réutilisables.

---

# 23. Principe directeur final

> **Ne pas construire un chatbot.**
>
> **Construire une infrastructure d'intelligence.**

Le modèle d'IA évoluera rapidement.

La plateforme, la mémoire, les données, les workflows, les agents, les évaluations et les systèmes de décision constituent l'actif durable.

---

## Conclusion

**LAMOUREUX AI PLATFORM** doit être conçue comme une architecture évolutive permettant de passer progressivement :

**Assistant IA local → Knowledge Engine → Investment Intelligence → Health Research → Distributed AI Platform → Self-Improving Infrastructure.**

Le premier objectif n'est pas de concurrencer OpenAI, Google ou Anthropic sur leurs modèles.

Le premier objectif est de construire **la couche d'intelligence privée qui peut utiliser les meilleurs modèles disponibles tout en conservant le contrôle des données, de la mémoire et des décisions.**

C'est cette architecture qui permettra ensuite de transformer les prototypes personnels en véritable produit technologique.