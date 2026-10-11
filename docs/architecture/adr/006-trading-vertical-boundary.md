# ADR-006 — Frontière infrastructure / vertical

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Le trading est le premier vertical mais ne doit pas contaminer l'infrastructure partagée. Si `ai-gateway` contient de la logique de portefeuille, il ne pourra plus servir la santé ou la recherche.

## Décision

Séparation stricte :

- **Infrastructure** (`ai-gateway`, `knowledge`, `memory`, `orchestrator`, `model-registry`, `evaluation`, `self-improvement`) : aucune logique métier verticale.
- **Vertical** (`trading_algo`, `services/api-core`, `apps/web`) : logique métier, workflows, UI.

## Règles

- Un service d'infrastructure ne connaît pas "AAPL", "portefeuille", "ordre".
- Un service vertical ne réimplémente pas de routage LLM, de RAG ou de mémoire.
- Toute remontée d'une logique verticale vers l'infrastructure nécessite un ADR.
- Les contrats API de l'infrastructure sont stables et versionnés.

## Conséquences

- L'infrastructure peut servir santé, recherche, PME sans modification.
- Le trading peut évoluer sans casser l'infrastructure.
- Coût : rigueur architecturale.

## Alternatives rejetées

- **Monolithe par vertical** : rejeté (duplication).
- **Infrastructure spécifique au trading** : rejeté (non réutilisable).
