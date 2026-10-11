# ADR-002 — AI Gateway comme point d'entrée unique

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Sans couche d'abstraction, chaque service appellerait directement Ollama, vLLM ou un provider distant. Cela crée :
- duplication de logique de routage
- incohérence de la politique de confidentialité
- impossibilité de changer de modèle sans toucher au code applicatif
- absence de métriques centralisées

## Décision

Tous les appels LLM passent par `services/ai-gateway`. Aucun service n'appelle un provider directement.

## Conséquences

**Positives :**
- Politique de privacy appliquée en un seul endroit
- Changement de modèle sans modification applicative
- Métriques unifiées (tokens, coût, latence)
- Fallback centralisé
- Audit unifié

**Négatives :**
- Point de défaillance unique (mitigé par réplicas)
- Latence additionnelle (<5ms)

## Règles

- Aucun service n'appelle Ollama/vLLM/remote directement.
- Toute nouvelle route LLM est ajoutée au gateway, pas ailleurs.
- Le gateway expose `/health` et `/v1/policy` pour audit.
