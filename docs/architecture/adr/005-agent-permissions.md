# ADR-005 — Permissions des agents

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Un agent qui hérite automatiquement des permissions de l'utilisateur peut faire plus de dégâts qu'un utilisateur (exécution en masse, appels réseau, accès fichiers). Le principe du moindre privilège doit être appliqué.

## Décision

Chaque agent possède une liste explicite de permissions :

- `tools` : liste blanche d'outils autorisés
- `classification_max` : niveau maximum qu'il peut traiter
- `targets` : cibles provider autorisées
- `budget_tokens` : limite par invocation
- `timeout_seconds`
- `sandbox` : oui/non

## Règles

- Aucun agent n'a accès direct au live trading.
- Aucun agent ne peut élever sa propre classification.
- Aucun agent ne peut ajouter un outil à sa liste.
- Toute demande d'élévation passe par ADR.
- Chaque invocation est auditée (agent, outils, budget consommé, résultat).

## Conséquences

- Les agents sont plus lents à créer (config explicite) mais sûrs.
- Un agent compromis ne peut pas dépasser son périmètre.
- Les permissions sont versionnées avec l'agent.
