# ADR-004 — Mémoire multi-niveaux

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Une mémoire unique crée trois problèmes : mélange de faits et d'hypothèses, impossibilité de tracer une décision, impossibilité de purger sélectivement.

## Décision

Quatre mémoires distinctes :

| Type | Storage | Nature |
|---|---|---|
| SHORT_TERM | Redis | Contexte conversation, TTL |
| LONG_TERM | PostgreSQL | Faits validés humainement |
| KNOWLEDGE | pgvector / Qdrant | Documents indexés |
| DECISION | PostgreSQL | Décisions + outcomes + évaluation |

## Règles

- Une décision enregistrée est **append-only**. On ajoute `outcome` et `evaluation`, jamais on ne modifie.
- Un fait passe de SHORT_TERM à LONG_TERM uniquement par validation humaine.
- KNOWLEDGE est adressable par tenant + classification.
- DECISION contient le modèle, la version de prompt, les sources et le confidence.

## Conséquences

- Traçabilité complète des décisions.
- Datasets d'apprentissage construits à partir de DECISION évaluée.
- Purge sélective possible (ex: purger SHORT_TERM sans toucher DECISION).
