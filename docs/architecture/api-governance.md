# API Governance

## Standards obligatoires

- Prefixe REST: `/api/v1`
- Headers institutionnels:
  - `X-Correlation-ID`
  - `Idempotency-Key`
  - `X-Tenant-ID`
- Endpoints d'operabilite:
  - `GET /health`
  - `GET /ready`
  - `GET /metrics`

## Politique de versioning

- Les breaking changes REST creent une nouvelle version (`/api/v2`).
- Les changements backward-compatible restent dans la version majeure en cours.
- Les schemas GraphQL, si introduits, doivent etre versionnes et traces.

## Politique de deprecation

- `Deprecated`: annonce au moins 90 jours avant suppression.
- `Sunset notice`: annonce au moins 30 jours avant suppression.
- Toute suppression doit inclure un chemin de migration documente.

## Etat actuel du repo

- `services/api-core` expose deja `/api/v1`.
- Les headers institutionnels sont propages par middleware en mode `transitional`.
- `POST /api/v1/orders` applique maintenant une idempotence persistante par `tenant_id + operation + key`.
- Les autres endpoints d'ecriture restent en mode transitoire pour l'idempotence metier.

## Prochaines etapes recommandees

1. Rendre `X-Tenant-ID` obligatoire au niveau gateway.
2. Etendre le stockage d'idempotence aux autres commandes critiques.
3. Standardiser les envelopes d'erreur (`code`, `message`, `correlation_id`, `details`).
