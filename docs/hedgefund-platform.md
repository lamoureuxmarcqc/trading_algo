# Hedge Fund Platform Blueprint

Ce dossier matérialise une base monorepo pour faire évoluer `trading_algo` vers une plateforme de type family office / prop / hedge fund.

## Gouvernance institutionnelle

Le cadre d'architecture v3.0 est maintenant explicité dans :

- [docs/architecture/README.md](/C:/Users/marc/source/projet_trading/docs/architecture/README.md)
- [docs/architecture/platform-blueprint.md](/C:/Users/marc/source/projet_trading/docs/architecture/platform-blueprint.md)
- [docs/architecture/api-governance.md](/C:/Users/marc/source/projet_trading/docs/architecture/api-governance.md)
- [docs/architecture/event-governance.md](/C:/Users/marc/source/projet_trading/docs/architecture/event-governance.md)
- [docs/architecture/adr/README.md](/C:/Users/marc/source/projet_trading/docs/architecture/adr/README.md)

## Architecture livrée

- `services/api-core`: API FastAPI modulaire avec endpoints auth, portfolio, trading, risk et AI.
- `apps/web`: cockpit frontend Next.js pour CIO / trader / risk.
- `infra/docker/docker-compose.hedgefund.yml`: stack locale PostgreSQL, Redis et API.
- `services/api-core/sql/001_init.sql`: schéma bootstrap pour auth, market data, portfolio et exécution.

## Positionnement

Le package historique `trading_algo` reste la brique quant existante.
Le nouveau socle lui ajoute une couche produit, sécurité, API et orchestration.

## Standards déjà matérialisés

- versioning API sous `/api/v1`
- propagation de `X-Correlation-ID`, `X-Tenant-ID` et `Idempotency-Key`
- endpoints d'exploitabilité `GET /health`, `GET /ready`, `GET /metrics`
- idempotence persistante sur la création d'ordres
- outbox événementiel persistant pour les événements métier critiques
- dispatcher d'outbox avec retry, statut de livraison et pilotage admin
- index ADR pour les décisions structurantes

## Cockpit frontend

Le frontend `apps/web` est maintenant branche sur `api-core` pour:

- `dashboard`
- `portfolio`
- `risk`
- `research`
- `trading`
- `admin`

Le mode degrade reste volontairement actif: si l'API est indisponible, les pages retombent sur un snapshot fallback explicite.

## Étapes suivantes

1. Brancher les endpoints du `api-core` sur PostgreSQL avec SQLAlchemy.
2. Remplacer les données de démonstration par les agrégats `trading_algo`.
3. Ajouter le realtime gateway NestJS / Socket.IO.
4. Câbler le frontend aux endpoints FastAPI.
5. Ajouter workers Celery, ingestion broker et contrôles de risque pré-trade.
