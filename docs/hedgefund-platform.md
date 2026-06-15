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
- strategie d'allocation `barbell` exploitable par API et cockpit

## Cockpit frontend

Le frontend `apps/web` est maintenant branche sur `api-core` pour:

- `dashboard`
- `portfolio`
- `risk`
- `research`
- `trading`
- `admin`

Le mode degrade reste volontairement actif: si l'API est indisponible, les pages retombent sur un snapshot fallback explicite.

La page `portfolio` expose maintenant une lecture institutionnelle de la strategie `barbell`:

- split defensif / opportuniste / cash
- poids cibles et poids courants
- instructions de rebalance suggerees
- justification de posture selon le regime de marche

Le terminal web servi directement par FastAPI a aussi ete durci:

- chargement agrege via un snapshot unique
- reduction du fan-out HTTP au chargement
- exposition native de la posture barbell
- visibilite immediate sur l'etat de livraison de l'outbox evenementielle
- surface de controle du snapshot: fraicheur marche, ordres ouverts, alertes risque/outbox
- filtre global cote client pour explorer rapidement les tableaux et scenarios sans nouvel appel API

## Risk Intelligence

Le domaine risque couvre maintenant:

- un catalogue de stress historiques multi-decennies
- contexte macro: inflation, chomage, taux et informations de crise
- impact estime sur le portefeuille courant
- matrice de correlation portefeuille
- exposition via `GET /api/v1/risk/scenarios` et `GET /api/v1/risk/correlations`

## Portfolio Construction

Le socle `api-core` embarque maintenant une implementation initiale de la strategie barbell:

- poche defensive pour la preservation du capital
- poche opportuniste pour la convexite et la croissance
- reserve de cash explicite pour absorber les chocs et financer le redeploiement
- allocation pilotee par le regime de marche et les signaux quantitatifs
- exposition via `GET /api/v1/portfolio/barbell` et `POST /api/v1/portfolio/barbell`

## Database Versioning

Alembic est maintenant prepare pour versionner `services/api-core`:

- metadata SQLAlchemy branchee dans `alembic/env.py`
- premiere revision de durcissement outbox/idempotence
- workflow cible: `alembic upgrade head` pour mettre les bases a niveau

## Étapes suivantes

1. Brancher les endpoints du `api-core` sur PostgreSQL avec SQLAlchemy.
2. Remplacer les données de démonstration par les agrégats `trading_algo`.
3. Ajouter le realtime gateway NestJS / Socket.IO.
4. Câbler le frontend aux endpoints FastAPI.
5. Ajouter workers Celery, ingestion broker et contrôles de risque pré-trade.
