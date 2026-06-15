# API Core

Service FastAPI central pour l'architecture hedge fund.

## Démarrage local

```bash
pip install -r services/api-core/requirements.txt
uvicorn app.main:app --reload --app-dir services/api-core
```

## Bootstrap PostgreSQL

Le backend crée les tables au démarrage si `AUTO_CREATE_TABLES=true`.

Pour forcer l'initialisation manuellement :

```bash
python services/api-core/scripts/bootstrap_db.py
```

Smoke test rapide :

```bash
python services/api-core/scripts/smoke_test_api.py
```

## Interface web intégrée

Sans lancer Next.js, vous pouvez utiliser l'interface servie par FastAPI :

```text
http://127.0.0.1:8000/
```

Ou directement :

```text
http://127.0.0.1:8000/api/v1/terminal
```

Le terminal integre charge maintenant un snapshot agrege unique via `GET /api/v1/terminal/snapshot`.

Seed inclus par défaut :

- utilisateur `cio@hedgefund.local`
- mot de passe applicatif `demo`
- compte `FO-001`
- positions `AAPL`, `MSFT`, `NVDA`

## Endpoints inclus

- `GET /` redirect vers le terminal web intégré
- `GET /api/v1/terminal`
- `GET /api/v1/terminal/snapshot`
- `GET /api/v1/health`
- `GET /api/v1/ready`
- `GET /api/v1/metrics`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/mfa`
- `GET /api/v1/auth/users/me`
- `GET /api/v1/portfolio`
- `GET /api/v1/portfolio/performance`
- `GET /api/v1/portfolio/positions`
- `GET /api/v1/portfolio/barbell`
- `POST /api/v1/portfolio/barbell`
- `POST /api/v1/portfolio/rebalance`
- `POST /api/v1/orders`
- `GET /api/v1/orders`
- `DELETE /api/v1/orders/{order_id}`
- `GET /api/v1/orders/fills`
- `GET /api/v1/risk/portfolio`
- `GET /api/v1/risk/positions`
- `GET /api/v1/risk/scenarios`
- `GET /api/v1/risk/scenario/{scenario_id}`
- `GET /api/v1/risk/correlations`
- `GET /api/v1/signals/{symbol}`
- `GET /api/v1/forecast/{symbol}`
- `GET /api/v1/regime`
- `GET /api/v1/admin/events`
- `GET /api/v1/admin/events/summary`
- `POST /api/v1/admin/events/dispatch`

## Standards institutionnels

Headers propagés par le middleware:

- `X-Correlation-ID`
- `X-Tenant-ID`
- `Idempotency-Key`

Le service est actuellement en mode `transitional`: les headers sont supportés et reflétés, mais seuls les contrôles non disruptifs sont activés par défaut.

### Idempotence des ordres

- `POST /api/v1/orders` supporte désormais l'idempotence persistante via `Idempotency-Key`.
- Une même clé, pour un même `X-Tenant-ID`, rejoue la réponse initiale sans recréer d'ordre.
- Une même clé réutilisée avec une payload différente retourne `409 Conflict`.
- Le header de réponse `X-Idempotency-Status` vaut `created` ou `replayed`.

## Robustesse du refresh portefeuille

- `POST /api/v1/portfolio/refresh` ne doit plus échouer pour un simple décalage de schéma de l'outbox.
- Si la table d'outbox n'est pas encore au bon niveau, l'API passe en mode `fail-open` pour préserver l'opération métier.
- La correction durable reste l'exécution des migrations Alembic.

## Event backbone applicatif

- `api-core` publie maintenant ses événements métier critiques dans une table `event_outbox`.
- Les événements sont normalisés avec `event_name`, `topic`, `tenant_id`, `correlation_id` et `payload`.
- Les premières publications branchées couvrent :
  - `com.terminal.orders.created.v1`
  - `com.terminal.orders.cancelled.v1`
  - `com.terminal.portfolio.refreshed.v1`
- `GET /api/v1/admin/events` permet d'inspecter l'outbox récente.
- `GET /api/v1/admin/events/summary` expose l'état `pending/failed/delivered`.
- `POST /api/v1/admin/events/dispatch` permet de vider manuellement la file.
- Un bus local en mémoire existe pour brancher des handlers applicatifs avant l'arrivée d'un broker Kafka/Redpanda.
- Le dispatcher marque désormais `attempt_count`, `last_error` et `dispatched_at` pour fiabiliser les retries.
- Au démarrage et après les écritures critiques, l'application tente automatiquement un dispatch léger de l'outbox.

## Risk & Scenario Intelligence

- La plateforme expose maintenant un catalogue de stress historiques:
  - `1929` grande depression
  - `1973_oil` choc petrolier
  - `1989` leverage crack
  - `2000_tech` bulle techno
  - `2008` crise financiere
  - `2020_pandemic` pandemie
  - `2022_inflation` inflation et remontée des taux
- Chaque scenario inclut:
  - contexte macro
  - drawdown et PnL estimes
  - impacts par poche du portefeuille
  - decomposition des chocs
- `GET /api/v1/risk/correlations` fournit une matrice de correlation portefeuille exploitable par les frontends.

## Barbell Strategy

- `GET /api/v1/portfolio/barbell` expose une allocation barbell par defaut, pilotee par le regime de marche.
- `POST /api/v1/portfolio/barbell` accepte des cibles personnalisees pour la poche defensive, la poche opportuniste et le cash buffer.
- Le moteur combine :
  - regime de marche
  - signaux et forecast par actif
  - scores de qualite / volatilite
  - liquidite
  - poids courants du portefeuille
- La reponse inclut :
  - les poids cibles par poche
  - les allocations par instrument
  - une reserve `CASH`
  - les instructions de rebalance suggerees

## Institutional Terminal Snapshot

- `GET /api/v1/terminal/snapshot` agrege en une seule reponse:
  - portfolio, performance, regime et risk snapshot
  - history, signals, forecasts et stress scenarios
  - correlation matrix et position risk
  - barbell allocation
  - orders, fills, research, users, audit logs
  - outbox event summary
- Le terminal web FastAPI consomme ce snapshot pour reduire la latence applicative et le bruit reseau.
- Le terminal affiche aussi une surface de controle locale:
  - age du snapshot et de la derniere donnee de marche
  - nombre d'ordres ouverts
  - alertes outbox, concentration, correlation, VaR et fraicheur de marche
  - filtre global cote client pour retrouver rapidement symboles, ordres, secteurs et utilisateurs

## Alembic

Baseline et durcissement outbox:

```bash
alembic upgrade head
```

Si la base existait deja avant l'introduction d'Alembic et que vous voulez seulement aligner l'historique local apres verification:

```bash
alembic stamp head
```

Workflow recommande ensuite:

1. modifier les modeles SQLAlchemy
2. generer une revision Alembic
3. relire la migration
4. appliquer avec `alembic upgrade head`
