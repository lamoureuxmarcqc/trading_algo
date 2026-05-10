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

Seed inclus par défaut :

- utilisateur `cio@hedgefund.local`
- mot de passe applicatif `demo`
- compte `FO-001`
- positions `AAPL`, `MSFT`, `NVDA`

## Endpoints inclus

- `GET /` redirect vers le terminal web intégré
- `GET /api/v1/terminal`
- `GET /api/v1/health`
- `GET /api/v1/ready`
- `GET /api/v1/metrics`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/mfa`
- `GET /api/v1/auth/users/me`
- `GET /api/v1/portfolio`
- `GET /api/v1/portfolio/performance`
- `GET /api/v1/portfolio/positions`
- `POST /api/v1/portfolio/rebalance`
- `POST /api/v1/orders`
- `GET /api/v1/orders`
- `DELETE /api/v1/orders/{order_id}`
- `GET /api/v1/orders/fills`
- `GET /api/v1/risk/portfolio`
- `GET /api/v1/risk/positions`
- `GET /api/v1/risk/scenario/{scenario_id}`
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
