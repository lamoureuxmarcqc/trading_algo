# Hedge Fund Platform Blueprint

> **Purpose:** define the current institutional architecture around `trading_algo` and establish the target operating model for a family-office / prop / hedge-fund style research and portfolio platform.

---

# 1. Executive Positioning

The repository is evolving from a standalone quantitative trading package into a modular institutional platform.

The architecture deliberately separates:

```text
Quant Research
      ↓
Signals / AI
      ↓
Risk Intelligence
      ↓
Portfolio Construction
      ↓
Paper Trading
      ↓
Future Execution Layer
```

The historical `trading_algo` package remains the quantitative engine. The institutional layer adds API contracts, persistence, orchestration, governance, risk and user-facing workflows.

---

# 2. Architecture at a Glance

```text
┌───────────────────────────────────────────────────────────┐
│                     Next.js Cockpit                       │
│ CIO │ Portfolio │ Risk │ Research │ Trading │ Admin      │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                       FastAPI API                         │
│ /auth │ /portfolio │ /orders │ /risk │ /signals │ /ai    │
└───────────────┬───────────────────────┬───────────────────┘
                │                       │
                ▼                       ▼
       ┌────────────────┐      ┌──────────────────┐
       │ Domain Services│      │ Trading Algo      │
       │ Risk / Portfolio│      │ Quant / ML Engine │
       └───────┬────────┘      └──────────────────┘
               │
        ┌──────┴────────┐
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│ PostgreSQL   │  │ Redis        │
│ State/Audit  │  │ Cache/Infra  │
└──────────────┘  └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ Event Outbox │
        │ + Dispatcher │
        └──────────────┘
```

---

# 3. Current Architecture

## Backend

`services/api-core` provides the institutional FastAPI layer with domains for:

- authentication
- portfolio
- trading/orders
- risk
- AI/signals

## Frontend

`apps/web` provides the Next.js cockpit for:

- `dashboard`
- `portfolio`
- `risk`
- `research`
- `trading`
- `admin`

## Infrastructure

The local stack includes:

- PostgreSQL
- Redis
- FastAPI

via:

```text
infra/docker/docker-compose.hedgefund.yml
```

## Database

Bootstrap SQL:

```text
services/api-core/sql/001_init.sql
```

Alembic provides database versioning.

---

# 4. Institutional Governance

The platform already establishes several institutional patterns.

## API Governance

- API versioning under `/api/v1`
- `X-Correlation-ID`
- `X-Tenant-ID`
- `Idempotency-Key`
- Health endpoint
- Readiness endpoint
- Metrics endpoint

## Order Safety

Order creation uses persistent idempotency controls.

This is important because retries, browser refreshes and network failures must not create duplicate orders.

## Event Governance

Critical business events use a persistent outbox.

The outbox supports:

- persistence
- delivery status
- retry
- dispatcher control
- operational visibility

## Architecture Decisions

Structural decisions should be recorded through ADRs.

Relevant documentation:

```text
docs/architecture/adr/
```

---

# 5. Frontend Operating Model

The cockpit is organized around institutional roles.

| Surface | Primary purpose |
|---|---|
| Dashboard | Executive overview |
| Portfolio | Allocation and positioning |
| Risk | Exposure, scenarios and correlations |
| Research | Quant/AI analysis |
| Trading | Orders and paper-trading workflow |
| Admin | Operational and governance controls |

## Degraded Mode

The frontend intentionally supports explicit fallback behavior when the API is unavailable.

The fallback must be visible to the user and must never be mistaken for live market state.

---

# 6. Portfolio Construction

The platform includes an initial **barbell allocation** strategy.

Conceptually:

```text
             Portfolio
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   Defensive  Opportunistic Cash
```

The strategy provides:

- defensive capital preservation
- opportunistic growth/convexity
- explicit cash reserve
- regime-aware allocation
- quantitative signal inputs

Current API surfaces:

```http
GET  /api/v1/portfolio/barbell
POST /api/v1/portfolio/barbell
```

The barbell should remain a **portfolio-construction framework**, not an implicit trading mandate.

---

# 7. Risk Intelligence

The risk domain currently includes:

- historical stress scenarios
- macro context
- portfolio impact estimates
- portfolio correlation matrix

Relevant API surfaces:

```http
GET /api/v1/risk/scenarios
GET /api/v1/risk/correlations
```

The next logical extension is to connect risk intelligence to:

- AI confidence
- model drift
- regime detection
- position sizing
- portfolio concentration
- scenario-aware limits

---

# 8. Trading Algo Integration

The first institutional integration point is:

```text
TradingAlgoTerminalService
```

Endpoint:

```http
POST /api/v1/terminal/trading-algo
```

Supported operations:

- `analyze`
- `compare`
- `screen`

The gateway calls Python modules directly rather than relying on arbitrary shell execution.

Current output includes:

- price
- trend
- recommendation
- RSI
- moving averages
- volatility
- Sharpe
- VaR
- maximum drawdown

This establishes a stable application boundary between the legacy quant engine and the institutional platform.

---

# 9. AI Integration Strategy

The AI layer is defined in:

```text
docs/ai-trading-model-integration-roadmap.md
```

Delivered V1 capabilities:

### Diagnostics

- Model version
- Training timestamp
- Inference timestamp
- Confidence
- Accuracy
- MAE/RMSE
- Signal Sharpe

### Explainability

- Explanation payloads
- Clearly labeled heuristic fallback

### Probabilistic Predictions

- 50%
- 80%
- 95%
- 1D / 5D / 30D horizons
- Prediction cones

### Risk Alignment

- Kelly-capped sizing
- Volatility-adjusted sizing
- Position limits
- Model/data/concept drift

### Backtesting

- PnL
- Drawdown
- Win/loss
- Trade count
- Benchmark comparison
- Costs and slippage

### Paper Trading

- Pending Orders
- Minimum probability
- Maximum notional
- Stop-loss
- Take-profit
- Idempotency
- Audit outbox

Remaining hardening:

- true SHAP/LIME values from model artifacts
- persistent diagnostics and backtest run history
- independent model validation logs
- institutional pre-trade compliance checks before broader paper-trading usage
- live execution, if ever required, under a separately governed permission layer

---

# 10. Separation of Concerns

| Layer | Responsibility |
|---|---|
| Market Data Feeds | Institutional feeds (Bloomberg B-PIPE, Refinitiv) with freshness monitoring |
| Feature Engineering | Real-time & historical feature transformations |
| AI/Quant Engine | Predictions, signal generation & strategy modeling |
| Explainability & Diagnostics | SHAP/LIME interpretation & model drift monitoring |
| Enterprise Risk Engine | Server-side pre-trade risk controls, VaR/SVaR limits, concentration checks |
| Compliance & Mandates | Regulatory compliance, ESG screens, restricted list filtering |
| Portfolio Engine | Multi-factor optimization, rebalancing & Pro-Forma order staging |
| OMS / FIX Gateway | FIX Protocol session management, order routing & execution reports |
| Audit & CAT Logging | Immutable event audit trail for regulatory compliance |
| UI Cockpit | Institutional cockpit (Next.js), heatmap visualizer & decision submission |

---

# 11. Data & Event Flow

A typical research-to-paper-trade workflow should become:

```text
Market Data
    ↓
Feature Generation
    ↓
Model Inference
    ↓
Diagnostics + Explanation
    ↓
Probability / Signal
    ↓
Risk Validation
    ↓
Position Sizing
    ↓
Backtest Evidence
    ↓
Paper Order
    ↓
Idempotency Check
    ↓
Pending Orders
    ↓
Audit Event
    ↓
Outbox
```

This flow is intentionally more conservative than a direct:

```text
AI → Order
```

architecture.

---

# 12. Database Versioning

Alembic is the migration mechanism for `services/api-core`.

Standard workflow:

```bash
alembic upgrade head
```

For a database that has already been manually aligned:

```bash
alembic stamp head
```

Every migration should be:

1. generated deliberately
2. reviewed
3. tested
4. reversible where practical
5. applied through the migration workflow

---

# 13. Operational Requirements

The platform should continuously expose:

- API health
- API readiness
- metrics
- market-data freshness
- order state
- outbox state
- risk alerts
- model freshness
- data drift
- degraded-mode status

Operational dashboards should distinguish:

```text
LIVE DATA
STALE DATA
FALLBACK DATA
UNAVAILABLE
```

rather than collapsing all states into a single success/failure indicator.

---

# 14. Security Principles

### Principle 1 — Least Privilege

AI/research components should not automatically inherit live trading permissions.

### Principle 2 — Server-Side Enforcement

Critical risk and trading rules must be enforced at the API boundary.

### Principle 3 — Explicit Live Permission

Any future live execution path must require a distinct permission boundary.

### Principle 4 — Idempotent Operations

Order-creation operations must remain safe under retries.

### Principle 5 — Auditability

Material trading and portfolio decisions should be reconstructable after the fact.

---

# 15. Roadmap

## Near Term

1. Persist AI diagnostics/backtest artifacts in PostgreSQL
2. Replace heuristic explanations with SHAP/LIME model artifacts
3. Calibrate prediction bands against historical forecast errors
4. Expand drift/staleness monitoring with validation logs
5. Add institutional pre-trade checks to AI-generated staging

## Medium Term

6. Persistent paper-trading policy artifacts
7. Expanded screening/model integration
8. Pro-forma blotter preview with cash and transaction-cost impact
9. Model stress testing under historical shocks

## Future

10. More sophisticated portfolio optimization
11. Multi-model ensembles
12. Regime-aware allocation
13. Institutional reporting
14. A separately governed execution layer, if ever required

---

# 16. Definition of Done

A capability is considered institutionally ready only when it has:

- A stable, versioned API contract (`/api/v1`)
- Unit and integration test coverage across all domains
- Explicit failure behavior, fallback mode, and observability (Prometheus/Grafana)
- Auditability adhering to CAT/MiFID II logging requirements
- Database migration support where state is persisted
- Pre-trade risk validation enforced at the API boundary
- Frontend handling including **Heatmap Visualizations** for cross-asset correlations and exposure
- **Pro-Forma Estimated Order Book Generation:** Before clicking `[Apply Recommended Allocation]`, the platform MUST display an estimated trade blotter (symbol, side, estimated shares, estimated dollar value, slippage estimate)
- Documented security boundaries and degraded mode

A feature that works only in the UI is not considered complete.

---

# 17. Documentation Map

| Document | Responsibility |
|---|---|
| `docs/architecture/README.md` | Architecture standards |
| `docs/architecture/platform-blueprint.md` | Detailed architecture |
| `docs/architecture/api-governance.md` | API standards |
| `docs/architecture/event-governance.md` | Events/outbox |
| `docs/architecture/adr/README.md` | Architecture decisions |
| `docs/hedgefund-platform.md` | Platform blueprint |
| `docs/ai-trading-model-integration-roadmap.md` | AI roadmap |

---

## Strategic Principle

> **Institutional quality is not created by adding more features.**
>
> It is created by making every important decision **observable, reproducible, risk-bounded, testable and auditable**.
