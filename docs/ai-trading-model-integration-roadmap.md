# AI Trading Model Integration Roadmap

> **Purpose:** transform the existing `trading_algo` prediction layer into an explainable, probabilistic and risk-aware decision-support service that can be validated through deterministic backtesting and paper trading.

## 1. Scope

This roadmap covers the next product layer for the Hedge Fund Platform Terminal:

**Model → Explanation → Probability → Risk → Sizing → Backtest → Paper Trading**

The objective is not to make the AI an autonomous trader. The objective is to make AI outputs **measurable, explainable, testable and bounded by deterministic risk controls**.

---

## 2. Current State

### Delivered

- `POST /api/v1/terminal/trading-algo`
- Terminal FastAPI integration
- Next.js integration
- `analyze`, `compare` and `screen`
- Price and trend information
- Recommendation
- RSI
- SMA 20 / SMA 50
- Volatility
- Sharpe
- VaR
- Maximum drawdown

### Not Yet Delivered

- Model diagnostics
- SHAP/LIME explanations
- Calibrated probabilistic prediction bands
- Kelly-based sizing
- Model/data/concept drift monitoring
- Interactive backtesting
- Paper-trading order staging

> **Important:** the API contracts below are design targets. They must not be interpreted as currently implemented endpoints.

---

# 3. Product Architecture

```text
                    ┌──────────────────────┐
                    │   Market Data        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Feature Pipeline   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    AI / Quant Model  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        Diagnostics       Explanation       Prediction
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Risk & Sizing Layer  │
                    │ deterministic rules  │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
            Backtesting                Paper Trading
                 │                           │
                 ▼                           ▼
          Decision Evidence           Pending Orders
                                             │
                                             ▼
                                      Audit / Outbox
```

---

# 4. Phase 1 — Model Diagnostics

## Objective

Every AI prediction must expose enough metadata to answer:

- Which model generated it?
- When was it trained?
- When was the prediction generated?
- What data/features were used?
- How has the model performed historically?
- Is the model or market data stale?

## Target fields

| Field | Purpose |
|---|---|
| `model_version` | Reproducibility |
| `trained_at` | Model freshness |
| `inference_at` | Prediction timestamp |
| `confidence_score` | Model confidence |
| `feature_set_version` | Feature reproducibility |
| `directional_accuracy_30d` | Recent directional accuracy |
| `directional_accuracy_90d` | Medium-term directional accuracy |
| `hit_rate` | Signal hit rate |
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `signal_sharpe_30d` | Recent signal quality |
| `signal_sharpe_90d` | Medium-term signal quality |

## AI Explanation

Add an **AI Explanation** drawer/modal containing:

- Top positive features
- Top negative features
- Normalized feature contributions
- Explanation source
- Explanation timestamp

Example:

```text
Sentiment          +12%
Momentum 14D        -5%
Earnings Beat       +8%
Volatility           -3%
```

Supported explanation sources:

```text
shap
lime
heuristic_fallback
```

The fallback must be clearly labeled. It must never be presented as SHAP/LIME output.

---

# 5. Phase 2 — Probabilistic Predictions

## Objective

Move from a single target price to a probability distribution.

### Horizons

- `1d`
- `5d`
- `30d`

### Confidence intervals

- 50%
- 80%
- 95%

Each prediction should expose:

- Central/median path
- Downside path
- Upside path
- Probability of reaching a target
- Probability of reaching a stop

## Target visualization

The UI should display:

1. Historical observed price
2. Current price
3. Median projected path
4. 50% interval
5. 80% interval
6. 95% interval
7. Target marker
8. Stop marker

> Prediction intervals must be statistically defined and backtested for calibration before being used as decision inputs.

---

# 6. Phase 3 — Risk Alignment & Position Sizing

AI output must be converted into a risk-bounded recommendation rather than an unrestricted order.

## Target sizing outputs

- Raw Kelly fraction
- Capped Kelly fraction
- Volatility-adjusted size
- Recommended dollar exposure
- Recommended quantity
- Risk budget consumed
- Stop-loss
- Take-profit

## Guardrails

Recommended initial controls:

```text
max_position_weight
max_notional
max_portfolio_risk
max_daily_turnover
minimum_confidence
minimum_probability
```

### Model-quality warnings

The service should expose:

- `model_stale`
- `market_data_stale`
- `data_drift_detected`
- `concept_drift_detected`
- `regime_mismatch`

A warning must be visible before a sizing or staging decision.

---

# 7. Phase 4 — Backtesting

## Objective

Every material AI strategy should have reproducible evidence before paper trading.

## Parameters

```json
{
  "symbols": ["AAPL", "MSFT", "NVDA"],
  "lookback": "180d",
  "buy_probability_threshold": 0.75,
  "max_position_weight": 0.12,
  "stop_loss": 0.08,
  "take_profit": 0.18,
  "slippage_bps": 5
}
```

## Required outputs

- Total PnL
- Annualized return where applicable
- Maximum drawdown
- Win/loss ratio
- Trade count
- Exposure time
- Average trade
- Fees
- Slippage
- Benchmark return
- Excess return
- Optional Sharpe / Sortino

The backtest must avoid look-ahead bias and should document the treatment of:

- Corporate actions
- Missing data
- Delisted securities where relevant
- Transaction costs
- Slippage
- Signal timing
- Position sizing

---

# 8. Phase 5 — Paper Trading

Paper trading is the first stage where AI outputs become staged orders.

## Guardrails

- Paper trading only
- Minimum buy-probability threshold
- Maximum notional
- Maximum position weight
- Mandatory stop-loss
- Optional/mandatory take-profit according to policy
- `Idempotency-Key` required
- Audit event required
- Orders enter `Pending Orders`
- No direct live execution

Example:

```json
{
  "symbols": ["AAPL", "MSFT"],
  "buy_probability_threshold": 0.75,
  "max_notional": 25000,
  "stop_loss": 0.08,
  "take_profit": 0.18,
  "policy_id": "paper-ai-guardrail-v1"
}
```

---

# 9. Target API Contracts

> These endpoints are design targets and are not necessarily implemented.

## Diagnostics

```http
GET /api/v1/ai/models/{symbol}/diagnostics
```

Example:

```json
{
  "symbol": "AAPL",
  "model_version": "lstm-catboost-2026.08.11",
  "trained_at": "2026-08-11T08:00:00Z",
  "inference_at": "2026-08-11T13:35:00Z",
  "confidence_score": 0.81,
  "directional_accuracy_30d": 0.63,
  "directional_accuracy_90d": 0.58,
  "mae": 2.41,
  "rmse": 3.87,
  "signal_sharpe_30d": 1.12,
  "signal_sharpe_90d": 0.84,
  "staleness": {
    "model_stale": false,
    "market_data_stale": false,
    "data_drift_detected": false,
    "concept_drift_detected": false
  }
}
```

## Prediction Bands

```http
GET /api/v1/ai/predictions/{symbol}/bands
```

## Backtests

```http
POST /api/v1/ai/backtests
```

## Paper Trading

```http
POST /api/v1/ai/paper-trading/stage
```

---

# 10. UI Surfaces

## Next.js

### `/research`

- AI prediction console
- Diagnostics
- Explanation drawer
- Feature contributions

### `/dashboard`

- Prediction bands
- Main opportunities
- Signal quality

### `/risk`

- Staleness
- Drift
- AI exposure
- Risk budget

### `/trading`

- Paper-trading guardrails
- Pending Orders
- Audit status

### `/backtesting`

- PnL
- Drawdown
- Trades
- Benchmark
- Scenario parameters

## FastAPI Terminal

The FastAPI terminal must remain usable independently of Next.js.

Minimum AI display:

- Model status
- Confidence
- Prediction bands
- Staleness
- Risk warnings

---

# 11. Implementation Phases

| Phase | Deliverable | Exit Criteria |
|---|---|---|
| 1 | Schemas + services | Unit-tested deterministic contracts |
| 2 | Diagnostics UI | Model/freshness/performance visible |
| 3 | Prediction bands | Calibrated 50/80/95 intervals |
| 4 | Sizing + guardrails | API-enforced risk controls |
| 5 | Backtesting | Reproducible performance evidence |
| 6 | Paper trading | Staging + idempotency + audit |
| 7 | Persistence | Diagnostics/backtests stored in PostgreSQL |

---

# 12. Security & Governance

The following rules are mandatory:

1. AI starts in **paper trading only**.
2. No live order without explicit confirmation.
3. Live execution requires separate permission.
4. Risk controls are enforced server-side.
5. Staged orders require `Idempotency-Key`.
6. Critical events are auditable.
7. Staleness and confidence are always visible.
8. A frontend toggle must never bypass an API guardrail.
9. AI explanations must identify their source.
10. Backtests must be reproducible.

---

# 13. Acceptance Criteria

A phase is not considered complete until:

- API tests pass.
- Frontend build passes.
- Deterministic services work without network access.
- AI outputs expose freshness and confidence.
- Prediction intervals are statistically validated.
- Backtests expose PnL and risk metrics.
- Benchmark comparison is available.
- Paper-trading staging is disabled by default.
- Order staging is idempotent.
- Audit events are emitted.
- No live execution path is implicitly enabled.

---

## Strategic Principle

> **The platform should never confuse a model's confidence with certainty.**
>
> AI produces signals and probabilities.  
> The risk engine determines what can be done with them.  
> The audit layer records what was decided.  
> Paper trading proves behavior before any future consideration of live execution.
