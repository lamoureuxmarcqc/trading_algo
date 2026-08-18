# Trading Algo — Quantitative Research & Institutional Trading Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Trading Algo** is a modular quantitative research and trading platform combining market-data extraction, quantitative analytics, machine learning and institutional portfolio/risk infrastructure.

## Executive Overview

The repository has evolved beyond a standalone trading-analysis package. It now contains two complementary layers:

1. **Quant Engine — `trading_algo`**
   - Market-data extraction and caching
   - Technical and quantitative indicators
   - Screening and comparative analysis
   - Machine-learning experimentation
   - Interactive research dashboards

2. **Institutional Platform — `services/api-core` + `apps/web`**
   - FastAPI application layer
   - Portfolio, trading, risk and AI domains
   - PostgreSQL persistence
   - Redis infrastructure
   - Next.js institutional cockpit
   - Event-driven governance and auditability
   - Risk scenarios and portfolio construction
   - Integration gateway for `trading_algo`

The strategic objective is to provide a **research → signal → risk → portfolio → paper-trading** workflow while keeping live execution explicitly separated from the AI/research layer.

---

## Current Platform Status

| Capability | Status | Notes |
|---|---|---|
| Quantitative analysis | ✅ Available | `trading_algo` engine |
| `analyze` / `compare` / `screen` | ✅ Available | Exposed through terminal/API |
| FastAPI institutional API | ✅ Available | `/api/v1` |
| Next.js cockpit | ✅ Available | CIO / portfolio / risk / research / trading / admin surfaces |
| PostgreSQL persistence | ✅ Available | Institutional schema + Alembic foundation |
| Redis infrastructure | ✅ Available | Local Docker stack |
| Correlation / stress analysis | ✅ Available | Risk domain |
| Barbell allocation | ✅ Available | API + portfolio cockpit |
| Idempotent order creation | ✅ Available | API-level control |
| Persistent outbox | ✅ Available | Event governance |
| AI diagnostics | 🟡 Roadmap | Version, confidence, accuracy, drift |
| Probabilistic prediction bands | 🟡 Roadmap | 50% / 80% / 95% |
| AI position sizing | 🟡 Roadmap | Kelly-capped + volatility-adjusted |
| Interactive backtesting | 🟡 Roadmap | PnL, drawdown, benchmark |
| Paper-trading staging | 🟡 Roadmap | Guardrails + audit |
| Live automated execution | 🔴 Not a current objective | Must remain separately permissioned |

See [`docs/hedgefund-platform.md`](docs/hedgefund-platform.md) and [`docs/ai-trading-model-integration-roadmap.md`](docs/ai-trading-model-integration-roadmap.md).

---

## Core Capabilities

### Quantitative Research

- Market-data extraction through `yfinance`
- Local caching to reduce repeated API calls
- S&P 500 screening
- Technical indicators and risk statistics
- Comparative analysis across securities
- Interactive Plotly/Dash reporting

### Machine Learning

Current experimentation includes:

- **LSTM** for time-series modelling
- **CatBoost**
- **SVR**
- **MLP**
- **Linear Regression**
- Auto-ML-style model selection through `find_best_model.py`

The AI roadmap extends this foundation with diagnostics, probabilistic predictions, explainability, risk-aware sizing and backtesting.

### Institutional Platform

The institutional layer currently provides:

- FastAPI domain services
- PostgreSQL persistence
- Redis infrastructure
- Next.js cockpit
- Health/readiness/metrics endpoints
- Correlation and tenant identifiers
- Persistent order idempotency
- Event outbox and retry mechanisms
- Risk scenarios and portfolio correlations
- Barbell portfolio allocation
- Trading Algo integration

---

## Repository Structure

```text
.
├── trading_algo/                    # Quantitative research and ML engine
│   ├── data/                        # Market-data extraction
│   ├── models/                      # ML architectures and training
│   ├── preprocessing/               # Cleaning, scaling and feature pipelines
│   ├── screening/                   # Universe screening
│   ├── visualization/               # Charts and reporting
│   ├── web_dashboard/               # Dash interface
│   ├── __init__.py
│   └── __main__.py
│
├── services/
│   └── api-core/                    # FastAPI institutional backend
│
├── apps/
│   └── web/                         # Next.js institutional cockpit
│
├── docs/
│   ├── architecture/                # Architecture, ADRs and governance
│   ├── hedgefund-platform.md        # Platform blueprint
│   └── ai-trading-model-integration-roadmap.md
│
├── infra/
│   └── docker/                      # Local infrastructure
│
├── models_saved/                    # Trained models / scalers / reports
├── dashboards/                      # Generated HTML dashboards
├── logs/                            # Runtime logs
├── pyproject.toml
└── .env                             # Local secrets/configuration — never commit
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lamoureuxmarcqc/trading_algo.git
cd trading_algo
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

```bash
python -m pip install --upgrade pip
pip install -e .
```

### 4. Configure environment variables

Create `.env` from the available example when present:

```bash
cp .env.example .env
```

> Never commit API keys, credentials or production secrets.

---

## Command-Line Usage

The primary CLI is `trading-algo`.

| Command | Purpose |
|---|---|
| `trading-algo` | Interactive menu |
| `trading-algo AAPL --advanced` | Advanced analysis |
| `trading-algo MSFT --mode train` | Train a model |
| `trading-algo AAPL,TSLA --mode compare` | Compare securities |
| `trading-algo --mode screen` | Screen the supported universe |

---

## Institutional Terminal

Start the FastAPI terminal locally:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --app-dir services/api-core
```

Open:

```text
http://127.0.0.1:8000/api/v1/terminal
```

The terminal integrates the existing quant engine through:

```http
POST /api/v1/terminal/trading-algo
```

Example:

```json
{
  "command": "compare",
  "symbols": ["AAPL", "MSFT", "NVDA"],
  "period": "1y",
  "max_symbols": 8
}
```

Supported commands:

- `analyze`
- `compare`
- `screen`

The integration currently exposes price, trend, recommendation, RSI, moving averages, volatility, Sharpe, VaR and maximum drawdown.

---

## AI Roadmap

The next major product layer is documented in [`docs/ai-trading-model-integration-roadmap.md`](docs/ai-trading-model-integration-roadmap.md).

Planned capabilities include:

- Model versioning and confidence
- Historical directional accuracy
- MAE / RMSE
- Signal Sharpe
- SHAP/LIME explanations
- Probabilistic prediction bands
- Kelly-capped and volatility-adjusted sizing
- Model/data/concept drift detection
- Interactive backtesting
- Paper-trading order staging
- Idempotency and audit controls

The design principle is:

> **AI generates research and decision support; risk controls remain deterministic and enforceable at the API boundary.**

---

## Web Dashboard

The repository also includes the existing Dash research interface.

```bash
python start.py
```

or:

```bash
python -m trading_algo.web_dashboard.app
```

Typical local address:

```text
http://127.0.0.1:8050
```

Useful environment variables:

- `DEBUG`
- `WEB_HOST`
- `WEB_PORT`
- `REDIS_URL`
- `REDIS_URI`

### Troubleshooting

If dashboard tabs fail to load:

1. Inspect startup logs for import failures.
2. Check heavy dependencies such as TensorFlow.
3. Verify `assets/` and `assets/custom.css`.
4. Use browser DevTools → Network.
5. Inspect `/_dash-layout` and `/_dash-update-component` responses.

---

## Database Migrations

The API layer uses Alembic.

Upgrade:

```bash
alembic upgrade head
```

If an existing database is already manually aligned:

```bash
alembic stamp head
```

Recommended workflow:

1. Modify SQLAlchemy models.
2. Generate a migration.
3. Review the generated migration.
4. Test it against a representative database.
5. Apply with `alembic upgrade head`.

---

## Testing

Run the Python test suite:

```bash
python -m unittest discover tests
```

Run API integration tests:

```powershell
.\.venv\Scripts\python.exe -m pytest services\api-core\tests\test_trading_algo_terminal.py -q
```

Before considering a feature complete, validate:

- Unit tests
- API contract tests
- Database migration behavior
- Frontend build
- Fallback/degraded mode
- Risk and order guardrails

---

## Operational Principles

### 1. Research ≠ Execution

Research models must not directly obtain unrestricted live-order authority.

### 2. Risk Controls Are Server-Side

Critical controls must be enforced by the API, not only by the frontend.

### 3. Auditability Is a First-Class Requirement

Order staging, critical decisions and business events should be traceable.

### 4. Safe Degradation

The frontend may fall back to an explicit snapshot when the API is unavailable. The UI must clearly identify degraded data.

### 5. No False Precision

AI predictions must expose confidence and data/model freshness rather than presenting point forecasts as certainty.

---

## Documentation Map

| Document | Purpose |
|---|---|
| [`docs/hedgefund-platform.md`](docs/hedgefund-platform.md) | Institutional platform architecture and current capabilities |
| [`docs/ai-trading-model-integration-roadmap.md`](docs/ai-trading-model-integration-roadmap.md) | AI integration roadmap and acceptance criteria |
| [`docs/architecture/README.md`](docs/architecture/README.md) | Architecture standards |
| [`docs/architecture/api-governance.md`](docs/architecture/api-governance.md) | API governance |
| [`docs/architecture/event-governance.md`](docs/architecture/event-governance.md) | Event/outbox governance |
| [`docs/architecture/adr/README.md`](docs/architecture/adr/README.md) | Architecture Decision Records |

---

## Outputs and Artifacts

- `models_saved/` — trained models, scalers and performance reports
- `dashboards/` — interactive HTML dashboards
- `logs/` — operational and error logs

---

## Contribution

1. Create a feature branch.
2. Keep changes focused and reviewable.
3. Follow PEP 8 for Python.
4. Add tests for behavioral changes.
5. Update relevant documentation.
6. Open a Pull Request.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).

---

**Developed by Marc Lamoureux**
