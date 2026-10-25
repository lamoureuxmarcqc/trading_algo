# Trading Algo — Quantitative Research & Institutional Trading Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Trading Algo** is the institutional trading vertical of the **LAMOUREUX AI PLATFORM** — a private, modular, model-agnostic AI infrastructure combining market-data extraction, quantitative analytics, machine learning and institutional portfolio/risk infrastructure.

## Executive Overview

The repository is organized as a **two-layer platform**:

1. **AI Infrastructure Layer — `services/`**
   - `ai-gateway` — model-agnostic routing with privacy enforcement
   - `knowledge` — RAG, embeddings, citations, knowledge graph
   - `memory` — short-term, long-term, knowledge, decision memory
   - `orchestrator` — supervisor + specialized agents
   - `model-registry` — versioning, rollback, comparison
   - `evaluation` — factuality, hallucination, calibration
   - `self-improvement` — controlled self-improvement engine

2. **Institutional Trading Layer — `trading_algo` + `services/api-core` + `apps/web`**
   - Market-data extraction and caching
   - Technical and quantitative indicators
   - Screening and comparative analysis
   - Machine-learning experimentation
   - FastAPI application layer
   - Portfolio, trading, risk and AI domains
   - PostgreSQL persistence, Redis infrastructure
   - Next.js institutional cockpit
   - Event-driven governance and auditability

The strategic principle:

> **The trading platform is the first vertical agent of the AI infrastructure — not the other way around.**
>
> The platform builds the intelligence infrastructure, not just another AI application.

See [`LAMOUREUX_AI_PLATFORM.md`](LAMOUREUX_AI_PLATFORM.md), [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md), [`docs/hedgefund-platform.md`](docs/hedgefund-platform.md) and [`docs/ai-trading-model-integration-roadmap.md`](docs/ai-trading-model-integration-roadmap.md).

---

## Self-Improvement Engine

The platform is designed to improve itself — under strict governance.

**Five improvers, one pipeline, seven stages:**

```text
CODE │ AI │ KNOWLEDGE │ FEATURE │ AGENT
              ↓
OBSERVE → DETECT → PROPOSE → SIMULATE → REVIEW → DEPLOY → EVALUATE
                                                    ↓
                                              KEEP / ROLLBACK
```

**Non-negotiable rules:**

- No self-deployment without human review.
- No modification of security, risk, orders or audit without double validation.
- No automatic knowledge deletion.
- No live trading access for improvers.
- Every change is reversible, tested, auditable.
- Rate limiting and circuit breakers enforced.

See [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) for the full architecture.

---

## Current Platform Status

### AI Infrastructure Layer

| Capability | Status | Notes |
|---|---|---|
| AI Gateway (model abstraction) | 🟡 Scaffolding | Privacy-enforced routing |
| Knowledge Layer (RAG) | 🟡 Phases 2c–2d livrées, 2e–2f à venir | pgvector + hybrid + reranker local opt-in + citations |
| Memory Layer (4 niveaux) | 🟡 Scaffolding | Short / long / knowledge / decision |
| Agent Orchestrator | 🟡 Scaffolding | Supervisor + vertical agents |
| Model Registry | 🔴 Planned | Versioning + rollback |
| Evaluation Framework | 🔴 Planned | Factuality + hallucination |
| Self-Improvement Engine | 🔴 Planned | 5 improvers + 7-stage pipeline |
| Local LLM Runtime (Ollama) | 🟡 Scaffolding | Local-first by default |
| GPU Inference (vLLM) | 🔴 Planned | Phase 6 — distributed |
| Data Classification Engine | 🟡 Scaffolding | 5 niveaux + enforcement |
| Audit Trail unifié | ✅ Available | Outbox + events |

### Institutional Trading Layer

| Capability | Status | Notes |
|---|---|---|
| Quantitative analysis | ✅ Available | `trading_algo` engine |
| `analyze` / `compare` / `screen` | ✅ Available | Exposed through terminal/API |
| FastAPI institutional API | ✅ Available | `/api/v1` |
| Next.js cockpit | ✅ Available | CIO / portfolio / risk / research / trading / admin |
| PostgreSQL persistence | ✅ Available | Institutional schema + Alembic |
| Redis infrastructure | ✅ Available | Local Docker stack |
| Correlation / stress analysis | ✅ Available | Risk domain |
| Barbell allocation | ✅ Available | API + portfolio cockpit |
| Idempotent order creation | ✅ Available | API-level control |
| Persistent outbox | ✅ Available | Event governance |
| AI diagnostics | ✅ Available | Version, confidence, accuracy, staleness, drift |
| Probabilistic prediction bands | ✅ Available | 50% / 80% / 95%, 1D / 5D / 30D |
| AI position sizing | ✅ Available | Kelly-capped + volatility-adjusted |
| Interactive backtesting | ✅ Available | PnL, drawdown, benchmark, costs |
| Paper-trading staging | ✅ Available | Guardrails + idempotence + audit |
| Live automated execution | 🔴 Not a current objective | Separately permissioned |

---

## Repository Structure

```text
.
├── AGENTS.md                        # Codex instructions
├── SELF_IMPROVEMENT.md              # Self-improvement architecture
├── LAMOUREUX_AI_PLATFORM.md         # Strategic AI platform blueprint
├── README.md
├── trading_algo/                    # Quantitative research and ML engine
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── screening/
│   ├── visualization/
│   ├── web_dashboard/
│   ├── __init__.py
│   └── __main__.py
│
├── services/
│   ├── api-core/                    # FastAPI institutional backend
│   ├── ai-gateway/                  # Model-agnostic routing + privacy
│   ├── knowledge/                   # RAG + embeddings + citations
│   ├── memory/                      # Multi-layer memory
│   ├── orchestrator/                # Agent supervisor
│   ├── model-registry/              # Model versioning
│   ├── evaluation/                  # Factuality + hallucination
│   └── self-improvement/            # Controlled self-improvement
│
├── apps/
│   └── web/                         # Next.js institutional cockpit
│
├── docs/
│   ├── architecture/
│   │   └── adr/
│   ├── hedgefund-platform.md
│   └── ai-trading-model-integration-roadmap.md
│
├── infra/
│   ├── docker/
│   ├── ollama/
│   ├── vllm/
│   └── monitoring/
│
├── models_saved/
├── dashboards/
├── logs/
├── pyproject.toml
└── .env
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
.\venv\Scripts\Activate.ps1
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

```bash
cp .env.example .env
```

> Never commit API keys, credentials or production secrets.

### 5. Start the local AI infrastructure

```bash
docker compose -f infra/docker/docker-compose.yml up -d
```

### 6. Start the local AI infrastructure

```
.\run.ps1 up            # Démarrer la stack
.\run.ps1 warmup        # Précharger Ollama (753 ms → 2.4s)
.\run.ps1 smoke         # Vérifier que le LLM répond
.\run.ps1 test          # Tester les 44 tests
.\run.ps1 health        # Statut gateway
.\run.ps1 policy        # Matrice privacy
.\run.ps1 down          # Arrêter la stack
```
---

## AI Gateway

All model calls go through a single abstraction layer:

```python
from ai_gateway import AIGateway, DataClassification

gw = AIGateway()
response = await gw.inference(
    prompt="Analyse ce document...",
    classification=DataClassification.HIGHLY_CONFIDENTIAL,
    tenant_id="lamoureux",
)
```

Routing rules:

| Classification | Allowed targets |
|---|---|
| `PUBLIC` | LOCAL, LOCAL_GPU, REMOTE |
| `INTERNAL` | LOCAL, LOCAL_GPU, REMOTE |
| `CONFIDENTIAL` | LOCAL, LOCAL_GPU |
| `HIGHLY_CONFIDENTIAL` | LOCAL, LOCAL_GPU |
| `RESTRICTED` | LOCAL only |

The gateway enforces these rules **before** any network call.

---

## Knowledge Layer (RAG)

Ingest documents, then query with citations:

```http
POST /api/v1/knowledge/ingest
{
  "file_path": "reports/NVDA_Q3.pdf",
  "tenant_id": "lamoureux",
  "classification": "CONFIDENTIAL"
}
```

```http
POST /api/v1/knowledge/query
{
  "query": "Quels sont les risques principaux de NVDA?",
  "tenant_id": "lamoureux",
  "top_k": 5
}
```

Response includes `sources` with `document_id`, `page`, `excerpt`, and a `confidence` score.

---

## Memory Layer

Four memory types, strictly separated:

| Type | Storage | Purpose |
|---|---|---|
| `SHORT_TERM` | Redis | Conversation context |
| `LONG_TERM` | PostgreSQL | Validated facts |
| `KNOWLEDGE` | pgvector / Qdrant | Indexed documents |
| `DECISION` | PostgreSQL | Decisions + outcomes + evaluation |

Decision memory is immutable — a recorded decision can only be evaluated, never modified.

---

## Self-Improvement

```http
GET  /api/v1/improvement/proposals
POST /api/v1/improvement/proposals/{id}/simulate
POST /api/v1/improvement/proposals/{id}/approve
POST /api/v1/improvement/proposals/{id}/reject
POST /api/v1/improvement/proposals/{id}/deploy
POST /api/v1/improvement/proposals/{id}/rollback
```

Every improvement passes through: OBSERVE → DETECT → PROPOSE → SIMULATE → REVIEW → DEPLOY → EVALUATE.

No auto-deployment. No exceptions.

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
- `simulate`

---

## AI Roadmap

The next major product layer is documented in [`docs/ai-trading-model-integration-roadmap.md`](docs/ai-trading-model-integration-roadmap.md).

The design principle:

> **AI generates research and decision support; risk controls remain deterministic and enforceable at the API boundary.**

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
- AI privacy enforcement (classification → routing)

---

## Operational Principles

### 1. Local First
Sensitive data stays local by default. Any remote call requires an explicit classification decision.

### 2. Model Agnostic
No dependency on a single provider. The AI Gateway abstracts local (Ollama, vLLM) and remote providers.

### 3. Research ≠ Execution
Research models must not obtain unrestricted live-order authority.

### 4. Risk Controls Are Server-Side
Critical controls are enforced by the API, not by the frontend.

### 5. Auditability Is a First-Class Requirement
Order staging, critical decisions and business events are traceable.

### 6. Safe Degradation
The frontend may fall back to an explicit snapshot when the API is unavailable. Degraded data must be labeled.

### 7. No False Precision
AI predictions expose confidence, freshness and source. No point forecast is presented as certainty.

### 8. Human in the Loop
High-classification decisions and all improvements require explicit validation.

### 9. Self-Improvement Is Governed
Every improvement is observed, proposed, simulated, validated, deployed, evaluated, reversible and audited.

---

## Documentation Map

| Document | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Codex instructions |
| [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) | Self-improvement architecture |
| [`docs/LAMOUREUX_AI_PLATFORM.md`](LAMOUREUX_AI_PLATFORM.md) | Strategic AI platform blueprint |
| [`docs/hedgefund-platform.md`](docs/hedgefund-platform.md) | Institutional platform architecture |
| [`docs/ai-trading-model-integration-roadmap.md`](docs/ai-trading-model-integration-roadmap.md) | AI integration roadmap |
| [`docs/architecture/README.md`](docs/architecture/README.md) | Architecture standards |
| [`docs/architecture/api-governance.md`](docs/architecture/api-governance.md) | API governance |
| [`docs/architecture/event-governance.md`](docs/architecture/event-governance.md) | Event/outbox governance |
| [`docs/architecture/adr/README.md`](docs/architecture/adr/README.md) | Architecture Decision Records |

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
