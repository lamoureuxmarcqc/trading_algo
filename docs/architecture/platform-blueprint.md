# Platform Blueprint

## Positionnement

Le repo combine deux couches:

- `trading_algo`: moteurs quant, analytics, dashboards historiques
- couche institutionnelle: `apps/`, `services/`, `infra/`, `docs/`

## Macro-architecture cible

```text
Frontend (Next.js)
        |
API Gateway
        |
Microservices Layer
Auth | Portfolio | Market Data | Pricing | Risk | OMS | Execution | AI | Notification
        |
Kafka / Redpanda
        |
PostgreSQL | TimescaleDB | Redis | Vector DB | S3/MinIO
        |
Python | Rust | Ray | GPU Inference
```

## Bounded contexts cibles

- Identity & Security
- Portfolio Management
- Market Data
- Pricing
- Risk
- OMS
- Execution
- AI Intelligence
- Quant Research
- Notifications

## Priorites de mise en oeuvre

1. Gouvernance d'architecture et standards communs
2. API core observable et securisable
3. Separation des contextes metier en services explicites
4. Event-driven backbone
5. hardening institutionnel: DR, chaos, audit, AI governance
