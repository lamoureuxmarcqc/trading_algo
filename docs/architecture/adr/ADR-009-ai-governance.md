# ADR-009: AI Governance as a First-Class Platform Concern

- Status: Accepted
- Date: 2026-05-08

## Context

The platform intends to use AI for insights, summaries, and decision support in a regulated, high-stakes environment.

## Decision

Treat AI governance as a mandatory architectural layer covering prompt logging, human review for trading actions, explainability, hallucination monitoring, and sensitive data masking.

## Consequences

- Makes AI usage auditable and safer by design.
- Increases implementation scope beyond basic model integration.
- Requires shared policy enforcement across services, data, and UX layers.
