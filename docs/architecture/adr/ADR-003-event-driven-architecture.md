# ADR-003: Event-Driven Architecture as a Core Integration Style

- Status: Accepted
- Date: 2026-05-08

## Context

The platform spans bounded contexts with different latency, coupling, and scaling profiles. Synchronous APIs alone would create tight dependencies between research, risk, OMS, and AI functions.

## Decision

Use event-driven integration for cross-service state propagation, notifications, workflow orchestration, and audit trails, while retaining synchronous APIs for command and query entry points.

## Consequences

- Improves decoupling and horizontal scalability.
- Requires stronger schema governance, observability, and replay discipline.
- Demands idempotent consumers and explicit failure-handling strategies.
