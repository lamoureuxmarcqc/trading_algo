# ADR-002: PostgreSQL with TimescaleDB for Operational and Time-Series Storage

- Status: Accepted
- Date: 2026-05-08

## Context

The platform needs transactional integrity for identities, orders, portfolios, and audit logs, while also storing dense market and signal time-series data.

## Decision

Standardize on PostgreSQL for core relational workloads and extend with TimescaleDB for time-series storage patterns where needed.

## Consequences

- Reduces cognitive load by keeping operational and time-series workloads in a coherent ecosystem.
- Preserves SQL tooling, backup, and replication patterns.
- Large-scale tick retention and analytics still require careful partitioning and lifecycle design.
