# ADR-006: Redis for Shared Caching and Short-Lived Coordination

- Status: Accepted
- Date: 2026-05-08

## Context

The platform needs low-latency caching, session support, and a place for transient coordination primitives without overloading the primary database.

## Decision

Use Redis for cache acceleration, ephemeral state, and namespace-based tenant isolation.

## Consequences

- Improves latency for read-heavy market and portfolio views.
- Supports session and short-lived token patterns.
- Requires careful key design, expiration policies, and failure-mode testing.
