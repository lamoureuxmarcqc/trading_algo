# ADR-008: Event Sourcing for OMS Lifecycle History

- Status: Proposed
- Date: 2026-05-08

## Context

OMS workflows benefit from immutable auditability, replayable lifecycle transitions, and historical reconstruction across submission, routing, fills, and cancellations.

## Decision

Adopt event sourcing principles for the OMS domain, with an append-only event log as the source of truth and derived read models for operational queries.

## Consequences

- Improves auditability and replay for order lifecycle reconstruction.
- Raises implementation complexity for projections, consistency, and migrations.
- Should be introduced first in OMS only, not forced on every bounded context.
