# ADR-010: Multi-Region Disaster Recovery Strategy

- Status: Proposed
- Date: 2026-05-08

## Context

The target operating model requires strong availability, low recovery times, and continuity for critical investment workflows.

## Decision

Design toward multi-AZ primary resilience immediately and a secondary region disaster recovery posture as the institutional hardening target.

## Consequences

- Aligns the platform with explicit RPO and RTO expectations.
- Avoids premature full multi-region complexity for early-stage environments.
- Requires periodic failover exercises, backup validation, and service dependency inventories.
