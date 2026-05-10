# ADR-004: Kubernetes as the Deployment Substrate

- Status: Accepted
- Date: 2026-05-08

## Context

Institutional workloads require controlled rollouts, autoscaling, isolation, and disaster recovery readiness across multiple services and compute profiles.

## Decision

Adopt Kubernetes as the standard runtime for deployable services, jobs, and supporting infrastructure integrations.

## Consequences

- Aligns with immutable infrastructure and GitOps-ready delivery.
- Supports autoscaling and multi-environment parity.
- Introduces platform engineering overhead that must be justified by service maturity.
