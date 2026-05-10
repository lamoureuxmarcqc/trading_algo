# ADR-001: Kafka / Redpanda as the Event Backbone

- Status: Accepted
- Date: 2026-05-08

## Context

The target platform needs replayable market, risk, execution, and audit events across multiple services. Order, portfolio, and AI flows require high-throughput asynchronous communication and durable event retention.

## Decision

Adopt Kafka-compatible infrastructure as the strategic event backbone, with Redpanda acceptable for streamlined operations in early stages.

## Consequences

- Enables ordered, replayable, event-driven workflows.
- Fits schema registry and consumer versioning requirements better than queue-centric patterns.
- Increases operational complexity versus simpler brokers and must be paired with schema governance.
