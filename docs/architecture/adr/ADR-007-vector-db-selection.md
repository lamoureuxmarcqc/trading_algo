# ADR-007: Vector Database as a Dedicated AI Retrieval Component

- Status: Proposed
- Date: 2026-05-08

## Context

AI-native workflows will need retrieval across research notes, prompts, decisions, and institutional memory. The best fit depends on scale, governance, and cloud posture.

## Decision

Reserve a dedicated vector store as a platform component, but defer final vendor selection until retrieval use cases, volume, and security constraints are validated.

## Consequences

- Keeps the AI architecture explicit without premature vendor lock-in.
- Allows evaluation of pgvector, Qdrant, Weaviate, or managed offerings.
- Requires a follow-up benchmark and governance ADR before production selection.
