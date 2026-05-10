# ADR-005: FastAPI as the Python API Standard

- Status: Accepted
- Date: 2026-05-08

## Context

The current repo already exposes a Python service layer and must move quickly while preserving strong typing, OpenAPI generation, and ergonomic developer workflows.

## Decision

Use FastAPI as the default framework for Python-based APIs in the institutional platform.

## Consequences

- Accelerates API delivery with schema-first contracts and good async support.
- Fits well with Pydantic-based validation and typed service boundaries.
- Requires explicit middleware and governance layers for institutional controls.
