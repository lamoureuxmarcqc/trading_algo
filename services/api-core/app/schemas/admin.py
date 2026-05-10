from typing import Any

from pydantic import BaseModel


class AdminUser(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    mfa_enabled: bool
    is_active: bool
    created_at: str


class AuditLogEntry(BaseModel):
    id: str
    event_type: str
    entity_type: str
    entity_id: str | None = None
    actor_email: str
    details: str | None = None
    created_at: str


class DomainEventEntry(BaseModel):
    id: str
    event_name: str
    topic: str
    event_version: int
    tenant_id: str
    correlation_id: str | None = None
    aggregate_type: str
    aggregate_id: str
    delivery_status: str
    attempt_count: int
    last_error: str | None = None
    dispatched_at: str | None = None
    payload: dict[str, Any]
    created_at: str


class DomainEventDispatchResponse(BaseModel):
    scanned: int
    delivered: int
    failed: int


class DomainEventSummary(BaseModel):
    pending: int
    failed: int
    delivered: int
