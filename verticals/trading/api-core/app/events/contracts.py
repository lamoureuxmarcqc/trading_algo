from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class DomainEvent:
    event_name: str
    topic: str
    aggregate_type: str
    aggregate_id: str
    tenant_id: str = "public"
    correlation_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    event_version: int = 1
    occurred_at: datetime = field(default_factory=utcnow)
