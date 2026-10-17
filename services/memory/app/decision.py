"""Decision memory : append-only, immuable."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .models import DecisionEvaluate, DecisionRecord


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class DecisionRow(Base):
    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    action: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evaluation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class DecisionImmutableError(Exception):
    """Levee quand on tente de modifier une decision."""


def record(payload: DecisionRecord) -> str:
    """Enregistre une decision (INSERT only, jamais UPDATE)."""
    return str(uuid.uuid4())
