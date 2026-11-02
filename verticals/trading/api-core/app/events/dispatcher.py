from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import EventOutbox
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class EventSink(Protocol):
    def publish(self, event: EventOutbox) -> None:
        """Publish one event to the downstream transport."""


class LoggingEventSink:
    def publish(self, event: EventOutbox) -> None:
        logger.info(
            "Dispatched domain event",
            extra={
                "event_id": event.id,
                "event_name": event.event_name,
                "topic": event.topic,
                "tenant_id": event.tenant_id,
                "correlation_id": event.correlation_id,
            },
        )


@dataclass
class DispatchResult:
    scanned: int = 0
    delivered: int = 0
    failed: int = 0


class OutboxDispatcher:
    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        sink: EventSink | None = None,
    ) -> None:
        self._session_factory = session_factory or SessionLocal
        self._sink = sink or LoggingEventSink()

    def dispatch_pending(self, limit: int | None = None) -> DispatchResult:
        batch_size = limit or settings.outbox_dispatch_batch_size
        result = DispatchResult()

        with self._session_factory() as session:
            events = (
                session.scalars(
                    select(EventOutbox)
                    .where(EventOutbox.delivery_status.in_(("pending", "failed")))
                    .order_by(EventOutbox.created_at.asc())
                    .limit(batch_size)
                )
                .all()
            )

            for event in events:
                result.scanned += 1
                event.attempt_count += 1
                try:
                    self._sink.publish(event)
                except Exception as exc:
                    event.delivery_status = "failed"
                    event.last_error = str(exc)
                    result.failed += 1
                else:
                    event.delivery_status = "delivered"
                    event.last_error = None
                    event.dispatched_at = datetime.now(timezone.utc)
                    result.delivered += 1

            session.commit()

        return result


outbox_dispatcher = OutboxDispatcher()
