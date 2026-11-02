from __future__ import annotations

from collections import defaultdict
from typing import Callable

from app.events.contracts import DomainEvent


DomainEventHandler = Callable[[DomainEvent], None]


class InProcessEventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[DomainEventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: DomainEventHandler) -> None:
        self._subscribers[event_name].append(handler)

    def publish(self, event: DomainEvent) -> None:
        for handler in self._subscribers[event.event_name]:
            handler(event)


event_bus = InProcessEventBus()
