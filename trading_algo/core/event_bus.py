from collections import defaultdict
from typing import Callable, Dict, List, Any


class EventBus:
    """
    Architecture event-driven légère (type Kafka simplifié).
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable):
        self.subscribers[event].append(handler)

    def publish(self, event: str, data: Any = None):
        for handler in self.subscribers[event]:
            handler(data)