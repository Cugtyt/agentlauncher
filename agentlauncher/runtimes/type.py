from typing import Any

from agentlauncher.events import EventBus, EventHandler, EventType


class RuntimeType:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def subscribe(
        self, event_type: type[EventType], handler: EventHandler[Any]
    ) -> None:
        self.event_bus.subscribe(event_type, handler)
