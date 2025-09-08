import asyncio
import enum
from typing import Any

from .type import EventHandler, EventType


class EventVerboseLevel(enum.IntEnum):
    SILENT = 0
    BASIC = 1
    DETAILED = 2


class EventBus:
    def __init__(self, verbose: EventVerboseLevel = EventVerboseLevel.SILENT):
        self._subscribers: dict[type[EventType], list[EventHandler[Any]]] = {}
        self._verbose = verbose

    def subscribe(
        self, event_type: type[EventType], handler: EventHandler[Any]
    ) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def emit(self, event: EventType) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        self.log_event(event)
        await asyncio.gather(*(handler(event) for handler in handlers))

    def log_event(self, event: EventType) -> None:
        if self._verbose == EventVerboseLevel.SILENT:
            return
        if self._verbose == EventVerboseLevel.BASIC:
            print(f"Event emitted: {event.__class__.__name__}")
            return

        print(f"----- Event emitted: {event.__class__.__name__} -----")
        print(event)
        print("-------------------------")
