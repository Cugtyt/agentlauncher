import asyncio
from typing import Any

from .type import EventHandler, EventType


class EventBus:
    def __init__(self, verbose: bool = False):
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
        if self._verbose:
            self.log_event(event)
        await asyncio.gather(*(handler(event) for handler in handlers))

    def verbose(self) -> None:
        self._verbose = True

    def silent(self) -> None:
        self._verbose = False

    def log_event(self, event: EventType) -> None:
        print(f"----- Event emitted: {event.__class__.__name__} -----")
        print(event)
        print("-------------------------")
