import asyncio
import enum
from datetime import datetime
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

    def emit(self, event: EventType) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        if self._verbose != EventVerboseLevel.SILENT:
            asyncio.create_task(asyncio.to_thread(self.log_event, event))
        for handler in handlers:
            asyncio.ensure_future(handler(event))

    def log_event(self, event: EventType) -> None:
        if self._verbose == EventVerboseLevel.SILENT:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if self._verbose == EventVerboseLevel.BASIC:
            agent_id = (
                f" (agent_id: {getattr(event, 'agent_id', 'N/A')})"
                if hasattr(event, "agent_id")
                else ""
            )
            print(f"[{timestamp}] Event emitted: {event.__class__.__name__}{agent_id}")
            return

        print(f"----- [{timestamp}] Event emitted: {event.__class__.__name__} -----")
        print(event)
        print("-------------------------")
