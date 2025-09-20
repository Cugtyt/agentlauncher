import asyncio
import enum
import logging
from collections import defaultdict
from typing import Any

from .type import EventHandler, EventType


class EventVerboseLevel(enum.IntEnum):
    SILENT = 0
    BASIC = 1
    DETAILED = 2


class EventBus:
    def __init__(self, verbose: EventVerboseLevel = EventVerboseLevel.SILENT):
        self._subscribers: dict[type[EventType], list[EventHandler[Any]]] = defaultdict(
            list
        )
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)

    def subscribe(
        self, event_type: type[EventType], handler: EventHandler[Any]
    ) -> None:
        self._subscribers[event_type].append(handler)

    async def emit(self, event: EventType) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        if self._verbose != EventVerboseLevel.SILENT:
            self.log_event(event)
        for handler in handlers:
            asyncio.create_task(handler(event))

    def log_event(self, event: EventType) -> None:
        if self._verbose == EventVerboseLevel.SILENT:
            return

        if self._verbose == EventVerboseLevel.BASIC:
            self._logger.info(
                f"[{event.agent_id}] Event emitted: {event.__class__.__name__}"
            )
            return

        self._logger.debug(f"----- Event emitted: {event.__class__.__name__} -----")
        self._logger.debug(f"{event}")
        self._logger.debug("-------------------------")
