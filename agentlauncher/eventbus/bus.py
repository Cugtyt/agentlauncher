import asyncio
import logging
from collections import defaultdict
from typing import Any

from .type import EventHandler, EventType


class EventBus:
    def __init__(self):
        self._subscribers: dict[type[EventType], list[EventHandler[Any]]] = defaultdict(
            list
        )
        self._logger = logging.getLogger(__name__)

    def subscribe(
        self, event_type: type[EventType], handler: EventHandler[Any]
    ) -> None:
        self._subscribers[event_type].append(handler)

    async def emit(self, event: EventType) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        self._log_event(event)
        for handler in handlers:
            asyncio.create_task(handler(event))

    def _log_event(self, event: EventType) -> None:
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                "[%s] Event emitted: %s", event.agent_id, event.__class__.__name__
            )
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("Event details: %s", event)
