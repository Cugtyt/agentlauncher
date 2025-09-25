import asyncio
import logging
from collections import defaultdict
from typing import Any

from agentlauncher.shared import get_primary_agent_id

from .type import EventHandler, EventHookCallback, EventType


class EventBus:
    def __init__(self):
        self._subscribers: dict[type[EventType], list[EventHandler[Any]]] = defaultdict(
            list
        )
        self._hooks: dict[str, EventHookCallback] = {}
        self._hook_lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    def subscribe(
        self, event_type: type[EventType], handler: EventHandler[Any]
    ) -> None:
        self._subscribers[event_type].append(handler)

    async def add_hook(self, agent_id: str, hook: EventHookCallback) -> None:
        async with self._hook_lock:
            self._hooks[agent_id] = hook

    async def remove_hook(self, agent_id: str) -> None:
        async with self._hook_lock:
            self._hooks.pop(agent_id, None)

    async def emit(self, event: EventType) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        self._log_event(event)
        for handler in handlers:
            asyncio.create_task(handler(event))
        asyncio.create_task(self._invoke_hook(event))

    async def _invoke_hook(self, event: EventType) -> None:
        async with self._hook_lock:
            hook = self._hooks.get(event.agent_id, None) or self._hooks.get(
                get_primary_agent_id(event.agent_id), None
            )
        if hook is None:
            return
        try:
            await hook(event)
        except Exception:
            self._logger.exception("Hook failed for agent %s", event.agent_id)

    def _log_event(self, event: EventType) -> None:
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                "[%s] Event emitted: %s", event.agent_id, event.__class__.__name__
            )
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("Event details: %s", event)
