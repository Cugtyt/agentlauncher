import asyncio
from typing import cast

from agentlauncher.events import (
    EventBus,
    LLMRequestEvent,
    LLMResponseEvent,
    LLMRuntimeErrorEvent,
)
from agentlauncher.llm_interface import (
    LLMHandler,
)
from agentlauncher.llm_interface.message import (
    AssistantMessage,
    ResponseMessageList,
)

from .shared import is_primary_agent
from .type import RuntimeType


class LLMRuntime(RuntimeType):
    def __init__(
        self,
        event_bus: EventBus,
    ):
        super().__init__(event_bus)
        self._primary_agent_llm_handler: LLMHandler | None = None
        self._sub_agent_llm_handler: LLMHandler | None = None
        self.event_bus.subscribe(LLMRequestEvent, self.handle_llm_request)
        self.event_bus.subscribe(LLMRuntimeErrorEvent, self.handle_llm_runtime_error)

    def set_primary_agent_handler(self, handler: LLMHandler) -> None:
        self._primary_agent_llm_handler = handler

    def set_sub_agent_handler(self, handler: LLMHandler) -> None:
        self._sub_agent_llm_handler = handler

    async def handle_llm_request(self, event: LLMRequestEvent) -> None:
        handler = (
            self._primary_agent_llm_handler
            if is_primary_agent(event.agent_id) or not self._sub_agent_llm_handler
            else self._sub_agent_llm_handler
        )
        if not handler:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="LLM handler not found.",
                    request_event=event,
                )
            )
            return
        try:
            if asyncio.iscoroutinefunction(handler):
                response = await handler(
                    event.messages, event.tool_schemas, event.agent_id, self.event_bus
                )
            else:
                response = await asyncio.to_thread(
                    handler,
                    event.messages,
                    event.tool_schemas,
                    event.agent_id,
                    self.event_bus,
                )
            await self.event_bus.emit(
                LLMResponseEvent(
                    agent_id=event.agent_id,
                    request_event=event,
                    response=cast(ResponseMessageList, response),
                )
            )
        except Exception as e:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(
                    agent_id=event.agent_id, error=str(e), request_event=event
                )
            )

    async def handle_llm_runtime_error(self, event: LLMRuntimeErrorEvent) -> None:
        if event.request_event.retry_count < 5:
            await self.event_bus.emit(
                LLMRequestEvent(
                    agent_id=event.request_event.agent_id,
                    messages=event.request_event.messages,
                    tool_schemas=event.request_event.tool_schemas,
                    retry_count=event.request_event.retry_count + 1,
                )
            )
        else:
            await self.event_bus.emit(
                LLMResponseEvent(
                    agent_id=event.request_event.agent_id,
                    request_event=event.request_event,
                    response=[
                        AssistantMessage(content="Runtime error: " + event.error)
                    ],
                )
            )
