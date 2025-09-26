import asyncio
from typing import cast

from agentlauncher.eventbus import EventBus, EventContext
from agentlauncher.events import (
    LLMRequestEvent,
    LLMResponseEvent,
    LLMRuntimeErrorEvent,
)
from agentlauncher.llm_interface import LLMProcessor
from agentlauncher.llm_interface.message import (
    AssistantMessage,
    ResponseMessageList,
)
from agentlauncher.shared import is_primary_agent

from .type import RuntimeType


class LLMRuntime(RuntimeType):
    def __init__(
        self,
        event_bus: EventBus,
    ):
        super().__init__(event_bus)
        self._primary_agent_llm_processor: LLMProcessor | None = None
        self._sub_agent_llm_processor: LLMProcessor | None = None
        self.event_bus.subscribe(LLMRequestEvent, self.handle_llm_request)
        self.event_bus.subscribe(LLMRuntimeErrorEvent, self.handle_llm_runtime_error)

    def set_primary_agent_llm_processor(self, processor: LLMProcessor) -> None:
        self._primary_agent_llm_processor = processor

    def set_sub_agent_llm_processor(self, processor: LLMProcessor) -> None:
        self._sub_agent_llm_processor = processor

    async def handle_llm_request(self, event: LLMRequestEvent) -> None:
        handler = (
            self._primary_agent_llm_processor
            if is_primary_agent(event.agent_id) or not self._sub_agent_llm_processor
            else self._sub_agent_llm_processor
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
            context = EventContext(agent_id=event.agent_id, event_bus=self.event_bus)
            if asyncio.iscoroutinefunction(handler):
                response = await handler(event.messages, event.tool_schemas, context)
            else:
                response = await asyncio.to_thread(
                    handler,
                    event.messages,
                    event.tool_schemas,
                    context,
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
