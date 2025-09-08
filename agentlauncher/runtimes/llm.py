from agentlauncher.event import (
    EventBus,
    LLMRequestEvent,
    LLMResponseEvent,
    LLMRuntimeErrorEvent,
)
from agentlauncher.llm import (
    LLMHandler,
)

from .shared import AGENT_0_NAME


class LLMRuntime:
    def __init__(
        self,
        event_bus: EventBus,
    ):
        self.event_bus = event_bus
        self.main_agent_llm_handler: LLMHandler | None = None
        self.sub_agent_llm_handler: LLMHandler | None = None
        self.event_bus.subscribe(LLMRequestEvent, self.handle_llm_request)

    async def set_main_agent_handler(self, handler: LLMHandler) -> None:
        self.main_agent_llm_handler = handler

    async def set_sub_agent_handler(self, handler: LLMHandler) -> None:
        self.sub_agent_llm_handler = handler

    async def handle_llm_request(self, event: LLMRequestEvent) -> None:
        handler = (
            self.main_agent_llm_handler
            if event.agent_id == AGENT_0_NAME or not self.sub_agent_llm_handler
            else self.sub_agent_llm_handler
        )
        if not handler:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="LLM handler not found.",
                )
            )
            return
        try:
            response = await handler(event.messages, event.tool_schemas)
            await self.event_bus.emit(
                LLMResponseEvent(
                    agent_id=event.agent_id,
                    response=response,
                )
            )
        except Exception as e:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(agent_id=event.agent_id, error=str(e))
            )
