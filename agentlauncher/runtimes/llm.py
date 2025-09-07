from agentlauncher.event import (
    EventBus,
    LLMRequestEvent,
    LLMResponseEvent,
    LLMRuntimeErrorEvent,
)
from agentlauncher.llm import (
    LLMHandler,
)


class LLMRuntime:
    def __init__(
        self,
        event_bus: EventBus,
    ):
        self.event_bus = event_bus
        self.llm_handlers: dict[str, LLMHandler] = {}
        self.event_bus.subscribe(LLMRequestEvent, self.handle_llm_request)

    async def register(self, name: str, handler: LLMHandler) -> None:
        if name in self.llm_handlers:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(
                    agent_id=None,
                    error=f"LLM handler '{name}' is already registered.",
                )
            )
        self.llm_handlers[name] = handler

    async def handle_llm_request(self, event: LLMRequestEvent) -> None:
        handler = self.llm_handlers.get(event.llm_handler_name)
        if not handler:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error=f"LLM handler '{event.llm_handler_name}' not found.",
                )
            )
            return
        try:
            response = await handler(event.messages, event.tool_schemas)
            await self.event_bus.emit(
                LLMResponseEvent(
                    agent_id=event.agent_id,
                    llm_handler_name=event.llm_handler_name,
                    response=response,
                )
            )
        except Exception as e:
            await self.event_bus.emit(
                LLMRuntimeErrorEvent(agent_id=event.agent_id, error=str(e))
            )
