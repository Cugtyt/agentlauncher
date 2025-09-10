from agentlauncher.event import (
    EventBus,
    LLMResponseEvent,
    MessageAddEvent,
    TaskCreateEvent,
    TaskFinishEvent,
)
from agentlauncher.llm import (
    AssistantMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)

from .shared import AGENT_0_NAME


class MessageRuntime:
    def __init__(self, event_bus: EventBus):
        self.history: list[
            UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage
        ] = []
        self.event_bus = event_bus
        self.event_bus.subscribe(LLMResponseEvent, self.handle_llm_response)
        self.event_bus.subscribe(TaskCreateEvent, self.handle_task_create)
        self.event_bus.subscribe(TaskFinishEvent, self.handle_task_finish)

    async def handle_llm_response(self, event: LLMResponseEvent) -> None:
        if event.agent_id != AGENT_0_NAME:
            return
        self.history.extend(event.response)
        await self.event_bus.emit(
            MessageAddEvent(agent_id=event.agent_id, messages=event.response)
        )

    async def handle_task_create(self, event: TaskCreateEvent) -> None:
        self.history.append(UserMessage(content=event.task))
        await self.event_bus.emit(
            MessageAddEvent(
                agent_id=AGENT_0_NAME, messages=[UserMessage(content=event.task)]
            )
        )

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        if event.agent_id != AGENT_0_NAME:
            return
        self.history.append(AssistantMessage(content=event.result))
        await self.event_bus.emit(
            MessageAddEvent(
                agent_id=AGENT_0_NAME, messages=[AssistantMessage(content=event.result)]
            )
        )
