from agentlauncher.events import (
    EventBus,
    LLMResponseEvent,
    MessageAddEvent,
    TaskCreateEvent,
    ToolsExecResultsEvent,
)
from agentlauncher.llm_interface import (
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
        self.event_bus.subscribe(ToolsExecResultsEvent, self.handle_tools_exec_results)

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

    async def handle_tools_exec_results(self, event: ToolsExecResultsEvent) -> None:
        if event.agent_id != AGENT_0_NAME:
            return
        messages: list[ToolResultMessage] = []
        for result in event.tool_results:
            messages.append(
                ToolResultMessage(
                    tool_call_id=result.tool_call_id,
                    tool_name=result.tool_name,
                    result=result.result,
                )
            )
        self.history.extend(messages)
        await self.event_bus.emit(
            MessageAddEvent(agent_id=event.agent_id, messages=messages)
        )
