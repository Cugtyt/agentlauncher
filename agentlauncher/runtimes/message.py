from collections.abc import Awaitable, Callable, Sequence

from agentlauncher.events import (
    EventBus,
    LLMResponseEvent,
    MessagesAddEvent,
    TaskCreateEvent,
    ToolsExecResultsEvent,
)
from agentlauncher.llm_interface import (
    Message,
    ResponseMessageList,
    ToolResultMessage,
    UserMessage,
)

from .shared import AGENT_0_NAME


class MessageRuntime:
    def __init__(self, event_bus: EventBus):
        self.history: list[Message] = []
        self.event_bus = event_bus
        self.event_bus.subscribe(LLMResponseEvent, self.handle_llm_response)
        self.event_bus.subscribe(TaskCreateEvent, self.handle_task_create)
        self.event_bus.subscribe(ToolsExecResultsEvent, self.handle_tools_exec_results)
        self.event_bus.subscribe(MessagesAddEvent, self.handle_conversation_update)
        self.response_message_handler: (
            Callable[[ResponseMessageList], Awaitable[ResponseMessageList]] | None
        ) = None
        self.conversation_handler: (
            Callable[[Sequence[Message]], Awaitable[Sequence[Message]]] | None
        ) = None

    async def handle_llm_response(self, event: LLMResponseEvent) -> None:
        if event.agent_id != AGENT_0_NAME:
            return
        if self.response_message_handler:
            event.response = await self.response_message_handler(event.response)
        self.history.extend(event.response)
        await self.event_bus.emit(
            MessagesAddEvent(agent_id=event.agent_id, messages=event.response)
        )

    async def handle_task_create(self, event: TaskCreateEvent) -> None:
        self.history.append(UserMessage(content=event.task))
        await self.event_bus.emit(
            MessagesAddEvent(
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
            MessagesAddEvent(agent_id=event.agent_id, messages=messages)
        )

    async def handle_conversation_update(self, event: MessagesAddEvent) -> None:
        if event.agent_id == AGENT_0_NAME and self.conversation_handler:
            self.history = list(await self.conversation_handler(self.history))

    def register_message_handler(
        self, handler: Callable[[ResponseMessageList], Awaitable[ResponseMessageList]]
    ) -> None:
        self.response_message_handler = handler

    def register_conversation_handler(
        self, handler: Callable[[Sequence[Message]], Awaitable[Sequence[Message]]]
    ) -> None:
        self.conversation_handler = handler
