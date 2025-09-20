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

from .shared import is_primary_agent
from .type import RuntimeType


class MessageRuntime(RuntimeType):
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.history: list[Message] = []
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
        if not is_primary_agent(event.agent_id):
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
                agent_id=event.agent_id, messages=[UserMessage(content=event.task)]
            )
        )

    async def handle_tools_exec_results(self, event: ToolsExecResultsEvent) -> None:
        if not is_primary_agent(event.agent_id):
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
        if is_primary_agent(event.agent_id) and self.conversation_handler:
            self.history = list(await self.conversation_handler(self.history))

    def register_message_handler(
        self, handler: Callable[[ResponseMessageList], Awaitable[ResponseMessageList]]
    ) -> None:
        self.response_message_handler = handler

    def register_conversation_handler(
        self, handler: Callable[[Sequence[Message]], Awaitable[Sequence[Message]]]
    ) -> None:
        self.conversation_handler = handler
