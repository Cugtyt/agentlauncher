from asyncio import iscoroutinefunction
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from agentlauncher.eventbus import EventBus
from agentlauncher.events import (
    AgentConversationProcessedEvent,
    AgentCreateEvent,
    AgentDeletedEvent,
    AgentFinishEvent,
    AgentLauncherShutdownEvent,
    AgentRuntimeErrorEvent,
    AgentStartEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    MessagesAddEvent,
    TaskCreateEvent,
    TaskFinishEvent,
    ToolCall,
    ToolsExecRequestEvent,
    ToolsExecResultsEvent,
)
from agentlauncher.events.tool import ToolResult
from agentlauncher.llm_interface import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

from .shared import PRIMARY_AGENT_SYSTEM_PROMPT, is_primary_agent
from .type import RuntimeType

type ConversationProcessor = Callable[
    [
        list[Message],
        str,  # agent_id
        EventBus,
    ],
    Awaitable[list[Message]] | list[Message],
]


class Agent:
    def __init__(
        self,
        agent_id: str,
        task: str,
        conversation: list[Message],
        tool_schemas: list[ToolSchema],
        event_bus: EventBus,
        system_prompt: str | None = None,
        conversation_processor: ConversationProcessor | None = None,
    ):
        self.agent_id = agent_id
        self.task = task
        self.conversation = conversation
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas
        self.event_bus = event_bus
        self.conversation_processor = conversation_processor

    async def start(self) -> None:
        await self.event_bus.emit(AgentStartEvent(agent_id=self.agent_id))
        user_message = UserMessage(content=self.task)
        self.conversation = [
            *self.conversation,
            user_message,
        ]
        await self.event_bus.emit(
            MessagesAddEvent(agent_id=self.agent_id, messages=[user_message])
        )
        if not self.system_prompt:
            message_list = [*self.conversation]
        else:
            message_list = [
                SystemMessage(content=self.system_prompt),
                *self.conversation,
            ]
        if self.conversation_processor:
            if iscoroutinefunction(self.conversation_processor):
                processed_message_list = await self.conversation_processor(
                    message_list, self.agent_id, self.event_bus
                )
            else:
                processed_message_list = self.conversation_processor(
                    message_list, self.agent_id, self.event_bus
                )
            processed_message_list = cast(list[Message], processed_message_list)
            await self.event_bus.emit(
                AgentConversationProcessedEvent(
                    agent_id=self.agent_id,
                    original_messages=message_list,
                    processed_messages=processed_message_list,
                )
            )
            message_list = processed_message_list
        await self.event_bus.emit(
            LLMRequestEvent(
                agent_id=self.agent_id,
                messages=message_list,
                tool_schemas=self.tool_schemas,
            )
        )

    async def handle_llm_response(
        self, response: Sequence[AssistantMessage | ToolCallMessage]
    ) -> None:
        await self.event_bus.emit(
            MessagesAddEvent(agent_id=self.agent_id, messages=response)
        )
        self.conversation.extend(response)
        tool_calls = [
            ToolCall(msg.tool_call_id, msg.tool_name, msg.arguments)
            for msg in response
            if isinstance(msg, ToolCallMessage)
        ]
        if not tool_calls:
            assistant_contents = [
                msg.content for msg in response if isinstance(msg, AssistantMessage)
            ]
            final_response = "\n".join(assistant_contents) if assistant_contents else ""
            await self.event_bus.emit(
                AgentFinishEvent(agent_id=self.agent_id, result=final_response)
            )
            return
        await self.event_bus.emit(
            ToolsExecRequestEvent(agent_id=self.agent_id, tool_calls=tool_calls)
        )

    async def handle_tools_exec_results(self, tool_results: list[ToolResult]) -> None:
        tool_result_messages = [
            ToolResultMessage(
                tool_call_id=tr.tool_call_id,
                tool_name=tr.tool_name,
                result=tr.result,
            )
            for tr in tool_results
        ]
        await self.event_bus.emit(
            MessagesAddEvent(agent_id=self.agent_id, messages=tool_result_messages)
        )
        self.conversation.extend(tool_result_messages)
        if not self.system_prompt:
            message_list = [*self.conversation]
        else:
            message_list = [
                SystemMessage(content=self.system_prompt),
                *self.conversation,
            ]
        await self.event_bus.emit(
            LLMRequestEvent(
                agent_id=self.agent_id,
                messages=message_list,
                tool_schemas=self.tool_schemas,
            )
        )


class AgentRuntime(RuntimeType):
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.event_bus.subscribe(AgentCreateEvent, self.handle_agent_create)
        self.event_bus.subscribe(LLMResponseEvent, self.handle_llm_response)
        self.event_bus.subscribe(ToolsExecResultsEvent, self.handle_tools_exec_results)
        self.event_bus.subscribe(AgentFinishEvent, self.handle_agent_finish)
        self.event_bus.subscribe(TaskCreateEvent, self.handle_task_create)
        self.event_bus.subscribe(TaskFinishEvent, self.handle_task_finish)
        self.event_bus.subscribe(
            AgentRuntimeErrorEvent, self.handle_agent_runtime_error
        )
        self.event_bus.subscribe(
            AgentLauncherShutdownEvent, self.handle_launcher_shutdown
        )
        self.agents: dict[str, Agent] = {}
        self.conversation_processor: ConversationProcessor | None = None

    def set_conversation_processor(self, processor: ConversationProcessor) -> None:
        self.conversation_processor = processor

    async def handle_task_create(self, event: TaskCreateEvent) -> None:
        await self.event_bus.emit(
            AgentCreateEvent(
                agent_id=event.agent_id,
                task=event.task,
                conversation=event.conversation or [],
                system_prompt=event.system_prompt or PRIMARY_AGENT_SYSTEM_PROMPT,
                tool_schemas=event.tool_schemas,
            )
        )

    async def handle_agent_create(self, event: AgentCreateEvent) -> None:
        if event.agent_id in self.agents:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent with this ID already exists.",
                )
            )
            return
        self.agents[event.agent_id] = Agent(
            agent_id=event.agent_id,
            task=event.task,
            conversation=event.conversation or [],
            system_prompt=event.system_prompt,
            event_bus=self.event_bus,
            tool_schemas=event.tool_schemas,
            conversation_processor=self.conversation_processor,
        )
        await self.agents[event.agent_id].start()

    async def handle_llm_response(self, event: LLMResponseEvent) -> None:
        agent = self.agents.get(event.agent_id)
        if not agent:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found for LLM response.",
                )
            )
            return

        await agent.handle_llm_response(event.response)

    async def handle_tools_exec_results(self, event: ToolsExecResultsEvent) -> None:
        agent = self.agents.get(event.agent_id)
        if not agent:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found for tool execution result.",
                )
            )
            return
        await agent.handle_tools_exec_results(event.tool_results)

    async def handle_agent_finish(self, event: AgentFinishEvent) -> None:
        if event.agent_id in self.agents:
            if not is_primary_agent(event.agent_id):
                del self.agents[event.agent_id]
                await self.event_bus.emit(AgentDeletedEvent(agent_id=event.agent_id))
            else:
                await self.event_bus.emit(
                    TaskFinishEvent(agent_id=event.agent_id, result=event.result)
                )
        else:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found on finish.",
                )
            )

    async def handle_agent_runtime_error(self, event: AgentRuntimeErrorEvent) -> None:
        if event.agent_id and event.agent_id in self.agents:
            del self.agents[event.agent_id]
            await self.event_bus.emit(AgentDeletedEvent(agent_id=event.agent_id))
        await self.event_bus.emit(
            TaskFinishEvent(
                agent_id=event.agent_id or "unknown",
                result=f"Agent encountered an error: {event.error}",
            )
        )

    async def handle_launcher_shutdown(self, event: AgentLauncherShutdownEvent) -> None:
        for agent_id in list(self.agents.keys()):
            del self.agents[agent_id]
            await self.event_bus.emit(AgentDeletedEvent(agent_id=agent_id))

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        if is_primary_agent(event.agent_id) and event.agent_id in self.agents:
            del self.agents[event.agent_id]
            await self.event_bus.emit(AgentDeletedEvent(agent_id=event.agent_id))
