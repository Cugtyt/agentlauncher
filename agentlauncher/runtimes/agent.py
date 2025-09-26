import asyncio
from asyncio import Lock
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from agentlauncher.eventbus import EventBus, EventContext
from agentlauncher.events import (
    AgentCreateEvent,
    AgentDeletedEvent,
    AgentFinishEvent,
    AgentLauncherShutdownEvent,
    AgentRuntimeErrorEvent,
    AgentStartEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    MessagesAddEvent,
    TaskCancelEvent,
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
from agentlauncher.shared import (
    PRIMARY_AGENT_SYSTEM_PROMPT,
    get_primary_agent_id,
    is_primary_agent,
)

from .type import RuntimeType

type ConversationProcessor = Callable[
    [
        list[Message],
        EventContext,
    ],
    Awaitable[list[Message]] | list[Message],
]


class Agent:
    def __init__(
        self,
        agent_id: str,
        conversation: list[Message],
        tool_schemas: list[ToolSchema],
        event_bus: EventBus,
        system_prompt: str | None = None,
        conversation_processor: ConversationProcessor | None = None,
    ):
        self.agent_id = agent_id
        self._message_cache: list[Message] = list(conversation)
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas
        self.event_bus = event_bus
        self.conversation_processor = conversation_processor

    async def _build_message_list(self) -> list[Message]:
        await self._process_conversation()
        base = list(self._message_cache)
        return (
            [SystemMessage(content=self.system_prompt)] + base
            if self.system_prompt
            else base
        )

    async def _process_conversation(self):
        if not self.conversation_processor:
            return
        context = EventContext(agent_id=self.agent_id, event_bus=self.event_bus)
        if asyncio.iscoroutinefunction(self.conversation_processor):
            self._message_cache = await self.conversation_processor(
                list(self._message_cache), context
            )
        else:
            self._message_cache = cast(
                list[Message],
                self.conversation_processor(list(self._message_cache), context),
            )

    async def start(self, task: str) -> None:
        await self.event_bus.emit(AgentStartEvent(agent_id=self.agent_id))
        user_message = UserMessage(content=task)
        self._message_cache.append(user_message)
        await self.event_bus.emit(
            MessagesAddEvent(agent_id=self.agent_id, messages=[user_message])
        )
        await self.event_bus.emit(
            LLMRequestEvent(
                agent_id=self.agent_id,
                messages=await self._build_message_list(),
                tool_schemas=self.tool_schemas,
            )
        )

    async def handle_llm_response(
        self, response: Sequence[AssistantMessage | ToolCallMessage]
    ) -> None:
        await self.event_bus.emit(
            MessagesAddEvent(agent_id=self.agent_id, messages=response)
        )
        self._message_cache.extend(response)
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
        self._message_cache.extend(tool_result_messages)

        await self.event_bus.emit(
            LLMRequestEvent(
                agent_id=self.agent_id,
                messages=await self._build_message_list(),
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
        self.event_bus.subscribe(TaskCancelEvent, self.handle_task_cancel)
        self.event_bus.subscribe(
            AgentRuntimeErrorEvent, self.handle_agent_runtime_error
        )
        self.event_bus.subscribe(
            AgentLauncherShutdownEvent, self.handle_launcher_shutdown
        )
        self.agents: dict[str, Agent] = {}
        self.conversation_processor: ConversationProcessor | None = None
        self._agents_lock = Lock()
        self._cancelled_agents: set[str] = set()

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
        async with self._agents_lock:
            if event.agent_id in self.agents:
                error_event = AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent with this ID already exists.",
                )
                agent = None
            else:
                agent = Agent(
                    agent_id=event.agent_id,
                    conversation=event.conversation or [],
                    system_prompt=event.system_prompt,
                    event_bus=self.event_bus,
                    tool_schemas=event.tool_schemas,
                    conversation_processor=self.conversation_processor,
                )
                self.agents[event.agent_id] = agent
                error_event = None

        if error_event is not None:
            await self.event_bus.emit(error_event)
            return

        if agent is None:
            return

        await agent.start(event.task)

    async def handle_llm_response(self, event: LLMResponseEvent) -> None:
        async with self._agents_lock:
            agent = self.agents.get(event.agent_id)
        if not agent:
            if event.agent_id in self._cancelled_agents:
                self._cancelled_agents.discard(event.agent_id)
                return
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found for LLM response.",
                )
            )
            return

        await agent.handle_llm_response(event.response)

    async def handle_tools_exec_results(self, event: ToolsExecResultsEvent) -> None:
        async with self._agents_lock:
            agent = self.agents.get(event.agent_id)
        if not agent:
            if event.agent_id in self._cancelled_agents:
                self._cancelled_agents.discard(event.agent_id)
                return
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found for tool execution result.",
                )
            )
            return
        await agent.handle_tools_exec_results(event.tool_results)

    async def handle_agent_finish(self, event: AgentFinishEvent) -> None:
        events_to_emit: list[
            AgentDeletedEvent | TaskFinishEvent | AgentRuntimeErrorEvent
        ] = []
        async with self._agents_lock:
            if event.agent_id in self.agents:
                del self.agents[event.agent_id]
                events_to_emit.append(AgentDeletedEvent(agent_id=event.agent_id))
                if is_primary_agent(event.agent_id):
                    events_to_emit.append(
                        TaskFinishEvent(agent_id=event.agent_id, result=event.result)
                    )
            else:
                events_to_emit.append(
                    AgentRuntimeErrorEvent(
                        agent_id=event.agent_id,
                        error="Agent not found on finish.",
                    )
                )

        for emit_event in events_to_emit:
            await self.event_bus.emit(emit_event)

    async def handle_agent_runtime_error(self, event: AgentRuntimeErrorEvent) -> None:
        events_to_emit: list[AgentDeletedEvent | TaskFinishEvent]
        events_to_emit = []
        async with self._agents_lock:
            if event.agent_id and event.agent_id in self.agents:
                del self.agents[event.agent_id]
                events_to_emit.append(AgentDeletedEvent(agent_id=event.agent_id))
        events_to_emit.append(
            TaskFinishEvent(
                agent_id=event.agent_id or "unknown",
                result=f"Agent encountered an error: {event.error}",
            )
        )
        for emit_event in events_to_emit:
            await self.event_bus.emit(emit_event)

    async def handle_launcher_shutdown(self, event: AgentLauncherShutdownEvent) -> None:
        async with self._agents_lock:
            agent_ids = list(self.agents.keys())
            self.agents.clear()
        for agent_id in agent_ids:
            await self.event_bus.emit(AgentDeletedEvent(agent_id=agent_id))

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        should_emit_deleted = False
        async with self._agents_lock:
            if is_primary_agent(event.agent_id) and event.agent_id in self.agents:
                del self.agents[event.agent_id]
                should_emit_deleted = True
        if should_emit_deleted:
            await self.event_bus.emit(AgentDeletedEvent(agent_id=event.agent_id))

    async def handle_task_cancel(self, event: TaskCancelEvent) -> None:
        to_delete: list[str] = []
        async with self._agents_lock:
            for agent_id in list(self.agents.keys()):
                if get_primary_agent_id(agent_id) == event.agent_id:
                    del self.agents[agent_id]
                    to_delete.append(agent_id)
            self._cancelled_agents.update(to_delete)
            self._cancelled_agents.add(event.agent_id)
        for agent_id in to_delete:
            await self.event_bus.emit(AgentDeletedEvent(agent_id=agent_id))
