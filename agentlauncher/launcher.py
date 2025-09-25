import asyncio
from typing import Any

from agentlauncher.eventbus import EventBus, EventHandler, EventType
from agentlauncher.events import (
    AgentLauncherShutdownEvent,
    AgentLauncherStopEvent,
    TaskCreateEvent,
    TaskFinishEvent,
)
from agentlauncher.llm_interface import ToolParamSchema
from agentlauncher.llm_interface.message import Message
from agentlauncher.runtimes import (
    PRIMARY_AGENT_SYSTEM_PROMPT,
    AgentRuntime,
    LLMRuntime,
    MessageRuntime,
    RuntimeType,
    ToolRuntime,
    generate_primary_agent_id,
)


class AgentLauncher:
    def __init__(
        self,
        system_prompt: str = PRIMARY_AGENT_SYSTEM_PROMPT,
        sub_agent_tool: bool = True,
    ):
        self.event_bus = EventBus()
        self.system_prompt = system_prompt
        self.agent_runtime = AgentRuntime(self.event_bus)
        self.llm_runtime = LLMRuntime(self.event_bus)
        self.tool_runtime = ToolRuntime(self.event_bus, sub_agent_tool=sub_agent_tool)
        self.message_runtime = MessageRuntime(self.event_bus)
        self.runtimes: list[RuntimeType] = []
        self.event_bus.subscribe(TaskFinishEvent, self.handle_task_finish)
        self._final_results: dict[str, asyncio.Future[str]] = {}
        self.primary_agents: set[str] = set()
        self._agent_lock = asyncio.Lock()

    def register_tool(
        self,
        name: str,
        function,
        description: str,
        parameters: dict[str, ToolParamSchema],
    ):
        self.tool_runtime.register(name, function, description, parameters)

    def tool(self, name: str, description: str, parameters: dict[str, dict]):
        def decorator(func):
            self.register_tool(
                name,
                func,
                description,
                {p: ToolParamSchema(**s) for p, s in parameters.items()},
            )

            return func

        return decorator

    def register_primary_agent_llm_handler(self, function):
        self.llm_runtime.set_primary_agent_handler(function)

    def primary_agent_llm_handler(self):
        def decorator(func):
            self.register_primary_agent_llm_handler(func)
            return func

        return decorator

    def register_sub_agent_llm_handler(self, function):
        self.llm_runtime.set_sub_agent_handler(function)

    def sub_agent_llm_handler(self):
        def decorator(func):
            self.register_sub_agent_llm_handler(func)
            return func

        return decorator

    def subscribe_event(self, event_type: type[EventType]):
        def decorator(func: EventHandler[Any]):
            self.event_bus.subscribe(event_type, func)
            return func

        return decorator

    def register_message_handler(self, function):
        self.message_runtime.register_message_handler(function)

    def message_handler(self):
        def decorator(func):
            self.register_message_handler(func)
            return func

        return decorator

    def register_conversation_handler(self, function):
        self.message_runtime.register_conversation_handler(function)

    def conversation_handler(self):
        def decorator(func):
            self.register_conversation_handler(func)
            return func

        return decorator

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        async with self._agent_lock:
            if (
                event.agent_id not in self.primary_agents
                or event.agent_id not in self._final_results
            ):
                return

            self._final_results[event.agent_id].set_result(event.result or "")
            await self.event_bus.emit(
                AgentLauncherStopEvent(
                    agent_id=event.agent_id, result=event.result or ""
                )
            )

    async def run(self, task: str, history: list[Message] | None = None) -> str:
        async with self._agent_lock:
            agent_id = generate_primary_agent_id(len(self.primary_agents))
            self.primary_agents.add(agent_id)
            self.tool_runtime.setup_sub_agent_tool()
            self._final_results[agent_id] = asyncio.get_event_loop().create_future()

        await self.event_bus.emit(
            TaskCreateEvent(
                agent_id=agent_id,
                task=task,
                conversation=history,
                system_prompt=self.system_prompt,
                tool_schemas=self.tool_runtime.get_tool_schemas(
                    list(self.tool_runtime.tools.keys())
                ),
            )
        )
        result = await self._final_results[agent_id]
        async with self._agent_lock:
            del self._final_results[agent_id]
        return result

    async def shutdown(self) -> None:
        async with self._agent_lock:
            primary_agents = list(self.primary_agents)
            self.primary_agents.clear()
        for agent_id in primary_agents:
            await self.event_bus.emit(AgentLauncherShutdownEvent(agent_id=agent_id))

    def register_runtime(self, runtime_type: type[RuntimeType]) -> None:
        self.runtimes.append(runtime_type(self.event_bus))
