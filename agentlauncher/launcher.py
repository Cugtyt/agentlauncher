import asyncio
from typing import Any

from agentlauncher.eventbus import EventBus, EventHandler, EventHookCallback, EventType
from agentlauncher.events import (
    AgentLauncherShutdownEvent,
    AgentLauncherStopEvent,
    TaskCancelEvent,
    TaskCreateEvent,
    TaskFinishEvent,
)
from agentlauncher.llm_interface import ToolParamSchema
from agentlauncher.llm_interface.message import Message
from agentlauncher.runtimes import (
    AgentRuntime,
    LLMRuntime,
    RuntimeType,
    ToolRuntime,
)
from agentlauncher.shared import PRIMARY_AGENT_SYSTEM_PROMPT, generate_primary_agent_id


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
        self.runtimes: list[RuntimeType] = []
        self.event_bus.subscribe(TaskFinishEvent, self.handle_task_finish)
        self._final_results: dict[str, asyncio.Future[str]] = {}
        self._result_lock = asyncio.Lock()

    def register_tool(
        self,
        name: str,
        function,
        description: str,
        parameters: dict[str, ToolParamSchema],
        context_key: str | None = None,
    ):
        self.tool_runtime.register(
            name, function, description, parameters, context_key=context_key
        )

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict],
        *,
        context_key: str | None = None,
    ):
        def decorator(func):
            self.register_tool(
                name,
                func,
                description,
                {p: ToolParamSchema(**s) for p, s in parameters.items()},
                context_key=context_key,
            )

            return func

        return decorator

    def set_primary_agent_llm_processor(self, function):
        self.llm_runtime.set_primary_agent_llm_processor(function)

    def primary_agent_llm_processor(self):
        def decorator(func):
            self.set_primary_agent_llm_processor(func)
            return func

        return decorator

    def set_sub_agent_llm_processor(self, function):
        self.llm_runtime.set_sub_agent_llm_processor(function)

    def sub_agent_llm_processor(self):
        def decorator(func):
            self.set_sub_agent_llm_processor(func)
            return func

        return decorator

    def subscribe_event(self, event_type: type[EventType]):
        def decorator(func: EventHandler[Any]):
            self.event_bus.subscribe(event_type, func)
            return func

        return decorator

    def set_conversation_processor(self, function):
        self.agent_runtime.set_conversation_processor(function)

    def conversation_processor(self):
        def decorator(func):
            self.set_conversation_processor(func)
            return func

        return decorator

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        async with self._result_lock:
            future = self._final_results.get(event.agent_id)
            if future is None or future.done():
                return
            future.set_result(event.result or "")

        await self.event_bus.emit(
            AgentLauncherStopEvent(agent_id=event.agent_id, result=event.result or "")
        )

    async def run(
        self,
        task: str,
        history: list[Message] | None = None,
        timeout: float | None = 600.0,
        event_callback: EventHookCallback | None = None,
    ) -> str | None:
        agent_id = generate_primary_agent_id()
        future = asyncio.get_running_loop().create_future()

        async with self._result_lock:
            self.tool_runtime.setup_sub_agent_tool()
            self._final_results[agent_id] = future
            tool_schemas = self.tool_runtime.get_tool_schemas(
                list(self.tool_runtime.tools.keys())
            )

        timeout_reason = (
            f"Task timed out after {timeout} seconds"
            if timeout is not None
            else "Task timed out"
        )
        result: str | None = None

        try:
            if event_callback is not None:
                await self.event_bus.add_hook(agent_id, event_callback)

            await self.event_bus.emit(
                TaskCreateEvent(
                    agent_id=agent_id,
                    task=task,
                    conversation=history,
                    system_prompt=self.system_prompt,
                    tool_schemas=tool_schemas,
                )
            )

            if timeout is None:
                result = await future
            else:
                result = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            await self.cancel(agent_id, reason=timeout_reason)
            return None
        finally:
            async with self._result_lock:
                self._final_results.pop(agent_id, None)
            await self.event_bus.remove_hook(agent_id)

        return result

    async def shutdown(self) -> None:
        async with self._result_lock:
            active_agents = list(self._final_results.keys())
        for agent_id in active_agents:
            await self.cancel(agent_id, reason="Launcher shutdown requested")
            await self.event_bus.emit(AgentLauncherShutdownEvent(agent_id=agent_id))

    def register_runtime(self, runtime_type: type[RuntimeType]) -> None:
        self.runtimes.append(runtime_type(self.event_bus))

    async def cancel(self, agent_id: str, reason: str | None = None) -> None:
        async with self._result_lock:
            future = self._final_results.pop(agent_id, None)
        was_primary = future is not None
        if future and not future.done():
            future.cancel()
        cancel_reason = reason or "Task cancelled"
        await self.event_bus.emit(
            TaskCancelEvent(agent_id=agent_id, reason=cancel_reason)
        )
        if was_primary:
            await self.event_bus.emit(
                AgentLauncherStopEvent(agent_id=agent_id, result=cancel_reason)
            )
