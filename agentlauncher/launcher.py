import asyncio
from typing import Any

from agentlauncher.events import (
    EventBus,
    EventHandler,
    EventType,
    EventVerboseLevel,
    TaskCreateEvent,
    TaskFinishEvent,
)
from agentlauncher.runtimes import (
    AGENT_0_NAME,
    AGENT_0_SYSTEM_PROMPT,
    AgentRuntime,
    LLMRuntime,
    MessageRuntime,
    ToolRuntime,
)


class AgentLauncher:
    def __init__(
        self,
        system_prompt: str = AGENT_0_SYSTEM_PROMPT,
        verbose: EventVerboseLevel = EventVerboseLevel.SILENT,
    ):
        self.event_bus = EventBus(verbose=verbose)
        self.system_prompt = system_prompt
        self.agent_runtime = AgentRuntime(self.event_bus)
        self.llm_runtime = LLMRuntime(self.event_bus)
        self.tool_runtime = ToolRuntime(self.event_bus)
        self.message_runtime = MessageRuntime(self.event_bus)
        self.event_bus.subscribe(TaskFinishEvent, self.handle_task_finish)
        self._final_result: asyncio.Future[str] | None = None

    async def register_tool(
        self, name: str, function, description: str, parameters: dict
    ):
        await self.tool_runtime.register(name, function, description, parameters)

    def tool(self, name: str, description: str, parameters: dict):
        def decorator(func):
            async def register_tool():
                await self.register_tool(name, func, description, parameters)

            asyncio.create_task(register_tool())
            return func

        return decorator

    async def register_main_agent_llm_handler(self, name: str, function):
        await self.llm_runtime.set_main_agent_handler(function)

    def main_agent_llm_handler(self, name: str):
        def decorator(func):
            async def register_handler():
                await self.register_main_agent_llm_handler(name, func)

            asyncio.create_task(register_handler())
            return func

        return decorator

    async def register_sub_agent_llm_handler(self, name: str, function):
        await self.llm_runtime.set_sub_agent_handler(function)

    async def sub_agent_llm_handler(self, name: str):
        def decorator(func):
            async def register_handler():
                await self.register_sub_agent_llm_handler(name, func)

            asyncio.create_task(register_handler())
            return func

        return decorator

    def subscribe_event(self, event_type: type[EventType]):
        def decorator(func: EventHandler[Any]):
            self.event_bus.subscribe(event_type, func)
            return func

        return decorator

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        if event.agent_id != AGENT_0_NAME:
            return
        if self._final_result and not self._final_result.done():
            self._final_result.set_result(event.result or "")

    async def run(
        self,
        task: str,
    ) -> str:
        self.tool_runtime.setup()
        self._final_result = asyncio.get_event_loop().create_future()
        await self.event_bus.emit(
            TaskCreateEvent(
                task=task,
                conversation=self.message_runtime.history,
                system_prompt=self.system_prompt,
                tool_schemas=self.tool_runtime.get_tool_schemas(
                    list(self.tool_runtime.tools.keys())
                ),
            )
        )
        return await self._final_result
