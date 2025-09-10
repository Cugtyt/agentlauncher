import asyncio

from agentlauncher.event import (
    EventBus,
    EventVerboseLevel,
    TaskCreateEvent,
    TaskFinishEvent,
)
from agentlauncher.runtimes import (
    AGENT_0_NAME,
    AGENT_0_SYSTEM_PROMPT,
    AgentRuntime,
    LLMRuntime,
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
        self.event_bus.subscribe(TaskFinishEvent, self.handle_task_finish)
        self._final_result: asyncio.Future[str] | None = None

    async def register_tool(
        self, name: str, function, description: str, parameters: dict
    ):
        await self.tool_runtime.register(name, function, description, parameters)

    async def register_main_agent_llm_handler(self, name: str, function):
        await self.llm_runtime.set_main_agent_handler(function)

    async def register_sub_agent_llm_handler(self, name: str, function):
        await self.llm_runtime.set_sub_agent_handler(function)

    async def handle_task_finish(self, event: TaskFinishEvent) -> None:
        if event.agent_id != AGENT_0_NAME:
            return
        if self._final_result and not self._final_result.done():
            self._final_result.set_result(event.result or "")

    async def run(
        self,
        task: str,
        conversation=None,
    ) -> str:
        self.tool_runtime.setup()
        self._final_result = asyncio.get_event_loop().create_future()
        if conversation is None:
            conversation = []
        await self.event_bus.emit(
            TaskCreateEvent(
                task=task,
                conversation=conversation,
                system_prompt=self.system_prompt,
                tool_schemas=self.tool_runtime.get_tool_schemas(
                    list(self.tool_runtime.tools.keys())
                ),
            )
        )
        return await self._final_result
