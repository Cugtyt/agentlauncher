import uuid

from agentlauncher.event import AgentCreateEvent, AgentFinishEvent, EventBus
from agentlauncher.runtimes import AgentRuntime, LLMRuntime, ToolRuntime


class AgentLauncher:
    def __init__(self, verbose: bool = False):
        self.event_bus = EventBus(verbose=verbose)

        self.agent_runtime = AgentRuntime(self.event_bus)
        self.llm_runtime = LLMRuntime(self.event_bus)
        self.tool_runtime = ToolRuntime(self.event_bus)
        self.event_bus.subscribe(AgentFinishEvent, self.handle_agent_finish)
        self.final_result: str | None = None

    async def register_tool(
        self, name: str, function, description: str, parameters: dict
    ):
        await self.tool_runtime.register(name, function, description, parameters)

    async def register_llm_handler(self, name: str, function):
        await self.llm_runtime.register(name, function)

    async def handle_agent_finish(self, event: AgentFinishEvent) -> None:
        self.final_result = event.result

    async def run(
        self,
        task: str,
        system_prompt: str,
        llm_handler_name: str,
        tool_names: list[str] | None = None,
        conversation=None,
    ) -> None:
        if conversation is None:
            conversation = []
        if tool_names is None:
            tool_names = list(self.tool_runtime.tools.keys())
        agent_id = str(uuid.uuid4())
        await self.event_bus.emit(
            AgentCreateEvent(
                agent_id=agent_id,
                task=task,
                conversation=conversation,
                system_prompt=system_prompt,
                llm_handler_name=llm_handler_name,
                tool_schemas=self.tool_runtime.get_tool_schemas(tool_names),
            )
        )
