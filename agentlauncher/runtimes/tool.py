import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from agentlauncher.events import (
    AgentFinishEvent,
    EventBus,
    ToolExecErrorEvent,
    ToolExecFinishEvent,
    ToolExecStartEvent,
    ToolResult,
    ToolRuntimeErrorEvent,
    ToolsExecRequestEvent,
    ToolsExecResultsEvent,
)
from agentlauncher.events.agent import AgentCreateEvent
from agentlauncher.llm_interface import ToolParamSchema, ToolSchema

from .shared import CREATE_SUB_AGENT_TOOL_NAME
from .type import RuntimeType


@dataclass
class Tool(ToolSchema):
    function: Callable[..., str | Awaitable[str]]


class ToolRuntime(RuntimeType):
    def __init__(
        self,
        event_bus: EventBus,
    ):
        super().__init__(event_bus)
        self.tools: dict[str, Tool] = {}

        self.event_bus.subscribe(ToolsExecRequestEvent, self.handle_tools_exec_request)
        self.event_bus.subscribe(AgentFinishEvent, self.handle_agent_finish)
        self.event_bus.subscribe(ToolRuntimeErrorEvent, self.handle_tool_runtime_error)
        self.sub_agent_futures: dict[str, asyncio.Future[str]] = {}

    def setup(self) -> None:
        self.tools[CREATE_SUB_AGENT_TOOL_NAME] = Tool(
            name=CREATE_SUB_AGENT_TOOL_NAME,
            function=self._create_sub_agent_tool,
            description="Create a sub-agent to handle a specific task.",
            parameters={
                "task": ToolParamSchema(
                    type="string",
                    description="The task for the sub-agent to accomplish.",
                    required=True,
                ),
                "tool_name_list": ToolParamSchema(
                    type="array",
                    items={"type": "string"},
                    description="List of tool names that the sub-agent can use, "
                    "the tool names are from your tool list."
                    f"available tools are: {', '.join(self.tools.keys())}",
                    required=True,
                ),
            },
        )

    async def handle_agent_finish(self, event: AgentFinishEvent) -> None:
        if event.agent_id not in self.sub_agent_futures:
            return
        future = self.sub_agent_futures[event.agent_id]
        if not future.done():
            future.set_result(event.result or "")

    async def _create_sub_agent_tool(self, task: str, tool_name_list: list[str]) -> str:
        agent_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self.sub_agent_futures[agent_id] = future

        await self.event_bus.emit(
            AgentCreateEvent(
                agent_id=agent_id,
                task=task,
                tool_schemas=self.get_tool_schemas(tool_name_list),
            )
        )

        try:
            result = await future
            return result
        finally:
            del self.sub_agent_futures[agent_id]

    def register(
        self,
        name: str,
        function: Callable[..., str | Awaitable[str]],
        description: str,
        parameters: dict[str, ToolParamSchema],
    ):
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        tool = Tool(
            name=name, function=function, description=description, parameters=parameters
        )
        self.tools[name] = tool

    async def tool_exec(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str,
        tool_call_id: str,
    ) -> str:
        await self.event_bus.emit(
            ToolExecStartEvent(
                agent_id=agent_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        try:
            tool = self.tools[tool_name]
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**arguments)
            else:
                result = await asyncio.to_thread(tool.function, **arguments)

            await self.event_bus.emit(
                ToolExecFinishEvent(
                    agent_id=agent_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=cast(str, result),
                )
            )
            return cast(str, result)
        except Exception as e:
            await self.event_bus.emit(
                ToolExecErrorEvent(
                    agent_id=agent_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    error=str(e),
                )
            )
            raise e

    async def handle_tools_exec_request(self, event: ToolsExecRequestEvent) -> None:
        missing_tools = [
            tc.tool_name for tc in event.tool_calls if tc.tool_name not in self.tools
        ]
        if missing_tools:
            await self.event_bus.emit(
                ToolRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error=f"Tool(s) not found: {', '.join(missing_tools)}",
                )
            )
            return

        tasks = [
            self.tool_exec(
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                agent_id=event.agent_id,
                tool_call_id=tool_call.tool_call_id,
            )
            for tool_call in event.tool_calls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_results = []
        for tool_call, result in zip(event.tool_calls, results, strict=False):
            if not isinstance(result, Exception):
                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        tool_name=tool_call.tool_name,
                        result=cast(str, result),
                    )
                )

        await self.event_bus.emit(
            ToolsExecResultsEvent(
                agent_id=event.agent_id,
                tool_results=tool_results,
            )
        )

    def get_tool_schemas(self, tool_names: list[str]) -> list[ToolSchema]:
        return [tool for name, tool in self.tools.items() if name in tool_names]

    async def handle_tool_runtime_error(self, event: ToolRuntimeErrorEvent) -> None:
        await self.event_bus.emit(
            ToolsExecResultsEvent(
                agent_id=event.agent_id,
                tool_results=[],
            )
        )
