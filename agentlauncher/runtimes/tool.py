import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from agentlauncher.event import (
    EventBus,
    ToolExecErrorEvent,
    ToolExecFinishEvent,
    ToolExecStartEvent,
    ToolResult,
    ToolRuntimeErrorEvent,
    ToolsExecRequestEvent,
    ToolsExecResultsEvent,
)
from agentlauncher.llm import ToolSchema


@dataclass
class Tool(ToolSchema):
    function: Callable[..., str | Awaitable[str]]


class ToolRuntime:
    def __init__(
        self,
        event_bus: EventBus,
    ):
        self.event_bus = event_bus
        self.tools: dict[str, Tool] = {}
        self.event_bus.subscribe(ToolsExecRequestEvent, self.handle_tool_exec_request)

    async def register(
        self,
        name: str,
        function: Callable[..., str | Awaitable[str]],
        description: str,
        parameters: dict[str, Any],
    ):
        if name in self.tools:
            await self.event_bus.emit(
                ToolRuntimeErrorEvent(
                    agent_id=None,
                    error=f"Tool '{name}' is already registered.",
                )
            )
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
                result = tool.function(**arguments)
                if asyncio.iscoroutine(result):
                    result = await result

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

    async def handle_tool_exec_request(self, event: ToolsExecRequestEvent) -> None:
        for tool_call in event.tool_calls:
            if tool_call.tool_name not in self.tools:
                await self.event_bus.emit(
                    ToolRuntimeErrorEvent(
                        agent_id=event.agent_id,
                        error=f"Tool '{tool_call.tool_name}' not found.",
                    )
                )
                return

        tasks = []
        for tool_call in event.tool_calls:
            tasks.append(
                self.tool_exec(
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    agent_id=event.agent_id,
                    tool_call_id=tool_call.tool_call_id,
                )
            )

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
