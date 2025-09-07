from dataclasses import dataclass
from typing import Any

from .type import EventType


@dataclass
class ToolCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolsExecRequestEvent(EventType):
    agent_id: str
    tool_calls: list[ToolCall]


@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    result: str


@dataclass
class ToolsExecResultsEvent(EventType):
    agent_id: str
    tool_results: list[ToolResult]


@dataclass
class ToolRuntimeErrorEvent(EventType):
    agent_id: str | None
    error: str


@dataclass
class ToolExecStartEvent(EventType):
    agent_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolExecFinishEvent(EventType):
    agent_id: str
    tool_call_id: str
    tool_name: str
    result: str


@dataclass
class ToolExecErrorEvent(EventType):
    agent_id: str
    tool_call_id: str
    tool_name: str
    error: str
