from dataclasses import dataclass

from agentlauncher.eventbus import EventType
from agentlauncher.llm_interface import (
    ToolSchema,
)


@dataclass
class TaskCreateEvent(EventType):
    task: str
    tool_schemas: list[ToolSchema]
    system_prompt: str | None = None


@dataclass
class TaskFinishEvent(EventType):
    result: str


@dataclass
class TaskCancelEvent(EventType):
    reason: str | None = None
