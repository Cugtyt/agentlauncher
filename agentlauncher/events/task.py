from dataclasses import dataclass

from agentlauncher.llm_interface import (
    Message,
    ToolSchema,
)

from .type import EventType


@dataclass
class TaskCreateEvent(EventType):
    task: str
    tool_schemas: list[ToolSchema]
    system_prompt: str | None = None
    conversation: list[Message] | None = None


@dataclass
class TaskFinishEvent(EventType):
    result: str
