from dataclasses import dataclass

from agentlauncher.eventbus import EventType
from agentlauncher.llm_interface import (
    Message,
    ToolSchema,
)


@dataclass
class TaskCreateEvent(EventType):
    task: str
    tool_schemas: list[ToolSchema]
    system_prompt: str | None = None
    conversation: list[Message] | None = None


@dataclass
class TaskFinishEvent(EventType):
    result: str
