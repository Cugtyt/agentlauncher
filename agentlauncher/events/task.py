from dataclasses import dataclass

from agentlauncher.llm_interface import (
    AssistantMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

from .type import EventType


@dataclass
class TaskCreateEvent(EventType):
    task: str
    tool_schemas: list[ToolSchema]
    system_prompt: str | None = None
    conversation: (
        list[UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage]
        | None
    ) = None


@dataclass
class TaskFinishEvent(EventType):
    agent_id: str
    result: str
