from collections.abc import Sequence
from dataclasses import dataclass

from agentlauncher.llm_interface import (
    AssistantMessage,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

from .type import EventType


@dataclass
class LLMRequestEvent(EventType):
    agent_id: str
    messages: list[
        UserMessage
        | AssistantMessage
        | SystemMessage
        | ToolCallMessage
        | ToolResultMessage
    ]
    tool_schemas: list[ToolSchema]
    retry_count: int = 0


@dataclass
class LLMResponseEvent(EventType):
    agent_id: str
    response: Sequence[AssistantMessage | ToolCallMessage]


@dataclass
class LLMRuntimeErrorEvent(EventType):
    agent_id: str | None
    error: str
    request_event: LLMRequestEvent
