from collections.abc import Sequence
from dataclasses import dataclass

from agentlauncher.eventbus import EventType
from agentlauncher.llm_interface import (
    AssistantMessage,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)


@dataclass
class LLMRequestEvent(EventType):
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
    request_event: LLMRequestEvent
    response: Sequence[AssistantMessage | ToolCallMessage]


@dataclass
class LLMRuntimeErrorEvent(EventType):
    error: str
    request_event: LLMRequestEvent
