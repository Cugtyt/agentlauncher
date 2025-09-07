from dataclasses import dataclass

from agentlauncher.llm import (
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
    llm_handler_name: str
    messages: list[
        UserMessage
        | AssistantMessage
        | SystemMessage
        | ToolCallMessage
        | ToolResultMessage
    ]
    tool_schemas: list[ToolSchema]


@dataclass
class LLMResponseEvent(EventType):
    agent_id: str
    llm_handler_name: str
    response: list[AssistantMessage | ToolCallMessage]


@dataclass
class LLMRuntimeErrorEvent(EventType):
    agent_id: str | None
    error: str
