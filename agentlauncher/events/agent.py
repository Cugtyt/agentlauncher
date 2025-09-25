from dataclasses import dataclass

from agentlauncher.eventbus import EventType
from agentlauncher.llm_interface import (
    Message,
    ToolSchema,
)


@dataclass
class AgentCreateEvent(EventType):
    task: str
    tool_schemas: list[ToolSchema]
    conversation: list[Message] | None = None
    system_prompt: str | None = None


@dataclass
class AgentStartEvent(EventType): ...


@dataclass
class AgentFinishEvent(EventType):
    result: str


@dataclass
class AgentRuntimeErrorEvent(EventType):
    error: str


@dataclass
class AgentDeletedEvent(EventType): ...


@dataclass
class AgentConversationProcessedEvent(EventType):
    original_messages: list[Message]
    processed_messages: list[Message]
