from dataclasses import dataclass

from agentlauncher.llm_interface import (
    Message,
    ToolSchema,
)

from .type import EventType


@dataclass
class AgentCreateEvent(EventType):
    agent_id: str
    task: str
    tool_schemas: list[ToolSchema]
    conversation: list[Message] | None = None
    system_prompt: str | None = None


@dataclass
class AgentStartEvent(EventType):
    agent_id: str


@dataclass
class AgentFinishEvent(EventType):
    agent_id: str
    result: str


@dataclass
class AgentRuntimeErrorEvent(EventType):
    agent_id: str
    error: str


@dataclass
class AgentDeletedEvent(EventType):
    agent_id: str
