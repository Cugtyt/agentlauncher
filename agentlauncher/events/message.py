from collections.abc import Sequence
from dataclasses import dataclass

from agentlauncher.llm_interface import (
    Message,
)

from .type import EventType


@dataclass
class MessagesAddEvent(EventType):
    agent_id: str
    messages: Sequence[Message]


@dataclass
class MessageStartStreamingEvent(EventType):
    agent_id: str


@dataclass
class MessageDeltaStreamingEvent(EventType):
    agent_id: str
    delta: str


@dataclass
class MessageDoneStreamingEvent(EventType):
    agent_id: str
    message: str


@dataclass
class MessageErrorStreamingEvent(EventType):
    agent_id: str


@dataclass
class ToolCallNameStreamingEvent(EventType):
    agent_id: str
    tool_call_id: str
    tool_name: str


@dataclass
class ToolCallArgumentsStartStreamingEvent(EventType):
    agent_id: str
    tool_call_id: str


@dataclass
class ToolCallArgumentsDeltaStreamingEvent(EventType):
    agent_id: str
    tool_call_id: str
    arguments_delta: str


@dataclass
class ToolCallArgumentsDoneStreamingEvent(EventType):
    agent_id: str
    tool_call_id: str
    arguments: str


@dataclass
class ToolCallArgumentsErrorStreamingEvent(EventType):
    agent_id: str
    tool_call_id: str
