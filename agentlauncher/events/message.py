from collections.abc import Sequence
from dataclasses import dataclass

from agentlauncher.llm_interface import (
    Message,
)

from .type import EventType


@dataclass
class MessagesAddEvent(EventType):
    messages: Sequence[Message]


@dataclass
class MessageStartStreamingEvent(EventType): ...


@dataclass
class MessageDeltaStreamingEvent(EventType):
    delta: str


@dataclass
class MessageDoneStreamingEvent(EventType):
    message: str


@dataclass
class MessageErrorStreamingEvent(EventType):
    error: str


@dataclass
class ToolCallNameStreamingEvent(EventType):
    tool_call_id: str
    tool_name: str


@dataclass
class ToolCallArgumentsStartStreamingEvent(EventType):
    agent_id: str
    tool_call_id: str


@dataclass
class ToolCallArgumentsDeltaStreamingEvent(EventType):
    tool_call_id: str
    arguments_delta: str


@dataclass
class ToolCallArgumentsDoneStreamingEvent(EventType):
    tool_call_id: str
    arguments: str


@dataclass
class ToolCallArgumentsErrorStreamingEvent(EventType):
    tool_call_id: str
    error: str
