from collections.abc import Sequence
from dataclasses import dataclass

from agentlauncher.llm import (
    AssistantMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)

from .type import EventType


@dataclass
class MessageAddEvent(EventType):
    agent_id: str
    messages: Sequence[
        UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage
    ]
