from dataclasses import dataclass

from .tool import ToolSchema


@dataclass
class UserMessage:
    content: str


@dataclass
class SystemMessage:
    content: str


@dataclass
class ToolCallMessage:
    tool_call_id: str
    tool_name: str
    arguments: dict


@dataclass
class AssistantMessage:
    content: str


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    result: str


type RequestMessageList = list[
    AssistantMessage | SystemMessage | ToolCallMessage | ToolResultMessage | UserMessage
]

type RequestToolList = list[ToolSchema]

type ResponseMessageList = list[AssistantMessage | ToolCallMessage]
