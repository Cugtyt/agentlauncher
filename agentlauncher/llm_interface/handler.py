from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlauncher.events import EventBus

from .message import (
    RequestMessageList,
    RequestToolList,
    ResponseMessageList,
)

type LLMHandler = Callable[
    [
        RequestMessageList,
        RequestToolList,
        str,  # agent_id
        "EventBus",
    ],
    Awaitable[ResponseMessageList] | ResponseMessageList,
]
