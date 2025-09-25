from collections.abc import Awaitable, Callable

from agentlauncher.eventbus import EventBus

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
        EventBus,
    ],
    Awaitable[ResponseMessageList] | ResponseMessageList,
]
