from collections.abc import Awaitable, Callable

from agentlauncher.eventbus import EventContext

from .message import (
    RequestMessageList,
    RequestToolList,
    ResponseMessageList,
)

type LLMProcessor = Callable[
    [
        RequestMessageList,
        RequestToolList,
        EventContext,
    ],
    Awaitable[ResponseMessageList] | ResponseMessageList,
]
