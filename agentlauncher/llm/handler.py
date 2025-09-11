from collections.abc import Awaitable, Callable

from .message import (
    RequestMessageList,
    RequestToolList,
    ResponseMessageList,
)

type LLMHandler = Callable[
    [
        RequestMessageList,
        RequestToolList,
    ],
    Awaitable[ResponseMessageList] | ResponseMessageList,
]
