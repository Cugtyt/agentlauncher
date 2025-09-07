from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar


class EventType(Protocol): ...


T = TypeVar("T", bound=EventType)
type EventHandler[T] = Callable[[T], Awaitable[None]]
