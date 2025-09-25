from abc import ABC
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar


@dataclass
class EventType(ABC):
    agent_id: str


T = TypeVar("T", bound=EventType)
type EventHandler[T] = Callable[[T], Coroutine[Any, Any, None]]
type EventHookCallback = Callable[[EventType], Coroutine[Any, Any, None]]
