from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class EventType(ABC):
    agent_id: str


T = TypeVar("T", bound=EventType)
type EventHandler[T] = Callable[[T], Awaitable[None]]
