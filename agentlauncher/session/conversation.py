from abc import ABC, abstractmethod
from typing import Any

from agentlauncher.llm_interface import Message

type SessionContext = dict[str, Any]


class ConversationSession(ABC):
    @classmethod
    @abstractmethod
    def create(
        cls, session_context: SessionContext | None = None
    ) -> "ConversationSession": ...

    @abstractmethod
    async def load(self) -> list[Message]: ...

    @abstractmethod
    async def prepare_messages(
        self,
    ) -> None: ...

    @abstractmethod
    async def append(self, messages: list[Message]) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
