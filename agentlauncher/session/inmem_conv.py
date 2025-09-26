from agentlauncher.llm_interface import Message

from .conversation import ConversationSession, SessionContext


class InMemoryConversationSession(ConversationSession):
    def __init__(self) -> None:
        self.messages: list[Message] = []

    @classmethod
    def create(
        cls, session_context: SessionContext | None = None
    ) -> "ConversationSession":
        session = InMemoryConversationSession()
        session.messages = (
            session_context.get("messages", []) if session_context else []
        )
        return session

    async def load(self) -> list[Message]:
        return list(self.messages)

    async def prepare_messages(
        self,
    ) -> None:
        pass

    async def append(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    async def close(self) -> None:
        self.messages.clear()
