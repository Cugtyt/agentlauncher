from agentlauncher.llm_interface import Message

from .conversation import ConversationSession, SessionContext


class InMemoryConversationSession(ConversationSession):
    def __init__(self):
        self.messages: list[Message] = []

    @classmethod
    def create(
        cls, session_context: SessionContext | None = None
    ) -> "ConversationSession":
        return InMemoryConversationSession()

    async def load(self) -> list[Message]:
        return self.messages

    async def process(
        self,
    ) -> None:
        pass

    async def append(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
