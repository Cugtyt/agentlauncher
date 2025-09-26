from dataclasses import dataclass

from .bus import EventBus


@dataclass
class EventContext:
    agent_id: str
    event_bus: EventBus
