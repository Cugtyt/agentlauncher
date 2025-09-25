from dataclasses import dataclass

from agentlauncher.eventbus import EventType


@dataclass
class AgentLauncherRunEvent(EventType):
    task: str


@dataclass
class AgentLauncherStopEvent(EventType):
    result: str


@dataclass
class AgentLauncherShutdownEvent(EventType):
    pass


@dataclass
class AgentLauncherErrorEvent(EventType):
    error: str
