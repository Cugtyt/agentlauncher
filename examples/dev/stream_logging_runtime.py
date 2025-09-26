from agentlauncher.eventbus import EventBus
from agentlauncher.events import (
    MessageDeltaStreamingEvent,
    MessageDoneStreamingEvent,
    MessageStartStreamingEvent,
    ToolCallArgumentsDeltaStreamingEvent,
    ToolCallArgumentsDoneStreamingEvent,
    ToolCallNameStreamingEvent,
)
from agentlauncher.runtimes import RuntimeType


class StreamLoggingRuntime(RuntimeType):
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.subscribe(
            MessageStartStreamingEvent, self.handle_message_start_streaming_event
        )
        self.subscribe(
            MessageDeltaStreamingEvent, self.handle_message_delta_streaming_event
        )
        self.subscribe(
            MessageDoneStreamingEvent, self.handle_message_done_streaming_event
        )
        self.subscribe(
            ToolCallNameStreamingEvent, self.handle_tool_call_name_streaming_event
        )
        self.subscribe(
            ToolCallArgumentsDeltaStreamingEvent,
            self.handle_tool_call_arguments_delta_streaming_event,
        )
        self.subscribe(
            ToolCallArgumentsDoneStreamingEvent,
            self.handle_tool_call_arguments_done_streaming_event,
        )

    async def handle_message_start_streaming_event(
        self, event: MessageStartStreamingEvent
    ):
        print(f"[{event.agent_id}] ", end="", flush=True)

    async def handle_message_delta_streaming_event(
        self, event: MessageDeltaStreamingEvent
    ):
        print(f"{event.delta}", end="", flush=True)

    async def handle_message_done_streaming_event(
        self, event: MessageDoneStreamingEvent
    ):
        print()

    async def handle_tool_call_name_streaming_event(
        self, event: ToolCallNameStreamingEvent
    ):
        print(f"\n[{event.agent_id}] Tool call started: {event.tool_name}")

    async def handle_tool_call_arguments_delta_streaming_event(
        self, event: ToolCallArgumentsDeltaStreamingEvent
    ):
        print(f"{event.arguments_delta}", end="", flush=True)

    async def handle_tool_call_arguments_done_streaming_event(
        self, event: ToolCallArgumentsDoneStreamingEvent
    ):
        print()
