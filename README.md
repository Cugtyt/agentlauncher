# AgentLauncher

AgentLauncher is a high-performance, stateless agent runtime framework that standardizes agent lifecycle management through event-driven orchestration. Built on Python asyncio with uvloop acceleration, it handles many concurrent task requests at full speed while maintaining responsive event processing. The framework accepts tasks and triggers the system to provision agents with a primary-agent-first pattern—optionally spawning ephemeral sub-agents to handle well-bounded subtasks and reduce primary agent workload. Its stateless design enables horizontal scaling across multiple servers for distributed deployments. The framework is fully extensible: listen to any lifecycle event, extend the event system, or integrate custom tools to support diverse scenarios.

## Architecture & core features

- **Stateless runtime framework** – accepts tasks and provisions agents on-demand with standardized lifecycle management, no persistent state between invocations.
- **Event-driven orchestration** – every lifecycle milestone emits structured events on a shared bus, enabling observability, auditing, custom workflows, or external system integration.
- **Primary agent + ephemeral helpers** – the main agent retains task ownership while the framework opportunistically provisions helper agents for contained subtasks, reducing primary agent cognitive load.
- **Extensible event system** – listen to any built-in event, emit custom events, or extend the event schema to support domain-specific orchestration patterns.
- **Pluggable tool ecosystem** – register domain-specific tools, swap LLM processors, or integrate custom runtimes without modifying the core framework.
- **High-performance async runtime** – built on Python asyncio with uvloop acceleration to handle concurrent task requests at scale, enabling full-speed event processing across many simultaneous agent workflows.
- **Horizontally scalable** – stateless design allows deployment across multiple servers or containers for distributed workloads and load balancing.

### Standardized lifecycle

1. **Task acceptance** – `launcher.run` emits `TaskCreateEvent`, triggering the runtime to provision agents with the specified system prompt, conversation history, and available tool schemas.
2. **Primary agent instantiation** – `AgentRuntime` creates the main agent and announces it with `AgentStartEvent`.
3. **Sub-agent provisioning (optional)** – when the primary agent determines that subtasks can reduce its workload, the framework spawns helper agents, tracks their progress through intermediate events, and retires them upon completion.
4. **Agent-runtime interaction** – agents emit `MessagesAddEvent` for prompts, then request completions via `LLMRequestEvent`.
5. **LLM processing** – `LLMRuntime` dispatches to the configured processor, emitting streaming events such as `MessageStartStreamingEvent` and `MessageDoneStreamingEvent`.
6. **Tool execution** – when agents invoke tools, `ToolRuntime` issues `ToolExecStartEvent`, `ToolExecFinishEvent`, and `ToolsExecResultsEvent` before routing results back to the requesting agent.
7. **Task completion** – once the primary agent completes its work, it publishes `AgentFinishEvent`; `TaskFinishEvent` signals final resolution and unblocks the caller.

The event contracts live under `agentlauncher/events`, while runtime implementations sit in `agentlauncher/runtimes`. Hooks and subscribers receive the same event objects.

### Key event types
- **Agent lifecycle**: `AgentStartEvent`, `AgentFinishEvent`, `AgentDeletedEvent`
- **Task management**: `TaskCreateEvent`, `TaskFinishEvent`, `TaskCancelEvent`  
- **LLM interaction**: `LLMRequestEvent`, `LLMResponseEvent`, `MessageDeltaStreamingEvent`
- **Tool execution**: `ToolExecStartEvent`, `ToolExecFinishEvent`, `ToolsExecResultsEvent`

## Installation & setup

**Prerequisites:**
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

**Install:**
```bash
git clone https://github.com/Cugtyt/agentlauncher.git
cd agentlauncher
uv sync
```

**LLM Configuration:**
The framework requires an LLM processor. The included examples use Azure OpenAI via `DefaultAzureCredential`:
- Set `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET` environment variables, or
- Use managed identity with Azure OpenAI access
- Update the endpoint in `examples/dev/gpt.py` if needed

For other LLM providers, implement the `LLMProcessor` interface (see `examples/dev/mock_gpt.py` for reference).

## Examples

### FastAPI server demo

The demo server (`examples/server/main.py`) demonstrates production-ready HTTP integration. It provides both synchronous and streaming endpoints:

```bash
uv run uvicorn examples.server.main:app --reload
```

Smoke-test the endpoints from another terminal:

```bash
curl -sS http://127.0.0.1:8000/health

curl -sS -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Plan a team offsite"}'

curl -sS -N -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Plan a team offsite", "stream": true}'
```

**Streaming response format:**
```
AgentStartEvent
MessagesAddEvent  
MessageStartStreamingEvent
MessageDoneStreamingEvent
ToolExecStartEvent
ToolExecFinishEvent
AgentFinishEvent
TaskFinishEvent
```

The server passes an `asyncio.Queue` directly to `launcher.run` as the `event_hook`, making it easy to adapt for WebSockets, Server-Sent Events, or other real-time transports.

### Direct integration

The `examples/dev` folder provides reusable building blocks. Here's a minimal integration with event streaming:

```python
import asyncio

from agentlauncher import AgentLauncher
from agentlauncher.eventbus.type import EventType
from examples.dev.gpt import gpt_handler
from examples.dev.helper import register_tools


async def stream_events(queue: asyncio.Queue[EventType | None]) -> None:
    while (event := await queue.get()) is not None:
        print(event.__class__.__name__)


async def main() -> None:
    launcher = AgentLauncher()
    register_tools(launcher)
    launcher.set_primary_agent_llm_processor(gpt_handler)

    events: asyncio.Queue[EventType | None] = asyncio.Queue()
    consumer = asyncio.create_task(stream_events(events))

    result = await launcher.run(
        task="Plan a team offsite",
        event_hook=events,
    )

    await consumer
    print("Final result:", result)


if __name__ == "__main__":
    asyncio.run(main())
```

Passing a queue as the `event_hook` lets you reuse the same consumption pattern as the FastAPI example. The launcher automatically pushes `None` when the task finishes, so consumers can exit cleanly.

## Extension patterns

### Custom tools
```python
@launcher.tool(
    name="my_tool", 
    description="My custom tool",
    parameters={"param": {"type": "string", "required": True}}
)
def my_tool(param: str, ctx: EventContext) -> str:
    return f"Processed: {param}"
```

### Event listeners
```python
@launcher.subscribe_event(AgentStartEvent)
async def on_agent_start(event: AgentStartEvent):
    print(f"Agent {event.agent_id} started")
```

### Custom LLM processors
```python
async def my_llm_processor(
    messages: list[Message], 
    tools: list[ToolSchema], 
    context: EventContext
) -> list[Message]:
    # Your LLM integration logic
    return [AssistantMessage(content="Response")]

launcher.set_primary_agent_llm_processor(my_llm_processor)
```

### Conversation middleware
```python
@launcher.conversation_processor()
async def conversation_filter(conversation: list[Message]) -> list[Message]:
    # Filter, transform, or log conversation history
    return conversation[-10:]  # Keep last 10 messages
```
