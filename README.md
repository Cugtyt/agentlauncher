# AgentLauncher

AgentLauncher is an event-driven, multi-agent framework for solving complex tasks by dynamically generating sub-agents.
The main agent coordinates strategy, while sub-agents handle specialized tasks.
Agent lifecycles are managed automatically, similar to jobs in Kubernetes—sub-agents are lightweight and ephemeral.


## How It Works

1. **Task Initialization**: Launching AgentLauncher with a task triggers a `TaskCreateEvent`.
2. **Agent Creation**: `AgentRuntime` responds and triggers `AgentCreateEvent`.
3. **Main Agent Management**: `AgentManager` creates the main agent (if needed) and triggers `AgentStartEvent`.
4. **Agent Startup**: The agent logs the initial message (`MessageAddEvent`) and requests an LLM response (`LLMRequestEvent`).
5. **LLM Interaction**: `LLMRuntime` calls the LLM API, triggering `LLMResponseEvent`.
6. **Message Handling**: The agent logs the LLM response and may request tool execution (`ToolsExecRequestEvent`).
7. **Tool Execution**: `ToolRuntime` executes tools, triggering `ToolExecStartEvent`, `ToolExecFinishEvent`, and `ToolsExecResultsEvent`.
8. **Result Processing**: The agent processes tool results and may request further LLM responses, repeating the cycle.
9. **Task Completion**: The agent triggers `AgentFinishEvent`. `AgentRuntime` marks completion, triggers `TaskFinishEvent`, and cleans up sub-agents.
10. **Final Output**: AgentLauncher returns the result. New tasks restart the flow.


## Features

- **Multi-Agent System**: Main agent delegates to sub-agents; standardized lifecycle and interactions.
- **Event-Driven Architecture**: All agent behavior is event-based.
- **Dynamic Agent Management**: Agents spawn sub-agents for specialized tasks; lifecycles are automatic.
- **Modular & Extensible**: Easily add new tools and LLM handlers by subscribing to events.
- **Fully Asynchronous**: Built on `asyncio` for efficient, non-blocking event handling.



## Example Usage

See `examples/dev/main.py` for a usage example. The snippet below mirrors the explicit registration flow used in the repo:

```python
import asyncio

from agentlauncher import AgentLauncher
from agentlauncher.events import MessageDeltaStreamingEvent
from agentlauncher.llm_interface import ToolParamSchema
from my_handlers import my_llm_handler


async def main() -> None:
    launcher = AgentLauncher()

    def calculate_tool(a: int, b: int, c: int) -> str:
        return str(a * b + c)

    launcher.register_tool(
        name="calculate",
        function=calculate_tool,
        description="Calculate the result of a * b + c.",
        parameters={
            "a": ToolParamSchema(type="integer", description="Multiplicand", required=True),
            "b": ToolParamSchema(type="integer", description="Multiplier", required=True),
            "c": ToolParamSchema(type="integer", description="Addend", required=True),
        },
    )

    launcher.set_primary_agent_llm_processor(my_llm_handler)
    # Optionally set a distinct processor for spawned sub-agents
    # launcher.set_sub_agent_llm_processor(my_sub_agent_handler)

    @launcher.subscribe_event(MessageDeltaStreamingEvent)
    async def handle_message_delta_streaming_event(event: MessageDeltaStreamingEvent):
        print(event.delta, end="", flush=True)

    # Register any extra runtimes that should react to events
    # launcher.register_runtime(MyCustomRuntime)

    result = await launcher.run("Book a trip to Tokyo in April")
    print("Final Result:\n", result)


if __name__ == "__main__":
    asyncio.run(main())
```

### FastAPI demo server

The `examples/server/main.py` module exposes an HTTP API powered by FastAPI. It
accepts task submissions and can either return the final result or stream live
events from the agent workflow using newline-delimited JSON. Run it locally with:

```bash
uv run uvicorn examples.server.main:app --reload
```

Use `POST /tasks` with a JSON payload such as `{"task": "Plan a team offsite"}`
to receive the final answer, or add `"stream": true` to opt into streaming
events.