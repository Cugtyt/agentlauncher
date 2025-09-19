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

See `examples/main.py` for a usage example. Register tools and LLM handlers, then run tasks interactively:

```python
from agentlauncher import AgentLauncher
from agentlauncher.events import (
    EventBus,
    MessageDeltaStreamingEvent,
)

launcher = AgentLauncher()

@launcher.tool(
    name="calculate",
    description="Calculate the result of the expression a * b + c.",
    parameters={
        "a": {
            "type": "integer",
            "description": "The first integer.",
            "required": True,
        },
        "b": {
            "type": "integer",
            "description": "The second integer.",
            "required": True,
        },
        "c": {
            "type": "integer",
            "description": "The third integer.",
            "required": True,
        },
    },
)
def calculate_tool(a: int, b: int, c: int) -> str:
    return str(a * b + c)

launcher.register_main_agent_llm_handler(your_llm_function)
launcher.register_sub_agent_llm_handler(your_llm_function) # optional

# you can subscribe any events you want
@launcher.subscribe_event(MessageDeltaStreamingEvent)
async def handle_message_delta_streaming_event(event: MessageDeltaStreamingEvent):
    print(f"{event.delta}", end="", flush=True)

# for complex scenarios, you can define your own runtime to handle multiple events
launcher.register_runtime(StreamLoggingRuntime)

while True:
    task = input("Enter your task (or 'exit' to quit): ")
    if task.lower() == "exit":
        break
    result = await launcher.run(task)
    print(f"Final Result: {result}")
```