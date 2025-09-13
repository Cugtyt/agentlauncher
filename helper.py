import asyncio
import json
from typing import Any

from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from agentlauncher import (
    AgentLauncher,
)
from agentlauncher.events import (
    EventBus,
    MessageDeltaStreamingEvent,
    MessageDoneStreamingEvent,
    MessageStartStreamingEvent,
    ToolCallArgumentsDeltaStreamingEvent,
    ToolCallArgumentsDoneStreamingEvent,
    ToolCallNameStreamingEvent,
)
from agentlauncher.llm_interface import (
    AssistantMessage,
    ResponseMessageList,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)
client = AzureOpenAI(
    base_url="https://smarttsg-gpt.openai.azure.com/openai/v1/",
    azure_ad_token_provider=token_provider,
    api_version="preview",
)


def gpt_handler(
    messages: list[
        UserMessage
        | AssistantMessage
        | SystemMessage
        | ToolCallMessage
        | ToolResultMessage
    ],
    tools: list[ToolSchema],
) -> ResponseMessageList:
    def convert_message(
        message: UserMessage
        | AssistantMessage
        | SystemMessage
        | ToolCallMessage
        | ToolResultMessage,
    ) -> Any:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AssistantMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, ToolCallMessage):
            return ResponseFunctionToolCallParam(
                arguments=json.dumps(message.arguments),
                call_id=message.tool_call_id,
                name=message.tool_name,
                type="function_call",
            )
        elif isinstance(message, ToolResultMessage):
            return FunctionCallOutput(
                call_id=message.tool_call_id,
                output=message.result,
                type="function_call_output",
            )
        else:
            raise ValueError("Unknown message type")

    gpt_messages = [convert_message(message) for message in messages]
    gpt_tools = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]
    resp = client.responses.create(
        model="gpt-4.1",
        tools=gpt_tools,  # type: ignore
        input=gpt_messages,  # type: ignore
        tool_choice="auto",
    )
    result: ResponseMessageList = []
    for output in resp.output:
        if isinstance(output, ResponseFunctionToolCall):
            result.append(
                ToolCallMessage(
                    tool_call_id=output.call_id,
                    tool_name=output.name,
                    arguments=json.loads(output.arguments),
                )
            )
        elif isinstance(output, ResponseOutputMessage):
            result.append(
                AssistantMessage(content=output.content[0].text)  # type: ignore
            )

    return result


async def gpt_stream_handler(
    messages: list[
        UserMessage
        | AssistantMessage
        | SystemMessage
        | ToolCallMessage
        | ToolResultMessage
    ],
    tools: list[ToolSchema],
    agent_id: str,
    event_bus: EventBus,
) -> ResponseMessageList:
    def convert_message(
        message: UserMessage
        | AssistantMessage
        | SystemMessage
        | ToolCallMessage
        | ToolResultMessage,
    ) -> Any:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AssistantMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, ToolCallMessage):
            return ResponseFunctionToolCallParam(
                arguments=json.dumps(
                    {
                        "type": "object",
                        **message.arguments,
                    }
                ),
                call_id=message.tool_call_id,
                name=message.tool_name,
                type="function_call",
            )
        elif isinstance(message, ToolResultMessage):
            return FunctionCallOutput(
                call_id=message.tool_call_id,
                output=message.result,
                type="function_call_output",
            )
        else:
            raise ValueError("Unknown message type")

    gpt_messages = [convert_message(message) for message in messages]
    gpt_tools = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {
                        "type": value.type,
                        "description": value.description,
                    } | ({"items": value.items} if value.type == "array" else {})
                    for key, value in tool.parameters.items()
                },
                "required": [
                    key for key, value in tool.parameters.items() if value.required
                ],
            },
        }
        for tool in tools
    ]
    resp = client.responses.create(
        model="gpt-4.1",
        tools=gpt_tools,  # type: ignore
        input=gpt_messages,  # type: ignore
        tool_choice="auto",
        stream=True,
    )
    result: ResponseMessageList = []
    partial_tool_call: ToolCallMessage | None = None
    partial_tool_call_arguments: str | None = None
    partial_assistant_message: AssistantMessage | None = None
    for chunk in resp:
        if isinstance(chunk, ResponseOutputItemAddedEvent):
            if isinstance(chunk.item, ResponseOutputMessage):
                partial_assistant_message = AssistantMessage(content="")
                await event_bus.emit(MessageStartStreamingEvent(agent_id=agent_id))
            elif isinstance(chunk.item, ResponseFunctionToolCall):
                partial_tool_call = ToolCallMessage(
                    tool_call_id=chunk.item.call_id,
                    tool_name=chunk.item.name,
                    arguments={},
                )
                partial_tool_call_arguments = ""
                await event_bus.emit(
                    ToolCallNameStreamingEvent(
                        agent_id=agent_id,
                        tool_name=chunk.item.name,
                        tool_call_id=chunk.item.call_id,
                    )
                )
        elif isinstance(chunk, ResponseTextDeltaEvent):
            if partial_assistant_message is None:
                raise ValueError("Received text delta without starting message")
            partial_assistant_message.content += chunk.delta
            await event_bus.emit(
                MessageDeltaStreamingEvent(agent_id=agent_id, delta=chunk.delta)
            )
        elif isinstance(chunk, ResponseTextDoneEvent):
            if partial_assistant_message is None:
                raise ValueError("Received text done without starting message")
            result.append(partial_assistant_message)
            partial_assistant_message = AssistantMessage(content="")
            await event_bus.emit(
                MessageDoneStreamingEvent(
                    agent_id=agent_id, message=partial_assistant_message.content
                )
            )
        elif isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
            if partial_tool_call is None or partial_tool_call_arguments is None:
                raise ValueError(
                    "Received function call arguments delta without starting tool call"
                )
            partial_tool_call_arguments += chunk.delta
            await event_bus.emit(
                ToolCallArgumentsDeltaStreamingEvent(
                    agent_id=agent_id,
                    tool_call_id=partial_tool_call.tool_call_id,
                    arguments_delta=chunk.delta,
                )
            )
        elif isinstance(chunk, ResponseFunctionCallArgumentsDoneEvent):
            if partial_tool_call is None or partial_tool_call_arguments is None:
                raise ValueError(
                    "Received function call arguments done without starting tool call"
                )
            partial_tool_call.arguments = json.loads(partial_tool_call_arguments)
            result.append(partial_tool_call)
            await event_bus.emit(
                ToolCallArgumentsDoneStreamingEvent(
                    agent_id=agent_id,
                    tool_call_id=partial_tool_call.tool_call_id,
                    arguments=partial_tool_call_arguments,
                )
            )
            partial_tool_call = None
            partial_tool_call_arguments = None
    return result


async def register(launcher: AgentLauncher) -> None:
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

    @launcher.tool(
        name="get_weather",
        description="Get the current weather for a given location.",
        parameters={
            "location": {
                "type": "string",
                "description": "The location to get the weather for.",
                "required": True,
            },
        },
    )
    def get_weather_tool(location: str) -> str:
        return f"The weather in {location} is sunny with a high of 75°F."

    @launcher.tool(
        name="convert_temperature",
        description="Convert temperature from Fahrenheit to Celsius.",
        parameters={
            "fahrenheit": {
                "type": "number",
                "description": "Temperature in Fahrenheit.",
                "required": True,
            },
        },
    )
    def convert_temperature_tool(fahrenheit: float) -> str:
        celsius = (fahrenheit - 32) * 5.0 / 9.0
        return f"{fahrenheit}°F is {celsius:.2f}°C."

    @launcher.tool(
        name="get_current_time",
        description="Get the current date and time.",
        parameters={},
    )
    def get_current_time_tool() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @launcher.tool(
        name="generate_random_number",
        description="Generate a random integer between min and max.",
        parameters={
            "min": {
                "type": "integer",
                "description": "Minimum value.",
                "required": True,
            },
            "max": {
                "type": "integer",
                "description": "Maximum value.",
                "required": True,
            },
        },
    )
    def generate_random_number_tool(min: int, max: int) -> str:
        import random

        return str(random.randint(min, max))

    @launcher.tool(
        name="search_web",
        description="Search the web for a given query.",
        parameters={
            "query": {
                "type": "string",
                "description": "The search query.",
                "required": True,
            },
        },
    )
    async def search_web_tool(query: str) -> str:
        await asyncio.sleep(1)
        return f"Search results for '{query}': Example result 1, Example result 2."

    @launcher.tool(
        name="get_stock_price",
        description="Get the current stock price for a given ticker symbol.",
        parameters={
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol.",
                "required": True,
            },
        },
    )
    async def get_stock_price_tool(ticker: str) -> str:
        await asyncio.sleep(1)
        return f"The current price of {ticker} is $150.00."

    @launcher.tool(
        name="text_analysis",
        description="Analyze the text and provide word count.",
        parameters={
            "text": {
                "type": "string",
                "description": "The text to analyze.",
                "required": True,
            },
        },
    )
    def text_analysis_tool(text: str) -> str:
        word_count = len(text.split())
        return f"The text contains {word_count} words."

    @launcher.tool(
        name="find_dates",
        description="Suggest three suitable dates in the given month "
        "(format: YYYY-MM).",
        parameters={
            "month": {
                "type": "string",
                "description": "Month in YYYY-MM format.",
                "required": True,
            },
        },
    )
    def find_dates_tool(month: str) -> str:
        return f"Suggested dates: {month}-10, {month}-17, {month}-24."

    @launcher.tool(
        name="suggest_speakers",
        description="Suggest two keynote speakers for a given topic.",
        parameters={
            "topic": {
                "type": "string",
                "description": "The topic for keynote speakers.",
                "required": True,
            },
        },
    )
    def suggest_speakers_tool(topic: str) -> str:
        return f"Keynote speakers in {topic}: Dr. Alice Smith, Prof. Bob Lee."

    @launcher.tool(
        name="draft_agenda",
        description="Prepare a draft agenda with a given number of sessions.",
        parameters={
            "sessions": {
                "type": "integer",
                "description": "Number of sessions.",
                "required": True,
            },
        },
    )
    def draft_agenda_tool(sessions: int) -> str:
        agenda = "\n".join([f"Session {i + 1}: Topic TBD" for i in range(sessions)])
        return f"Draft agenda:\n{agenda}"

    @launcher.tool(
        name="list_platforms",
        description="List three online platforms suitable for hosting a conference.",
        parameters={},
    )
    def list_platforms_tool() -> str:
        return "Online platforms: Zoom, Microsoft Teams, Hopin."

    @launcher.tool(
        name="estimate_budget",
        description="Estimate a budget for the event, including speaker fees,"
        " platform costs, and marketing.",
        parameters={
            "speakers": {
                "type": "integer",
                "description": "Number of speakers.",
                "required": True,
            },
            "platform": {
                "type": "string",
                "description": "Platform name.",
                "required": True,
            },
            "marketing": {
                "type": "integer",
                "description": "Marketing budget in USD.",
                "required": True,
            },
        },
    )
    def estimate_budget_tool(speakers: int, platform: str, marketing: int) -> str:
        total = speakers * 1000 + 500 + marketing
        return (
            f"Estimated budget: Speaker fees ${speakers * 1000}, "
            f"Platform ({platform}) $500, Marketing ${marketing}, Total ${total}."
        )

    @launcher.tool(
        name="draft_email",
        description="Draft an invitation email for the event.",
        parameters={
            "event_name": {
                "type": "string",
                "description": "Name of the event.",
                "required": True,
            },
        },
    )
    def draft_email_tool(event_name: str) -> str:
        return (
            f"Subject: Invitation to {event_name}\n"
            "Dear Attendee,\nYou are invited to our virtual conference. "
            "More details to follow."
        )

    @launcher.main_agent_llm_handler(name="gpt-4")
    async def main_agent_handler(
        messages: list[
            UserMessage
            | AssistantMessage
            | SystemMessage
            | ToolCallMessage
            | ToolResultMessage
        ],
        tools: list[ToolSchema],
        agent_id: str,
        event_bus: EventBus,
    ) -> ResponseMessageList:
        return await gpt_stream_handler(messages, tools, agent_id, event_bus)

    @launcher.subscribe_event(MessageStartStreamingEvent)
    async def handle_message_start_streaming_event(event: MessageStartStreamingEvent):
        print(f"[{event.agent_id}] ", end="", flush=True)

    @launcher.subscribe_event(MessageDeltaStreamingEvent)
    async def handle_message_delta_streaming_event(event: MessageDeltaStreamingEvent):
        print(f"{event.delta}", end="", flush=True)

    @launcher.subscribe_event(MessageDoneStreamingEvent)
    async def handle_message_done_streaming_event(event: MessageDoneStreamingEvent):
        print()

    @launcher.subscribe_event(ToolCallNameStreamingEvent)
    async def handle_tool_call_name_streaming_event(event: ToolCallNameStreamingEvent):
        print(f"\n[{event.agent_id}] Tool call started: {event.tool_name}")

    @launcher.subscribe_event(ToolCallArgumentsDeltaStreamingEvent)
    async def handle_tool_call_arguments_delta_streaming_event(
        event: ToolCallArgumentsDeltaStreamingEvent,
    ):
        print(f"{event.arguments_delta}", end="", flush=True)

    @launcher.subscribe_event(ToolCallArgumentsDoneStreamingEvent)
    async def handle_tool_call_arguments_done_streaming_event(
        event: ToolCallArgumentsDoneStreamingEvent,
    ):
        print()
