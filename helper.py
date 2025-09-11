import asyncio
import json
from typing import Any

from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseOutputMessage,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from agentlauncher import (
    AgentLauncher,
)
from agentlauncher.llm import (
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


def calculate(a: int, b: int, c: int) -> str:
    return str(a * b + c)


def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny with a high of 75°F."


def convert_temperature(fahrenheit: float) -> str:
    celsius = (fahrenheit - 32) * 5.0 / 9.0
    return f"{fahrenheit}°F is {celsius:.2f}°C."


def get_current_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def search_web(query: str) -> str:
    await asyncio.sleep(1)
    return f"Search results for '{query}': Example result 1, Example result 2."


def generate_random_number(min: int, max: int) -> str:
    import random

    return str(random.randint(min, max))


async def get_stock_price(ticker: str) -> str:
    await asyncio.sleep(1)
    return f"The current price of {ticker} is $150.00."


def text_analysis(text: str) -> str:
    word_count = len(text.split())
    return f"The text contains {word_count} words."


async def register(launcher: AgentLauncher) -> None:
    await launcher.register_tool(
        name="calculate",
        function=calculate,
        description="Calculate the result of the expression a * b + c.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first integer."},
                "b": {"type": "integer", "description": "The second integer."},
                "c": {"type": "integer", "description": "The third integer."},
            },
            "required": ["a", "b", "c"],
        },
    )

    await launcher.register_tool(
        name="get_weather",
        function=get_weather,
        description="Get the current weather for a given location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for.",
                },
            },
            "required": ["location"],
        },
    )

    await launcher.register_tool(
        name="convert_temperature",
        function=convert_temperature,
        description="Convert temperature from Fahrenheit to Celsius.",
        parameters={
            "type": "object",
            "properties": {
                "fahrenheit": {
                    "type": "number",
                    "description": "Temperature in Fahrenheit.",
                },
            },
            "required": ["fahrenheit"],
        },
    )

    await launcher.register_tool(
        name="get_current_time",
        function=get_current_time,
        description="Get the current date and time.",
        parameters={
            "type": "object",
            "properties": {},
        },
    )

    await launcher.register_tool(
        name="generate_random_number",
        function=generate_random_number,
        description="Generate a random integer between min and max.",
        parameters={
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "Minimum value."},
                "max": {"type": "integer", "description": "Maximum value."},
            },
            "required": ["min", "max"],
        },
    )

    await launcher.register_tool(
        name="search_web",
        function=search_web,
        description="Search the web for a given query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    )

    await launcher.register_tool(
        name="get_stock_price",
        function=get_stock_price,
        description="Get the current stock price for a given ticker symbol.",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol."},
            },
            "required": ["ticker"],
        },
    )

    await launcher.register_tool(
        name="text_analysis",
        function=text_analysis,
        description="Analyze the text and provide word count.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze."},
            },
            "required": ["text"],
        },
    )

    await launcher.register_main_agent_llm_handler(
        name="gpt-4",
        function=gpt_handler,
    )
