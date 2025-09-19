import asyncio

from agentlauncher import (
    AgentLauncher,
)
from agentlauncher.llm_interface import (
    AssistantMessage,
    Message,
)


def register_tools(launcher: AgentLauncher) -> None:
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


def register_message_handlers(launcher: AgentLauncher) -> None:
    @launcher.message_handler()
    async def count_words_handler(
        messages: list[Message],
    ) -> list[Message]:
        for message in messages:
            if isinstance(message, AssistantMessage):
                word_count = len(message.content.split())
                print(f"{'>' * 20} (Word count: {word_count})")
        return messages

    @launcher.conversation_handler()
    async def trim_conversation_handler(
        conversation: list[Message],
    ) -> list[Message]:
        max_messages = 10
        if len(conversation) > max_messages and isinstance(
            conversation[-1], AssistantMessage
        ):
            print(f"{'>' * 20} Trimming conversation to {max_messages} messages.")
            return conversation[-max_messages:]
        return conversation
