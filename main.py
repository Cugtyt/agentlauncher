import asyncio

from agentlauncher import (
    AgentLauncher,
)
from helper import register


async def main() -> None:
    launcher = AgentLauncher(verbose=True)

    await register(launcher)
    await launcher.run(
        task=(
            "I need you to help me with several tasks:\n"
            "1. Calculate what's 15 * 23 + 67\n"
            "2. Check the weather in Tokyo and London\n"
            "3. Convert 98.6°F to Celsius\n"
            "4. Get the current time\n"
            "5. Generate a random number between 50 and 150\n"
            "6. Search the web for 'python programming tips'\n"
            "7. Get stock prices for AAPL and TSLA\n"
            "8. Analyze this text: 'The quick brown fox jumps over the lazy dog'\n"
            "Please use the appropriate tools for each task and provide a summary."
        ),
        system_prompt=(
            "You are a helpful AI assistant with access to various tools. "
            "For each user request, think about which tools you need and call them efficiently. "  # noqa: E501
            "You can call multiple tools at once when appropriate. "
            "Always explain your reasoning and provide clear, organized results."
        ),
        tool_names=[
            "calculate",
            "get_weather",
            "convert_temperature",
            "get_current_time",
            "generate_random_number",
            "search_web",
            "get_stock_price",
            "text_analysis",
        ],
        llm_handler_name="gpt-4",
    )
    print(launcher.final_result)


if __name__ == "__main__":
    asyncio.run(main())
