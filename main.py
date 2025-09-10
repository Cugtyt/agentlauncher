import asyncio

from agentlauncher import (
    AgentLauncher,
    EventVerboseLevel,
)
from helper import register


async def main() -> None:
    launcher = AgentLauncher(
        verbose=EventVerboseLevel.DETAILED,
    )

    await register(launcher)
    result = await launcher.run(
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
    )
    print("Final Result:\n", result)


if __name__ == "__main__":
    asyncio.run(main())
