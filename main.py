import asyncio

from agentlauncher import (
    AgentLauncher,
    EventVerboseLevel,
)
from helper import register

test_task = """I need you to help me with several tasks:
1. Calculate what's 15 * 23 + 67
2. Check the weather in Tokyo and London
3. Convert 98.6°F to Celsius
4. Get the current time
5. Generate a random number between 50 and 150
6. Search the web for 'python programming tips'
7. Get stock prices for AAPL and TSLA
8. Analyze this text: 'The quick brown fox jumps over the lazy dog'
Please use the appropriate tools for each task and provide a summary."""


async def main() -> None:
    launcher = AgentLauncher(
        verbose=EventVerboseLevel.BASIC,
    )

    await register(launcher)
    await launcher.run(test_task)

    # while True:
    #     task = input("Enter your task (or 'exit' to quit): ")
    #     if task.lower() in ("exit", "quit"):
    #         break

    #     if task == '1':
    #         task = test_task

    #     result = await launcher.run(task=task)
    #     print("Final Result:\n", result)


if __name__ == "__main__":
    asyncio.run(main())
