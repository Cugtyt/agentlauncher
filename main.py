import asyncio

from agentlauncher import (
    AgentLauncher,
    EventVerboseLevel,
)
from helper import register

test_task = """You are to help me organize a virtual conference. Please:
1. Find three suitable dates in the next month for the event.
2. Research and suggest two keynote speakers in AI.
3. Prepare a draft agenda with at least five sessions.
4. List three online platforms suitable for hosting the conference.
5. Estimate a budget for the event including speaker fees, platform costs and marketing.
6. Draft an invitation email for potential attendees.
7. Summarize all findings and provide a recommended plan of action.
Each step may require different tools or information sources. Provide a clear summary.
"""


async def main() -> None:
    launcher = AgentLauncher(
        verbose=EventVerboseLevel.SILENT,
    )

    register(launcher)
    await launcher.run(test_task)
    # for message in launcher.message_runtime.history:
    #     print(message)

    # while True:
    #     task = input("Enter your task (or 'exit' to quit): ")
    #     if task.lower() in ("exit", "quit"):
    #         break

    #     if task == '1':
    #         task = test_task

    #     result = await launcher.run(task=task)
        # print("Final Result:\n", result)
    await launcher.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
