import asyncio
import logging

from gpt import gpt_handler
from helper import register_conversation_counter, register_tools
from stream_logging_runtime import StreamLoggingRuntime

from agentlauncher import AgentLauncher
from agentlauncher.events import (
    MessagesAddEvent,
)
from agentlauncher.llm_interface import (
    AssistantMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)

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
    launcher = AgentLauncher()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("agentlauncher").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    register_tools(launcher)
    # register_message_handlers(launcher)
    launcher.set_primary_agent_llm_processor(gpt_handler)
    launcher.register_runtime(StreamLoggingRuntime)
    register_conversation_counter(launcher)

    @launcher.subscribe_event(MessagesAddEvent)
    async def handle_messages_add_event(event: MessagesAddEvent):
        for message in event.messages:
            if isinstance(message, UserMessage):
                logger.info(f"[{event.agent_id}] User: {message.content}")
            elif isinstance(message, AssistantMessage):
                logger.info(f"[{event.agent_id}] Assistant: {message.content}")
            elif isinstance(message, ToolCallMessage):
                logger.info(
                    f"[{event.agent_id}] ToolCall: {message.tool_name} "
                    f"with arguments {message.arguments}"
                )
            elif isinstance(message, ToolResultMessage):
                logger.info(
                    f"[{event.agent_id}] ToolResult: {message.tool_name} "
                    f"result: {message.result}"
                )

    tasks = [asyncio.create_task(launcher.run(test_task)) for _ in range(3)]
    results = await asyncio.gather(*tasks)

    # for result in results:
    #     print("Result:\n", result)

    # for message in launcher.message_runtime.history:
    #     print(message)

    # while True:
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
    # await launcher.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
