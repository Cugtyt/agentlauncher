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

from agentlauncher.eventbus import EventBus
from agentlauncher.events import (
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
            "parameters": {
                "type": "object",
                "properties": {
                    key: {
                        "type": value.type,
                        "description": value.description,
                    }
                    | ({"items": value.items} if value.type == "array" else {})
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
                    }
                    | ({"items": value.items} if value.type == "array" else {})
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
