from agentlauncher.event import (
    AgentCreateEvent,
    AgentFinishEvent,
    AgentRuntimeErrorEvent,
    AgentStartEvent,
    EventBus,
    LLMRequestEvent,
    LLMResponseEvent,
    ToolCall,
    ToolsExecRequestEvent,
    ToolsExecResultsEvent,
)
from agentlauncher.event.tool import ToolResult
from agentlauncher.llm import (
    AssistantMessage,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)


class Agent:
    def __init__(
        self,
        agent_id: str,
        task: str,
        conversation: list[
            UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage
        ],
        system_prompt: str,
        llm_handler_name: str,
        tool_schemas: list[ToolSchema],
        event_bus: EventBus,
    ):
        self.agent_id = agent_id
        self.task = task
        self.conversation = conversation
        self.system_prompt = system_prompt
        self.llm_handler_name = llm_handler_name
        self.tool_schemas = tool_schemas
        self.event_bus = event_bus

    async def start(self) -> None:
        await self.event_bus.emit(AgentStartEvent(agent_id=self.agent_id))
        self.conversation = [
            *self.conversation,
            UserMessage(content=self.task),
        ]
        message_list = [
            SystemMessage(content=self.system_prompt),
            *self.conversation,
        ]
        await self.event_bus.emit(
            LLMRequestEvent(
                agent_id=self.agent_id,
                messages=message_list,
                tool_schemas=self.tool_schemas,
                llm_handler_name=self.llm_handler_name,
            )
        )

    async def handle_llm_response(
        self, response: list[AssistantMessage | ToolCallMessage]
    ) -> None:
        self.conversation.extend(response)
        tool_calls = [
            ToolCall(msg.tool_call_id, msg.tool_name, msg.arguments)
            for msg in response
            if isinstance(msg, ToolCallMessage)
        ]
        if not tool_calls:
            final_response = "\n".join(
                msg.content for msg in response if isinstance(msg, AssistantMessage)
            )
            await self.event_bus.emit(
                AgentFinishEvent(agent_id=self.agent_id, result=final_response)
            )
            return
        await self.event_bus.emit(
            ToolsExecRequestEvent(agent_id=self.agent_id, tool_calls=tool_calls)
        )

    async def handle_tool_exec_result(self, tool_results: list[ToolResult]) -> None:
        self.conversation.extend(
            [
                ToolResultMessage(
                    tool_call_id=tr.tool_call_id,
                    tool_name=tr.tool_name,
                    result=tr.result,
                )
                for tr in tool_results
            ]
        )
        message_list = [
            SystemMessage(content=self.system_prompt),
            *self.conversation,
        ]
        await self.event_bus.emit(
            LLMRequestEvent(
                agent_id=self.agent_id,
                messages=message_list,
                tool_schemas=self.tool_schemas,
                llm_handler_name=self.llm_handler_name,
            )
        )


class AgentRuntime:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe(AgentCreateEvent, self.handle_agent_start)
        self.event_bus.subscribe(LLMResponseEvent, self.handle_llm_response)
        self.event_bus.subscribe(ToolsExecResultsEvent, self.handle_tool_exec_result)
        self.agents: dict[str, Agent] = {}

    async def handle_agent_start(self, event: AgentCreateEvent) -> None:
        if event.agent_id in self.agents:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent with this ID already exists.",
                )
            )
            return
        self.agents[event.agent_id] = Agent(
            agent_id=event.agent_id,
            task=event.task,
            conversation=event.conversation or [],
            system_prompt=event.system_prompt or "You are a helpful assistant.",
            event_bus=self.event_bus,
            llm_handler_name=event.llm_handler_name,
            tool_schemas=event.tool_schemas,
        )
        await self.agents[event.agent_id].start()

    async def handle_llm_response(self, event: LLMResponseEvent) -> None:
        agent = self.agents.get(event.agent_id)
        if not agent:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found for LLM response.",
                )
            )
            return

        await agent.handle_llm_response(event.response)

    async def handle_tool_exec_result(self, event: ToolsExecResultsEvent) -> None:
        agent = self.agents.get(event.agent_id)
        if not agent:
            await self.event_bus.emit(
                AgentRuntimeErrorEvent(
                    agent_id=event.agent_id,
                    error="Agent not found for tool execution result.",
                )
            )
            return
        await agent.handle_tool_exec_result(event.tool_results)
