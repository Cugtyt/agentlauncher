import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentlauncher.eventbus import EventBus
from agentlauncher.llm_interface import (
    AssistantMessage,
    ResponseMessageList,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)

MockMessage = (
    UserMessage | AssistantMessage | SystemMessage | ToolCallMessage | ToolResultMessage
)


def _last_user_message_content(messages: list[MockMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message.content
    return None


def _mock_value_for_type(param_type: str, name: str) -> Any:
    if param_type == "string":
        return f"mock-{name}"
    if param_type in {"integer", "number"}:
        return random.randint(1, 100)
    if param_type == "boolean":
        return random.choice([True, False])
    if param_type == "array":
        return [f"mock-item-for-{name}"]
    if param_type == "object":
        return {"example": f"mock-{name}-value"}
    return f"mock-{name}"


def _mock_arguments_for_tool(tool: ToolSchema) -> dict[str, Any]:
    return {
        name: _mock_value_for_type(schema.type, name)
        for name, schema in tool.parameters.items()
    }


def _build_tool_arguments(
    tool: ToolSchema, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    arguments = _mock_arguments_for_tool(tool)
    if overrides:
        arguments.update(overrides)
    return arguments


def _make_tool_call(tool_name: str, arguments: dict[str, Any]) -> ToolCallMessage:
    return ToolCallMessage(
        tool_call_id=f"mock-tool-{uuid.uuid4()}",
        tool_name=tool_name,
        arguments=arguments,
    )


@dataclass
class MockSession:
    responses: list[ResponseMessageList]
    cursor: int = 0

    def next_response(self) -> ResponseMessageList:
        if self.cursor >= len(self.responses):
            return [
                AssistantMessage(
                    content=(
                        "Mock conversation already concluded. Let me know if you "
                        "need to start over."
                    )
                )
            ]

        response = self.responses[self.cursor]
        self.cursor += 1
        return response

    def is_complete(self) -> bool:
        return self.cursor >= len(self.responses)


ToolMap = dict[str, ToolSchema]
ScenarioBuilder = Callable[[ToolMap, str], list[ResponseMessageList]]


def _truncate_topic(topic: str, limit: int = 32) -> str:
    topic = topic.strip()
    if len(topic) <= limit:
        return topic
    return topic[: limit - 3] + "..."


def _build_conference_scenario(
    tool_map: ToolMap, topic: str
) -> list[ResponseMessageList]:
    month = f"2025-{random.randint(1, 12):02d}"
    speakers = random.randint(2, 3)
    marketing = random.choice([1200, 1500, 1800])
    sessions = random.randint(4, 6)
    platform = random.choice(["Zoom", "Microsoft Teams", "Hopin"])
    find_dates_tool = tool_map["find_dates"]
    suggest_speakers_tool = tool_map["suggest_speakers"]
    draft_agenda_tool = tool_map["draft_agenda"]
    list_platforms_tool = tool_map["list_platforms"]
    estimate_budget_tool = tool_map["estimate_budget"]
    draft_email_tool = tool_map["draft_email"]

    return [
        [
            _make_tool_call(
                find_dates_tool.name,
                {"month": month},
            )
        ],
        [
            _make_tool_call(
                suggest_speakers_tool.name,
                {"topic": topic},
            )
        ],
        [
            _make_tool_call(
                draft_agenda_tool.name,
                {"sessions": sessions},
            ),
            _make_tool_call(list_platforms_tool.name, {}),
        ],
        [
            _make_tool_call(
                estimate_budget_tool.name,
                {
                    "speakers": speakers,
                    "platform": platform,
                    "marketing": marketing,
                },
            )
        ],
        [
            _make_tool_call(
                draft_email_tool.name,
                {
                    "event_name": f"{_truncate_topic(topic, 20)} Summit",
                },
            )
        ],
        [
            AssistantMessage(
                content=(
                    "Conference plan ready! I've scheduled tentative dates in "
                    f"{month}, requested speaker suggestions, drafted an agenda "
                    f"with {sessions} sessions, and estimated costs using "
                    f"{platform}. Check the tool results for details."
                )
            )
        ],
    ]


def _build_weather_scenario(tool_map: ToolMap, topic: str) -> list[ResponseMessageList]:
    location = (
        topic.split(" in ")[-1].strip()
        if " in " in topic
        else random.choice(["Seattle", "Austin", "Berlin", "Tokyo"])
    )
    fahrenheit = random.randint(60, 90)
    weather_tool = tool_map["get_weather"]
    convert_tool = tool_map["convert_temperature"]
    return [
        [
            _make_tool_call(
                weather_tool.name,
                {"location": location},
            )
        ],
        [
            _make_tool_call(
                convert_tool.name,
                {"fahrenheit": float(fahrenheit)},
            ),
            AssistantMessage(
                content="Converting that forecast into Celsius for clarity."
            ),
        ],
        [
            AssistantMessage(
                content=(
                    f"Here's the weather outlook for {location}. I've also provided "
                    "the Celsius conversion so you can compare easily."
                )
            )
        ],
    ]


def _build_research_scenario(
    tool_map: ToolMap, topic: str
) -> list[ResponseMessageList]:
    query = f"Key facts about {topic}" if topic else "Interesting AI updates"
    analysis_text = f"Summary request: {topic}" if topic else "General research summary"
    search_tool = tool_map["search_web"]
    analysis_tool = tool_map["text_analysis"]
    random_tool = tool_map["generate_random_number"]
    return [
        [
            _make_tool_call(
                search_tool.name,
                {"query": query},
            )
        ],
        [
            _make_tool_call(
                analysis_tool.name,
                {"text": analysis_text},
            )
        ],
        [
            _make_tool_call(
                random_tool.name,
                {"min": 100, "max": 999},
            )
        ],
        [
            AssistantMessage(
                content=(
                    "Research complete. I gathered highlights, analyzed the text, "
                    "and tagged the findings with a reference code from the random "
                    "number generator."
                )
            )
        ],
    ]


def _build_calculation_scenario(
    tool_map: ToolMap, topic: str
) -> list[ResponseMessageList]:
    a, b, c = random.randint(2, 9), random.randint(2, 9), random.randint(1, 5)
    expression = f"{a} * {b} + {c}"
    calculate_tool = tool_map["calculate"]
    return [
        [
            _make_tool_call(
                calculate_tool.name,
                {"a": a, "b": b, "c": c},
            )
        ],
        [
            AssistantMessage(
                content=(
                    f"The calculation {expression} is complete. Check the tool "
                    "result for the numeric value, and let me know if you'd like "
                    "me to apply it elsewhere."
                )
            )
        ],
    ]


def _build_generic_tool_scenario(
    tool_map: ToolMap, topic: str
) -> list[ResponseMessageList]:
    if tool_map:
        tool = random.choice(list(tool_map.values()))
        arguments = _build_tool_arguments(tool)
        return [
            [_make_tool_call(tool.name, arguments)],
            [
                AssistantMessage(
                    content=(
                        f"I invoked {tool.name} using mock inputs to progress on "
                        f"{topic or 'the current request'}. Review the tool output "
                        "for details."
                    )
                )
            ],
        ]

    explanation = topic or "your latest request"
    return [
        [
            AssistantMessage(
                content=(
                    f"Starting a mock reasoning pass about {explanation}. I'll "
                    "keep this conversation short and sweet."
                )
            )
        ],
        [
            AssistantMessage(
                content=(
                    "All set! Nothing else is required on my side, but feel free "
                    "to ask for another mock run."
                )
            )
        ],
    ]


SCENARIO_DEFINITIONS: list[tuple[set[str], ScenarioBuilder]] = [
    (
        {
            "find_dates",
            "suggest_speakers",
            "draft_agenda",
            "list_platforms",
            "estimate_budget",
            "draft_email",
        },
        _build_conference_scenario,
    ),
    ({"get_weather", "convert_temperature"}, _build_weather_scenario),
    (
        {"search_web", "text_analysis", "generate_random_number"},
        _build_research_scenario,
    ),
    ({"calculate"}, _build_calculation_scenario),
]


_mock_sessions: dict[str, MockSession] = {}


def _choose_scenario(tool_map: ToolMap, topic: str) -> list[ResponseMessageList]:
    tool_names = set(tool_map.keys())
    candidates = [
        builder
        for required, builder in SCENARIO_DEFINITIONS
        if required.issubset(tool_names)
    ]

    if candidates:
        builder = random.choice(candidates)
        return builder(tool_map, topic)

    return _build_generic_tool_scenario(tool_map, topic)


async def gpt_mock_handler(
    messages: list[MockMessage],
    tools: list[ToolSchema],
    agent_id: str,
    event_bus: EventBus,
) -> ResponseMessageList:

    last_user_content = _last_user_message_content(messages) or "your request"
    tool_map: ToolMap = {tool.name: tool for tool in tools}

    session = _mock_sessions.get(agent_id)
    if session is None or session.is_complete():
        scenario_responses = _choose_scenario(tool_map, last_user_content)
        session_responses: list[ResponseMessageList]
        if scenario_responses:
            session_responses = scenario_responses
        else:
            fallback_response: ResponseMessageList = [
                AssistantMessage(
                    content=(
                        "Mock handler fallback response: no scenario matched, but "
                        "I'm acknowledging your request."
                    )
                )
            ]
            session_responses = [fallback_response]
        session = MockSession(responses=session_responses)
        _mock_sessions[agent_id] = session

    response = session.next_response()
    if session.is_complete():
        _mock_sessions.pop(agent_id, None)

    return response
