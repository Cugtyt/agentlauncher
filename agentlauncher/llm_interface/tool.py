from dataclasses import dataclass


@dataclass
class ToolParamSchema:
    type: str
    description: str
    required: bool
    items: dict | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, ToolParamSchema]
