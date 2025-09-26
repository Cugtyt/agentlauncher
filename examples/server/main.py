import asyncio
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agentlauncher import AgentLauncher
from agentlauncher.eventbus.type import EventType
from examples.dev.gpt import gpt_handler
from examples.dev.helper import register_tools


def build_launcher() -> AgentLauncher:
    launcher = AgentLauncher()
    register_tools(launcher)

    launcher.set_primary_agent_llm_processor(gpt_handler)
    return launcher


app = FastAPI(title="AgentLauncher Server", version="0.1.0")
launcher = build_launcher()


class TaskRequest(BaseModel):
    task: str
    stream: bool = False


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _serialise_event(event: object) -> dict[str, Any]:
    if is_dataclass(event) and not isinstance(event, type):
        payload = asdict(event)
    else:
        payload = {
            key: value for key, value in vars(event).items() if not key.startswith("_")
        }
    return {
        "type": type(event).__name__,
        "payload": payload,
    }


async def _stream_task(task: str) -> StreamingResponse:
    queue: asyncio.Queue[EventType | None] = asyncio.Queue()

    asyncio.create_task(launcher.run(task=task, event_hook=queue))

    async def event_generator() -> AsyncIterator[str]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield type(item).__name__ + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/tasks", response_model=None)
async def submit_task(request: TaskRequest) -> Response:
    if request.stream:
        return await _stream_task(request.task)
    result = await launcher.run(task=request.task)
    return JSONResponse({"result": result})
