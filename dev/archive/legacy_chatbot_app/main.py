# Minimal chatbot backend with FastAPI and Anthropic streaming.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="Minimal Chatbot")

SYSTEM_PROMPT = "You are a helpful assistant. Be concise and clear."
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = 1024

api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key) if api_key else None


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


def sse_payload(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_name": MODEL},
    )


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set. Add it before starting the chatbot.",
        )

    messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]

    def stream() -> Iterator[str]:
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as response_stream:
                for text in response_stream.text_stream:
                    yield sse_payload({"text": text})
        except Exception as exc:  # pragma: no cover - network/API failure path
            yield sse_payload({"error": str(exc)})

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
