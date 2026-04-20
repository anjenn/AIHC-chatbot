# Minimal local chatbot app for AIHC.

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.chatbot_pipeline import run_chatbot_pipeline
from src.prompts import STRUCTURED_INTAKE_PROMPT


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="AIHC Chatbot")

MODEL = os.environ.get("CHATBOT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str


def _latest_user_message(messages: list[Message]) -> str:
    # Pull the newest user message because the pipeline is one structured case at a time.
    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()
    return ""


def run_chat(messages: list[Message]) -> str:
    # Run the structured intake and retrieve-then-reason pipeline for the latest case.
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message:
        return STRUCTURED_INTAKE_PROMPT

    result = run_chatbot_pipeline(latest_user_message, model=MODEL)
    return str(result.get("reply", "")).strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": MODEL,
            "structured_intake_prompt": STRUCTURED_INTAKE_PROMPT,
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    try:
        reply = run_chat(req.messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not reply:
        raise HTTPException(
            status_code=502,
            detail="The diagnostic support pipeline returned an empty reply.",
        )
    return ChatResponse(reply=reply)
