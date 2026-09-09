import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import ask_agent


app = FastAPI(
    title="Mini AI Agent API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None


@app.get("/api")
def home():
    return {
        "status": "online",
        "name": "Mini AI Agent",
        "message": "Mini AI Agent API is running successfully."
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = (request.question or "").strip()

    if not question:
        return {
            "success": False,
            "answer": "Please enter a question."
        }

    try:
        result = ask_agent(
            question=question,
            chat_id=request.chat_id
        )

        return {
            "success": True,
            "answer": result.get("answer", ""),
            "tool": result.get("tool", "general"),
            "tool_name": result.get(
                "tool_name",
                "🤖 Gemini AI"
            ),
            "source": result.get("source", ""),
            "sources": result.get("sources", []),
            "chat_id": result.get("chat_id")
        }

    except Exception as error:
        return {
            "success": False,
            "answer": "❌ Agent error: " + str(error)
        }