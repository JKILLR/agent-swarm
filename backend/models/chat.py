"""Pydantic models for chat functionality."""

from pydantic import BaseModel


class ChatMessageModel(BaseModel):
    """Model for a single chat message."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    agent: str | None = None
    thinking: str | None = None


class ChatSession(BaseModel):
    """Model for a chat session."""
    id: str
    title: str
    swarm: str | None = None
    created_at: str
    updated_at: str
    messages: list[ChatMessageModel] = []
