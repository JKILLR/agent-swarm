"""Pydantic models for chat functionality."""

from typing import Optional, List
from pydantic import BaseModel


class ChatMessageModel(BaseModel):
    """Model for a single chat message."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    agent: Optional[str] = None
    thinking: Optional[str] = None


class ChatSession(BaseModel):
    """Model for a chat session."""
    id: str
    title: str
    swarm: Optional[str] = None
    created_at: str
    updated_at: str
    messages: List[ChatMessageModel] = []
