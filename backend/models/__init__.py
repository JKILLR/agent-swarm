"""Pydantic models for Agent Swarm API."""

from .requests import (
    SwarmCreate,
    ChatMessage,
    JobCreate,
    WorkCreateRequest,
    MessageSendRequest,
    HandoffRequest,
    EscalationCreateRequest,
    AgentExecuteRequest,
    SearchRequest,
    FetchRequest,
)
from .responses import (
    SwarmResponse,
    AgentInfo,
    HealthResponse,
)
from .chat import (
    ChatMessageModel,
    ChatSession,
)

__all__ = [
    # Requests
    "SwarmCreate",
    "ChatMessage",
    "JobCreate",
    "WorkCreateRequest",
    "MessageSendRequest",
    "HandoffRequest",
    "EscalationCreateRequest",
    "AgentExecuteRequest",
    "SearchRequest",
    "FetchRequest",
    # Responses
    "SwarmResponse",
    "AgentInfo",
    "HealthResponse",
    # Chat
    "ChatMessageModel",
    "ChatSession",
]
