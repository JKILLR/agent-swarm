"""Pydantic models for API request bodies."""

from typing import Optional, Dict
from pydantic import BaseModel


class SwarmCreate(BaseModel):
    """Request to create a new swarm."""
    name: str
    description: str = ""
    template: str = "_template"


class ChatMessage(BaseModel):
    """Request to send a chat message."""
    message: str
    swarm: Optional[str] = None


class JobCreate(BaseModel):
    """Request to create a background job."""
    type: str = "chat"  # "chat", "swarm_directive", "task"
    prompt: str
    swarm: Optional[str] = None
    session_id: Optional[str] = None


class WorkCreateRequest(BaseModel):
    """Request to create a work item."""
    title: str
    description: str
    work_type: str = "task"
    priority: str = "medium"
    parent_id: Optional[str] = None
    swarm_name: Optional[str] = None
    context: Optional[Dict] = None


class MessageSendRequest(BaseModel):
    """Request to send a message to an agent."""
    from_agent: str
    to_agent: str
    subject: str
    body: str
    message_type: str = "request"  # request, response, notification, handoff, escalation
    priority: str = "normal"  # low, normal, high, urgent
    thread_id: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Optional[Dict] = None


class HandoffRequest(BaseModel):
    """Request for structured agent handoff."""
    from_agent: str
    to_agent: str
    task_summary: str
    completed_work: str
    remaining_work: str
    context: Optional[Dict] = None
    priority: str = "normal"


class EscalationCreateRequest(BaseModel):
    """Request to create an escalation."""
    from_level: str  # "agent", "coo"
    to_level: str  # "coo", "ceo"
    reason: str  # "stuck", "decision", "conflict", "resource", "error", "timeout"
    priority: str = "medium"  # "low", "medium", "high", "critical"
    context: dict
    swarm_name: Optional[str] = None
    agent_name: Optional[str] = None
    work_id: Optional[str] = None


class AgentExecuteRequest(BaseModel):
    """Request to execute an agent task."""
    swarm: str
    agent: str
    prompt: str
    max_turns: int = 25
    timeout: float = 600.0


class SearchRequest(BaseModel):
    """Request to search the web."""
    query: str
    num_results: int = 5


class FetchRequest(BaseModel):
    """Request to fetch a URL."""
    url: str
    extract_text: bool = True
