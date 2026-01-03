"""Pydantic models for API responses."""

from pydantic import BaseModel


class SwarmResponse(BaseModel):
    """Response containing swarm information."""
    name: str
    description: str
    status: str
    agent_count: int
    priorities: list[str]


class AgentInfo(BaseModel):
    """Response containing agent information."""
    name: str
    type: str
    model: str
    background: bool
    description: str


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    swarm_count: int
    agent_count: int
