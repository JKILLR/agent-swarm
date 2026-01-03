"""Agent execution context dataclass.

This module defines the AgentExecutionContext which encapsulates all
configuration and metadata needed to execute an agent in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentExecutionContext:
    """Context for agent execution.

    Contains all configuration needed to execute an agent with proper
    isolation, permissions, and tracking.

    Attributes:
        agent_name: Unique name of the agent (e.g., "implementer", "critic")
        agent_type: Type classification (orchestrator, implementer, critic, researcher, etc.)
        swarm_name: Name of the swarm this agent belongs to
        workspace: Working directory for the agent
        allowed_tools: List of tools the agent can use
        permission_mode: Claude CLI permission mode (acceptEdits, default, readonly)
        git_credentials: Whether to pass git credentials to the agent
        web_access: Whether the agent can access web resources
        max_turns: Maximum conversation turns before stopping
        timeout: Execution timeout in seconds
        job_id: Optional job ID for tracking in the job system
        parent_agent: Name of parent agent for task delegation tracing
    """

    agent_name: str
    agent_type: str
    swarm_name: str
    workspace: Path

    # Permissions
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep"
    ])
    permission_mode: str = "default"

    # Credentials (passed as env vars)
    git_credentials: bool = False
    web_access: bool = False

    # Limits
    max_turns: int = 25
    timeout: float = 600.0

    # Tracking
    job_id: str | None = None
    parent_agent: str | None = None

    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure workspace is a Path object
        if isinstance(self.workspace, str):
            self.workspace = Path(self.workspace)

        # Validate permission mode
        valid_modes = {"acceptEdits", "default", "readonly"}
        if self.permission_mode not in valid_modes:
            raise ValueError(
                f"Invalid permission_mode: {self.permission_mode}. "
                f"Must be one of: {valid_modes}"
            )

        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got: {self.timeout}")

        # Validate max_turns
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got: {self.max_turns}")

    @property
    def full_name(self) -> str:
        """Get full agent identifier including swarm."""
        return f"{self.swarm_name}/{self.agent_name}"

    @property
    def is_privileged(self) -> bool:
        """Check if this agent has privileged access (swarm_dev)."""
        return self.swarm_name == "swarm_dev"

    def to_dict(self) -> dict:
        """Convert context to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "swarm_name": self.swarm_name,
            "workspace": str(self.workspace),
            "allowed_tools": self.allowed_tools,
            "permission_mode": self.permission_mode,
            "git_credentials": self.git_credentials,
            "web_access": self.web_access,
            "max_turns": self.max_turns,
            "timeout": self.timeout,
            "job_id": self.job_id,
            "parent_agent": self.parent_agent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentExecutionContext":
        """Create context from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            agent_type=data["agent_type"],
            swarm_name=data["swarm_name"],
            workspace=Path(data["workspace"]),
            allowed_tools=data.get("allowed_tools", []),
            permission_mode=data.get("permission_mode", "default"),
            git_credentials=data.get("git_credentials", False),
            web_access=data.get("web_access", False),
            max_turns=data.get("max_turns", 25),
            timeout=data.get("timeout", 600.0),
            job_id=data.get("job_id"),
            parent_agent=data.get("parent_agent"),
        )
