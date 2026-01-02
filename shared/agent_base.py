"""Base agent implementation wrapping Claude Agent SDK."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Import the real agent executor
from .agent_executor import stream_agent

# Check if execution is available (API key or CLI)
CLAUDE_SDK_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))


logger = logging.getLogger(__name__)


# Default tools by role
ROLE_TOOLS: dict[str, list[str]] = {
    "orchestrator": ["Read", "Glob", "Bash"],
    "worker": ["Read", "Write", "Edit", "Bash", "Glob"],
    "critic": ["Read", "Glob"],
}


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    role: str
    model: str = "claude-opus-4-5-20251101"
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    tools: list[str] | None = None
    max_turns: int = 25
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default tools based on role if not specified."""
        if self.tools is None:
            self.tools = ROLE_TOOLS.get(self.role, ["Read", "Glob"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Create AgentConfig from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            role=data.get("role", "worker"),
            model=data.get("model", "claude-opus-4-5-20251101"),
            system_prompt=data.get("system_prompt"),
            system_prompt_file=data.get("system_prompt_file"),
            tools=data.get("tools"),
            max_turns=data.get("max_turns", 25),
            settings=data.get("settings", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "system_prompt_file": self.system_prompt_file,
            "tools": self.tools,
            "max_turns": self.max_turns,
            "settings": self.settings,
        }


class BaseAgent:
    """Base agent wrapping Claude Agent SDK."""

    def __init__(
        self,
        config: AgentConfig,
        swarm_path: Path | None = None,
        logs_dir: Path | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration
            swarm_path: Path to the swarm directory (for loading system prompts)
            logs_dir: Directory for conversation logs
        """
        self.config = config
        self.swarm_path = swarm_path
        self.logs_dir = logs_dir or Path("./logs")
        self.conversation_history: list[dict[str, str]] = []
        self._system_prompt: str | None = None

        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load system prompt
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load system prompt from file or config."""
        if self.config.system_prompt:
            self._system_prompt = self.config.system_prompt
        elif self.config.system_prompt_file and self.swarm_path:
            prompt_path = self.swarm_path / self.config.system_prompt_file
            if prompt_path.exists():
                self._system_prompt = prompt_path.read_text()
                logger.debug(f"Loaded system prompt from {prompt_path}")
            else:
                logger.warning(f"System prompt file not found: {prompt_path}")
                self._system_prompt = self._get_default_prompt()
        else:
            self._system_prompt = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default system prompt based on role."""
        role_prompts = {
            "orchestrator": (
                f"You are {self.config.name}, the orchestrator agent. "
                "Your role is to coordinate tasks, delegate work to other agents, "
                "and ensure smooth operation of the swarm. Make strategic decisions "
                "and maintain awareness of all ongoing activities."
            ),
            "worker": (
                f"You are {self.config.name}, a worker agent. "
                "Your role is to execute tasks assigned by the orchestrator. "
                "Focus on implementation, writing code, and completing concrete work items. "
                "Report your progress and any blockers encountered."
            ),
            "critic": (
                f"You are {self.config.name}, the critic agent. "
                "Your role is to review and challenge proposals, code, and decisions. "
                "Be constructively critical - identify potential issues, edge cases, "
                "and improvements. You must push back on weak proposals."
            ),
        }
        return role_prompts.get(self.config.role, role_prompts["worker"])

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return self._system_prompt or self._get_default_prompt()

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name

    @property
    def role(self) -> str:
        """Get agent role."""
        return self.config.role

    def _log_conversation(
        self,
        prompt: str,
        response: str,
        workspace: str | None = None,
    ) -> None:
        """Log conversation to file."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent": self.config.name,
            "role": self.config.role,
            "workspace": workspace,
            "prompt": prompt,
            "response": response,
        }

        # Add to history
        self.conversation_history.append(log_entry)

        # Write to log file
        log_file = self.logs_dir / f"{self.config.name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    async def run(
        self,
        prompt: str,
        workspace: str | None = None,
    ) -> AsyncIterator[str]:
        """Run the agent with a prompt.

        Args:
            prompt: The prompt to send to the agent
            workspace: Optional workspace directory for the agent

        Yields:
            Response chunks from the agent
        """
        logger.info(f"Agent {self.name} running with prompt: {prompt[:100]}...")

        full_response = ""
        workspace_path = Path(workspace) if workspace else None

        # Check if execution is available
        if not CLAUDE_SDK_AVAILABLE:
            # Return mock response for local development
            mock_response = (
                f"[Mock response from {self.name}]\n\n"
                "Agent execution is not available. Set either:\n"
                "- ANTHROPIC_API_KEY for API access\n"
                "- CLAUDE_CODE_OAUTH_TOKEN for CLI access\n\n"
                f"Prompt received: {prompt[:200]}..."
            )
            yield mock_response
            self._log_conversation(prompt, mock_response, workspace)
            return

        # Use the real agent executor
        async for event in stream_agent(
            prompt=prompt,
            system_prompt=self.system_prompt,
            tools=self.config.tools,
            workspace=workspace_path,
        ):
            event_type = event.get("type", "")

            if event_type == "content":
                chunk = event.get("content", "")
                if chunk:
                    full_response += chunk
                    yield chunk
            elif event_type == "thinking":
                # Could yield thinking if needed
                pass
            elif event_type == "error":
                error_msg = event.get("content", "Unknown error")
                yield f"\n[Error: {error_msg}]\n"
                full_response += f"\n[Error: {error_msg}]\n"

        # Log the conversation
        self._log_conversation(prompt, full_response, workspace)
        logger.debug(f"Agent {self.name} completed response")

    async def run_sync(
        self,
        prompt: str,
        workspace: str | None = None,
    ) -> str:
        """Run the agent and return complete response.

        Args:
            prompt: The prompt to send to the agent
            workspace: Optional workspace directory

        Returns:
            Complete response string
        """
        chunks = []
        async for chunk in self.run(prompt, workspace):
            chunks.append(chunk)
        return "".join(chunks)

    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name!r}, role={self.role!r}, model={self.config.model!r})"


def load_agent_from_yaml(
    yaml_path: Path,
    swarm_path: Path | None = None,
    logs_dir: Path | None = None,
) -> BaseAgent:
    """Load an agent from a YAML config file.

    Args:
        yaml_path: Path to agent YAML config
        swarm_path: Path to swarm directory
        logs_dir: Directory for logs

    Returns:
        Configured BaseAgent instance
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    config = AgentConfig.from_dict(data)
    return BaseAgent(config, swarm_path, logs_dir)
