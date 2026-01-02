"""Central registry of AgentDefinition objects for reuse across swarms."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from claude_agent_sdk import AgentDefinition as SDKAgentDefinition
except ImportError:
    SDKAgentDefinition = None


@dataclass
class AgentDefinition:
    """Definition of an agent's capabilities and configuration.

    This wraps the Claude Agent SDK's AgentDefinition with additional
    metadata for the swarm system.
    """

    name: str
    description: str
    prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    model: str = "opus"
    agent_type: str = "worker"
    background: bool = False
    wake_enabled: bool = False

    def to_sdk_definition(self) -> Any:
        """Convert to Claude Agent SDK AgentDefinition if available."""
        if SDKAgentDefinition is None:
            return self
        return SDKAgentDefinition(
            description=self.description,
            prompt=self.prompt,
            tools=self.tools,
            model=self.model,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "tools": self.tools,
            "model": self.model,
            "type": self.agent_type,
            "background": self.background,
            "wake_enabled": self.wake_enabled,
        }


# Base agent type definitions with default configurations
AGENT_TYPES: Dict[str, AgentDefinition] = {
    "orchestrator": AgentDefinition(
        name="orchestrator",
        description="Coordinates work. Spawns subagents in parallel via Task tool.",
        tools=["Read", "Glob", "Bash", "Task"],
        model="opus",
        agent_type="orchestrator",
        background=False,
        wake_enabled=True,
    ),
    "researcher": AgentDefinition(
        name="researcher",
        description="Research specialist. Run in BACKGROUND. Wakes main thread when done.",
        tools=["Read", "Bash", "Grep", "Glob", "WebSearch"],
        model="opus",
        agent_type="researcher",
        background=True,
        wake_enabled=True,
    ),
    "implementer": AgentDefinition(
        name="implementer",
        description="Implementation specialist. Run in BACKGROUND for coding tasks.",
        tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        model="opus",
        agent_type="implementer",
        background=True,
        wake_enabled=True,
    ),
    "worker": AgentDefinition(
        name="worker",
        description="General worker agent for implementation tasks.",
        tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        model="opus",
        agent_type="worker",
        background=False,
        wake_enabled=False,
    ),
    "critic": AgentDefinition(
        name="critic",
        description="Adversarial reviewer. Run PROACTIVELY after proposals. READ-ONLY.",
        tools=["Read", "Grep", "Glob"],
        model="opus",
        agent_type="critic",
        background=True,
        wake_enabled=True,
    ),
    "monitor": AgentDefinition(
        name="monitor",
        description="Background monitor. Watches for errors. WAKES main thread on problems only.",
        tools=["Bash", "Read", "Grep"],
        model="opus",
        agent_type="monitor",
        background=True,
        wake_enabled=True,
    ),
    "coordinator": AgentDefinition(
        name="coordinator",
        description="Task coordination and cross-swarm handoffs.",
        tools=["Read", "Glob", "Grep", "Write", "Edit"],
        model="opus",
        agent_type="coordinator",
        background=False,
        wake_enabled=True,
    ),
    "quality": AgentDefinition(
        name="quality",
        description="Quality assurance, standards enforcement, and audits.",
        tools=["Read", "Glob", "Grep", "Write", "Edit", "Bash"],
        model="opus",
        agent_type="quality",
        background=False,
        wake_enabled=True,
    ),
    "architect": AgentDefinition(
        name="architect",
        description="System design and architecture planning.",
        tools=["Read", "Glob", "Grep", "Write", "Edit"],
        model="opus",
        agent_type="architect",
        background=True,
        wake_enabled=True,
    ),
    "reviewer": AgentDefinition(
        name="reviewer",
        description="Code review for correctness and best practices.",
        tools=["Read", "Glob", "Grep"],
        model="opus",
        agent_type="reviewer",
        background=True,
        wake_enabled=True,
    ),
}


def parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content with optional frontmatter

    Returns:
        Tuple of (frontmatter dict, remaining content)
    """
    frontmatter = {}
    body = content

    # Check for frontmatter delimiter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            except yaml.YAMLError:
                pass

    return frontmatter, body


def load_agent(
    agent_type: str,
    prompt_file: Path,
    name: Optional[str] = None,
) -> AgentDefinition:
    """Load an agent definition from a prompt file.

    Args:
        agent_type: Type of agent (orchestrator, researcher, etc.)
        prompt_file: Path to the markdown prompt file
        name: Optional name override

    Returns:
        Configured AgentDefinition
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}. Valid types: {list(AGENT_TYPES.keys())}")

    base = AGENT_TYPES[agent_type]

    # Read and parse the prompt file
    content = prompt_file.read_text()
    frontmatter, prompt = parse_frontmatter(content)

    # Merge frontmatter with base definition
    return AgentDefinition(
        name=name or frontmatter.get("name", base.name),
        description=frontmatter.get("description", base.description),
        prompt=prompt,
        tools=frontmatter.get("tools", base.tools) if isinstance(frontmatter.get("tools"), list)
              else base.tools,
        model=frontmatter.get("model", base.model),
        agent_type=frontmatter.get("type", agent_type),
        background=frontmatter.get("background", base.background),
        wake_enabled=frontmatter.get("wake_enabled", base.wake_enabled),
    )


def load_agent_from_file(prompt_file: Path) -> AgentDefinition:
    """Load an agent definition from a prompt file, inferring type from frontmatter or filename.

    Args:
        prompt_file: Path to the markdown prompt file

    Returns:
        Configured AgentDefinition
    """
    content = prompt_file.read_text()
    frontmatter, prompt = parse_frontmatter(content)

    # Get agent type from frontmatter or infer from filename
    agent_type = frontmatter.get("type")
    if not agent_type:
        filename = prompt_file.stem.lower()
        for type_name in AGENT_TYPES:
            if type_name in filename:
                agent_type = type_name
                break
        if not agent_type:
            agent_type = "worker"  # Default to worker

    return load_agent(agent_type, prompt_file, name=frontmatter.get("name", prompt_file.stem))


def get_agent_type(name: str) -> AgentDefinition:
    """Get a base agent type definition.

    Args:
        name: Name of the agent type

    Returns:
        Base AgentDefinition for that type
    """
    if name not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {name}")
    return AGENT_TYPES[name]


def list_agent_types() -> List[str]:
    """List all available agent types."""
    return list(AGENT_TYPES.keys())
