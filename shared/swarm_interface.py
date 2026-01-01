"""Swarm interface and configuration."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .agent_base import AgentConfig, BaseAgent
from .consensus import ConsensusResult, ConsensusProtocol

logger = logging.getLogger(__name__)


@dataclass
class SwarmConfig:
    """Configuration for a swarm parsed from swarm.yaml."""

    name: str
    description: str
    version: str = "0.1.0"
    status: str = "active"  # active, paused, archived
    agents: List[Dict[str, Any]] = field(default_factory=list)
    workspace_path: str = "./workspace"
    settings: Dict[str, Any] = field(default_factory=dict)
    priorities: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SwarmConfig":
        """Load SwarmConfig from a YAML file.

        Args:
            yaml_path: Path to swarm.yaml

        Returns:
            SwarmConfig instance
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            name=data.get("name", yaml_path.parent.name),
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            status=data.get("status", "active"),
            agents=data.get("agents", []),
            workspace_path=data.get("workspace", "./workspace"),
            settings=data.get("settings", {}),
            priorities=data.get("priorities", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status,
            "agents": self.agents,
            "workspace": self.workspace_path,
            "settings": self.settings,
            "priorities": self.priorities,
        }

    def save(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class SwarmInterface(ABC):
    """Abstract interface for swarm implementations."""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current swarm status.

        Returns:
            Dictionary with status information
        """
        pass

    @abstractmethod
    def get_priorities(self) -> List[str]:
        """Get current priorities.

        Returns:
            List of priority strings
        """
        pass

    @abstractmethod
    async def receive_directive(self, directive: str) -> str:
        """Receive and process a directive from the supreme orchestrator.

        Args:
            directive: The directive to process

        Returns:
            Response string
        """
        pass

    @abstractmethod
    def report_progress(self) -> Dict[str, Any]:
        """Report current progress on tasks.

        Returns:
            Dictionary with progress information
        """
        pass

    @abstractmethod
    async def request_consensus(self, proposal: str) -> ConsensusResult:
        """Request consensus from swarm agents on a proposal.

        Args:
            proposal: The proposal to vote on

        Returns:
            ConsensusResult with voting outcome
        """
        pass


class Swarm(SwarmInterface):
    """Swarm implementation that manages agents and handles directives."""

    def __init__(
        self,
        config: SwarmConfig,
        swarm_path: Path,
        logs_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the swarm.

        Args:
            config: Swarm configuration
            swarm_path: Path to the swarm directory
            logs_dir: Directory for logs
        """
        self.config = config
        self.swarm_path = swarm_path
        self.logs_dir = logs_dir or Path("./logs")
        self.agents: Dict[str, BaseAgent] = {}
        self.consensus_protocol = ConsensusProtocol(
            logs_dir=self.logs_dir / "consensus"
        )
        self._current_tasks: List[Dict[str, Any]] = []
        self._progress: Dict[str, Any] = {}

        # Load agents
        self._load_agents()

    def _load_agents(self) -> None:
        """Load and initialize agents from configuration."""
        agents_dir = self.swarm_path / "agents"

        for agent_def in self.config.agents:
            if isinstance(agent_def, str):
                # Simple string reference to agent file
                agent_name = agent_def
                prompt_file = f"agents/{agent_name}.md"
                role = self._infer_role(agent_name)
                agent_config = AgentConfig(
                    name=agent_name,
                    role=role,
                    system_prompt_file=prompt_file,
                )
            elif isinstance(agent_def, dict):
                # Full agent configuration
                agent_name = agent_def.get("name", "unnamed")
                prompt_file = agent_def.get("prompt_file", f"agents/{agent_name}.md")
                agent_config = AgentConfig(
                    name=agent_name,
                    role=agent_def.get("role", self._infer_role(agent_name)),
                    model=agent_def.get("model", "claude-sonnet-4-5-20250929"),
                    system_prompt_file=prompt_file,
                    tools=agent_def.get("tools"),
                    max_turns=agent_def.get("max_turns", 25),
                    settings=agent_def.get("settings", {}),
                )
            else:
                logger.warning(f"Invalid agent definition: {agent_def}")
                continue

            agent = BaseAgent(
                config=agent_config,
                swarm_path=self.swarm_path,
                logs_dir=self.logs_dir,
            )
            self.agents[agent_name] = agent
            logger.debug(f"Loaded agent: {agent_name} ({agent_config.role})")

        # Auto-discover agents from agents/ directory if none configured
        if not self.agents and agents_dir.exists():
            self._auto_discover_agents(agents_dir)

    def _auto_discover_agents(self, agents_dir: Path) -> None:
        """Auto-discover agent configs from agents/ subdirectory."""
        for prompt_file in agents_dir.glob("*.md"):
            agent_name = prompt_file.stem
            if agent_name.startswith("_"):
                continue  # Skip files starting with underscore

            role = self._infer_role(agent_name)
            agent_config = AgentConfig(
                name=agent_name,
                role=role,
                system_prompt_file=f"agents/{prompt_file.name}",
            )

            agent = BaseAgent(
                config=agent_config,
                swarm_path=self.swarm_path,
                logs_dir=self.logs_dir,
            )
            self.agents[agent_name] = agent
            logger.debug(f"Auto-discovered agent: {agent_name} ({role})")

    def _infer_role(self, name: str) -> str:
        """Infer agent role from name."""
        name_lower = name.lower()
        if "orchestrator" in name_lower or "coordinator" in name_lower:
            return "orchestrator"
        elif "critic" in name_lower or "reviewer" in name_lower:
            return "critic"
        return "worker"

    @property
    def name(self) -> str:
        """Get swarm name."""
        return self.config.name

    @property
    def workspace(self) -> Path:
        """Get absolute workspace path."""
        workspace = Path(self.config.workspace_path)
        if workspace.is_absolute():
            return workspace
        return self.swarm_path / workspace

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self.agents.get(name)

    def get_orchestrator(self) -> Optional[BaseAgent]:
        """Get the orchestrator agent if one exists."""
        for agent in self.agents.values():
            if agent.role == "orchestrator":
                return agent
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "status": self.config.status,
            "agents": [
                {"name": a.name, "role": a.role}
                for a in self.agents.values()
            ],
            "agent_count": len(self.agents),
            "priorities": self.config.priorities,
            "workspace": str(self.workspace),
            "current_tasks": len(self._current_tasks),
        }

    def get_priorities(self) -> List[str]:
        """Get current priorities."""
        return self.config.priorities.copy()

    def set_priorities(self, priorities: List[str]) -> None:
        """Set priorities."""
        self.config.priorities = priorities

    async def receive_directive(self, directive: str) -> str:
        """Receive and process a directive from the supreme orchestrator."""
        logger.info(f"Swarm {self.name} received directive: {directive[:100]}...")

        # Check if swarm is active
        if self.config.status != "active":
            return f"Swarm {self.name} is {self.config.status}, cannot process directive."

        # Get orchestrator or first available agent
        orchestrator = self.get_orchestrator()
        if not orchestrator:
            orchestrator = next(iter(self.agents.values()), None)

        if not orchestrator:
            return f"Swarm {self.name} has no agents to process directive."

        # Have the orchestrator process the directive
        prompt = f"""You have received the following directive from the Supreme Orchestrator:

{directive}

Current swarm priorities:
{chr(10).join(f"- {p}" for p in self.config.priorities) or "No priorities set"}

Available agents in this swarm:
{chr(10).join(f"- {a.name} ({a.role})" for a in self.agents.values())}

Please acknowledge this directive and outline how you will proceed."""

        response = await orchestrator.run_sync(prompt, str(self.workspace))
        return response

    def report_progress(self) -> Dict[str, Any]:
        """Report current progress on tasks."""
        return {
            "swarm": self.config.name,
            "status": self.config.status,
            "tasks": self._current_tasks,
            "progress": self._progress,
            "agent_activities": {
                name: len(agent.conversation_history)
                for name, agent in self.agents.items()
            },
        }

    async def request_consensus(self, proposal: str) -> ConsensusResult:
        """Request consensus from swarm agents on a proposal."""
        # Get agents eligible to vote (all agents by default)
        voters = list(self.agents.values())

        if len(voters) < self.config.settings.get("min_voters", 1):
            return ConsensusResult(
                approved=False,
                outcome="insufficient_voters",
                votes={},
                discussion=[],
                proposal=proposal,
            )

        return await self.consensus_protocol.run_consensus(
            topic=f"Proposal for {self.name}",
            proposal=proposal,
            voters=voters,
        )

    def add_task(self, task: Dict[str, Any]) -> None:
        """Add a task to current tasks."""
        self._current_tasks.append(task)

    def complete_task(self, task_id: str) -> None:
        """Mark a task as complete."""
        self._current_tasks = [
            t for t in self._current_tasks
            if t.get("id") != task_id
        ]

    def __repr__(self) -> str:
        return f"Swarm(name={self.name!r}, agents={len(self.agents)}, status={self.config.status!r})"


def load_swarm(swarm_path: Path, logs_dir: Optional[Path] = None) -> Swarm:
    """Load a swarm from a directory.

    Args:
        swarm_path: Path to swarm directory containing swarm.yaml
        logs_dir: Directory for logs

    Returns:
        Initialized Swarm instance
    """
    yaml_path = swarm_path / "swarm.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"swarm.yaml not found in {swarm_path}")

    config = SwarmConfig.from_yaml(yaml_path)
    return Swarm(config, swarm_path, logs_dir)
