"""Supreme Orchestrator for managing all swarms."""

from __future__ import annotations

import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml

from shared.agent_base import AgentConfig, BaseAgent
from shared.agent_definitions import AgentDefinition, load_agent_from_file
from shared.swarm_interface import Swarm, SwarmConfig, load_swarm

try:
    from claude_agent_sdk import ClaudeAgentOptions, query
except ImportError:
    query = None
    ClaudeAgentOptions = None

logger = logging.getLogger(__name__)


class SupremeOrchestrator:
    """Supreme orchestrator that manages all swarms."""

    def __init__(
        self,
        base_path: Path,
        config_path: Path | None = None,
        logs_dir: Path | None = None,
    ) -> None:
        """Initialize the Supreme Orchestrator.

        Args:
            base_path: Base path for the agent-swarm system
            config_path: Path to main config.yaml
            logs_dir: Directory for logs
        """
        self.base_path = Path(base_path)
        self.swarms_dir = self.base_path / "swarms"
        self.config_path = config_path or (self.base_path / "config.yaml")
        self.logs_dir = logs_dir or (self.base_path / "logs")
        self.swarms: dict[str, Swarm] = {}
        self._config: dict[str, Any] = {}

        # Agent definition for supreme orchestrator
        self.agent: AgentDefinition | None = None
        # All agents across all swarms (for cross-swarm dispatch)
        self.all_agents: dict[str, AgentDefinition] = {}

        # Load configuration
        if self.config_path.exists():
            self._load_config()

        # Load supreme agent
        self._load_supreme_agent()

        # Create routing agent for backward compatibility
        self._routing_agent = self._create_routing_agent()

        # Discover swarms
        self.discover_swarms()

    def _load_config(self) -> None:
        """Load configuration from config.yaml."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from {self.config_path}")

    def _load_supreme_agent(self) -> None:
        """Load the supreme orchestrator agent definition."""
        prompt_file = self.base_path / "supreme" / "agents" / "supreme.md"

        if prompt_file.exists():
            try:
                self.agent = load_agent_from_file(prompt_file)
                self.all_agents["supreme"] = self.agent
                logger.debug("Loaded supreme orchestrator agent from file")
            except Exception as e:
                logger.warning(f"Failed to load supreme agent: {e}")
                self._create_default_supreme_agent()
        else:
            self._create_default_supreme_agent()

    def _create_default_supreme_agent(self) -> None:
        """Create default supreme agent definition."""
        self.agent = AgentDefinition(
            name="supreme",
            description="Supreme orchestrator. Routes to swarms, spawns parallel agents.",
            prompt=self._get_default_prompt(),
            tools=["Task", "Read", "Bash", "Glob"],
            model="opus",
            agent_type="orchestrator",
        )
        self.all_agents["supreme"] = self.agent

    def _get_default_prompt(self) -> str:
        """Get default supreme orchestrator prompt."""
        return """You are the Supreme Orchestrator managing multiple project swarms.

## Parallel Execution Patterns

When given a complex task, spawn subagents IN PARALLEL using the Task tool:

### Standard Pattern
Spawn in parallel:
1. [swarm]/researcher (background): Gather information
2. [swarm]/implementer (background): Begin implementation
3. [swarm]/critic (background): Prepare challenges
Wait for all, then synthesize.

### Cross-Swarm Pattern
For tasks affecting multiple projects, spawn agents from different swarms in parallel.

### Monitor Pattern
Spawn in background (don't wait):
1. [swarm]/monitor: Watch for problems, wake me if issues
Then continue with main task.

## Wake Handling
When subagents wake you with findings, synthesize all before responding to user.

## Available Swarms
{swarms_info}

Analyze requests and route to appropriate swarms. For complex tasks, use parallel execution."""

    def _create_routing_agent(self) -> BaseAgent:
        """Create the routing agent for backward compatibility."""
        model = self._config.get("models", {}).get("orchestrator", "claude-opus-4-5-20251101")

        config = AgentConfig(
            name="supreme_orchestrator",
            role="orchestrator",
            model=model,
            system_prompt=self._get_default_prompt().format(swarms_info="[No swarms loaded yet]"),
            tools=["Task", "Read", "Bash", "Glob"],
            max_turns=25,
        )

        return BaseAgent(config, logs_dir=self.logs_dir)

    def _collect_all_agents(self) -> None:
        """Collect all agent definitions from all swarms."""
        self.all_agents = {"supreme": self.agent} if self.agent else {}

        for swarm_name, swarm in self.swarms.items():
            for agent_name, agent_def in swarm.agent_definitions.items():
                qualified_name = f"{swarm_name}/{agent_name}"
                self.all_agents[qualified_name] = agent_def

        logger.debug(f"Collected {len(self.all_agents)} agents across all swarms")

    def _update_routing_agent_prompt(self) -> None:
        """Update the routing agent's system prompt with current swarm info."""
        swarms_info = self._format_swarms_info()
        prompt = self._get_default_prompt().format(swarms_info=swarms_info)
        self._routing_agent._system_prompt = prompt

        # Also update the agent definition
        if self.agent:
            self.agent.prompt = prompt

    def _format_swarms_info(self) -> str:
        """Format swarm information for prompts."""
        if not self.swarms:
            return "No swarms available. Consider creating one."

        lines = []
        for name, swarm in self.swarms.items():
            status = swarm.get_status()
            lines.append(f"### {name}")
            lines.append(f"- Description: {status['description']}")
            lines.append(f"- Status: {status['status']}")

            # List agents with their types
            agent_list = []
            for agent_info in status["agents"]:
                agent_type = agent_info.get("type", agent_info.get("role", "worker"))
                bg = " [bg]" if agent_info.get("background") else ""
                agent_list.append(f"{agent_info['name']} ({agent_type}){bg}")
            lines.append(f"- Agents: {', '.join(agent_list)}")

            if status["priorities"]:
                # Handle both string and dict priorities
                priority_strs = []
                for p in status["priorities"][:3]:
                    if isinstance(p, dict):
                        priority_strs.append(p.get("task", str(p)))
                    else:
                        priority_strs.append(str(p))
                lines.append(f"- Priorities: {', '.join(priority_strs)}")
            lines.append("")

        return "\n".join(lines)

    def discover_swarms(self) -> list[str]:
        """Discover and load all valid swarms from the swarms directory.

        Returns:
            List of discovered swarm names
        """
        discovered = []

        if not self.swarms_dir.exists():
            logger.warning(f"Swarms directory not found: {self.swarms_dir}")
            return discovered

        for item in self.swarms_dir.iterdir():
            if not item.is_dir():
                continue

            # Skip template and hidden directories
            if item.name.startswith("_") or item.name.startswith("."):
                continue

            swarm_yaml = item / "swarm.yaml"
            if swarm_yaml.exists():
                try:
                    swarm = load_swarm(item, self.logs_dir)
                    self.swarms[swarm.name] = swarm
                    discovered.append(swarm.name)
                    logger.info(f"Discovered swarm: {swarm.name}")
                except Exception as e:
                    logger.error(f"Failed to load swarm from {item}: {e}")

        # Collect all agents and update prompts
        self._collect_all_agents()
        self._update_routing_agent_prompt()

        return discovered

    def list_swarms(self) -> list[SwarmConfig]:
        """List all loaded swarm configurations.

        Returns:
            List of SwarmConfig objects
        """
        return [swarm.config for swarm in self.swarms.values()]

    def get_swarm(self, name: str) -> Swarm | None:
        """Get a swarm by name.

        Args:
            name: Name of the swarm

        Returns:
            Swarm instance or None if not found
        """
        return self.swarms.get(name)

    def get_all_status(self) -> dict[str, Any]:
        """Get status of all swarms.

        Returns:
            Dictionary with status of all swarms
        """
        return {
            "total_swarms": len(self.swarms),
            "total_agents": len(self.all_agents),
            "swarms": {name: swarm.get_status() for name, swarm in self.swarms.items()},
        }

    async def chat(self, user_input: str) -> AsyncIterator[Any]:
        """Chat with the Supreme Orchestrator using parallel agent dispatch.

        Args:
            user_input: User's input

        Yields:
            Messages from the orchestrator and subagents
        """
        # Build SDK agents dict
        sdk_agents = {}
        for name, defn in self.all_agents.items():
            sdk_agents[name] = defn.to_sdk_definition()

        # Use routing agent - SDK API compatibility TBD
        response = await self.route_request(user_input)
        yield {"type": "text", "content": response}

    async def route_request(self, user_input: str) -> str:
        """Route a user request to the appropriate swarm.

        Args:
            user_input: The user's input/request

        Returns:
            Response from the routing process
        """
        # Update routing agent with current swarm info
        self._update_routing_agent_prompt()

        # Add context about available swarms
        context = f"""User Request: {user_input}

Available Swarms:
{self._format_swarms_info()}

Available agents for parallel dispatch:
{", ".join(self.all_agents.keys())}

Analyze this request and determine which swarm should handle it.
For complex tasks, spawn multiple agents IN PARALLEL using the Task tool.
For meta-questions about the system, answer directly."""

        response = await self._routing_agent.run_sync(context)
        return response

    async def send_directive(self, swarm_name: str, directive: str) -> str:
        """Send a directive to a specific swarm.

        Args:
            swarm_name: Name of the swarm
            directive: The directive to send

        Returns:
            Response from the swarm
        """
        swarm = self.get_swarm(swarm_name)
        if not swarm:
            return f"Swarm '{swarm_name}' not found."

        return await swarm.receive_directive(directive)

    async def run_parallel_on_swarm(
        self,
        swarm_name: str,
        directive: str,
    ) -> AsyncIterator[Any]:
        """Run parallel agents on a specific swarm.

        Args:
            swarm_name: Name of the swarm
            directive: The directive/task

        Yields:
            Messages from agents
        """
        swarm = self.get_swarm(swarm_name)
        if not swarm:
            yield {"type": "error", "content": f"Swarm '{swarm_name}' not found."}
            return

        # Build tasks for parallel execution
        tasks = []
        for agent_name, defn in swarm.agent_definitions.items():
            if defn.agent_type == "orchestrator":
                continue  # Skip orchestrator for parallel tasks

            tasks.append(
                {
                    "agent": agent_name,
                    "prompt": f"{defn.agent_type.title()}: {directive}",
                    "background": defn.background,
                }
            )

        async for message in swarm.run_parallel(tasks):
            yield message

    def create_swarm(
        self,
        name: str,
        description: str = "",
        template: str = "_template",
    ) -> Swarm:
        """Create a new swarm from a template.

        Args:
            name: Name for the new swarm
            description: Description of the swarm
            template: Template directory name to copy from

        Returns:
            New Swarm instance
        """
        template_path = self.swarms_dir / template
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        new_swarm_path = self.swarms_dir / name.lower().replace(" ", "_")
        if new_swarm_path.exists():
            raise FileExistsError(f"Swarm directory already exists: {new_swarm_path}")

        # Copy template
        shutil.copytree(template_path, new_swarm_path)

        # Update swarm.yaml with new name and description
        swarm_yaml = new_swarm_path / "swarm.yaml"
        with open(swarm_yaml) as f:
            config_data = yaml.safe_load(f)

        config_data["name"] = name
        config_data["description"] = description or f"Swarm for {name}"
        config_data["status"] = "active"

        with open(swarm_yaml, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        # Load and register the new swarm
        swarm = load_swarm(new_swarm_path, self.logs_dir)
        self.swarms[swarm.name] = swarm

        # Update agent collections
        self._collect_all_agents()
        self._update_routing_agent_prompt()

        logger.info(f"Created new swarm: {name}")
        return swarm

    def archive_swarm(self, name: str) -> bool:
        """Archive a swarm (set status to archived).

        Args:
            name: Name of the swarm to archive

        Returns:
            True if successful
        """
        swarm = self.get_swarm(name)
        if not swarm:
            return False

        swarm.config.status = "archived"
        swarm.config.save(swarm.swarm_path / "swarm.yaml")

        logger.info(f"Archived swarm: {name}")
        return True

    def pause_swarm(self, name: str) -> bool:
        """Pause a swarm.

        Args:
            name: Name of the swarm to pause

        Returns:
            True if successful
        """
        swarm = self.get_swarm(name)
        if not swarm:
            return False

        swarm.config.status = "paused"
        swarm.config.save(swarm.swarm_path / "swarm.yaml")

        logger.info(f"Paused swarm: {name}")
        return True

    def activate_swarm(self, name: str) -> bool:
        """Activate a swarm.

        Args:
            name: Name of the swarm to activate

        Returns:
            True if successful
        """
        swarm = self.get_swarm(name)
        if not swarm:
            return False

        swarm.config.status = "active"
        swarm.config.save(swarm.swarm_path / "swarm.yaml")

        logger.info(f"Activated swarm: {name}")
        return True
