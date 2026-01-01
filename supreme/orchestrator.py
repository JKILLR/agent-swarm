"""Supreme Orchestrator for managing all swarms."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from shared.agent_base import AgentConfig, BaseAgent
from shared.swarm_interface import Swarm, SwarmConfig, load_swarm

logger = logging.getLogger(__name__)


ROUTING_SYSTEM_PROMPT = """You are the Supreme Orchestrator, the top-level coordinator for a multi-swarm AI system. Your role is to intelligently route user requests to the appropriate swarm.

## Available Swarms
{swarms_info}

## Your Responsibilities

1. **Request Analysis**: Understand what the user is asking for
2. **Swarm Selection**: Choose the most appropriate swarm to handle the request
3. **Directive Formulation**: Formulate clear directives for the selected swarm
4. **Status Monitoring**: Provide overviews of all swarm activities when asked

## Routing Guidelines

- Consider each swarm's description and priorities
- Route to the swarm with the most relevant expertise
- If no swarm is suitable, suggest creating a new one
- For meta-questions about the system, answer directly

## Response Format

When routing a request, respond with:
1. Your analysis of the request
2. Which swarm should handle it and why
3. The formulated directive for that swarm

When providing status, give a clear overview of all swarms.
"""


class SupremeOrchestrator:
    """Supreme orchestrator that manages all swarms."""

    def __init__(
        self,
        swarms_dir: Path,
        config_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the Supreme Orchestrator.

        Args:
            swarms_dir: Directory containing swarm directories
            config_path: Path to main config.yaml
            logs_dir: Directory for logs
        """
        self.swarms_dir = Path(swarms_dir)
        self.config_path = config_path
        self.logs_dir = logs_dir or Path("./logs")
        self.swarms: Dict[str, Swarm] = {}
        self._config: Dict[str, Any] = {}

        # Load configuration
        if config_path and config_path.exists():
            self._load_config()

        # Create routing agent
        self._routing_agent = self._create_routing_agent()

        # Discover swarms
        self.discover_swarms()

    def _load_config(self) -> None:
        """Load configuration from config.yaml."""
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from {self.config_path}")

    def _create_routing_agent(self) -> BaseAgent:
        """Create the routing agent."""
        model = self._config.get("orchestrator", {}).get(
            "routing_model",
            self._config.get("models", {}).get("default", "claude-sonnet-4-5-20250929"),
        )

        config = AgentConfig(
            name="supreme_orchestrator",
            role="orchestrator",
            model=model,
            system_prompt=ROUTING_SYSTEM_PROMPT.format(swarms_info="[No swarms loaded yet]"),
            max_turns=15,
        )

        return BaseAgent(config, logs_dir=self.logs_dir)

    def _update_routing_agent_prompt(self) -> None:
        """Update the routing agent's system prompt with current swarm info."""
        swarms_info = self._format_swarms_info()
        self._routing_agent._system_prompt = ROUTING_SYSTEM_PROMPT.format(
            swarms_info=swarms_info
        )

    def _format_swarms_info(self) -> str:
        """Format swarm information for the routing prompt."""
        if not self.swarms:
            return "No swarms available. Consider creating one."

        lines = []
        for name, swarm in self.swarms.items():
            status = swarm.get_status()
            lines.append(f"### {name}")
            lines.append(f"- Description: {status['description']}")
            lines.append(f"- Status: {status['status']}")
            lines.append(f"- Agents: {status['agent_count']}")
            if status['priorities']:
                lines.append(f"- Priorities: {', '.join(status['priorities'][:3])}")
            lines.append("")

        return "\n".join(lines)

    def discover_swarms(self) -> List[str]:
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

        # Update routing agent with new swarm info
        self._update_routing_agent_prompt()

        return discovered

    def list_swarms(self) -> List[SwarmConfig]:
        """List all loaded swarm configurations.

        Returns:
            List of SwarmConfig objects
        """
        return [swarm.config for swarm in self.swarms.values()]

    def get_swarm(self, name: str) -> Optional[Swarm]:
        """Get a swarm by name.

        Args:
            name: Name of the swarm

        Returns:
            Swarm instance or None if not found
        """
        return self.swarms.get(name)

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all swarms.

        Returns:
            Dictionary with status of all swarms
        """
        return {
            "total_swarms": len(self.swarms),
            "swarms": {
                name: swarm.get_status()
                for name, swarm in self.swarms.items()
            },
        }

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

Analyze this request and determine which swarm should handle it, or respond directly if it's a meta-question about the system."""

        response = await self._routing_agent.run_sync(context)

        # Parse response to see if we should forward to a swarm
        # For now, return the routing agent's analysis
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

        # Update routing agent
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
