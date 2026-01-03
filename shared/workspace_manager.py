"""Workspace management for agent isolation.

This module provides the WorkspaceManager class that handles workspace
isolation, path validation, and permission lookup for agents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages workspace isolation for agents.

    Handles workspace path resolution, security boundary validation,
    and permission lookup based on agent type and swarm.

    Attributes:
        project_root: Root directory of the agent-swarm project
        swarms_dir: Directory containing all swarm definitions
    """

    def __init__(self, project_root: Path):
        """Initialize the workspace manager.

        Args:
            project_root: Root directory of the agent-swarm project
        """
        self.project_root = project_root.resolve()
        self.swarms_dir = self.project_root / "swarms"

        logger.info(f"WorkspaceManager initialized with root: {self.project_root}")

    def get_workspace(self, swarm_name: str) -> Path:
        """Get the workspace path for a swarm.

        Creates the workspace directory if it doesn't exist.
        Special case: swarm_dev uses PROJECT_ROOT as its workspace.

        Args:
            swarm_name: Name of the swarm

        Returns:
            Path to the swarm's workspace directory
        """
        # Special case: swarm_dev gets access to project root
        if swarm_name == "swarm_dev":
            logger.debug(f"Swarm dev using project root: {self.project_root}")
            return self.project_root

        swarm_path = self.swarms_dir / swarm_name
        workspace = swarm_path / "workspace"

        # Create workspace if it doesn't exist
        workspace.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Workspace for {swarm_name}: {workspace}")
        return workspace

    def validate_path_access(
        self,
        path: Path,
        workspace: Path,
        allow_project_root: bool = False,
    ) -> bool:
        """Check if a path is within allowed boundaries.

        Security check to prevent file access escapes. Validates that
        the given path is within the workspace or (optionally) the
        project root.

        Args:
            path: Path to validate
            workspace: Agent's workspace directory
            allow_project_root: Whether to allow access to the full project

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            resolved = path.resolve()
            workspace_resolved = workspace.resolve()

            # Always allow workspace access
            if self._is_subpath(resolved, workspace_resolved):
                return True

            # Some agents (swarm_dev) can access project root
            if allow_project_root:
                if self._is_subpath(resolved, self.project_root):
                    return True

            logger.warning(
                f"Path access denied: {resolved} not in workspace {workspace_resolved}"
            )
            return False

        except (ValueError, OSError) as e:
            logger.error(f"Path validation error: {e}")
            return False

    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is a subpath of parent.

        Args:
            path: Path to check
            parent: Potential parent path

        Returns:
            True if path is under parent
        """
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def get_agent_permissions(
        self,
        agent_type: str,
        swarm_name: str,
    ) -> dict[str, Any]:
        """Get permissions based on agent type and swarm.

        Returns a dictionary of permission flags for the given agent.
        Swarm Dev agents get special elevated permissions for
        self-modification capability.

        Args:
            agent_type: Type of agent (orchestrator, implementer, critic, etc.)
            swarm_name: Name of the swarm

        Returns:
            Dictionary with permission flags:
            - allow_project_root: Can access files outside workspace
            - git_access: Can perform git operations
            - bash_allowed: Can execute bash commands
            - permission_mode: Claude CLI permission mode
            - web_access: Can access web resources
            - allowed_tools: List of allowed tool names
        """
        # Swarm Dev gets special permissions for self-modification
        if swarm_name == "swarm_dev":
            return {
                "allow_project_root": True,
                "git_access": True,
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"
                ],
            }

        # Permission levels by agent type for regular swarms
        permissions_by_type: dict[str, dict[str, Any]] = {
            "orchestrator": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"
                ],
            },
            "implementer": {
                "allow_project_root": False,
                "git_access": True,  # Can commit in their workspace
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep"
                ],
            },
            "architect": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep"
                ],
            },
            "critic": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": False,  # Read-only by design
                "permission_mode": "default",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Glob", "Grep"
                ],
            },
            "reviewer": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,  # Can run tests
                "permission_mode": "default",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Bash", "Glob", "Grep"
                ],
            },
            "researcher": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,
                "permission_mode": "default",
                "web_access": True,
                "allowed_tools": [
                    "Read", "Write", "Bash", "Glob", "Grep"
                ],
            },
            "tester": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
                "web_access": False,
                "allowed_tools": [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep"
                ],
            },
            "monitor": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,  # Limited commands
                "permission_mode": "default",
                "web_access": False,
                "allowed_tools": [
                    "Read", "Bash", "Glob", "Grep"
                ],
            },
        }

        # Return type-specific permissions or default
        if agent_type in permissions_by_type:
            return permissions_by_type[agent_type]

        # Default permissions for unknown agent types
        return {
            "allow_project_root": False,
            "git_access": False,
            "bash_allowed": True,
            "permission_mode": "default",
            "web_access": True,
            "allowed_tools": [
                "Read", "Write", "Edit", "Bash", "Glob", "Grep"
            ],
        }

    def get_state_file(self, swarm_name: str) -> Path:
        """Get the STATE.md file path for a swarm.

        Args:
            swarm_name: Name of the swarm

        Returns:
            Path to the swarm's STATE.md file
        """
        workspace = self.get_workspace(swarm_name)
        return workspace / "STATE.md"

    def ensure_workspace_structure(self, swarm_name: str) -> None:
        """Ensure the workspace has the expected directory structure.

        Creates workspace directory and any required subdirectories
        if they don't exist.

        Args:
            swarm_name: Name of the swarm
        """
        workspace = self.get_workspace(swarm_name)

        # Create workspace if needed
        workspace.mkdir(parents=True, exist_ok=True)

        # Create STATE.md if it doesn't exist
        state_file = workspace / "STATE.md"
        if not state_file.exists():
            state_file.write_text(f"# {swarm_name} - STATE.md\n\nInitialized.\n")
            logger.info(f"Created STATE.md for swarm: {swarm_name}")


# Module-level singleton with thread-safe initialization
import threading

_workspace_manager: WorkspaceManager | None = None
_workspace_manager_lock = threading.Lock()


def get_workspace_manager(project_root: Path | None = None) -> WorkspaceManager:
    """Get or create the global workspace manager.

    Thread-safe singleton pattern with double-checked locking.

    Args:
        project_root: Optional project root (required for first call)

    Returns:
        The workspace manager singleton

    Raises:
        ValueError: If project_root not provided on first call
    """
    global _workspace_manager

    if _workspace_manager is None:
        with _workspace_manager_lock:
            # Double-check after acquiring lock
            if _workspace_manager is None:
                if project_root is None:
                    raise ValueError(
                        "project_root must be provided on first call to get_workspace_manager"
                    )
                _workspace_manager = WorkspaceManager(project_root)

    return _workspace_manager
