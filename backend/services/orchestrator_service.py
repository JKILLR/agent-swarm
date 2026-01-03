"""Supreme Orchestrator management service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supreme.orchestrator import SupremeOrchestrator

# Global orchestrator instance
_orchestrator: "SupremeOrchestrator | None" = None


def get_orchestrator(project_root: Path | None = None, config_path: Path | None = None) -> "SupremeOrchestrator":
    """Get or create the Supreme Orchestrator.

    Args:
        project_root: Path to project root. Only used on first call.
        config_path: Path to config.yaml. Only used on first call.

    Returns:
        The global SupremeOrchestrator instance.
    """
    global _orchestrator
    if _orchestrator is None:
        if project_root is None:
            raise ValueError("project_root required on first call to get_orchestrator()")

        from supreme.orchestrator import SupremeOrchestrator

        _orchestrator = SupremeOrchestrator(
            base_path=project_root,
            config_path=config_path or project_root / "config.yaml",
            logs_dir=project_root / "logs",
        )
    return _orchestrator
