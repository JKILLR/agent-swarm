"""Auto-Spawn on Work Detection.

This module provides automatic agent spawning when new work items are
added to the Work Ledger without an assigned agent.

Quick Win Implementation:
- When work is added to the ledger, check if an appropriate agent should be spawned
- Uses swarm configuration to determine agent types
- Integrates with AgentExecutorPool for actual spawning

Usage:
    from shared.auto_spawn import enable_auto_spawn

    # Enable auto-spawn for the global work ledger
    enable_auto_spawn()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .work_models import WorkItem, WorkType

logger = logging.getLogger(__name__)


# Mapping from WorkType to suggested agent role
WORK_TYPE_TO_AGENT_ROLE: dict[WorkType, str] = {
    WorkType.TASK: "implementer",
    WorkType.FEATURE: "implementer",
    WorkType.BUG: "implementer",
    WorkType.RESEARCH: "researcher",
    WorkType.REVIEW: "reviewer",
    WorkType.DESIGN: "architect",
    WorkType.REFACTOR: "refactorer",
    WorkType.TEST: "implementer",
    WorkType.DOCUMENTATION: "implementer",
    WorkType.ESCALATION: "orchestrator",
}


def get_agent_for_work_type(work_type: WorkType) -> str:
    """Get the appropriate agent role for a work type.

    Args:
        work_type: The type of work item

    Returns:
        Agent role name (e.g., "implementer", "architect")
    """
    return WORK_TYPE_TO_AGENT_ROLE.get(work_type, "implementer")


def should_auto_spawn(work_item: WorkItem) -> bool:
    """Determine if an agent should be auto-spawned for this work item.

    Conditions for auto-spawn:
    - Work item has no owner (unclaimed)
    - Work item has a swarm_name (we need to know which swarm)
    - Work item is not an escalation (those need human/COO review)

    Args:
        work_item: The newly created work item

    Returns:
        True if an agent should be spawned
    """
    # Must be unclaimed
    if work_item.owner is not None:
        return False

    # Must have a swarm to spawn into
    if not work_item.swarm_name:
        logger.debug(f"Work item {work_item.id} has no swarm_name, skipping auto-spawn")
        return False

    # Don't auto-spawn for escalations (need COO/CEO review)
    if work_item.type == WorkType.ESCALATION:
        logger.debug(f"Work item {work_item.id} is escalation, skipping auto-spawn")
        return False

    return True


async def spawn_agent_for_work(
    work_item: WorkItem,
    swarm_config_dir: Path | None = None,
) -> bool:
    """Spawn an agent to handle a work item.

    This is the core auto-spawn function that:
    1. Determines the appropriate agent type
    2. Creates an execution context
    3. Spawns the agent via AgentExecutorPool

    Args:
        work_item: The work item to process
        swarm_config_dir: Directory containing swarm configurations

    Returns:
        True if agent was spawned successfully
    """
    from .agent_executor_pool import get_executor_pool
    from .workspace_manager import get_workspace_manager
    from .execution_context import AgentExecutionContext
    from .work_ledger import get_work_ledger

    try:
        agent_role = get_agent_for_work_type(work_item.type)
        swarm_name = work_item.swarm_name

        if not swarm_name:
            logger.warning(f"Cannot spawn agent for {work_item.id}: no swarm_name")
            return False

        # Get workspace manager
        workspace_mgr = get_workspace_manager()

        # Check if swarm exists and has the agent type
        swarm_config = workspace_mgr.get_swarm_config(swarm_name)
        if not swarm_config:
            logger.warning(f"Swarm {swarm_name} not found, cannot auto-spawn")
            return False

        # Find matching agent in swarm
        agents = swarm_config.get("agents", [])
        agent_config = None
        for agent in agents:
            if agent.get("role") == agent_role or agent.get("name") == agent_role:
                agent_config = agent
                break

        if not agent_config:
            # Fallback to implementer if specific role not found
            for agent in agents:
                if agent.get("role") == "implementer" or agent.get("name") == "implementer":
                    agent_config = agent
                    break

        if not agent_config:
            logger.warning(f"No suitable agent found in {swarm_name} for {work_item.type}")
            return False

        agent_name = agent_config.get("name", agent_role)

        # Claim the work first
        ledger = get_work_ledger()
        claimed = ledger.claim_work(work_item.id, agent_name)
        if not claimed:
            logger.warning(f"Could not claim work {work_item.id} for {agent_name}")
            return False

        # Resolve workspace
        workspace = workspace_mgr.resolve_workspace(swarm_name)

        # Build execution context
        context = AgentExecutionContext(
            agent_name=agent_name,
            agent_type=agent_config.get("role", "worker"),
            swarm_name=swarm_name,
            workspace=workspace,
            max_turns=agent_config.get("max_turns", 20),
            timeout=600,  # 10 minute timeout
            permission_mode="bypassPermissions",  # Consistent with current setup
        )

        # Build prompt with work context
        prompt = _build_work_prompt(work_item)

        # Get executor pool
        pool = get_executor_pool()

        # Spawn agent (fire and forget for now)
        logger.info(f"Auto-spawning {agent_name} for work item {work_item.id}")

        async def _execute_and_complete():
            """Execute agent and update work status on completion."""
            result_content = ""
            success = False
            try:
                async for event in pool.execute(context, prompt):
                    event_type = event.get("type", "")
                    if event_type == "agent_execution_complete":
                        success = event.get("success", False)
                    elif event_type == "content":
                        result_content += event.get("content", "")
                    elif event_type == "agent_delta":
                        result_content += event.get("delta", "")

                # Update work item status
                if success:
                    ledger.complete_work(
                        work_item.id,
                        agent_name,
                        result={"summary": result_content[:500]}
                    )
                else:
                    ledger.fail_work(
                        work_item.id,
                        agent_name,
                        error="Agent execution failed"
                    )
            except Exception as e:
                logger.error(f"Error in auto-spawned agent for {work_item.id}: {e}")
                ledger.fail_work(work_item.id, agent_name, error=str(e))

        # Create background task
        asyncio.create_task(_execute_and_complete())

        return True

    except Exception as e:
        logger.error(f"Error spawning agent for {work_item.id}: {e}")
        return False


def _build_work_prompt(work_item: WorkItem) -> str:
    """Build a prompt for the agent based on the work item.

    Args:
        work_item: The work item to process

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        f"# Work Item: {work_item.title}",
        f"**ID**: {work_item.id}",
        f"**Type**: {work_item.type.value}",
        f"**Priority**: {work_item.priority.value}",
        "",
        "## Description",
        work_item.description,
    ]

    if work_item.context:
        prompt_parts.extend([
            "",
            "## Additional Context",
        ])
        for key, value in work_item.context.items():
            prompt_parts.append(f"- **{key}**: {value}")

    prompt_parts.extend([
        "",
        "## Instructions",
        "1. Read the relevant STATE.md file for context",
        "2. Complete the work described above",
        "3. Update STATE.md with your progress",
        "4. Ensure all tests pass if applicable",
    ])

    return "\n".join(prompt_parts)


def on_work_created_handler(work_item: WorkItem) -> None:
    """Handler for work creation events - triggers auto-spawn.

    This is the callback registered with WorkLedger.
    It checks if auto-spawn should happen and initiates it.

    Args:
        work_item: The newly created work item
    """
    if not should_auto_spawn(work_item):
        return

    # Schedule async spawn in the event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(spawn_agent_for_work(work_item))
        else:
            # If no loop running, we're likely in sync context
            # Just log for now - the work will be picked up by polling
            logger.info(f"Work item {work_item.id} ready for auto-spawn (deferred)")
    except RuntimeError:
        # No event loop available
        logger.debug(f"Work item {work_item.id} ready for auto-spawn (no event loop)")


def enable_auto_spawn() -> None:
    """Enable auto-spawn for the global work ledger.

    Registers the on_work_created_handler with the global WorkLedger
    singleton so that agents are automatically spawned for new work.

    Usage:
        from shared.auto_spawn import enable_auto_spawn
        enable_auto_spawn()
    """
    from .work_ledger import get_work_ledger

    ledger = get_work_ledger()
    ledger.set_on_work_created(on_work_created_handler)
    logger.info("Auto-spawn enabled for work ledger")


def disable_auto_spawn() -> None:
    """Disable auto-spawn for the global work ledger."""
    from .work_ledger import get_work_ledger

    ledger = get_work_ledger()
    ledger.set_on_work_created(None)
    logger.info("Auto-spawn disabled for work ledger")
