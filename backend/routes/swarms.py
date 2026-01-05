"""Swarm management API endpoints.

This module provides endpoints for managing swarms including:
- Listing swarms
- Getting swarm details
- Creating swarms
- Listing agents in a swarm
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from models.requests import SwarmCreate
from services.orchestrator_service import get_orchestrator

router = APIRouter(prefix="/api/swarms", tags=["swarms"])


@router.get("")
async def list_swarms() -> list[dict[str, Any]]:
    """List all swarms with status.

    Returns:
        List of swarm summary dictionaries
    """
    orch = get_orchestrator()
    swarms = []

    for _name, swarm in orch.swarms.items():
        status = swarm.get_status()
        priorities = status.get("priorities", [])
        # Handle dict priorities
        priority_strs = []
        for p in priorities[:3]:
            if isinstance(p, dict):
                priority_strs.append(p.get("task", str(p)))
            else:
                priority_strs.append(str(p))

        swarms.append(
            {
                "name": status["name"],
                "description": status["description"],
                "status": status["status"],
                "agent_count": status["agent_count"],
                "priorities": priority_strs,
                "version": status.get("version", "0.1.0"),
            }
        )

    return swarms


@router.get("/{name}")
async def get_swarm(name: str) -> dict[str, Any]:
    """Get detailed swarm information.

    Args:
        name: The swarm name

    Returns:
        Swarm status dictionary

    Raises:
        HTTPException: If swarm not found
    """
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    status = swarm.get_status()
    return status


@router.post("")
async def create_swarm(data: SwarmCreate) -> dict[str, Any]:
    """Create a new swarm from template.

    Args:
        data: Swarm creation request

    Returns:
        Dictionary with success status and swarm name

    Raises:
        HTTPException: If swarm already exists or template not found
    """
    orch = get_orchestrator()

    try:
        swarm = orch.create_swarm(data.name, data.description, data.template)
        return {
            "success": True,
            "name": swarm.name,
            "message": f"Created swarm: {swarm.name}",
        }
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{name}/agents")
async def list_agents(name: str) -> list[dict[str, Any]]:
    """List agents in a swarm.

    Args:
        name: The swarm name

    Returns:
        List of agent info dictionaries

    Raises:
        HTTPException: If swarm not found
    """
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    agents = []
    for agent_name, defn in swarm.agent_definitions.items():
        agents.append(
            {
                "name": agent_name,
                "type": defn.agent_type,
                "model": defn.model,
                "background": defn.background,
                "description": defn.description,
            }
        )

    return agents
