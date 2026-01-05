"""Escalation Protocol API endpoints.

This module provides endpoints for the escalation protocol including:
- Listing escalations
- Creating escalations
- Resolving escalations
- Status updates
"""

from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException

from models.requests import EscalationCreateRequest
from shared.escalation_protocol import (
    get_escalation_manager,
    EscalationLevel,
    EscalationReason,
    EscalationPriority,
    EscalationStatus,
)

router = APIRouter(prefix="/api/escalations", tags=["escalations"])


@router.get("")
async def list_escalations(
    status: Optional[str] = None,
    level: Optional[str] = None,
    swarm: Optional[str] = None,
) -> List[Dict]:
    """List escalations with optional filters.

    Args:
        status: Filter by status (pending, blocking)
        level: Filter by escalation level
        swarm: Filter by swarm name

    Returns:
        List of escalation dictionaries
    """
    manager = get_escalation_manager()

    if status == "pending":
        target_level = EscalationLevel(level) if level else None
        items = manager.get_pending(level=target_level)
    elif swarm:
        items = manager.get_by_swarm(swarm)
    elif status == "blocking":
        items = manager.get_blocked_work()
    else:
        # Return all pending by default
        items = manager.get_pending()

    return [item.to_dict() for item in items]


@router.post("")
async def create_escalation(request: EscalationCreateRequest) -> dict:
    """Create a new escalation.

    Args:
        request: Escalation creation request

    Returns:
        Dictionary with success status and escalation details

    Raises:
        HTTPException: If escalation parameters are invalid
    """
    manager = get_escalation_manager()

    try:
        escalation = manager.create_escalation(
            from_level=EscalationLevel(request.from_level),
            to_level=EscalationLevel(request.to_level),
            reason=EscalationReason(request.reason),
            title=request.context.get("title", "Escalation"),
            description=request.context.get("description", ""),
            created_by=request.agent_name or "unknown",
            priority=EscalationPriority(request.priority),
            swarm_name=request.swarm_name,
            blocked_tasks=[request.work_id] if request.work_id else None,
            related_files=None,
            context=request.context,
        )

        return {"success": True, "escalation_id": escalation.id, "escalation": escalation.to_dict()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{escalation_id}")
async def get_escalation(escalation_id: str) -> dict:
    """Get a specific escalation.

    Args:
        escalation_id: The escalation ID

    Returns:
        Escalation dictionary

    Raises:
        HTTPException: If escalation not found
    """
    manager = get_escalation_manager()

    # Access internal dict since there's no get_by_id method
    if escalation_id not in manager._escalations:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")

    return manager._escalations[escalation_id].to_dict()


@router.post("/{escalation_id}/resolve")
async def resolve_escalation(
    escalation_id: str,
    resolution: str,
    resolved_by: str,
) -> dict:
    """Resolve an escalation.

    Args:
        escalation_id: The escalation ID
        resolution: Resolution description
        resolved_by: Who resolved the escalation

    Returns:
        Dictionary with success status and escalation details

    Raises:
        HTTPException: If escalation not found
    """
    manager = get_escalation_manager()
    escalation = manager.resolve_escalation(escalation_id, resolution, resolved_by)

    if not escalation:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")

    return {"success": True, "escalation": escalation.to_dict()}


@router.post("/{escalation_id}/status")
async def update_escalation_status(escalation_id: str, status: str) -> dict:
    """Update an escalation's status.

    Args:
        escalation_id: The escalation ID
        status: New status value

    Returns:
        Dictionary with success status and escalation details

    Raises:
        HTTPException: If escalation not found
    """
    manager = get_escalation_manager()
    escalation = manager.update_status(escalation_id, EscalationStatus(status))

    if not escalation:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")

    return {"success": True, "escalation": escalation.to_dict()}
