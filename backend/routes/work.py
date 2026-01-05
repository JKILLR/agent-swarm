"""Work Ledger API endpoints.

This module provides endpoints for persistent work tracking including:
- Listing and filtering work items
- Creating work items
- Claiming and completing work
- Progress tracking
- Recovery of orphaned work
"""

from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException

from models.requests import WorkCreateRequest
from shared.work_ledger import get_work_ledger
from shared.work_models import WorkType, WorkPriority

router = APIRouter(prefix="/api/work", tags=["work"])


@router.get("")
async def list_work_items(
    status: Optional[str] = None,
    swarm: Optional[str] = None,
    limit: int = 50,
) -> List[Dict]:
    """List work items with optional filters.

    Args:
        status: Filter by status (pending, in_progress, blocked, ready)
        swarm: Filter by swarm name
        limit: Maximum number of items to return

    Returns:
        List of work item dictionaries
    """
    ledger = get_work_ledger()

    if status == "pending":
        items = ledger.get_pending(swarm_name=swarm)
    elif status == "in_progress":
        items = ledger.get_in_progress()
    elif status == "blocked":
        items = ledger.get_blocked()
    elif status == "ready":
        items = ledger.get_ready_to_start(swarm_name=swarm)
    elif swarm:
        items = ledger.get_by_swarm(swarm)
    else:
        # Return recent items across all statuses
        items = []
        items.extend(ledger.get_in_progress())
        items.extend(ledger.get_pending())
        items.extend(ledger.get_blocked())

    return [item.to_dict() for item in items[:limit]]


@router.post("")
async def create_work_item(request: WorkCreateRequest) -> dict:
    """Create a new work item.

    Args:
        request: Work creation request

    Returns:
        Dictionary with success status and work item details
    """
    ledger = get_work_ledger()

    # Map string to enum
    work_type = WorkType(request.work_type)
    priority = WorkPriority(request.priority)

    item = ledger.create_work(
        title=request.title,
        description=request.description,
        work_type=work_type,
        priority=priority,
        parent_id=request.parent_id,
        swarm_name=request.swarm_name,
        context=request.context,
    )

    return {"success": True, "work_id": item.id, "work": item.to_dict()}


@router.get("/{work_id}")
async def get_work_item(work_id: str) -> dict:
    """Get a specific work item.

    Args:
        work_id: The work item ID

    Returns:
        Work item dictionary

    Raises:
        HTTPException: If work item not found
    """
    ledger = get_work_ledger()
    item = ledger.get_work(work_id)

    if not item:
        raise HTTPException(status_code=404, detail=f"Work item {work_id} not found")

    return item.to_dict()


@router.post("/{work_id}/claim")
async def claim_work_item(work_id: str, owner: str) -> dict:
    """Claim a work item for processing.

    Args:
        work_id: The work item ID
        owner: The agent claiming the work

    Returns:
        Dictionary with success status and work item

    Raises:
        HTTPException: If work item cannot be claimed
    """
    ledger = get_work_ledger()
    item = ledger.claim_work(work_id, owner)

    if not item:
        raise HTTPException(status_code=400, detail=f"Could not claim work item {work_id}")

    return {"success": True, "work": item.to_dict()}


@router.post("/{work_id}/complete")
async def complete_work_item(work_id: str, owner: str, result: Optional[Dict] = None) -> dict:
    """Mark a work item as completed.

    Args:
        work_id: The work item ID
        owner: The agent completing the work
        result: Optional result data

    Returns:
        Dictionary with success status and work item

    Raises:
        HTTPException: If work item cannot be completed
    """
    ledger = get_work_ledger()
    item = ledger.complete_work(work_id, owner, result)

    if not item:
        raise HTTPException(status_code=400, detail=f"Could not complete work item {work_id}")

    return {"success": True, "work": item.to_dict()}


@router.post("/{work_id}/fail")
async def fail_work_item(work_id: str, owner: str, error: str) -> dict:
    """Mark a work item as failed.

    Args:
        work_id: The work item ID
        owner: The agent marking the failure
        error: Error description

    Returns:
        Dictionary with success status and work item

    Raises:
        HTTPException: If work item cannot be marked as failed
    """
    ledger = get_work_ledger()
    item = ledger.fail_work(work_id, owner, error)

    if not item:
        raise HTTPException(status_code=400, detail=f"Could not fail work item {work_id}")

    return {"success": True, "work": item.to_dict()}


@router.get("/{work_id}/progress")
async def get_work_progress(work_id: str) -> dict:
    """Get progress summary for a work item including children.

    Args:
        work_id: The work item ID

    Returns:
        Progress summary dictionary
    """
    ledger = get_work_ledger()
    return ledger.get_progress(work_id)


@router.post("/recover")
async def recover_orphaned_work(timeout_minutes: int = 60) -> dict:
    """Recover orphaned work items that were abandoned.

    Args:
        timeout_minutes: Consider work orphaned if not updated in this time

    Returns:
        Dictionary with recovery status and recovered items
    """
    ledger = get_work_ledger()
    recovered = ledger.recover_orphaned_work(timeout_minutes)

    return {
        "success": True,
        "recovered_count": len(recovered),
        "recovered": [item.to_dict() for item in recovered],
    }
