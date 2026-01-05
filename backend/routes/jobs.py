"""Job management API endpoints.

This module provides endpoints for managing background jobs including:
- Listing jobs
- Creating new jobs
- Getting job details
- Cancelling jobs
"""

from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException

from models.requests import JobCreate
from jobs import get_job_queue, get_job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """List background jobs.

    Args:
        session_id: Filter by session ID
        status: Filter by job status
        limit: Maximum number of jobs to return

    Returns:
        List of job dictionaries
    """
    queue = get_job_queue()

    if session_id:
        jobs = queue.get_session_jobs(session_id)
    else:
        jobs = queue.get_recent_jobs(limit)

    # Filter by status if specified
    if status:
        jobs = [j for j in jobs if j.status.value == status]

    return [j.to_dict() for j in jobs]


@router.post("")
async def create_job(data: JobCreate) -> dict:
    """Create a new background job.

    Args:
        data: Job creation request

    Returns:
        Dictionary with success status and job details
    """
    manager = get_job_manager()

    job = await manager.submit_job(
        job_type=data.type,
        prompt=data.prompt,
        swarm=data.swarm,
        session_id=data.session_id,
    )

    return {
        "success": True,
        "job": job.to_dict(),
        "message": f"Job {job.id} queued",
    }


@router.get("/status")
async def get_job_manager_status() -> dict:
    """Get job manager status.

    Returns:
        Dictionary with manager status information
    """
    manager = get_job_manager()
    return manager.get_status()


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    """Get job details.

    Args:
        job_id: The job ID to look up

    Returns:
        Job details dictionary

    Raises:
        HTTPException: If job not found
    """
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job.to_dict()


@router.delete("/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Dictionary with success status

    Raises:
        HTTPException: If job not found
    """
    manager = get_job_manager()
    success = await manager.cancel_job(job_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "success": True,
        "message": f"Job {job_id} cancelled",
    }
