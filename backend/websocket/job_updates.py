"""Job update WebSocket endpoint and broadcasting.

This module handles WebSocket subscriptions for job status updates
and broadcasting updates to connected clients.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from jobs import Job

logger = logging.getLogger(__name__)

# Job update subscribers: job_id -> list of WebSocket connections
# Special key "all" for subscribers wanting all updates
job_update_subscribers: dict[str, list[WebSocket]] = {}


async def websocket_jobs(websocket: WebSocket):
    """WebSocket endpoint for job updates.

    Clients can subscribe to specific job updates or all updates.

    Actions:
        - subscribe: Subscribe to a specific job_id
        - subscribe_all: Subscribe to all job updates
        - unsubscribe: Unsubscribe from a specific job_id
        - list: Get current running and pending jobs
    """
    # Import here to avoid circular imports
    from jobs import get_job_queue

    await websocket.accept()
    subscribed_jobs: set[str] = set()

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "subscribe":
                # Subscribe to a specific job's updates
                job_id = data.get("job_id")
                if job_id:
                    if job_id not in job_update_subscribers:
                        job_update_subscribers[job_id] = []
                    job_update_subscribers[job_id].append(websocket)
                    subscribed_jobs.add(job_id)

                    # Send current job status
                    queue = get_job_queue()
                    job = queue.get_job(job_id)
                    if job:
                        await websocket.send_json({
                            "type": "job_status",
                            "job": job.to_dict(),
                        })

            elif action == "subscribe_all":
                # Subscribe to all job updates
                if "all" not in job_update_subscribers:
                    job_update_subscribers["all"] = []
                job_update_subscribers["all"].append(websocket)
                subscribed_jobs.add("all")

            elif action == "unsubscribe":
                job_id = data.get("job_id")
                if job_id and job_id in subscribed_jobs:
                    if job_id in job_update_subscribers:
                        try:
                            job_update_subscribers[job_id].remove(websocket)
                        except ValueError:
                            pass
                    subscribed_jobs.discard(job_id)

            elif action == "list":
                # Get current jobs
                queue = get_job_queue()
                running = queue.get_running_jobs()
                pending = queue.get_pending_jobs()
                await websocket.send_json({
                    "type": "job_list",
                    "running": [j.to_dict() for j in running],
                    "pending": [j.to_dict() for j in pending],
                })

    except WebSocketDisconnect:
        pass
    finally:
        # Clean up subscriptions
        for job_id in subscribed_jobs:
            if job_id in job_update_subscribers:
                if websocket in job_update_subscribers[job_id]:
                    try:
                        job_update_subscribers[job_id].remove(websocket)
                    except ValueError:
                        pass


async def broadcast_job_update(job: "Job"):
    """Broadcast job update to subscribers.

    Sends update to:
    - Job-specific subscribers (by job.id)
    - "all" subscribers who want all updates

    Args:
        job: The job that was updated
    """
    job_dict = job.to_dict()
    message = {"type": "job_update", "job": job_dict}

    # Send to job-specific subscribers
    if job.id in job_update_subscribers:
        for ws in job_update_subscribers[job.id][:]:
            try:
                await ws.send_json(message)
            except Exception:
                try:
                    job_update_subscribers[job.id].remove(ws)
                except ValueError:
                    pass

    # Send to "all" subscribers
    if "all" in job_update_subscribers:
        for ws in job_update_subscribers["all"][:]:
            try:
                await ws.send_json(message)
            except Exception:
                try:
                    job_update_subscribers["all"].remove(ws)
                except ValueError:
                    pass
