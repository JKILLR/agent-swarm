"""
Background Job System for Agent Swarm

Provides:
- Job queue with SQLite persistence
- Concurrent job execution
- Progress tracking via WebSocket
- Job resumption after restart
"""

import asyncio
import json
import logging
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# Add project root to path for imports
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.workspace_manager import get_workspace_manager
from shared.agent_executor_pool import get_executor_pool
from shared.execution_context import AgentExecutionContext

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a background job."""
    id: str
    type: str  # "chat", "task", "swarm_directive"
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Job parameters
    prompt: str = ""
    swarm: Optional[str] = None
    session_id: Optional[str] = None

    # Progress tracking
    progress: int = 0  # 0-100
    current_activity: str = ""
    activities: list[dict] = field(default_factory=list)

    # Results
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "swarm": self.swarm,
            "session_id": self.session_id,
            "progress": self.progress,
            "current_activity": self.current_activity,
            "activities": self.activities[-10:],  # Last 10 activities
            "result": self.result[:500] if self.result and len(self.result) > 500 else self.result,
            "error": self.error,
        }


class JobQueue:
    """SQLite-backed job queue with persistence."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    prompt TEXT,
                    swarm TEXT,
                    session_id TEXT,
                    progress INTEGER DEFAULT 0,
                    current_activity TEXT,
                    activities TEXT,
                    result TEXT,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id)
            """)
            conn.commit()

    def create_job(
        self,
        job_type: str,
        prompt: str,
        swarm: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Job:
        """Create a new job."""
        job = Job(
            id=str(uuid.uuid4()),
            type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            prompt=prompt,
            swarm=swarm,
            session_id=session_id,
        )
        self._save_job(job)
        logger.info(f"Created job {job.id}: {job_type}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row:
                return self._row_to_job(row)
        return None

    def get_pending_jobs(self) -> list[Job]:
        """Get all pending jobs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at",
                (JobStatus.PENDING.value,)
            ).fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_running_jobs(self) -> list[Job]:
        """Get all running jobs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY started_at",
                (JobStatus.RUNNING.value,)
            ).fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_session_jobs(self, session_id: str) -> list[Job]:
        """Get all jobs for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,)
            ).fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_recent_jobs(self, limit: int = 20) -> list[Job]:
        """Get recent jobs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._row_to_job(row) for row in rows]

    def update_job(self, job: Job):
        """Update a job in the database."""
        self._save_job(job)

    def _save_job(self, job: Job):
        """Save a job to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs
                (id, type, status, created_at, started_at, completed_at,
                 prompt, swarm, session_id, progress, current_activity,
                 activities, result, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id,
                job.type,
                job.status.value,
                job.created_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.prompt,
                job.swarm,
                job.session_id,
                job.progress,
                job.current_activity,
                json.dumps(job.activities),
                job.result,
                job.error,
            ))
            conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert a database row to a Job object."""
        return Job(
            id=row["id"],
            type=row["type"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            prompt=row["prompt"] or "",
            swarm=row["swarm"],
            session_id=row["session_id"],
            progress=row["progress"] or 0,
            current_activity=row["current_activity"] or "",
            activities=json.loads(row["activities"]) if row["activities"] else [],
            result=row["result"],
            error=row["error"],
        )


class JobManager:
    """
    Manages concurrent job execution.

    Features:
    - Runs multiple jobs in parallel
    - Tracks progress and sends updates
    - Handles job cancellation
    - Persists state for resumption
    """

    def __init__(
        self,
        queue: JobQueue,
        max_concurrent: int = 3,
        on_job_update: Optional[Callable[[Job], None]] = None,
    ):
        self.queue = queue
        self.max_concurrent = max_concurrent
        self.on_job_update = on_job_update
        self.use_pool = True  # Use AgentExecutorPool for better isolation

        # Track running tasks
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._job_processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

        # Event for new jobs
        self._new_job_event = asyncio.Event()

    async def start(self):
        """Start the job manager worker loop."""
        logger.info("Starting job manager...")

        # Mark any previously "running" jobs as pending (for restart recovery)
        for job in self.queue.get_running_jobs():
            logger.info(f"Recovering job {job.id} from previous run")
            job.status = JobStatus.PENDING
            job.current_activity = "Queued for retry"
            self.queue.update_job(job)

        # Start worker loop
        asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        """Main worker loop that processes jobs."""
        while True:
            try:
                # Check for pending jobs if we have capacity
                if len(self._running_tasks) < self.max_concurrent:
                    pending = self.queue.get_pending_jobs()
                    for job in pending:
                        if len(self._running_tasks) >= self.max_concurrent:
                            break
                        if job.id not in self._running_tasks:
                            # Start the job
                            task = asyncio.create_task(self._execute_job(job))
                            self._running_tasks[job.id] = task

                # Clean up completed tasks
                completed = [
                    job_id for job_id, task in self._running_tasks.items()
                    if task.done()
                ]
                for job_id in completed:
                    del self._running_tasks[job_id]

                # Wait for new jobs or check periodically
                try:
                    await asyncio.wait_for(self._new_job_event.wait(), timeout=1.0)
                    self._new_job_event.clear()
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)

    async def submit_job(
        self,
        job_type: str,
        prompt: str,
        swarm: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Job:
        """Submit a new job to the queue."""
        job = self.queue.create_job(job_type, prompt, swarm, session_id)
        self._new_job_event.set()  # Wake up worker
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.queue.get_job(job_id)
        if not job:
            return False

        self._cancelled.add(job_id)

        # Kill the process if running
        if job_id in self._job_processes:
            process = self._job_processes[job_id]
            try:
                process.kill()
                await process.wait()
            except Exception as e:
                logger.error(f"Error killing job process: {e}")

        # Cancel the task
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()

        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        job.current_activity = "Cancelled by user"
        self.queue.update_job(job)

        if self.on_job_update:
            self.on_job_update(job)

        logger.info(f"Cancelled job {job_id}")
        return True

    async def _execute_job(self, job: Job):
        """Execute a single job."""
        logger.info(f"Starting job {job.id}: {job.type}")

        # Update status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        job.current_activity = "Starting..."
        self.queue.update_job(job)

        if self.on_job_update:
            self.on_job_update(job)

        try:
            # Execute based on job type
            if job.type == "chat":
                result = await self._execute_chat_job(job)
            elif job.type == "swarm_directive":
                result = await self._execute_swarm_directive(job)
            else:
                result = await self._execute_generic_job(job)

            # Mark completed
            if job.id not in self._cancelled:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result = result
                job.progress = 100
                job.current_activity = "Completed"

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.current_activity = "Cancelled"

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = str(e)
            job.current_activity = f"Failed: {str(e)[:100]}"

        finally:
            # Clean up
            if job.id in self._job_processes:
                del self._job_processes[job.id]
            if job.id in self._cancelled:
                self._cancelled.discard(job.id)

            self.queue.update_job(job)

            if self.on_job_update:
                self.on_job_update(job)

    async def _execute_chat_job(self, job: Job) -> str:
        """Execute a chat job using Claude CLI."""
        # Use pool if enabled
        if self.use_pool:
            return await self._execute_with_pool(job)

        # Original implementation below (fallback)
        from main import stream_claude_response

        job.current_activity = "Initializing Claude..."
        self.queue.update_job(job)

        # Start Claude CLI process
        process = await stream_claude_response(
            prompt=job.prompt,
            swarm_name=job.swarm,
            workspace=PROJECT_ROOT,
        )

        self._job_processes[job.id] = process

        async def drain_stderr():
            """Drain stderr to prevent buffer deadlock."""
            if process.stderr:
                try:
                    while True:
                        chunk = await process.stderr.read(65536)
                        if not chunk:
                            break
                except Exception:
                    pass

        # Start draining stderr concurrently to prevent deadlock
        stderr_task = asyncio.create_task(drain_stderr())

        # Collect response
        full_response = ""

        try:
            if process.stdout:
                job.current_activity = "Processing..."
                self.queue.update_job(job)

                while True:
                    if job.id in self._cancelled:
                        break

                    try:
                        line = await asyncio.wait_for(
                            process.stdout.readline(),
                            timeout=1.0
                        )
                        if not line:
                            break

                        # Parse and track progress
                        try:
                            event = json.loads(line.decode().strip())
                            event_type = event.get("type", "")

                            # Track tool use as activity
                            if event_type == "content_block_start":
                                block = event.get("content_block", {})
                                if block.get("type") == "tool_use":
                                    tool_name = block.get("name", "tool")
                                    job.current_activity = f"Using {tool_name}..."
                                    job.activities.append({
                                        "tool": tool_name,
                                        "time": datetime.now().isoformat(),
                                    })
                                    self.queue.update_job(job)
                                    if self.on_job_update:
                                        self.on_job_update(job)

                            # Collect text response
                            elif event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    full_response += delta.get("text", "")

                            elif event_type == "result":
                                full_response = event.get("result", full_response)

                        except json.JSONDecodeError:
                            pass

                    except asyncio.TimeoutError:
                        # Check if process is still running
                        if process.returncode is not None:
                            break
                        continue
        finally:
            # Ensure stderr task completes
            try:
                await asyncio.wait_for(stderr_task, timeout=5.0)
            except asyncio.TimeoutError:
                stderr_task.cancel()

        await process.wait()
        return full_response

    async def _execute_swarm_directive(self, job: Job) -> str:
        """Execute a swarm directive job."""
        # Similar to chat but targeted at a specific swarm
        job.current_activity = f"Directing {job.swarm}..."
        self.queue.update_job(job)

        # Use the same chat execution with swarm context
        return await self._execute_chat_job(job)

    async def _execute_generic_job(self, job: Job) -> str:
        """Execute a generic job."""
        job.current_activity = "Processing..."
        self.queue.update_job(job)

        return await self._execute_chat_job(job)

    async def _execute_with_pool(self, job: Job) -> str:
        """Execute a job using the AgentExecutorPool for better isolation."""
        workspace_manager = get_workspace_manager(PROJECT_ROOT)
        pool = get_executor_pool(workspace_manager=workspace_manager)

        # Determine workspace and agent type
        swarm_name = job.swarm or "swarm_dev"  # Default to swarm_dev for COO tasks
        agent_type = "orchestrator" if job.type == "chat" else "implementer"

        workspace = workspace_manager.get_workspace(swarm_name)
        permissions = workspace_manager.get_agent_permissions(agent_type, swarm_name)

        # Build execution context
        context = AgentExecutionContext(
            agent_name=f"job_{job.id[:8]}",
            agent_type=agent_type,
            swarm_name=swarm_name,
            workspace=workspace,
            allowed_tools=permissions["allowed_tools"],
            permission_mode=permissions["permission_mode"],
            git_credentials=permissions["git_access"],
            web_access=permissions["web_access"],
            job_id=job.id,
        )

        job.current_activity = "Executing with pool..."
        self.queue.update_job(job)

        full_response = ""

        async for event in pool.execute(context, job.prompt):
            event_type = event.get("type", "")

            # Update job progress based on events
            if event_type == "tool_start":
                tool_name = event.get("tool", "tool")
                job.current_activity = f"Using {tool_name}..."
                job.activities.append({
                    "tool": tool_name,
                    "time": datetime.now().isoformat(),
                })
                self.queue.update_job(job)
                if self.on_job_update:
                    self.on_job_update(job)

            elif event_type == "agent_delta":
                full_response += event.get("delta", "")

            elif event_type == "content":
                full_response += event.get("content", "")

            elif event_type == "agent_execution_complete":
                if not event.get("success"):
                    raise RuntimeError(event.get("result_summary", "Agent execution failed"))

        return full_response

    def get_status(self) -> dict:
        """Get job manager status."""
        return {
            "running_jobs": len(self._running_tasks),
            "max_concurrent": self.max_concurrent,
            "pending_count": len(self.queue.get_pending_jobs()),
            "running_ids": list(self._running_tasks.keys()),
        }


# Singleton instances
_job_queue: Optional[JobQueue] = None
_job_manager: Optional[JobManager] = None


def get_job_queue(db_path: Optional[Path] = None) -> JobQueue:
    """Get the singleton job queue."""
    global _job_queue
    if _job_queue is None:
        if db_path is None:
            db_path = Path(__file__).parent.parent / "logs" / "jobs.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _job_queue = JobQueue(db_path)
    return _job_queue


def get_job_manager() -> JobManager:
    """Get the singleton job manager."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager(get_job_queue())
    return _job_manager
