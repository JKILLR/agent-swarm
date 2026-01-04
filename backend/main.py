"""FastAPI backend for Agent Swarm web interface."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import logging
import mimetypes
import os
import re
import sqlite3
import sys
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Request correlation ID context variable - accessible throughout the request lifecycle
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Import workspace manager and executor pool for agent isolation
from shared.workspace_manager import get_workspace_manager
from shared.agent_executor_pool import get_executor_pool, get_tool_description
from shared.execution_context import AgentExecutionContext
from shared.work_ledger import get_work_ledger
from shared.work_models import WorkType, WorkPriority, WorkStatus
from shared.agent_mailbox import (
    get_mailbox_manager,
    MessageType,
    MessagePriority,
    MessageStatus,
)
from shared.escalation_protocol import (
    get_escalation_manager,
    EscalationLevel,
    EscalationReason,
    EscalationPriority,
    EscalationStatus,
)
from shared.auto_spawn import enable_auto_spawn

# Add parent directory to path for imports
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env")

# Add both project root and backend dir to path for flexible imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

# Use absolute imports (work both as module and direct execution)
from memory import get_memory_manager
from supreme.orchestrator import SupremeOrchestrator
from jobs import get_job_queue, get_job_manager, JobStatus
from session_manager import get_session_manager

# Configure logging - structured format with correlation IDs
LOG_FILE = PROJECT_ROOT / "logs" / "backend.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds request_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured key=value log lines."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", "-")
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        message = record.getMessage().replace('"', '\\"')

        log_parts = [
            timestamp,
            f"level={record.levelname}",
            f"request_id={request_id}",
            f"logger={record.name}",
            f'message="{message}"',
        ]

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            exc_escaped = exc_text.replace('"', '\\"').replace("\n", "\\n")
            log_parts.append(f'exception="{exc_escaped}"')

        return " ".join(log_parts)


def setup_logging() -> logging.Logger:
    """Configure structured logging with correlation ID support."""
    correlation_filter = CorrelationIdFilter()

    console_handler = logging.StreamHandler()
    console_handler.addFilter(correlation_filter)
    console_handler.setFormatter(StructuredFormatter())

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.addFilter(correlation_filter)
    file_handler.setFormatter(StructuredFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
    )

    return logging.getLogger(__name__)


logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Agent Swarm API",
    description="API for managing hierarchical AI agent swarms",
    version="0.1.0",
)

# Enable CORS for frontend (allow local network access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local network access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that generates and propagates request correlation IDs."""

    async def dispatch(self, request: Request, call_next):
        # Check for existing correlation ID in header, or generate new one
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]

        # Set in context variable for logging
        request_id_var.set(correlation_id)

        # Log the incoming request
        logger.info(f"Request started: {request.method} {request.url.path}")

        # Process request and add correlation ID to response
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id

        logger.info(f"Request completed: {request.method} {request.url.path} status={response.status_code}")

        return response


# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Global orchestrator instance
orchestrator: SupremeOrchestrator | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Agent Swarm backend...")

    # Initialize coordination database for hooks
    db_path = PROJECT_ROOT / ".claude" / "coordination.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_log (
                id INTEGER PRIMARY KEY,
                agent TEXT,
                prompt TEXT,
                started_at TEXT,
                completed_at TEXT,
                status TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY,
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                agent TEXT,
                timestamp TEXT,
                UNIQUE(namespace, key)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_agent ON task_log(agent, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ns ON decisions(namespace)")
        conn.commit()
        conn.close()
        logger.info(f"Coordination database initialized at {db_path}")
    except Exception as e:
        logger.warning(f"Failed to initialize coordination DB: {e}")

    # Initialize job manager with broadcast callback
    job_manager = get_job_manager()

    # Set up job update callback (will be defined later in the file)
    def on_job_update(job):
        # Schedule the async broadcast in the event loop
        asyncio.create_task(_broadcast_job_update_safe(job))

    job_manager.on_job_update = on_job_update
    await job_manager.start()
    logger.info("Job manager started")

    # Initialize workspace manager and executor pool
    workspace_manager = get_workspace_manager(PROJECT_ROOT)
    executor_pool = get_executor_pool(max_concurrent=5, workspace_manager=workspace_manager)

    # Set up executor pool event broadcasting
    def on_executor_event(event: dict):
        """Broadcast executor pool events to WebSocket subscribers."""
        asyncio.create_task(_broadcast_executor_event_safe(event))

    executor_pool.set_event_callback(on_executor_event)
    logger.info("Workspace manager and executor pool initialized")

    # Initialize Work Ledger for persistent work tracking
    ledger_dir = PROJECT_ROOT / "workspace" / "ledger"
    work_ledger = get_work_ledger(ledger_dir)
    logger.info(f"Work ledger initialized at {ledger_dir}")

    # Initialize Mailbox system for agent communication
    mailboxes_dir = PROJECT_ROOT / "workspace" / "mailboxes"
    mailbox_manager = get_mailbox_manager(mailboxes_dir)
    logger.info(f"Mailbox manager initialized at {mailboxes_dir}")

    # Initialize Escalation Protocol
    escalation_logs_dir = PROJECT_ROOT / "logs" / "escalations"
    escalation_manager = get_escalation_manager(escalation_logs_dir)
    logger.info(f"Escalation manager initialized at {escalation_logs_dir}")

    # Enable auto-spawn: When new work items are created, automatically spawn agents
    enable_auto_spawn()
    logger.info("Auto-spawn enabled for work ledger")

    # Recover orphaned work from previous session crashes
    # This picks up any work items that were IN_PROGRESS when the server stopped
    try:
        recovered_count = work_ledger.recover_orphaned_work(
            timeout_minutes=30,  # Work stale after 30 minutes
            actor="startup_recovery"
        )
        if recovered_count > 0:
            logger.info(f"Recovered {recovered_count} orphaned work items from previous session")
    except Exception as e:
        logger.warning(f"Failed to recover orphaned work: {e}")


async def _broadcast_job_update_safe(job):
    """Safely broadcast job updates (handles import order)."""
    try:
        await broadcast_job_update(job)
    except Exception as e:
        logger.error(f"Failed to broadcast job update: {e}")


async def _broadcast_executor_event_safe(event: dict):
    """Safely broadcast executor pool events to subscribers."""
    try:
        await broadcast_executor_pool_event(event)
    except Exception as e:
        logger.error(f"Failed to broadcast executor event: {e}")


def get_orchestrator() -> SupremeOrchestrator:
    """Get or create the Supreme Orchestrator."""
    global orchestrator
    if orchestrator is None:
        orchestrator = SupremeOrchestrator(
            base_path=PROJECT_ROOT,
            config_path=PROJECT_ROOT / "config.yaml",
            logs_dir=PROJECT_ROOT / "logs",
        )
    return orchestrator


# Pydantic models
class SwarmCreate(BaseModel):
    name: str
    description: str = ""
    template: str = "_template"


class ChatMessage(BaseModel):
    message: str
    swarm: str | None = None


class SwarmResponse(BaseModel):
    name: str
    description: str
    status: str
    agent_count: int
    priorities: list[str]


class AgentInfo(BaseModel):
    name: str
    type: str
    model: str
    background: bool
    description: str


class HealthResponse(BaseModel):
    status: str
    swarm_count: int
    agent_count: int


# Chat History Models
class ChatMessageModel(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    agent: str | None = None
    thinking: str | None = None


class ChatSession(BaseModel):
    id: str
    title: str
    swarm: str | None = None
    created_at: str
    updated_at: str
    messages: list[ChatMessageModel] = []


class ChatHistoryManager:
    """Manages chat history storage on disk."""

    def __init__(self, base_path: Path):
        self.chat_dir = base_path / "logs" / "chat"
        self.chat_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.chat_dir / f"{session_id}.json"

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all chat sessions (without full messages)."""
        sessions = []
        for file in sorted(self.chat_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                data = json.loads(file.read_text())
                # Return summary without full messages
                sessions.append(
                    {
                        "id": data["id"],
                        "title": data.get("title", "Untitled"),
                        "swarm": data.get("swarm"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to read chat session {file}: {e}")
        return sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a chat session with all messages."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def create_session(self, swarm: str | None = None, title: str | None = None) -> dict[str, Any]:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        session = {
            "id": session_id,
            "title": title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "swarm": swarm,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._save_session(session)
        return session

    def add_message(
        self, session_id: str, role: str, content: str, agent: str | None = None, thinking: str | None = None
    ) -> dict[str, Any]:
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "thinking": thinking,
        }
        session["messages"].append(message)
        session["updated_at"] = datetime.now().isoformat()

        # Auto-update title from first user message if still default
        if role == "user" and len(session["messages"]) == 1:
            session["title"] = content[:50] + ("..." if len(content) > 50 else "")

        self._save_session(session)
        return message

    def update_session(self, session_id: str, **kwargs) -> dict[str, Any] | None:
        """Update session metadata (title, swarm, etc.)."""
        session = self.get_session(session_id)
        if not session:
            return None

        for key in ["title", "swarm"]:
            if key in kwargs:
                session[key] = kwargs[key]
        session["updated_at"] = datetime.now().isoformat()
        self._save_session(session)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def _save_session(self, session: dict[str, Any]):
        """Save session to disk."""
        path = self._session_path(session["id"])
        path.write_text(json.dumps(session, indent=2))


# Global chat history manager
chat_history: ChatHistoryManager | None = None


def get_chat_history() -> ChatHistoryManager:
    """Get or create the chat history manager."""
    global chat_history
    if chat_history is None:
        chat_history = ChatHistoryManager(PROJECT_ROOT)
    return chat_history


# REST Endpoints
@app.get("/api/status", response_model=HealthResponse)
async def get_status():
    """Get system health status."""
    orch = get_orchestrator()
    return HealthResponse(
        status="healthy",
        swarm_count=len(orch.swarms),
        agent_count=len(orch.all_agents),
    )


# Job Management Endpoints
class JobCreate(BaseModel):
    """Request to create a background job."""
    type: str = "chat"  # "chat", "swarm_directive", "task"
    prompt: str
    swarm: str | None = None
    session_id: str | None = None


@app.get("/api/jobs")
async def list_jobs(
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List background jobs."""
    queue = get_job_queue()

    if session_id:
        jobs = queue.get_session_jobs(session_id)
    else:
        jobs = queue.get_recent_jobs(limit)

    # Filter by status if specified
    if status:
        jobs = [j for j in jobs if j.status.value == status]

    return [j.to_dict() for j in jobs]


@app.post("/api/jobs")
async def create_job(data: JobCreate) -> dict:
    """Create a new background job."""
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


@app.get("/api/jobs/status")
async def get_job_manager_status() -> dict:
    """Get job manager status."""
    manager = get_job_manager()
    return manager.get_status()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    """Get job details."""
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel a running job."""
    manager = get_job_manager()
    success = await manager.cancel_job(job_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "success": True,
        "message": f"Job {job_id} cancelled",
    }


# ============================================================
# WORK LEDGER ENDPOINTS (persistent work tracking)
# ============================================================


class WorkCreateRequest(BaseModel):
    """Request to create a work item."""
    title: str
    description: str
    work_type: str = "task"
    priority: str = "medium"
    parent_id: str | None = None
    swarm_name: str | None = None
    context: dict | None = None


@app.get("/api/work")
async def list_work_items(
    status: str | None = None,
    swarm: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List work items with optional filters."""
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


@app.post("/api/work")
async def create_work_item(request: WorkCreateRequest) -> dict:
    """Create a new work item."""
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


@app.get("/api/work/{work_id}")
async def get_work_item(work_id: str) -> dict:
    """Get a specific work item."""
    ledger = get_work_ledger()
    item = ledger.get_work(work_id)

    if not item:
        raise HTTPException(status_code=404, detail=f"Work item {work_id} not found")

    return item.to_dict()


@app.post("/api/work/{work_id}/claim")
async def claim_work_item(work_id: str, owner: str) -> dict:
    """Claim a work item for processing."""
    ledger = get_work_ledger()
    item = ledger.claim_work(work_id, owner)

    if not item:
        raise HTTPException(status_code=400, detail=f"Could not claim work item {work_id}")

    return {"success": True, "work": item.to_dict()}


@app.post("/api/work/{work_id}/complete")
async def complete_work_item(work_id: str, owner: str, result: dict | None = None) -> dict:
    """Mark a work item as completed."""
    ledger = get_work_ledger()
    item = ledger.complete_work(work_id, owner, result)

    if not item:
        raise HTTPException(status_code=400, detail=f"Could not complete work item {work_id}")

    return {"success": True, "work": item.to_dict()}


@app.post("/api/work/{work_id}/fail")
async def fail_work_item(work_id: str, owner: str, error: str) -> dict:
    """Mark a work item as failed."""
    ledger = get_work_ledger()
    item = ledger.fail_work(work_id, owner, error)

    if not item:
        raise HTTPException(status_code=400, detail=f"Could not fail work item {work_id}")

    return {"success": True, "work": item.to_dict()}


@app.get("/api/work/{work_id}/progress")
async def get_work_progress(work_id: str) -> dict:
    """Get progress summary for a work item including children."""
    ledger = get_work_ledger()
    return ledger.get_progress(work_id)


@app.post("/api/work/recover")
async def recover_orphaned_work(timeout_minutes: int = 60) -> dict:
    """Recover orphaned work items that were abandoned."""
    ledger = get_work_ledger()
    recovered = ledger.recover_orphaned_work(timeout_minutes)

    return {
        "success": True,
        "recovered_count": len(recovered),
        "recovered": [item.to_dict() for item in recovered],
    }


# ============================================================
# MAILBOX ENDPOINTS (agent communication)
# ============================================================


class MessageSendRequest(BaseModel):
    """Request to send a message."""
    from_agent: str
    to_agent: str
    subject: str
    body: str
    message_type: str = "request"
    priority: str = "normal"
    swarm_name: str | None = None
    payload: dict | None = None
    reply_to: str | None = None
    tags: list[str] | None = None


class HandoffRequest(BaseModel):
    """Request to send a handoff."""
    from_agent: str
    to_agent: str
    subject: str
    work_completed: str
    current_state: str
    next_steps: list[str]
    files_modified: list[str] | None = None
    blockers: list[str] | None = None
    swarm_name: str | None = None
    priority: str = "normal"


@app.get("/api/mailbox/{agent_name}")
async def check_mailbox(
    agent_name: str,
    unread_only: bool = True,
    message_type: str | None = None,
) -> list[dict]:
    """Check an agent's mailbox for messages."""
    mailbox = get_mailbox_manager()

    message_types = None
    if message_type:
        message_types = [MessageType(message_type)]

    messages = mailbox.check_mailbox(
        agent_name=agent_name,
        unread_only=unread_only,
        message_types=message_types,
    )

    return [msg.to_dict() for msg in messages]


@app.get("/api/mailbox/{agent_name}/count")
async def get_mailbox_count(agent_name: str) -> dict:
    """Get count of pending messages by priority."""
    mailbox = get_mailbox_manager()
    return mailbox.get_pending_count(agent_name)


@app.post("/api/mailbox/send")
async def send_message(request: MessageSendRequest) -> dict:
    """Send a message to an agent's mailbox."""
    mailbox = get_mailbox_manager()

    message = mailbox.send(
        from_agent=request.from_agent,
        to_agent=request.to_agent,
        subject=request.subject,
        body=request.body,
        message_type=MessageType(request.message_type),
        priority=MessagePriority[request.priority.upper()],
        swarm_name=request.swarm_name,
        payload=request.payload,
        reply_to=request.reply_to,
        tags=request.tags,
    )

    return {"success": True, "message_id": message.id, "message": message.to_dict()}


@app.post("/api/mailbox/handoff")
async def send_handoff(request: HandoffRequest) -> dict:
    """Send a structured handoff to another agent."""
    from shared.agent_mailbox import HandoffContext

    mailbox = get_mailbox_manager()

    context = HandoffContext(
        work_completed=request.work_completed,
        current_state=request.current_state,
        next_steps=request.next_steps,
        files_modified=request.files_modified or [],
        blockers=request.blockers or [],
    )

    message = mailbox.handoff(
        from_agent=request.from_agent,
        to_agent=request.to_agent,
        subject=request.subject,
        handoff_context=context,
        priority=MessagePriority[request.priority.upper()],
        swarm_name=request.swarm_name,
    )

    return {"success": True, "message_id": message.id, "message": message.to_dict()}


@app.post("/api/mailbox/message/{message_id}/read")
async def read_message(message_id: str) -> dict:
    """Mark a message as read."""
    mailbox = get_mailbox_manager()
    message = mailbox.read_message(message_id)

    if not message:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")

    return {"success": True, "message": message.to_dict()}


@app.post("/api/mailbox/message/{message_id}/complete")
async def complete_message(message_id: str, archive: bool = True) -> dict:
    """Mark a message as completed."""
    mailbox = get_mailbox_manager()
    message = mailbox.mark_completed(message_id, archive=archive)

    if not message:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")

    return {"success": True, "message": message.to_dict()}


@app.post("/api/mailbox/message/{message_id}/reply")
async def reply_to_message(message_id: str, from_agent: str, body: str, payload: dict | None = None) -> dict:
    """Reply to a message."""
    mailbox = get_mailbox_manager()
    message = mailbox.reply(message_id, from_agent, body, payload)

    if not message:
        raise HTTPException(status_code=404, detail=f"Original message {message_id} not found")

    return {"success": True, "message": message.to_dict()}


@app.get("/api/mailbox/thread/{thread_id}")
async def get_message_thread(thread_id: str) -> list[dict]:
    """Get all messages in a conversation thread."""
    mailbox = get_mailbox_manager()
    messages = mailbox.get_thread(thread_id)
    return [msg.to_dict() for msg in messages]


# ============================================================
# ESCALATION PROTOCOL ENDPOINTS
# ============================================================


class EscalationCreateRequest(BaseModel):
    """Request to create an escalation."""
    from_level: str  # "agent" or "coo"
    to_level: str    # "coo" or "ceo"
    reason: str
    title: str
    description: str
    created_by: str
    priority: str = "medium"
    swarm_name: str | None = None
    blocked_tasks: list[str] | None = None
    related_files: list[str] | None = None
    context: dict | None = None


@app.get("/api/escalations")
async def list_escalations(
    status: str | None = None,
    level: str | None = None,
    swarm: str | None = None,
) -> list[dict]:
    """List escalations with optional filters.

    Query parameters:
        status: 'pending', 'blocking', 'resolved', 'all', or specific EscalationStatus value
        level: 'coo' or 'ceo' - filter by target level
        swarm: Filter by swarm name
    """
    manager = get_escalation_manager()

    if status == "pending":
        target_level = EscalationLevel(level) if level else None
        items = manager.get_pending(level=target_level)
    elif swarm:
        items = manager.get_by_swarm(swarm)
    elif status == "blocking":
        items = manager.get_blocked_work()
    elif status == "all":
        items = manager.get_all()
    elif status == "resolved":
        items = manager.get_all(status=EscalationStatus.RESOLVED)
    else:
        # Return all pending by default
        items = manager.get_pending()

    return [item.to_dict() for item in items]


@app.post("/api/escalations")
async def create_escalation(request: EscalationCreateRequest) -> dict:
    """Create a new escalation."""
    esc_manager = get_escalation_manager()

    try:
        escalation = esc_manager.create_escalation(
            from_level=EscalationLevel(request.from_level),
            to_level=EscalationLevel(request.to_level),
            reason=EscalationReason(request.reason),
            title=request.title,
            description=request.description,
            created_by=request.created_by,
            priority=EscalationPriority(request.priority),
            swarm_name=request.swarm_name,
            blocked_tasks=request.blocked_tasks,
            related_files=request.related_files,
            context=request.context,
        )

        # Broadcast WebSocket event for real-time updates
        await broadcast_escalation_event("escalation_created", escalation.to_dict())

        return {"success": True, "escalation_id": escalation.id, "escalation": escalation.to_dict()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/escalations/{escalation_id}")
async def get_escalation(escalation_id: str) -> dict:
    """Get a specific escalation."""
    manager = get_escalation_manager()
    escalation = manager.get_by_id(escalation_id)

    if not escalation:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")

    return escalation.to_dict()


@app.post("/api/escalations/{escalation_id}/resolve")
async def resolve_escalation(
    escalation_id: str,
    resolution: str,
    resolved_by: str,
) -> dict:
    """Resolve an escalation."""
    esc_manager = get_escalation_manager()
    escalation = esc_manager.resolve_escalation(escalation_id, resolution, resolved_by)

    if not escalation:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")

    # Broadcast WebSocket event for real-time updates
    await broadcast_escalation_event("escalation_resolved", escalation.to_dict())

    return {"success": True, "escalation": escalation.to_dict()}


@app.post("/api/escalations/{escalation_id}/status")
async def update_escalation_status(escalation_id: str, status: str) -> dict:
    """Update an escalation's status."""
    esc_manager = get_escalation_manager()
    escalation = esc_manager.update_status(escalation_id, EscalationStatus(status))

    if not escalation:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found")

    # Broadcast WebSocket event for real-time updates
    await broadcast_escalation_event("escalation_updated", escalation.to_dict())

    return {"success": True, "escalation": escalation.to_dict()}


@app.get("/api/escalations/pending/coo")
async def get_coo_pending_escalations() -> list[dict]:
    """Get pending escalations for COO."""
    manager = get_escalation_manager()
    items = manager.get_pending(level=EscalationLevel.COO)
    return [item.to_dict() for item in items]


@app.get("/api/escalations/pending/ceo")
async def get_ceo_pending_escalations() -> list[dict]:
    """Get pending escalations for CEO (human)."""
    manager = get_escalation_manager()
    items = manager.get_pending(level=EscalationLevel.CEO)
    return [item.to_dict() for item in items]


# ============================================================
# REVIEW WORKFLOW ENDPOINTS
# ============================================================


@app.get("/api/swarms/{swarm_name}/workflow/{work_id}")
async def get_workflow_status(swarm_name: str, work_id: str) -> dict:
    """Get workflow status for a piece of work."""
    orch = get_orchestrator()

    if swarm_name not in orch.swarms:
        raise HTTPException(status_code=404, detail=f"Swarm '{swarm_name}' not found")

    swarm = orch.swarms[swarm_name]
    return swarm.get_workflow_status(work_id)


@app.post("/api/swarms/{swarm_name}/workflow/{work_id}/complete-step")
async def complete_workflow_step(
    swarm_name: str,
    work_id: str,
    step_name: str,
    agent_name: str | None = None,
    result: dict | None = None,
) -> dict:
    """Mark a workflow step as completed."""
    orch = get_orchestrator()

    if swarm_name not in orch.swarms:
        raise HTTPException(status_code=404, detail=f"Swarm '{swarm_name}' not found")

    swarm = orch.swarms[swarm_name]

    try:
        status = swarm.complete_workflow_step(work_id, step_name, result, agent_name)
        return {"success": True, "workflow_status": status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/swarms/{swarm_name}/workflow/{work_id}/can-complete")
async def can_complete_work(swarm_name: str, work_id: str) -> dict:
    """Check if work can be marked as complete (all required steps done)."""
    orch = get_orchestrator()

    if swarm_name not in orch.swarms:
        raise HTTPException(status_code=404, detail=f"Swarm '{swarm_name}' not found")

    swarm = orch.swarms[swarm_name]
    can_proceed, blocking_reason = swarm.can_complete_work(work_id)

    return {
        "can_complete": can_proceed,
        "blocking_reason": blocking_reason,
    }


@app.get("/api/swarms/{swarm_name}/workflow/{work_id}/next-step")
async def get_next_required_step(swarm_name: str, work_id: str) -> dict:
    """Get the next required step for a piece of work."""
    orch = get_orchestrator()

    if swarm_name not in orch.swarms:
        raise HTTPException(status_code=404, detail=f"Swarm '{swarm_name}' not found")

    swarm = orch.swarms[swarm_name]
    next_step = swarm.get_next_required_step(work_id)

    if next_step:
        return {
            "has_next": True,
            "step": next_step.step,
            "agent": next_step.agent,
            "required": next_step.required,
        }
    return {
        "has_next": False,
        "step": None,
        "agent": None,
        "required": None,
    }


@app.post("/api/swarms/{swarm_name}/workflow/{work_id}/reset")
async def reset_workflow(swarm_name: str, work_id: str) -> dict:
    """Reset workflow tracking for a piece of work."""
    orch = get_orchestrator()

    if swarm_name not in orch.swarms:
        raise HTTPException(status_code=404, detail=f"Swarm '{swarm_name}' not found")

    swarm = orch.swarms[swarm_name]
    swarm.reset_workflow(work_id)

    return {"success": True, "message": f"Workflow reset for {work_id}"}


@app.get("/api/swarms/{swarm_name}/workflow-config")
async def get_workflow_config(swarm_name: str) -> dict:
    """Get the review workflow configuration for a swarm."""
    orch = get_orchestrator()

    if swarm_name not in orch.swarms:
        raise HTTPException(status_code=404, detail=f"Swarm '{swarm_name}' not found")

    swarm = orch.swarms[swarm_name]
    workflow = swarm.config.review_workflow

    return {
        "swarm": swarm_name,
        "has_workflow": len(workflow) > 0,
        "steps": [
            {"step": s.step, "agent": s.agent, "required": s.required}
            for s in workflow
        ],
        "required_steps": [
            {"step": s.step, "agent": s.agent}
            for s in workflow if s.required
        ],
    }


# ============================================================
# AGENT EXECUTION ENDPOINTS (executor pool integration)
# ============================================================

class AgentExecuteRequest(BaseModel):
    """Request to execute an agent."""
    swarm: str
    agent: str
    prompt: str
    max_turns: int = 25
    timeout: float = 600.0


@app.post("/api/agents/execute")
async def execute_agent(request: AgentExecuteRequest) -> dict:
    """Execute an agent with proper workspace isolation.

    This endpoint runs an agent using the AgentExecutorPool, which provides:
    - Workspace isolation (agents can only access their swarm's workspace)
    - Concurrent execution limits
    - Process lifecycle management
    - Streaming event collection

    Args:
        request: AgentExecuteRequest with swarm, agent, prompt, and options

    Returns:
        Dictionary with success status, collected events, and workspace path
    """
    # Validate swarm exists
    orch = get_orchestrator()
    if request.swarm not in orch.swarms and request.swarm != "swarm_dev":
        raise HTTPException(status_code=404, detail=f"Swarm '{request.swarm}' not found")

    try:
        workspace_manager = get_workspace_manager(PROJECT_ROOT)
        pool = get_executor_pool()

        # Get workspace and permissions
        workspace = workspace_manager.get_workspace(request.swarm)
        permissions = workspace_manager.get_agent_permissions(request.agent, request.swarm)

        # Build execution context
        context = AgentExecutionContext(
            agent_name=request.agent,
            agent_type=request.agent,  # Will be refined later with agent definitions
            swarm_name=request.swarm,
            workspace=workspace,
            allowed_tools=permissions["allowed_tools"],
            permission_mode=permissions["permission_mode"],
            git_credentials=permissions["git_access"],
            web_access=permissions["web_access"],
            max_turns=request.max_turns,
            timeout=request.timeout,
        )

        # Execute and collect results
        results = []
        async for event in pool.execute(context, request.prompt):
            results.append(event)

        # Determine success from events
        success = True
        for event in results:
            if event.get("type") == "agent_execution_complete":
                success = event.get("success", True)
                break
            if event.get("type") == "error":
                success = False

        return {
            "success": success,
            "events": results,
            "workspace": str(workspace),
        }

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/pool/status")
async def get_pool_status() -> dict:
    """Get executor pool status.

    Returns the current state of the agent executor pool including:
    - Number of currently running agents
    - Available execution slots
    - Maximum concurrent capacity

    Returns:
        Dictionary with pool status information
    """
    try:
        pool = get_executor_pool()
        return {
            "active_count": pool.active_count,
            "available_slots": pool.available_slots,
            "max_concurrent": pool.max_concurrent,
        }
    except ValueError:
        # Pool not initialized yet
        return {
            "active_count": 0,
            "available_slots": 5,
            "max_concurrent": 5,
            "initialized": False,
        }


# WebSocket subscribers for executor pool events
executor_pool_subscribers: list[WebSocket] = []


async def broadcast_executor_pool_event(event: dict):
    """Broadcast executor pool events to all subscribers and main chat connections."""
    # Send to dedicated executor pool subscribers
    for ws in executor_pool_subscribers[:]:
        try:
            await ws.send_json(event)
        except Exception:
            try:
                executor_pool_subscribers.remove(ws)
            except ValueError:
                pass

    # Also send to main chat WebSocket connections for parallel agent tracking
    # Map executor pool event types to chat WebSocket event types
    event_type = event.get("type", "")
    if event_type == "agent_execution_start":
        # Send as agent_spawn for consistency with existing frontend
        chat_event = {
            "type": "agent_spawn",
            "agent": event.get("agent", "Unknown Agent"),
            "description": f"Executing in {event.get('swarm', 'workspace')}",
            "parentAgent": "COO",
            "executionId": event.get("execution_id", ""),
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass
    elif event_type == "agent_execution_complete":
        # Send as agent_complete_subagent for consistency
        chat_event = {
            "type": "agent_complete_subagent",
            "agent": event.get("agent", "Unknown Agent"),
            "success": event.get("success", False),
            "executionId": event.get("execution_id", ""),
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass
    elif event_type in ("tool_start", "tool_complete"):
        # Pass through tool events with agent attribution
        chat_event = {
            "type": event_type,
            "tool": event.get("tool", "unknown"),
            "description": event.get("description", ""),
            "agentName": event.get("agent", "Unknown Agent"),
            "success": event.get("success", True),
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass


@app.websocket("/ws/executor-pool")
async def websocket_executor_pool(websocket: WebSocket):
    """WebSocket endpoint for executor pool events.

    Clients receive real-time updates when:
    - Agents start executing
    - Agents complete (success or failure)
    - Tool usage by agents
    - Progress updates
    """
    await websocket.accept()
    executor_pool_subscribers.append(websocket)
    logger.info(f"Executor pool WebSocket connected. Total subscribers: {len(executor_pool_subscribers)}")

    try:
        # Send initial status
        try:
            pool = get_executor_pool()
            await websocket.send_json({
                "type": "executor_pool_status",
                "activeCount": pool.active_count,
                "availableSlots": pool.available_slots,
                "maxConcurrent": pool.max_concurrent,
            })
        except ValueError:
            await websocket.send_json({
                "type": "executor_pool_status",
                "activeCount": 0,
                "availableSlots": 5,
                "maxConcurrent": 5,
                "initialized": False,
            })

        # Keep connection alive and handle any client messages
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "get_status":
                    try:
                        pool = get_executor_pool()
                        await websocket.send_json({
                            "type": "executor_pool_status",
                            "activeCount": pool.active_count,
                            "availableSlots": pool.available_slots,
                            "maxConcurrent": pool.max_concurrent,
                        })
                    except ValueError:
                        await websocket.send_json({
                            "type": "executor_pool_status",
                            "activeCount": 0,
                            "availableSlots": 5,
                            "maxConcurrent": 5,
                            "initialized": False,
                        })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"Executor pool WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        try:
            executor_pool_subscribers.remove(websocket)
        except ValueError:
            pass
        logger.info(f"Executor pool WebSocket disconnected. Total subscribers: {len(executor_pool_subscribers)}")


@app.get("/api/swarms")
async def list_swarms() -> list[dict[str, Any]]:
    """List all swarms with status."""
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


@app.get("/api/swarms/{name}")
async def get_swarm(name: str) -> dict[str, Any]:
    """Get detailed swarm information."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    status = swarm.get_status()
    return status


@app.post("/api/swarms")
async def create_swarm(data: SwarmCreate) -> dict[str, Any]:
    """Create a new swarm from template."""
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


@app.get("/api/swarms/{name}/agents")
async def list_agents(name: str) -> list[dict[str, Any]]:
    """List agents in a swarm."""
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


@app.post("/api/chat")
async def chat(data: ChatMessage) -> dict[str, Any]:
    """Send a chat message (non-streaming)."""
    orch = get_orchestrator()

    try:
        response = await orch.route_request(data.message)
        return {
            "success": True,
            "response": response,
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File Management Endpoints
def get_file_info(file_path: Path, workspace: Path) -> dict[str, Any]:
    """Get file information including type and size."""
    relative_path = file_path.relative_to(workspace)
    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))

    # Determine file category
    if mime_type:
        if mime_type.startswith("image/"):
            category = "image"
        elif mime_type.startswith("text/") or mime_type in ["application/json", "application/javascript"]:
            category = "text"
        elif mime_type == "application/pdf":
            category = "pdf"
        else:
            category = "binary"
    else:
        # Check by extension
        ext = file_path.suffix.lower()
        if ext in [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".md",
            ".txt",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".cfg",
            ".ini",
            ".sh",
            ".css",
            ".html",
        ]:
            category = "text"
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico"]:
            category = "image"
        else:
            category = "binary"

    return {
        "name": file_path.name,
        "path": str(relative_path),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_dir": file_path.is_dir(),
        "mime_type": mime_type,
        "category": category,
    }


@app.get("/api/swarms/{name}/files")
async def list_files(name: str, path: str = "") -> dict[str, Any]:
    """List files in a swarm's workspace."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    workspace = swarm.workspace
    target_path = workspace / path if path else workspace

    # Auto-create workspace directory if it doesn't exist (only for root workspace)
    if not workspace.exists():
        logger.info(f"Creating missing workspace directory: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

    # Security check - ensure we're within workspace
    try:
        target_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    files = []
    dirs = []

    for item in sorted(target_path.iterdir()):
        if item.name.startswith("."):
            continue  # Skip hidden files

        try:
            info = get_file_info(item, workspace)
            if item.is_dir():
                # Count items in directory
                try:
                    info["item_count"] = len(list(item.iterdir()))
                except PermissionError:
                    info["item_count"] = 0
                dirs.append(info)
            else:
                files.append(info)
        except (PermissionError, OSError):
            continue

    return {
        "current_path": path,
        "workspace": str(workspace),
        "directories": dirs,
        "files": files,
    }


@app.get("/api/swarms/{name}/files/content")
async def get_file_content(name: str, path: str) -> dict[str, Any]:
    """Get content of a file in the workspace."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    workspace = swarm.workspace
    file_path = workspace / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if file_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is a directory: {path}")

    # Security check
    try:
        file_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    info = get_file_info(file_path, workspace)

    # Return content based on category
    if info["category"] == "text":
        try:
            content = file_path.read_text()
            return {**info, "content": content, "encoding": "utf-8"}
        except UnicodeDecodeError:
            content = base64.b64encode(file_path.read_bytes()).decode()
            return {**info, "content": content, "encoding": "base64"}
    elif info["category"] == "image":
        content = base64.b64encode(file_path.read_bytes()).decode()
        return {**info, "content": content, "encoding": "base64"}
    else:
        # Binary file - return base64
        content = base64.b64encode(file_path.read_bytes()).decode()
        return {**info, "content": content, "encoding": "base64"}


@app.post("/api/swarms/{name}/files")
async def upload_file(
    name: str,
    file: UploadFile = File(None),
    path: str = Form(""),
    filename: str = Form(None),
    content: str = Form(None),
    is_text: bool = Form(True),
) -> dict[str, Any]:
    """Upload or create a file in the workspace."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    workspace = swarm.workspace

    # Determine target path and filename
    if file:
        # File upload
        target_filename = filename or file.filename
        target_path = workspace / path / target_filename if path else workspace / target_filename
        file_content = await file.read()
    elif content is not None and filename:
        # Text content creation
        target_path = workspace / path / filename if path else workspace / filename
        if is_text:
            file_content = content.encode("utf-8")
        else:
            # Base64 encoded content
            file_content = base64.b64decode(content)
    else:
        raise HTTPException(status_code=400, detail="Must provide either file upload or content with filename")

    # Security check
    try:
        target_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    # Create parent directories if needed
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    target_path.write_bytes(file_content)

    logger.info(f"Created file: {target_path}")

    return {
        "success": True,
        "path": str(target_path.relative_to(workspace)),
        "size": len(file_content),
    }


@app.post("/api/swarms/{name}/files/directory")
async def create_directory(name: str, path: str = Form(...)) -> dict[str, Any]:
    """Create a directory in the workspace."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    workspace = swarm.workspace
    target_path = workspace / path

    # Security check
    try:
        target_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    target_path.mkdir(parents=True, exist_ok=True)

    return {
        "success": True,
        "path": path,
    }


@app.delete("/api/swarms/{name}/files")
async def delete_file(name: str, path: str) -> dict[str, Any]:
    """Delete a file or empty directory from workspace."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    workspace = swarm.workspace
    target_path = workspace / path

    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    # Security check
    try:
        target_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace")

    # Don't allow deleting workspace root
    if target_path.resolve() == workspace.resolve():
        raise HTTPException(status_code=403, detail="Cannot delete workspace root")

    if target_path.is_dir():
        if any(target_path.iterdir()):
            raise HTTPException(status_code=400, detail="Directory is not empty")
        target_path.rmdir()
    else:
        target_path.unlink()

    logger.info(f"Deleted: {target_path}")

    return {
        "success": True,
        "path": path,
    }


# Chat History Endpoints
@app.get("/api/chat/sessions")
async def list_chat_sessions() -> list[dict[str, Any]]:
    """List all chat sessions."""
    history = get_chat_history()
    return history.list_sessions()


@app.post("/api/chat/sessions")
async def create_chat_session(
    swarm: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create a new chat session."""
    history = get_chat_history()
    return history.create_session(swarm=swarm, title=title)


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session(session_id: str) -> dict[str, Any]:
    """Get a chat session with all messages."""
    history = get_chat_history()
    session = history.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@app.put("/api/chat/sessions/{session_id}")
async def update_chat_session(
    session_id: str,
    title: str | None = None,
    swarm: str | None = None,
) -> dict[str, Any]:
    """Update chat session metadata."""
    history = get_chat_history()
    kwargs = {}
    if title is not None:
        kwargs["title"] = title
    if swarm is not None:
        kwargs["swarm"] = swarm

    session = history.update_session(session_id, **kwargs)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str) -> dict[str, Any]:
    """Delete a chat session."""
    history = get_chat_history()
    if history.delete_session(session_id):
        return {"success": True, "id": session_id}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.post("/api/chat/sessions/{session_id}/messages")
async def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    agent: str | None = None,
    thinking: str | None = None,
) -> dict[str, Any]:
    """Add a message to a chat session."""
    history = get_chat_history()
    try:
        return history.add_message(session_id, role, content, agent, thinking)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# WEB SEARCH & FETCH ENDPOINTS (for agent access via curl)
# ============================================================


class SearchRequest(BaseModel):
    query: str
    num_results: int = 5

class FetchRequest(BaseModel):
    url: str
    extract_text: bool = True

@app.get("/api/search")
async def web_search_get(q: str, n: int = 5) -> dict[str, Any]:
    """
    Search the web via DuckDuckGo (GET version for easy curl access).

    Usage by agents:
        curl "http://localhost:8000/api/search?q=atomic+semantics&n=5"
    """
    return await _do_web_search(q, n)

@app.post("/api/search")
async def web_search_post(request: SearchRequest) -> dict[str, Any]:
    """Search the web via DuckDuckGo (POST version)."""
    return await _do_web_search(request.query, request.num_results)

async def _do_web_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """Execute web search using DuckDuckGo."""
    if not query:
        raise HTTPException(status_code=400, detail="No search query provided")

    num_results = min(num_results, 10)

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AgentSwarm/1.0)"})

        loop = asyncio.get_event_loop()
        html = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        )

        # Parse search results
        results = []
        pattern = r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]*)</a>'

        links = re.findall(pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (link, title) in enumerate(links[:num_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            if "uddg=" in link:
                actual_url = urllib.parse.unquote(link.split("uddg=")[-1].split("&")[0])
            else:
                actual_url = link
            results.append({
                "title": title.strip(),
                "url": actual_url,
                "snippet": snippet.strip()
            })

        logger.info(f"Web search: '{query}' returned {len(results)} results")
        return {"query": query, "results": results}

    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/fetch")
async def web_fetch_get(url: str, extract_text: bool = True) -> dict[str, Any]:
    """
    Fetch content from a URL (GET version for easy curl access).

    Usage by agents:
        curl "http://localhost:8000/api/fetch?url=https://example.com"
    """
    return await _do_web_fetch(url, extract_text)

@app.post("/api/fetch")
async def web_fetch_post(request: FetchRequest) -> dict[str, Any]:
    """Fetch content from a URL (POST version)."""
    return await _do_web_fetch(request.url, request.extract_text)

async def _do_web_fetch(url: str, extract_text: bool = True) -> dict[str, Any]:
    """Fetch and optionally extract text from a URL."""
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AgentSwarm/1.0)"})

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
        )

        if extract_text:
            # Simple HTML to text conversion
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            content = content[:10000]  # Limit size

        logger.info(f"Web fetch: '{url}' returned {len(content)} chars")
        return {"url": url, "content": content, "length": len(content)}

    except Exception as e:
        logger.error(f"Web fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


# WebSocket for streaming chat
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already removed
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_event(self, websocket: WebSocket, event_type: str, data: dict[str, Any]):
        """Send a structured event to the client."""
        try:
            if websocket not in self.active_connections:
                return  # Connection already closed
            await websocket.send_json(
                {
                    "type": event_type,
                    **data,
                }
            )
        except (RuntimeError, Exception) as e:
            if "close message" in str(e).lower():
                logger.debug(f"Skipped send to closed WebSocket: {e}")
            else:
                logger.error(f"Error sending event: {e}")


manager = ConnectionManager()


async def stream_claude_response(
    prompt: str,
    swarm_name: str | None = None,
    workspace: Path | None = None,
    chat_id: str | None = None,
    system_prompt: str | None = None,
    disallowed_tools: list[str] | None = None,
) -> asyncio.subprocess.Process:
    """
    Start a claude CLI process and return it for streaming.

    Uses 'claude -p --output-format stream-json' which outputs JSON lines
    that we can parse and stream to the frontend.

    Args:
        prompt: The user message/request
        system_prompt: Custom system prompt (COO role, context, etc.)
        chat_id: Session ID for continuity
        disallowed_tools: List of tool names to disable (e.g., ["Write", "Edit"] for COO)
    """
    # Build the command with prompt as argument (more reliable than stdin)
    cmd = [
        "claude",
        "-p",  # Print mode (non-interactive)
        "--output-format",
        "stream-json",
        "--verbose",  # Required for stream-json output
        "--permission-mode",
        "acceptEdits",  # Allow file writes without interactive approval
    ]

    # Add tool restrictions (e.g., COO cannot use Write/Edit)
    if disallowed_tools:
        cmd.extend(["--disallowedTools", ",".join(disallowed_tools)])

    # Add custom system prompt for COO role (append to keep Claude's tool knowledge)
    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    # NOTE: Session continuity disabled - was causing "session doesn't exist" errors
    # that confused the COO. Conversation history in prompt is sufficient context.
    # if chat_id:
    #     session_mgr = get_session_manager()
    #     continue_flags = session_mgr.get_continue_flags(chat_id)
    #     if continue_flags:
    #         cmd.extend(continue_flags)

    # Add user prompt as final argument
    cmd.append(prompt)

    # Set working directory to workspace if specified
    cwd = str(workspace) if workspace else None

    # Build environment - remove API key so CLI uses Max subscription
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # Force CLI to use Max subscription

    logger.info(f"Starting Claude CLI in {cwd or 'current dir'}")

    # Start the process with explicit environment
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,  # Don't use stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    return process


async def parse_claude_stream(
    process: asyncio.subprocess.Process,
    websocket: WebSocket,
    manager: ConnectionManager,
    chat_id: str | None = None,
) -> dict:
    """
    Parse streaming JSON output from claude CLI and send events to WebSocket.
    Captures session ID for continuity and returns dict with full response text and thinking.
    """
    # Use a dict to accumulate response (mutable, passed by reference)
    context = {
        "full_response": "",
        "full_thinking": "",
        "current_block_type": None,
        "session_id": None,  # Will be captured from Claude output
    }

    # Session manager disabled - was causing confusion
    session_mgr = None

    if not process.stdout:
        return {"response": "", "thinking": ""}

    async def drain_stderr():
        """Drain stderr to prevent buffer deadlock."""
        if process.stderr:
            try:
                while True:
                    chunk = await process.stderr.read(65536)
                    if not chunk:
                        break
                    # Log stderr output for debugging
                    stderr_text = chunk.decode(errors='ignore').strip()
                    if stderr_text:
                        logger.debug(f"Claude CLI stderr: {stderr_text[:200]}")
            except Exception as e:
                logger.debug(f"Stderr drain error: {e}")

    # Start draining stderr concurrently to prevent deadlock
    stderr_task = asyncio.create_task(drain_stderr())

    # Read all output at once to avoid buffer issues, then parse line by line
    buffer = b""
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(process.stdout.read(65536), timeout=1.0)
                if not chunk:
                    break
                buffer += chunk

                # Process complete lines from buffer
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                        # Process event and send to websocket (with session capture)
                        await _process_cli_event(event, websocket, manager, context, session_mgr, chat_id)
                    except json.JSONDecodeError:
                        continue
            except asyncio.TimeoutError:
                # Check if process is still running
                if process.returncode is not None:
                    break
                continue
            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                break

        # Process any remaining data in buffer
        if buffer:
            for line in buffer.decode().split("\n"):
                line_str = line.strip()
                if line_str:
                    try:
                        event = json.loads(line_str)
                        await _process_cli_event(event, websocket, manager, context, session_mgr, chat_id)
                    except json.JSONDecodeError:
                        pass
    finally:
        # Ensure stderr task completes
        try:
            await asyncio.wait_for(stderr_task, timeout=5.0)
        except asyncio.TimeoutError:
            stderr_task.cancel()

    # Wait for process to complete
    await process.wait()

    return {"response": context["full_response"], "thinking": context["full_thinking"]}


def _get_file_info(tool_name: str, tool_input: dict) -> tuple[str | None, str | None]:
    """Extract file path and operation type from tool input."""
    file_path = tool_input.get('file_path')

    if tool_name == 'Read':
        return file_path, 'read'
    elif tool_name == 'Write':
        return file_path, 'write'
    elif tool_name == 'Edit':
        return file_path, 'edit'
    return None, None


async def _process_cli_event(event: dict, websocket: WebSocket, manager, context: dict, session_mgr=None, chat_id: str = None):
    """Process a single CLI event and forward to WebSocket.

    Agent Tracking Logic:
    - context["agent_stack"] tracks the hierarchy of active agents (COO at bottom, sub-agents on top)
    - When Task tool starts, we push the sub-agent name onto the stack
    - When Task tool completes, we pop it off
    - All tool events are attributed to the current top-of-stack agent
    """
    event_type = event.get("type", "")

    # Initialize agent tracking stack if needed
    if "agent_stack" not in context:
        context["agent_stack"] = ["COO"]  # COO is always the base

    def get_current_agent():
        """Get the currently active agent (top of stack)."""
        return context["agent_stack"][-1] if context["agent_stack"] else "COO"

    try:
        # Capture session ID from Claude output for session continuity
        if event_type in ("init", "system", "session_start") and session_mgr and chat_id:
            session_id = event.get("session_id") or event.get("sessionId")
            if session_id:
                context["session_id"] = session_id
                asyncio.create_task(session_mgr.register_session(chat_id, session_id))

        if event_type == "assistant" and not event.get("parent_tool_use_id"):
            # Assistant message in agentic loop - may have text, thinking, or tool_use blocks
            # Skip if this is a subagent message (has parent_tool_use_id) - handled separately
            message = event.get("message", {})
            content_blocks = message.get("content", [])
            for block in content_blocks:
                if block.get("type") == "thinking":
                    text = block.get("thinking", "")
                    if text:
                        context["full_thinking"] = context.get("full_thinking", "") + text
                elif block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        await manager.send_event(
                            websocket,
                            "agent_delta",
                            {
                                "agent": "Supreme Orchestrator",
                                "agent_type": "orchestrator",
                                "delta": text,
                            },
                        )
                        context["full_response"] = context.get("full_response", "") + text
                elif block.get("type") == "tool_use":
                    # Fallback: detect tool_use from final assistant message
                    # This handles cases where streaming events didn't fire
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    tool_use_id = block.get("id", "")

                    # Get current agent from stack
                    current_agent = get_current_agent()

                    # Only handle Task agent spawning if we didn't already do it via streaming
                    if tool_name == "Task" and tool_use_id not in context.get("pending_tasks", {}):
                        subagent = tool_input.get("subagent_type") or tool_input.get("agent", "")
                        description = tool_input.get("description", tool_input.get("prompt", ""))[:100]
                        if subagent:
                            # Push to agent stack
                            context["agent_stack"].append(subagent)
                            # Track for cleanup
                            if "pending_tasks" not in context:
                                context["pending_tasks"] = {}
                            context["pending_tasks"][tool_use_id] = subagent
                            # Send agent_spawn event
                            await manager.send_event(
                                websocket,
                                "agent_spawn",
                                {
                                    "agent": subagent,
                                    "description": description,
                                    "parentAgent": current_agent,
                                },
                            )
                            current_agent = subagent  # Use subagent for this tool

                    # Only send tool events if we didn't already send via streaming
                    if not context.get(f"sent_tool_{tool_name}_{id(block)}"):
                        await manager.send_event(
                            websocket,
                            "tool_start",
                            {
                                "tool": tool_name,
                                "description": get_tool_description(tool_name, tool_input),
                                "input": tool_input,
                                "agentName": current_agent,
                            },
                        )
                        await manager.send_event(
                            websocket,
                            "tool_complete",
                            {
                                "tool": tool_name,
                                "success": True,
                                "summary": f"Completed: {tool_name}",
                                "agentName": current_agent,
                            },
                        )
                        context[f"sent_tool_{tool_name}_{id(block)}"] = True

        elif event_type == "content_block_start":
            content_block = event.get("content_block", {})
            block_type = content_block.get("type", "text")
            context["current_block_type"] = block_type

            if block_type == "thinking":
                await manager.send_event(
                    websocket,
                    "thinking_start",
                    {"agent": "Supreme Orchestrator"},
                )
            elif block_type == "tool_use":
                tool_name = content_block.get("name", "unknown")
                tool_use_id = content_block.get("id", "")
                context["current_tool"] = tool_name
                context["current_tool_use_id"] = tool_use_id
                # Reset streamed input accumulator
                context["current_tool_input_json"] = ""
                context["agent_spawn_sent"] = False

                # NOTE: tool_input is typically empty at content_block_start because
                # the input is streamed via input_json_delta events. Agent spawning
                # for Task tools is handled in input_json_delta when we have the full input.

                # Get current agent from stack for initial tool_start event
                current_agent = get_current_agent()

                # Mark as sent to avoid duplicate from fallback
                context[f"sent_tool_{tool_name}"] = True

                # Send initial tool_start event (description will be updated when input available)
                tool_event = {
                    "tool": tool_name,
                    "description": f"Starting {tool_name}...",
                    "input": {},
                    "agentName": current_agent,
                }

                await manager.send_event(websocket, "tool_start", tool_event)

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                context["full_thinking"] = context.get("full_thinking", "") + text
                await manager.send_event(
                    websocket,
                    "thinking_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "delta": text,
                    },
                )
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                context["full_response"] = context.get("full_response", "") + text
                await manager.send_event(
                    websocket,
                    "agent_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "delta": text,
                    },
                )
            elif delta_type == "input_json_delta":
                # Tool input is being streamed - accumulate it
                partial_json = delta.get("partial_json", "")
                context["current_tool_input_json"] = context.get("current_tool_input_json", "") + partial_json

                # Try to parse and detect agent spawning for Task tool
                if context.get("current_tool") == "Task" and not context.get("agent_spawn_sent"):
                    try:
                        partial_input = json.loads(context["current_tool_input_json"])
                        # Check both subagent_type (standard) and agent (legacy) fields
                        agent_name = partial_input.get("subagent_type") or partial_input.get("agent", "")
                        if agent_name:
                            # Use 'description' field if available, otherwise truncate 'prompt'
                            desc = partial_input.get("description", "") or partial_input.get("prompt", "")[:100]

                            # Get current agent BEFORE pushing to stack
                            parent_agent = get_current_agent()

                            # Push to agent stack for proper tool attribution
                            context["agent_stack"].append(agent_name)

                            # Track in pending_tasks for cleanup on completion
                            tool_use_id = context.get("current_tool_use_id", "")
                            if "pending_tasks" not in context:
                                context["pending_tasks"] = {}
                            context["pending_tasks"][tool_use_id] = agent_name

                            # Send agent_spawn event
                            await manager.send_event(
                                websocket,
                                "agent_spawn",
                                {
                                    "agent": agent_name,
                                    "description": desc,
                                    "parentAgent": parent_agent,
                                },
                            )
                            context["agent_spawn_sent"] = True

                            # Track delegation in Work Ledger for persistence across sessions
                            try:
                                ledger = get_work_ledger()
                                work_item = ledger.create_work(
                                    title=f"Delegation: {desc[:80]}",
                                    work_type=WorkType.TASK,
                                    priority=WorkPriority.HIGH,
                                    description=partial_input.get("prompt", desc),
                                    created_by=parent_agent or "COO",
                                    swarm_name="swarm_dev",  # TODO: get from context
                                    context={
                                        "tool_use_id": tool_use_id,
                                        "subagent_type": agent_name,
                                        "parent_agent": parent_agent,
                                    },
                                )
                                # Claim it immediately since we're executing it
                                ledger.claim_work(work_item.id, agent_name)
                                context["delegation_work_id"] = work_item.id
                                logger.info(f"Tracked delegation in Work Ledger: {work_item.id}")
                            except Exception as e:
                                logger.warning(f"Failed to track delegation in Work Ledger: {e}")
                    except json.JSONDecodeError:
                        pass  # JSON not complete yet

        elif event_type == "content_block_stop":
            block_type = context.get("current_block_type")
            if block_type == "thinking":
                await manager.send_event(
                    websocket,
                    "thinking_complete",
                    {
                        "agent": "Supreme Orchestrator",
                        "thinking": context.get("full_thinking", ""),
                    },
                )
            elif block_type == "tool_use":
                # Tool completed
                tool_name = context.get("current_tool", "unknown")
                tool_use_id = context.get("current_tool_use_id", "")
                current_agent = get_current_agent()

                await manager.send_event(
                    websocket,
                    "tool_complete",
                    {
                        "tool": tool_name,
                        "success": True,
                        "summary": f"Completed: {tool_name}",
                        "agentName": current_agent,
                    },
                )

                # If this was a Task tool, pop the agent from stack and send agent_complete
                if tool_name == "Task" and tool_use_id in context.get("pending_tasks", {}):
                    completed_agent = context["pending_tasks"].pop(tool_use_id)
                    if context["agent_stack"] and context["agent_stack"][-1] == completed_agent:
                        context["agent_stack"].pop()
                    # Send agent completion event
                    await manager.send_event(
                        websocket,
                        "agent_complete_subagent",
                        {
                            "agent": completed_agent,
                            "success": True,
                        },
                    )

                    # Mark work item as completed in Work Ledger
                    if context.get("delegation_work_id"):
                        try:
                            ledger = get_work_ledger()
                            ledger.complete_work(
                                context["delegation_work_id"],
                                completed_agent,
                                result={"status": "completed", "agent": completed_agent}
                            )
                            logger.info(f"Marked delegation {context['delegation_work_id']} as complete")
                            context["delegation_work_id"] = None
                        except Exception as e:
                            logger.warning(f"Failed to mark delegation complete: {e}")

                # Reset tool context
                context["current_tool"] = None
                context["current_tool_use_id"] = ""
                context["current_tool_input_json"] = ""
                context["agent_spawn_sent"] = False

        elif event_type == "result":
            # Final result from CLI
            text = event.get("result", "")
            if text:
                # Always send the final result - this is the complete response
                # Previous streaming may have only captured partial content
                context["full_response"] = text
                await manager.send_event(
                    websocket,
                    "agent_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "delta": text,
                    },
                )

        elif event_type == "tool_result":
            # Explicit tool result event (may have error info)
            tool_name = context.get("current_tool", "unknown")
            success = event.get("is_error", False) is False
            await manager.send_event(
                websocket,
                "tool_complete",
                {
                    "tool": tool_name,
                    "success": success,
                    "summary": f"{'Completed' if success else 'Failed'}: {tool_name}",
                },
            )
            context["current_tool"] = None

        elif event_type == "user":
            # User message containing tool results - this includes subagent activity!
            # When parent_tool_use_id is set, this is a subagent's tool result
            parent_tool_id = event.get("parent_tool_use_id")
            tool_use_result = event.get("tool_use_result")
            message = event.get("message", {})
            content_list = message.get("content", [])

            if parent_tool_id and parent_tool_id in context.get("pending_tasks", {}):
                # This is a subagent's tool result!
                subagent_name = context["pending_tasks"][parent_tool_id]

                for content_item in content_list:
                    if content_item.get("type") == "tool_result":
                        tool_use_id = content_item.get("tool_use_id", "")
                        result_content = content_item.get("content", "")
                        is_error = content_item.get("is_error", False)

                        # Look up what tool this was from our tracking
                        tool_info = context.get("subagent_tools", {}).get(tool_use_id, {})
                        tool_name = tool_info.get("name", "Tool")

                        # Send tool_complete for subagent
                        await manager.send_event(
                            websocket,
                            "tool_complete",
                            {
                                "tool": tool_name,
                                "success": not is_error,
                                "summary": result_content[:100] if isinstance(result_content, str) else "Completed",
                                "agentName": subagent_name,
                            },
                        )

            # Also handle nested assistant messages from subagents
            # (when the subagent makes tool calls, we see them here)

        elif event_type == "assistant" and event.get("parent_tool_use_id"):
            # This is a subagent's assistant message!
            parent_tool_id = event.get("parent_tool_use_id")
            if parent_tool_id in context.get("pending_tasks", {}):
                subagent_name = context["pending_tasks"][parent_tool_id]
                message = event.get("message", {})
                content_blocks = message.get("content", [])

                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        tool_use_id = block.get("id", "")

                        # Track this tool for later completion
                        if "subagent_tools" not in context:
                            context["subagent_tools"] = {}
                        context["subagent_tools"][tool_use_id] = {
                            "name": tool_name,
                            "agent": subagent_name,
                        }

                        # Emit tool_start for the subagent
                        await manager.send_event(
                            websocket,
                            "tool_start",
                            {
                                "tool": tool_name,
                                "description": get_tool_description(tool_name, tool_input),
                                "input": tool_input,
                                "agentName": subagent_name,
                            },
                        )
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            # Subagent is producing text output - emit as agent_delta
                            await manager.send_event(
                                websocket,
                                "agent_delta",
                                {
                                    "agent": subagent_name,
                                    "agent_type": "subagent",
                                    "delta": text,
                                },
                            )

    except Exception as e:
        logger.error(f"Error processing CLI event: {e}")


# Job update subscribers
job_update_subscribers: dict[str, list[WebSocket]] = {}


@app.websocket("/ws/jobs")
async def websocket_jobs(websocket: WebSocket):
    """WebSocket endpoint for job updates."""
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
                        job_update_subscribers[job_id].remove(websocket)
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
                    job_update_subscribers[job_id].remove(websocket)


async def broadcast_job_update(job):
    """Broadcast job update to subscribers."""
    job_dict = job.to_dict()
    message = {"type": "job_update", "job": job_dict}

    # Send to job-specific subscribers
    if job.id in job_update_subscribers:
        for ws in job_update_subscribers[job.id][:]:
            try:
                await ws.send_json(message)
            except Exception:
                job_update_subscribers[job.id].remove(ws)

    # Send to "all" subscribers
    if "all" in job_update_subscribers:
        for ws in job_update_subscribers["all"][:]:
            try:
                await ws.send_json(message)
            except Exception:
                job_update_subscribers["all"].remove(ws)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await manager.connect(websocket)
    orch = get_orchestrator()
    history = get_chat_history()

    # Generate a connection-level correlation ID for the WebSocket session
    ws_correlation_id = str(uuid.uuid4())[:8]
    request_id_var.set(ws_correlation_id)
    logger.info("WebSocket chat session started")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            _swarm_name = data.get("swarm")  # Reserved for future swarm-specific routing
            session_id = data.get("session_id")
            attachments = data.get("attachments", [])

            # Generate a new correlation ID for each message in the WebSocket session
            msg_correlation_id = str(uuid.uuid4())[:8]
            request_id_var.set(msg_correlation_id)
            logger.info(f"Processing chat message: {message[:100]}...")

            if not message:
                continue

            # Process image attachments - save to temp files for Claude CLI
            image_paths = []
            if attachments:
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / "agent_swarm_images"
                temp_dir.mkdir(exist_ok=True)

                for att in attachments:
                    if att.get("type") == "image" and att.get("content"):
                        # Save base64 image to temp file
                        img_data = base64.b64decode(att["content"])
                        ext = ".png"
                        if att.get("mimeType"):
                            if "jpeg" in att["mimeType"] or "jpg" in att["mimeType"]:
                                ext = ".jpg"
                            elif "gif" in att["mimeType"]:
                                ext = ".gif"
                            elif "webp" in att["mimeType"]:
                                ext = ".webp"
                        img_path = temp_dir / f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{att.get('name', 'image')}{ext}"
                        img_path.write_bytes(img_data)
                        image_paths.append(str(img_path))
                        logger.info(f"Saved image attachment to: {img_path}")

                # Add image paths to the message for Claude to read
                if image_paths:
                    message += f"\n\n[Images saved for analysis: {', '.join(image_paths)}]\nUse the Read tool to view these images."

            # Send acknowledgment
            await manager.send_event(
                websocket,
                "chat_start",
                {
                    "message": message,
                },
            )

            # Send thinking indicator
            await manager.send_event(
                websocket,
                "agent_start",
                {
                    "agent": "Supreme Orchestrator",
                    "agent_type": "orchestrator",
                },
            )

            try:
                # Memory manager for session summaries (NOT loading full context - too slow)
                memory = get_memory_manager()

                # Build conversation history - ONLY last 2 messages to avoid context pollution
                conversation_history = ""
                if session_id:
                    session = history.get_session(session_id)
                    if session and session.get("messages"):
                        messages = session["messages"]
                        # Only use last 2 messages to keep context fresh and avoid pollution
                        recent_messages = messages[-2:] if len(messages) > 2 else messages

                        history_lines = []
                        for msg in recent_messages:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            content = msg["content"]
                            # Truncate very long messages
                            if len(content) > 1000:
                                content = content[:1000] + "..."
                            history_lines.append(f"{role}: {content}")

                        if history_lines:
                            conversation_history = (
                                "\n\n## Recent Context\n" + "\n\n".join(history_lines) + "\n\n---\n"
                            )

                # Build system prompt for the COO - keep it MINIMAL
                all_swarms = []
                for name, s in orch.swarms.items():
                    agents_list = list(s.agents.keys())
                    all_swarms.append(f"  - {name}: {', '.join(agents_list)}")
                all_swarms_str = "\n".join(all_swarms) if all_swarms else "  No swarms defined"

                # System prompt for COO - REST API for real agents, Task tool for quick research only
                # NOTE: Write and Edit tools are DISABLED via --disallowedTools flag
                system_prompt = f"""You are the Supreme Orchestrator (COO) - a fully autonomous AI orchestrator.

## TOOL RESTRICTIONS - HARD ENFORCED

**The Write and Edit tools are DISABLED for you.** Attempting to use them will fail.

You MUST delegate ALL file modifications to agents.

## PRIMARY DELEGATION: REST API (RECOMMENDED)

**ALWAYS use the REST API for implementation work.** This spawns REAL agents with:
- Custom prompts loaded from `swarms/SWARM/agents/AGENT.md`
- Isolated workspace with proper permissions
- Tracking in the executor pool
- Full tool access

```bash
curl -X POST http://localhost:8000/api/agents/execute \\
  -H "Content-Type: application/json" \\
  -d '{{"swarm": "swarm_dev", "agent": "implementer", "prompt": "Read workspace/STATE.md. Then implement X. Update STATE.md when done."}}'
```

### Available Swarm Agents (via REST API)

**swarm_dev** (for agent-swarm development):
- **implementer** - Write code, create/modify files
- **architect** - Design solutions, create plans
- **critic** - Review code for bugs/issues
- **reviewer** - Code review and quality checks

**operations** (for cross-swarm coordination):
- **ops_coordinator** - Multi-swarm coordination, status reports
- **qa_agent** - Quality audits, standards enforcement

### When to Use REST API
- ANY file creation or modification
- Implementation tasks
- Code reviews requiring workspace access
- Multi-step tasks requiring context persistence
- Cross-swarm coordination

## SECONDARY: Task Tool (Quick Research Only)

The Task tool is for **quick, read-only** operations that don't need custom agent behavior:
- Quick web searches
- Simple file reads and exploration
- One-off questions

**LIMITATION**: Task tool does NOT load custom agent prompts or provide workspace isolation.

```
Task(subagent_type="researcher", prompt="Search for X and summarize findings")
```

## COORDINATION MODEL

### Tier 1 (DEFAULT) - Swarm Dev via REST API
For all agent-swarm system work:
```bash
curl -X POST http://localhost:8000/api/agents/execute \\
  -H "Content-Type: application/json" \\
  -d '{{"swarm": "swarm_dev", "agent": "AGENT_NAME", "prompt": "Your task here"}}'
```

### Tier 2 (ESCALATE) - Operations via REST API
Engage when ANY apply:
1. Spans multiple swarms?
2. Cross-swarm dependencies?
3. Changes core infrastructure?
4. Priority 1-2 (critical/high)?
5. Could conflict with ongoing work?

```bash
curl -X POST http://localhost:8000/api/agents/execute \\
  -H "Content-Type: application/json" \\
  -d '{{"swarm": "operations", "agent": "ops_coordinator", "prompt": "Tier 2: [describe task]. Coordinate and report back."}}'
```

## Your Capabilities

You CAN use:
- **Read** - Read any file to understand context
- **Glob/Grep** - Search files and code
- **Bash** - Run commands (git, tests, **curl for REST API delegation**)
- **Task** - Quick research only (read-only, no custom prompts)
- **Web Search**: `curl -s "http://localhost:8000/api/search?q=QUERY" | jq`

You CANNOT use (BLOCKED):
- **Write** - DISABLED (delegate via REST API)
- **Edit** - DISABLED (delegate via REST API)

## STATE.md Exception

You MAY update STATE.md files directly via Bash:
```bash
cat >> workspace/STATE.md << 'EOF'
### Progress Entry
...
EOF
```

## Standard Delegation Pipeline (All via REST API)

1. **swarm_dev/architect** → Design the solution
2. **swarm_dev/implementer** → Write the code
3. **swarm_dev/critic** → Review for bugs/issues
4. **swarm_dev/reviewer** → Final code review

## Swarm Workspaces
{all_swarms_str}
Files at: swarms/<swarm_name>/workspace/

## STATE.md - Shared Memory
- Read it before acting to understand current state
- Tell delegated agents to read and update it
- Update it after completing significant work (via Bash)

## Operations Reference
- Protocols: `swarms/operations/protocols/coordination_model.md`
- Quick reference: `swarms/operations/protocols/coo_quick_reference.md`

## Project Root: {PROJECT_ROOT}

## Your Approach
1. Understand what the user wants
2. **Delegate via REST API** - spawn REAL agents with proper isolation
3. Use Task tool ONLY for quick read-only research
4. Synthesize results and report back clearly
5. Update STATE.md with progress (via Bash)

**Remember: REST API = Real agents. Task tool = Quick research only.**"""

                user_message = message

                # Build user prompt with conversation context if needed
                if conversation_history:
                    user_prompt = f"""## Previous Conversation
{conversation_history}

---

**Current request:** {user_message}"""
                else:
                    user_prompt = user_message

                # Use Claude CLI (Max subscription handles authentication automatically)
                result = None
                process = None

                # Use Claude CLI with Write/Edit tools DISABLED for COO
                try:
                    process = await stream_claude_response(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        swarm_name=None,
                        workspace=PROJECT_ROOT,
                        chat_id=session_id,
                        disallowed_tools=["Write", "Edit"],  # COO CANNOT write/edit files
                    )

                    # Stream and parse the response
                    result = await asyncio.wait_for(
                        parse_claude_stream(process, websocket, manager, chat_id=session_id),
                        timeout=3600.0,  # 1 hour timeout
                    )
                except asyncio.TimeoutError:
                    if process:
                        process.kill()
                    raise RuntimeError("COO timed out after 1 hour")
                except Exception as e:
                    logger.error(f"Claude CLI failed: {e}")
                    raise RuntimeError(f"**Claude CLI Error:** {e}")

                # Check if we got a result
                if result is None:
                    raise RuntimeError("**Failed to get response from Claude CLI.**")

                # Send the complete response
                final_content = result["response"]
                if not final_content:
                    final_content = "(No response generated)"

                await manager.send_event(
                    websocket,
                    "agent_complete",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "content": final_content,
                        "thinking": result.get("thinking", ""),
                    },
                )

                # Send completion
                await manager.send_event(
                    websocket,
                    "chat_complete",
                    {
                        "success": True,
                    },
                )

                # Save session summary to memory (lightweight, don't block)
                try:
                    session_summary = f"**User**: {message[:200]}{'...' if len(message) > 200 else ''}\n\n**COO Response**: {final_content[:500]}{'...' if len(final_content) > 500 else ''}"
                    memory.save_session_summary(
                        session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                        summary=session_summary,
                        swarm_name=None,  # COO-level, not swarm-specific
                    )
                except Exception as mem_err:
                    logger.warning(f"Failed to save session summary: {mem_err}")

            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                error_msg = str(e)

                # Make error message user-friendly
                if "ANTHROPIC_API_KEY" in error_msg:
                    pass  # Already formatted nicely
                elif "401" in error_msg or "authentication" in error_msg.lower():
                    error_msg = (
                        "**Authentication failed.**\n\n"
                        "Your API key may be invalid.\n\n"
                        "**To fix:**\n"
                        "1. Check your API key at https://console.anthropic.com\n"
                        "2. Update `backend/.env` with: `ANTHROPIC_API_KEY=your_key`\n"
                        "3. Restart the backend server"
                    )

                # Wrap error-path sends in try/except to prevent cascade failures
                try:
                    await manager.send_event(
                        websocket,
                        "agent_complete",
                        {
                            "agent": "Supreme Orchestrator",
                            "agent_type": "orchestrator",
                            "content": error_msg,
                        },
                    )
                    await manager.send_event(
                        websocket,
                        "chat_complete",
                        {
                            "success": False,
                        },
                    )
                except Exception as send_err:
                    logger.debug(f"Failed to send error response (client may have disconnected): {send_err}")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# ============================================================
# ESCALATION WEBSOCKET ENDPOINT
# ============================================================

# WebSocket subscribers for escalation events
escalation_subscribers: list[WebSocket] = []


async def broadcast_escalation_event(event_type: str, escalation_data: dict):
    """Broadcast escalation events to all subscribers.

    Event types:
        - escalation_created: New escalation created
        - escalation_updated: Escalation status changed
        - escalation_resolved: Escalation was resolved
    """
    event = {
        "type": event_type,
        **escalation_data,
    }

    for ws in escalation_subscribers[:]:
        try:
            await ws.send_json(event)
        except Exception:
            try:
                escalation_subscribers.remove(ws)
            except ValueError:
                pass

    # Also notify main chat for CEO escalations or critical priority
    priority = escalation_data.get("priority", "medium")
    to_level = escalation_data.get("to_level", "")
    if to_level == "ceo" or priority == "critical":
        chat_event = {
            "type": "escalation_notification",
            "event_type": event_type,
            "escalation_id": escalation_data.get("id", ""),
            "title": escalation_data.get("title", "Escalation"),
            "priority": priority,
            "to_level": to_level,
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass


@app.websocket("/ws/escalations")
async def websocket_escalations(websocket: WebSocket):
    """WebSocket endpoint for escalation events.

    Clients receive real-time updates when:
    - Escalations are created
    - Escalation status changes
    - Escalations are resolved

    Actions:
        - get_pending: Get pending escalations for a level (coo or ceo)
        - get_blocking: Get escalations blocking work
        - get_status: Get summary counts
    """
    await websocket.accept()
    escalation_subscribers.append(websocket)
    logger.info(f"Escalation WebSocket connected. Total subscribers: {len(escalation_subscribers)}")

    try:
        # Send initial summary
        esc_manager = get_escalation_manager()
        coo_pending = esc_manager.get_pending(level=EscalationLevel.COO)
        ceo_pending = esc_manager.get_pending(level=EscalationLevel.CEO)
        blocking = esc_manager.get_blocked_work()

        await websocket.send_json({
            "type": "escalation_status",
            "pending_coo": len(coo_pending),
            "pending_ceo": len(ceo_pending),
            "blocking_work": len(blocking),
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "get_pending":
                    level = data.get("level", "coo")
                    target = EscalationLevel.COO if level == "coo" else EscalationLevel.CEO
                    items = esc_manager.get_pending(level=target)
                    await websocket.send_json({
                        "type": "pending_escalations",
                        "level": level,
                        "escalations": [e.to_dict() for e in items],
                    })

                elif action == "get_blocking":
                    items = esc_manager.get_blocked_work()
                    await websocket.send_json({
                        "type": "blocking_escalations",
                        "escalations": [e.to_dict() for e in items],
                    })

                elif action == "get_status":
                    coo_pending = esc_manager.get_pending(level=EscalationLevel.COO)
                    ceo_pending = esc_manager.get_pending(level=EscalationLevel.CEO)
                    blocking = esc_manager.get_blocked_work()
                    await websocket.send_json({
                        "type": "escalation_status",
                        "pending_coo": len(coo_pending),
                        "pending_ceo": len(ceo_pending),
                        "blocking_work": len(blocking),
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"Escalation WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        try:
            escalation_subscribers.remove(websocket)
        except ValueError:
            pass
        logger.info(f"Escalation WebSocket disconnected. Total subscribers: {len(escalation_subscribers)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
