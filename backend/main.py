"""FastAPI backend for Agent Swarm web interface."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Configure logging - both console and file for COO diagnostics
LOG_FILE = PROJECT_ROOT / "logs" / "backend.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler(LOG_FILE, mode='a'),  # File for COO to read
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent Swarm API",
    description="API for managing hierarchical AI agent swarms",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: SupremeOrchestrator | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    import sqlite3
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


async def _broadcast_job_update_safe(job):
    """Safely broadcast job updates (handles import order)."""
    try:
        await broadcast_job_update(job)
    except Exception as e:
        logger.error(f"Failed to broadcast job update: {e}")


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
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_event(self, websocket: WebSocket, event_type: str, data: dict[str, Any]):
        """Send a structured event to the client."""
        await websocket.send_json(
            {
                "type": event_type,
                **data,
            }
        )


manager = ConnectionManager()


async def stream_claude_response(
    prompt: str,
    swarm_name: str | None = None,
    workspace: Path | None = None,
    chat_id: str | None = None,
    system_prompt: str | None = None,
) -> asyncio.subprocess.Process:
    """
    Start a claude CLI process and return it for streaming.

    Uses 'claude -p --output-format stream-json' which outputs JSON lines
    that we can parse and stream to the frontend.

    Args:
        prompt: The user message/request
        system_prompt: Custom system prompt (COO role, context, etc.)
        chat_id: Session ID for continuity
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

    # Build environment - keep API key for now since Max subscription OAuth has issues
    # TODO: Remove this once Max subscription OAuth is working
    # To use Max subscription instead of API credits, uncomment the line below:
    # env.pop("ANTHROPIC_API_KEY", None)  # Force CLI to use Max subscription
    env = os.environ.copy()

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


def _get_tool_description(tool_name: str, tool_input: dict) -> str:
    """Generate human-readable description for a tool call."""
    descriptions = {
        "Read": lambda i: f"Reading {i.get('file_path', 'file')[:50]}",
        "Write": lambda i: f"Writing to {i.get('file_path', 'file')[:50]}",
        "Edit": lambda i: f"Editing {i.get('file_path', 'file')[:50]}",
        "Bash": lambda i: f"Running: {i.get('command', '')[:40]}{'...' if len(i.get('command', '')) > 40 else ''}",
        "Glob": lambda i: f"Searching for {i.get('pattern', 'files')}",
        "Grep": lambda i: f"Searching for '{i.get('pattern', '')[:30]}'",
        "Task": lambda i: f"Delegating to {i.get('agent', 'agent')}: {i.get('prompt', '')[:40]}...",
        "WebSearch": lambda i: f"Searching web: {i.get('query', '')[:40]}",
        "WebFetch": lambda i: f"Fetching {i.get('url', 'URL')[:40]}",
    }

    if tool_name in descriptions:
        try:
            return descriptions[tool_name](tool_input)
        except Exception:
            pass

    return f"Using {tool_name}"


async def _process_cli_event(event: dict, websocket: WebSocket, manager, context: dict, session_mgr=None, chat_id: str = None):
    """Process a single CLI event and forward to WebSocket."""
    event_type = event.get("type", "")

    try:
        # Capture session ID from Claude output for session continuity
        if event_type in ("init", "system", "session_start") and session_mgr and chat_id:
            session_id = event.get("session_id") or event.get("sessionId")
            if session_id:
                context["session_id"] = session_id
                asyncio.create_task(session_mgr.register_session(chat_id, session_id))

        if event_type == "assistant":
            # Assistant message in agentic loop - may have text, thinking, or tool_use blocks
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
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})

                    # Track which agent is active (for attribution)
                    current_agent = context.get("current_subagent", "COO")
                    if tool_name == "Task":
                        # Extract subagent name from Task input
                        subagent = tool_input.get("subagent_type") or tool_input.get("agent", "")
                        if subagent:
                            context["current_subagent"] = subagent
                            current_agent = subagent

                    # Only send if we didn't already send via streaming
                    if not context.get(f"sent_tool_{tool_name}_{id(block)}"):
                        await manager.send_event(
                            websocket,
                            "tool_start",
                            {
                                "tool": tool_name,
                                "description": _get_tool_description(tool_name, tool_input),
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
                tool_input = content_block.get("input", {})
                context["current_tool"] = tool_name

                # Track which agent is active (for attribution)
                current_agent = context.get("current_subagent", "COO")
                if tool_name == "Task":
                    # Extract subagent name from Task input
                    subagent = tool_input.get("subagent_type") or tool_input.get("agent", "")
                    if subagent:
                        context["current_subagent"] = subagent
                        current_agent = subagent

                # Mark as sent to avoid duplicate from fallback
                context[f"sent_tool_{tool_name}"] = True
                # Send tool_start event for real-time visibility
                await manager.send_event(
                    websocket,
                    "tool_start",
                    {
                        "tool": tool_name,
                        "description": _get_tool_description(tool_name, tool_input),
                        "input": tool_input,
                        "agentName": current_agent,
                    },
                )

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
                if context.get("current_tool") == "Task":
                    try:
                        import json
                        partial_input = json.loads(context["current_tool_input_json"])
                        agent_name = partial_input.get("agent", "")
                        if agent_name and not context.get("agent_spawn_sent"):
                            # Send agent_spawn event
                            await manager.send_event(
                                websocket,
                                "agent_spawn",
                                {
                                    "agent": agent_name,
                                    "description": partial_input.get("prompt", "")[:100],
                                },
                            )
                            context["agent_spawn_sent"] = True
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
                current_agent = context.get("current_subagent", "COO")
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
                # Reset tool context
                context["current_tool"] = None
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

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            _swarm_name = data.get("swarm")  # Reserved for future swarm-specific routing
            session_id = data.get("session_id")
            attachments = data.get("attachments", [])

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

                # System prompt for COO - uses CLI's built-in Task tool
                system_prompt = f"""You are the COO coordinating work across specialized agents.

## Your Job
DELEGATE work to agents. Don't implement code yourself.

## Available Agents (use with Task tool)
- researcher: Research topics, analyze code, gather information
- architect: Design solutions, plan implementations
- implementer: Write code, create files, make changes
- critic: Review code, find bugs, suggest improvements
- tester: Write tests, verify implementations

## How to Delegate
Use the Task tool with agent name and detailed prompt:
Task(subagent_type="researcher", prompt="Analyze the trading bot code in swarms/trading_bots/workspace/ and explain how it works")
Task(subagent_type="implementer", prompt="Add --yolo flag to skip confirmation in advanced_arb_bot.py")

## Swarm Workspaces
{all_swarms_str}
Files are at: swarms/<swarm_name>/workspace/

## STATE.md - Shared Context
Each swarm has a `workspace/STATE.md` file that maintains shared context across agents:
- **Before delegating**: Read STATE.md to understand current state
- **In agent prompts**: Tell agents to read and update STATE.md
- **If STATE.md doesn't exist**: Create it from the template when starting work on a swarm

The STATE.md contains: objectives, progress log, key files, architecture decisions, known issues, and next steps.
All agents are instructed to read it first and update it after completing work.

## Diagnostics
If something isn't working or the user asks about system status:
- Read `logs/backend.log` for recent backend activity and errors
- This shows WebSocket connections, CLI spawns, and any errors
- Use this to diagnose issues and explain what's happening

## Rules
1. DELEGATE - use Task to spawn agents for implementation work
2. Read STATE.md first to understand current swarm state
3. Be specific in prompts - tell agents exactly what to do and where STATE.md is
4. Synthesize agent results into clear summaries for the user
5. If errors occur, check logs/backend.log to understand and explain the issue"""

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

                # Use Claude CLI
                try:
                    process = await stream_claude_response(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        swarm_name=None,
                        workspace=PROJECT_ROOT,
                        chat_id=session_id,
                    )

                    # Stream and parse the response
                    result = await asyncio.wait_for(
                        parse_claude_stream(process, websocket, manager, chat_id=session_id),
                        timeout=900.0,  # 15 minute timeout
                    )
                except asyncio.TimeoutError:
                    if process:
                        process.kill()
                    raise RuntimeError("COO timed out after 15 minutes")
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

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
