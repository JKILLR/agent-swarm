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

sys.path.insert(0, str(PROJECT_ROOT))

# Try to import Anthropic SDK
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from tools import ToolExecutor, get_tool_definitions

from memory import get_memory_manager
from supreme.orchestrator import SupremeOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
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


def parse_agent_output(content: str) -> list[dict[str, Any]]:
    """Parse raw agent output into structured events."""
    import re

    events = []

    # Try to detect agent sections
    sections = re.split(
        r"\n(?=#{1,3}\s+|\*\*(?:Researcher|Implementer|Critic|Summary|Key Finding|Recommendation))", content
    )

    current_agent = "Supreme Orchestrator"
    current_type = "orchestrator"

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Detect agent type from section header
        if re.match(r"(?:#{1,3}\s+)?\*?Researcher", section, re.I):
            current_agent = "Researcher"
            current_type = "researcher"
        elif re.match(r"(?:#{1,3}\s+)?\*?Implementer", section, re.I):
            current_agent = "Implementer"
            current_type = "implementer"
        elif re.match(r"(?:#{1,3}\s+)?\*?Critic", section, re.I):
            current_agent = "Critic"
            current_type = "critic"
        elif re.match(r"(?:#{1,3}\s+)?\*?(?:Summary|Final|Recommendation)", section, re.I):
            current_agent = "Summary"
            current_type = "summary"

        events.append(
            {
                "agent": current_agent,
                "agent_type": current_type,
                "content": section,
            }
        )

    return events if events else [{"agent": "Supreme Orchestrator", "agent_type": "orchestrator", "content": content}]


async def run_agentic_chat(
    system_prompt: str,
    user_message: str,
    websocket: WebSocket,
    manager: ConnectionManager,
    orchestrator: SupremeOrchestrator,
) -> dict:
    """
    Run an agentic chat loop with tool execution.
    The COO can delegate to swarm agents using the Task tool.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("Anthropic SDK not installed")

    client = anthropic.Anthropic(api_key=api_key)
    tool_executor = ToolExecutor(orchestrator, websocket, manager)
    tools = get_tool_definitions()

    full_response = ""
    full_thinking = ""

    messages = [{"role": "user", "content": user_message}]

    # Agentic loop - continue until no more tool calls
    max_iterations = 50  # Higher limit for complex multi-agent tasks
    for iteration in range(max_iterations):
        logger.info(f"Agentic loop iteration {iteration + 1}")

        # Make API call with tools
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=8192,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Process response content
        text_content = ""
        tool_uses = []

        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text
                # Stream text to frontend
                await manager.send_event(
                    websocket,
                    "agent_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "delta": block.text,
                    },
                )
            elif block.type == "tool_use":
                tool_uses.append(block)
                # Notify frontend about tool use
                await manager.send_event(
                    websocket,
                    "agent_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "delta": f"\n\n🔧 *Using {block.name}...*\n",
                    },
                )

        full_response += text_content

        # If no tool calls, we're done
        if response.stop_reason == "end_turn" or not tool_uses:
            break

        # Execute tools and collect results
        if tool_uses:
            # Add assistant message with tool uses
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_use in tool_uses:
                logger.info(f"Executing tool: {tool_use.name}")

                # Execute the tool
                result = await tool_executor.execute(tool_use.name, tool_use.input)

                # Stream tool result summary to frontend
                result_preview = result[:200] + "..." if len(result) > 200 else result
                await manager.send_event(
                    websocket,
                    "agent_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "delta": f"\n📋 *{tool_use.name} result:* {result_preview}\n\n",
                    },
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    }
                )

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

    return {"response": full_response, "thinking": full_thinking}


async def stream_anthropic_response(
    prompt: str,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> dict:
    """
    Stream response using the Anthropic SDK (fallback without tools).
    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("Anthropic SDK not installed")

    client = anthropic.Anthropic(api_key=api_key)

    full_response = ""
    full_thinking = ""

    # Use streaming
    with client.messages.stream(
        model="claude-opus-4-5-20251101",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_start":
                    block = event.content_block
                    if hasattr(block, "type") and block.type == "thinking":
                        await manager.send_event(
                            websocket,
                            "thinking_start",
                            {
                                "agent": "Claude",
                            },
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "type"):
                        if delta.type == "thinking_delta":
                            text = delta.thinking
                            full_thinking += text
                            await manager.send_event(
                                websocket,
                                "thinking_delta",
                                {
                                    "agent": "Claude",
                                    "delta": text,
                                },
                            )
                        elif delta.type == "text_delta":
                            text = delta.text
                            full_response += text
                            await manager.send_event(
                                websocket,
                                "agent_delta",
                                {
                                    "agent": "Claude",
                                    "agent_type": "assistant",
                                    "delta": text,
                                },
                            )

                elif event.type == "content_block_stop":
                    if full_thinking:
                        await manager.send_event(
                            websocket,
                            "thinking_complete",
                            {
                                "agent": "Claude",
                                "thinking": full_thinking,
                            },
                        )

    return {"response": full_response, "thinking": full_thinking}


async def stream_claude_response(
    prompt: str,
    swarm_name: str | None = None,
    workspace: Path | None = None,
) -> asyncio.subprocess.Process:
    """
    Start a claude CLI process and return it for streaming.

    Uses 'claude -p --output-format stream-json' which outputs JSON lines
    that we can parse and stream to the frontend.
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
        prompt,  # Pass prompt as argument
    ]

    # Set working directory to workspace if specified
    cwd = str(workspace) if workspace else None

    # Build environment with OAuth token for subprocess authentication
    env = os.environ.copy()
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        logger.info("Claude CLI using OAuth token from environment")
    else:
        logger.warning("CLAUDE_CODE_OAUTH_TOKEN not set - CLI may fail to authenticate")

    logger.info(f"Starting Claude CLI in {cwd or 'current dir'}")

    # Start the process with explicit environment
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,  # Don't use stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,  # Pass environment with token
    )

    return process


async def parse_claude_stream(
    process: asyncio.subprocess.Process,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> dict:
    """
    Parse streaming JSON output from claude CLI and send events to WebSocket.
    Returns dict with full response text and thinking.
    """
    # Use a dict to accumulate response (mutable, passed by reference)
    context = {
        "full_response": "",
        "full_thinking": "",
        "current_block_type": None,
    }

    if not process.stdout:
        return {"response": "", "thinking": ""}

    # Read all output at once to avoid buffer issues, then parse line by line
    buffer = b""
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
                    # Process event and send to websocket
                    await _process_cli_event(event, websocket, manager, context)
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
                    await _process_cli_event(event, websocket, manager, context)
                except json.JSONDecodeError:
                    pass

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


async def _process_cli_event(event: dict, websocket: WebSocket, manager, context: dict):
    """Process a single CLI event and update response/thinking."""
    event_type = event.get("type", "")

    try:
        if event_type == "assistant":
            # Final assistant message - content was already streamed via deltas
            # Just accumulate for the final response, don't re-send deltas
            message = event.get("message", {})
            content_blocks = message.get("content", [])
            for block in content_blocks:
                if block.get("type") == "thinking":
                    text = block.get("thinking", "")
                    # Only set if we didn't get it from streaming
                    if not context.get("full_thinking"):
                        context["full_thinking"] = text
                elif block.get("type") == "text":
                    text = block.get("text", "")
                    # Only set if we didn't get it from streaming
                    if not context.get("full_response"):
                        context["full_response"] = text

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
                # Send tool_start event for real-time visibility
                await manager.send_event(
                    websocket,
                    "tool_start",
                    {
                        "tool": tool_name,
                        "description": _get_tool_description(tool_name, tool_input),
                        "input": tool_input,
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
                await manager.send_event(
                    websocket,
                    "tool_complete",
                    {
                        "tool": tool_name,
                        "success": True,
                        "summary": f"Completed: {tool_name}",
                    },
                )
                # Reset tool context
                context["current_tool"] = None
                context["current_tool_input_json"] = ""
                context["agent_spawn_sent"] = False

        elif event_type == "result":
            # Final result from CLI - only use if we didn't stream content
            text = event.get("result", "")
            if text and not context.get("full_response"):
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

            if not message:
                continue

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
                # Load memory context for COO
                memory = get_memory_manager()
                memory_context = memory.load_coo_context()

                # Build conversation history from session (with summarization for long conversations)
                conversation_history = ""
                if session_id:
                    session = history.get_session(session_id)
                    if session and session.get("messages"):
                        messages = session["messages"]

                        # Check if conversation needs summarization
                        if memory.needs_summarization(messages, max_tokens=40000):
                            logger.info(f"Session {session_id} needs summarization ({len(messages)} messages)")

                            # Use summary + recent messages
                            summary_context = memory.get_context_with_summary(
                                session_id=session_id,
                                recent_messages=messages,
                                max_recent=5,  # Keep last 5 messages in full
                            )

                            if summary_context:
                                conversation_history = (
                                    "\n\n## Conversation Context (Summarized)\n" + summary_context + "\n---\n"
                                )
                        else:
                            # Full history for shorter conversations
                            history_lines = []
                            for msg in messages:
                                role = "User" if msg["role"] == "user" else "Assistant"
                                content = msg["content"]
                                # Truncate very long messages in history
                                if len(content) > 2000:
                                    content = content[:2000] + "... [truncated]"
                                history_lines.append(f"{role}: {content}")

                            if history_lines:
                                conversation_history = (
                                    "\n\n## Previous Conversation\n" + "\n\n".join(history_lines) + "\n\n---\n"
                                )

                # Build system prompt for the COO
                all_swarms = []
                for name, s in orch.swarms.items():
                    s_status = s.get_status()
                    agents_list = list(s.agents.keys())
                    all_swarms.append(f"  - **{name}**: {s_status.get('description', 'No description')}")
                    all_swarms.append(f"    Agents: {', '.join(agents_list)}")
                all_swarms_str = "\n".join(all_swarms) if all_swarms else "  No swarms defined yet"

                system_prompt = f"""You are the Supreme Orchestrator (COO) of an AI agent swarm organization.

**CRITICAL: NEVER DUPLICATE CONTENT. Each piece of information should appear EXACTLY ONCE in your response.**

## Your Role
You are the Chief Operating Officer. The CEO (human) gives you directives, and you coordinate the swarms to execute them. You have deep knowledge of the organization's vision, priorities, and current state.

## Your Tools
You have access to powerful tools:
- **Task**: Spawn subagents to do work. Format: Task(agent="swarm_name/agent_name", prompt="what to do")
- **Read/Write/Edit/Bash/Glob/Grep**: Direct file and command operations
- **ListSwarms/GetSwarmStatus**: Get information about the organization
- **GitCommit/GitSync/GitStatus**: Git operations for code changes

## IMPORTANT: Delegation
When the CEO asks you to do something that requires specialized work:
1. **USE THE TASK TOOL** to delegate to the appropriate swarm/agent
2. Don't just describe what you would do - actually do it by calling tools
3. For development work → delegate to swarm_dev (implementer, architect, etc.)
4. For research → delegate to the appropriate swarm's researcher
5. For operational tasks → delegate to operations swarm

## Organization Structure

**Swarms Available:**
{all_swarms_str}

---

{memory_context}

---

{conversation_history}

## Communication Style
- Be concise and actionable
- Use ⚡ **CEO DECISION REQUIRED** for decisions needing approval
- Never repeat yourself within a response
- Execute with tools, don't just advise"""

                user_message = message

                # Combine system prompt and user message for CLI
                full_prompt = f"""{system_prompt}

---

**User request:** {user_message}"""

                # Try Claude CLI first (uses Max subscription)
                result = None
                api_key = os.environ.get("ANTHROPIC_API_KEY")

                logger.info("Starting Claude CLI for COO chat...")
                try:
                    process = await stream_claude_response(
                        prompt=full_prompt,
                        swarm_name=None,
                        workspace=PROJECT_ROOT,
                    )

                    # Stream and parse the response
                    result = await asyncio.wait_for(
                        parse_claude_stream(process, websocket, manager),
                        timeout=900.0,  # 15 minute timeout for complex multi-agent tasks
                    )
                except asyncio.TimeoutError:
                    if process:
                        process.kill()
                    raise RuntimeError("Claude CLI timed out after 15 minutes")
                except Exception as e:
                    logger.warning(f"Claude CLI failed: {e}")
                    # Fall back to API if available
                    if api_key and ANTHROPIC_AVAILABLE:
                        logger.info("Falling back to Anthropic API...")
                        try:
                            result = await run_agentic_chat(
                                system_prompt=system_prompt,
                                user_message=user_message,
                                websocket=websocket,
                                manager=manager,
                                orchestrator=orch,
                            )
                        except Exception as api_err:
                            logger.error(f"API fallback also failed: {api_err}")
                            raise RuntimeError(f"Both CLI and API failed. CLI error: {e}")
                    else:
                        raise

                # Check if we got a result
                if result is None:
                    raise RuntimeError(
                        "**Failed to get response.**\n\n"
                        "Make sure Claude CLI is authenticated (run `claude` in terminal first)."
                    )

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
