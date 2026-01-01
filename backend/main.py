"""FastAPI backend for Agent Swarm web interface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import base64
import mimetypes

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from supreme.orchestrator import SupremeOrchestrator
from shared.swarm_interface import load_swarm

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
orchestrator: Optional[SupremeOrchestrator] = None


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
    swarm: Optional[str] = None


class SwarmResponse(BaseModel):
    name: str
    description: str
    status: str
    agent_count: int
    priorities: List[str]


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
async def list_swarms() -> List[Dict[str, Any]]:
    """List all swarms with status."""
    orch = get_orchestrator()
    swarms = []

    for name, swarm in orch.swarms.items():
        status = swarm.get_status()
        priorities = status.get("priorities", [])
        # Handle dict priorities
        priority_strs = []
        for p in priorities[:3]:
            if isinstance(p, dict):
                priority_strs.append(p.get("task", str(p)))
            else:
                priority_strs.append(str(p))

        swarms.append({
            "name": status["name"],
            "description": status["description"],
            "status": status["status"],
            "agent_count": status["agent_count"],
            "priorities": priority_strs,
            "version": status.get("version", "0.1.0"),
        })

    return swarms


@app.get("/api/swarms/{name}")
async def get_swarm(name: str) -> Dict[str, Any]:
    """Get detailed swarm information."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    status = swarm.get_status()
    return status


@app.post("/api/swarms")
async def create_swarm(data: SwarmCreate) -> Dict[str, Any]:
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
async def list_agents(name: str) -> List[Dict[str, Any]]:
    """List agents in a swarm."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    agents = []
    for agent_name, defn in swarm.agent_definitions.items():
        agents.append({
            "name": agent_name,
            "type": defn.agent_type,
            "model": defn.model,
            "background": defn.background,
            "description": defn.description,
        })

    return agents


@app.post("/api/chat")
async def chat(data: ChatMessage) -> Dict[str, Any]:
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
def get_file_info(file_path: Path, workspace: Path) -> Dict[str, Any]:
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
        if ext in [".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".css", ".html"]:
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
async def list_files(name: str, path: str = "") -> Dict[str, Any]:
    """List files in a swarm's workspace."""
    orch = get_orchestrator()
    swarm = orch.get_swarm(name)

    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm '{name}' not found")

    workspace = swarm.workspace
    target_path = workspace / path if path else workspace

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
async def get_file_content(name: str, path: str) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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
async def create_directory(name: str, path: str = Form(...)) -> Dict[str, Any]:
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
async def delete_file(name: str, path: str) -> Dict[str, Any]:
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


# WebSocket for streaming chat
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_event(self, websocket: WebSocket, event_type: str, data: Dict[str, Any]):
        """Send a structured event to the client."""
        await websocket.send_json({
            "type": event_type,
            **data,
        })


manager = ConnectionManager()


def parse_agent_output(content: str) -> List[Dict[str, Any]]:
    """Parse raw agent output into structured events."""
    import re

    events = []

    # Try to detect agent sections
    sections = re.split(r"\n(?=#{1,3}\s+|\*\*(?:Researcher|Implementer|Critic|Summary|Key Finding|Recommendation))", content)

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

        events.append({
            "agent": current_agent,
            "agent_type": current_type,
            "content": section,
        })

    return events if events else [{"agent": "Supreme Orchestrator", "agent_type": "orchestrator", "content": content}]


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await manager.connect(websocket)
    orch = get_orchestrator()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                continue

            # Send acknowledgment
            await manager.send_event(websocket, "chat_start", {
                "message": message,
            })

            # Send thinking indicator
            await manager.send_event(websocket, "agent_start", {
                "agent": "Supreme Orchestrator",
                "agent_type": "orchestrator",
            })

            try:
                # Get response from orchestrator
                response = await orch.route_request(message)

                # Parse response into agent sections
                events = parse_agent_output(response)

                # Send each agent's output
                for event in events:
                    await manager.send_event(websocket, "agent_complete", event)
                    await asyncio.sleep(0.1)  # Small delay for UI animation

                # Send completion
                await manager.send_event(websocket, "chat_complete", {
                    "success": True,
                })

            except Exception as e:
                logger.error(f"Chat error: {e}")
                await manager.send_event(websocket, "error", {
                    "message": str(e),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
