"""FastAPI backend for Agent Swarm web interface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
