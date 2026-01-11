"""Chat API endpoints.

This module provides endpoints for chat functionality including:
- Non-streaming chat
- Chat session management
- Conversation memory extraction
"""

import logging
from typing import Optional, Any, List, Dict

from fastapi import APIRouter, HTTPException

from models.requests import ChatMessage
from services.orchestrator_service import get_orchestrator
from services.chat_history import get_chat_history
from services.conversation_memory import get_conversation_memory_service

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


# =============================================================================
# Simple Voice Chat Endpoint
# =============================================================================

from pydantic import BaseModel

class SimpleChat(BaseModel):
    message: str
    system: Optional[str] = None


@router.post("/simple")
async def simple_chat(data: SimpleChat) -> Dict[str, Any]:
    """Simple chat endpoint for voice assistant - bypasses orchestrator.

    Uses Claude API directly for fast, lightweight responses.
    """
    import anthropic
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = data.system or "You are a helpful voice assistant. Keep responses concise and conversational - they will be spoken aloud. Aim for 1-3 sentences unless more detail is needed."

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": data.message}]
        )
        return {
            "success": True,
            "response": response.content[0].text
        }
    except Exception as e:
        logger.error(f"Simple chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def chat(data: ChatMessage) -> Dict[str, Any]:
    """Send a chat message (non-streaming).

    Args:
        data: Chat message request

    Returns:
        Dictionary with success status and response

    Raises:
        HTTPException: If chat fails
    """
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


@router.get("/sessions")
async def list_sessions() -> List[Dict]:
    """List chat sessions.

    Returns:
        List of session summary dictionaries
    """
    history = get_chat_history()
    return history.list_sessions()


@router.post("/sessions")
async def create_session(swarm: Optional[str] = None, title: Optional[str] = None) -> dict:
    """Create a new chat session.

    Args:
        swarm: Optional swarm to associate with session
        title: Optional session title

    Returns:
        New session dictionary
    """
    history = get_chat_history()
    session = history.create_session(swarm=swarm, title=title)
    return session


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get a chat session with all messages.

    Args:
        session_id: The session ID

    Returns:
        Session dictionary with messages

    Raises:
        HTTPException: If session not found
    """
    history = get_chat_history()
    session = history.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return session


@router.put("/sessions/{session_id}")
async def update_session(session_id: str, title: Optional[str] = None, swarm: Optional[str] = None) -> dict:
    """Update session metadata.

    Args:
        session_id: The session ID
        title: New title
        swarm: New swarm association

    Returns:
        Updated session dictionary

    Raises:
        HTTPException: If session not found
    """
    history = get_chat_history()
    session = history.update_session(session_id, title=title, swarm=swarm)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a chat session.

    Args:
        session_id: The session ID

    Returns:
        Dictionary with success status

    Raises:
        HTTPException: If session not found
    """
    history = get_chat_history()
    success = history.delete_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"success": True, "message": f"Session {session_id} deleted"}


# =============================================================================
# Conversation Memory Extraction
# =============================================================================

@router.post("/sessions/{session_id}/extract-memories")
async def extract_memories(session_id: str) -> dict:
    """Extract memories from a chat session and add to MindGraph.

    This analyzes the conversation and extracts:
    - Explicit memories (pattern-matched: "remember that...", "my name is...", etc.)
    - Implicit memories (LLM-extracted if enabled)
    - Creates episodic memory node linking all extracted memories

    Args:
        session_id: The session ID to process

    Returns:
        Dictionary with extracted memory nodes

    Raises:
        HTTPException: If session not found or extraction fails
    """
    history = get_chat_history()
    session = history.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    messages = session.get("messages", [])
    if not messages:
        return {
            "success": True,
            "session_id": session_id,
            "extracted_count": 0,
            "nodes": [],
            "message": "No messages to extract from",
        }

    try:
        # Get the conversation memory service (initializes on first call)
        memory_service = get_conversation_memory_service(enable_llm_extraction=False)

        # Process the conversation
        created_nodes = await memory_service.process_conversation(session_id, messages)

        return {
            "success": True,
            "session_id": session_id,
            "extracted_count": len(created_nodes),
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type.value,
                    "description": node.description,
                }
                for node in created_nodes
            ],
        }

    except Exception as e:
        logger.error(f"Memory extraction failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Memory extraction failed: {str(e)}")


@router.get("/sessions/{session_id}/memories")
async def get_session_memories(session_id: str) -> dict:
    """Get all memories extracted from a specific session.

    Queries the MindGraph for nodes that have this session_id in their provenance.

    Args:
        session_id: The session ID

    Returns:
        Dictionary with memory nodes from this session
    """
    from services.mind_graph import get_mind_graph

    graph = get_mind_graph()

    # Find nodes with this session in provenance
    session_nodes = []
    for node in graph._nodes.values():
        provenance = node.provenance or {}
        if provenance.get("session_id") == session_id:
            session_nodes.append({
                "id": node.id,
                "label": node.label,
                "type": node.node_type.value,
                "description": node.description,
                "importance": node.metadata.get("importance") if node.metadata else None,
                "created_at": node.created_at.isoformat() if node.created_at else None,
            })

    return {
        "session_id": session_id,
        "memory_count": len(session_nodes),
        "memories": session_nodes,
    }
