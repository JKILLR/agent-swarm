"""Chat API endpoints.

This module provides endpoints for chat functionality including:
- Non-streaming chat
- Chat session management
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import ChatMessage
from ..services.orchestrator_service import get_orchestrator
from ..services.chat_history import get_chat_history

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("")
async def chat(data: ChatMessage) -> dict[str, Any]:
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
async def list_sessions() -> list[dict]:
    """List chat sessions.

    Returns:
        List of session summary dictionaries
    """
    history = get_chat_history()
    return history.list_sessions()


@router.post("/sessions")
async def create_session(swarm: str | None = None, title: str | None = None) -> dict:
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
async def update_session(session_id: str, title: str | None = None, swarm: str | None = None) -> dict:
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
