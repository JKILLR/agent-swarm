# backend/routes/context.py
"""API routes for personal context management."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from backend.services.context_service import get_context_service

router = APIRouter(prefix="/api/context", tags=["context"])


class WorkingContextUpdate(BaseModel):
    """Model for updating working context."""
    key: str
    value: Any


class ProfileUpdate(BaseModel):
    """Model for updating profile."""
    updates: dict[str, Any]


class StyleUpdate(BaseModel):
    """Model for updating communication style."""
    updates: dict[str, Any]


class LearnPhraseRequest(BaseModel):
    """Model for learning a phrase."""
    phrase_type: str  # greeting, closing, transition
    phrase: str


@router.get("/foundation")
async def get_foundation():
    """Get all foundation context (profile, style, preferences)."""
    service = get_context_service()
    return service.get_foundation()


@router.get("/project/{project_name}")
async def get_project(project_name: str):
    """Get project-specific context."""
    service = get_context_service()
    project = service.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
    return project


@router.get("/projects")
async def list_projects():
    """List all available projects."""
    service = get_context_service()
    return {"projects": service.list_projects()}


@router.get("/working")
async def get_working():
    """Get current working context."""
    service = get_context_service()
    return service.get_working()


@router.post("/working")
async def update_working(update: WorkingContextUpdate):
    """Update working context with a key-value pair."""
    service = get_context_service()
    service.update_working(update.key, update.value)
    return {"status": "updated", "key": update.key}


@router.delete("/working")
async def clear_working():
    """Clear working context."""
    service = get_context_service()
    service.clear_working()
    return {"status": "cleared"}


@router.get("/combined")
async def get_combined(project: Optional[str] = None):
    """Get merged context for LLM prompt injection."""
    service = get_context_service()
    return service.get_combined_context(project)


@router.get("/summary")
async def get_summary(project: Optional[str] = None):
    """Get a text summary suitable for LLM system prompts."""
    service = get_context_service()
    summary = service.get_context_summary(project)
    return {"summary": summary}


@router.patch("/profile")
async def update_profile(update: ProfileUpdate):
    """Update profile with new values."""
    service = get_context_service()
    success = service.update_profile(update.updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update profile")
    return {"status": "updated"}


@router.patch("/style")
async def update_style(update: StyleUpdate):
    """Update communication style with new values."""
    service = get_context_service()
    success = service.update_communication_style(update.updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update style")
    return {"status": "updated"}


@router.post("/learn-phrase")
async def learn_phrase(request: LearnPhraseRequest):
    """Learn a new phrase from user behavior."""
    valid_types = ["greetings", "closings", "transitions"]
    if request.phrase_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid phrase_type. Must be one of: {valid_types}"
        )
    
    service = get_context_service()
    success = service.learn_phrase(request.phrase_type, request.phrase)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to learn phrase")
    return {"status": "learned", "type": request.phrase_type, "phrase": request.phrase}


@router.post("/invalidate")
async def invalidate_cache(project: Optional[str] = None):
    """Invalidate context cache to force reload."""
    service = get_context_service()
    service.invalidate_cache(project)
    return {"status": "invalidated", "project": project or "all"}
