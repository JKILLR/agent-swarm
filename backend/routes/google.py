"""Google Workspace Integration API Routes."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.google_integration import get_google_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/google", tags=["google"])


# ==================== Request/Response Models ====================

class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    cc: Optional[str] = None
    bcc: Optional[str] = None
    html: bool = False


class CreateEventRequest(BaseModel):
    summary: str
    start_time: str  # ISO format
    end_time: str    # ISO format
    description: str = ""
    location: str = ""
    calendar_id: str = "primary"
    attendees: Optional[List[str]] = None


class CreateTaskRequest(BaseModel):
    title: str
    notes: str = ""
    due: Optional[str] = None  # ISO format date
    task_list_id: str = "@default"


# ==================== Auth Endpoints ====================

@router.get("/status")
async def get_status():
    """Get Google integration status."""
    google = get_google_service()
    status = google.get_status()

    if status["authenticated"]:
        try:
            profile = await google.get_user_profile()
            status["user"] = profile
        except Exception:
            pass

    return status


@router.get("/auth")
async def start_auth():
    """Start OAuth flow - redirects to Google."""
    from fastapi.responses import RedirectResponse
    google = get_google_service()

    if not google.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Google credentials not configured. Please add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env or config/google_credentials.json"
        )

    return RedirectResponse(url=google.get_auth_url("http://localhost:8000/api/google/callback"))


@router.get("/auth/url")
async def get_auth_url(redirect_uri: str = "http://localhost:8000/api/google/callback"):
    """Get OAuth authorization URL (returns JSON)."""
    google = get_google_service()

    if not google.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Google credentials not configured. Please add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env or config/google_credentials.json"
        )

    return {"auth_url": google.get_auth_url(redirect_uri)}


@router.get("/callback")
async def oauth_callback(code: str, redirect_uri: str = "http://localhost:8000/api/google/callback"):
    """Handle OAuth callback and exchange code for tokens."""
    google = get_google_service()

    success = await google.exchange_code(code, redirect_uri)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

    return {
        "success": True,
        "message": "Successfully authenticated with Google!",
        "status": google.get_status()
    }


# ==================== Gmail Endpoints ====================

@router.get("/gmail/messages")
async def get_emails(
    max_results: int = Query(10, ge=1, le=100),
    query: str = "",
    label_ids: Optional[str] = None
):
    """Get emails from Gmail.

    Query examples:
    - "is:unread" - Unread emails
    - "from:someone@email.com" - From specific sender
    - "subject:meeting" - Subject contains "meeting"
    - "newer_than:1d" - Emails from last day
    """
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    labels = label_ids.split(",") if label_ids else None
    emails = await google.get_emails(max_results, query, labels)

    return {"emails": emails, "count": len(emails)}


@router.get("/gmail/messages/{message_id}")
async def get_email_content(message_id: str):
    """Get full email content including body."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    email = await google.get_email_content(message_id)

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    return email


@router.post("/gmail/send")
async def send_email(request: SendEmailRequest):
    """Send an email via Gmail."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    result = await google.send_email(
        to=request.to,
        subject=request.subject,
        body=request.body,
        cc=request.cc,
        bcc=request.bcc,
        html=request.html
    )

    if not result:
        raise HTTPException(status_code=500, detail="Failed to send email")

    return {"success": True, "message_id": result.get("id")}


@router.post("/gmail/messages/{message_id}/read")
async def mark_as_read(message_id: str):
    """Mark an email as read."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    success = await google.mark_as_read(message_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to mark email as read")

    return {"success": True}


# ==================== Calendar Endpoints ====================

@router.get("/calendar/list")
async def get_calendars():
    """Get list of calendars."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    calendars = await google.get_calendars()
    return {"calendars": calendars}


@router.get("/calendar/events")
async def get_calendar_events(
    calendar_id: str = "primary",
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = Query(10, ge=1, le=100)
):
    """Get calendar events."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    t_min = datetime.fromisoformat(time_min) if time_min else None
    t_max = datetime.fromisoformat(time_max) if time_max else None

    events = await google.get_calendar_events(calendar_id, t_min, t_max, max_results)

    return {"events": events, "count": len(events)}


@router.post("/calendar/events")
async def create_calendar_event(request: CreateEventRequest):
    """Create a calendar event."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    start = datetime.fromisoformat(request.start_time)
    end = datetime.fromisoformat(request.end_time)

    event = await google.create_calendar_event(
        summary=request.summary,
        start_time=start,
        end_time=end,
        description=request.description,
        location=request.location,
        calendar_id=request.calendar_id,
        attendees=request.attendees
    )

    if not event:
        raise HTTPException(status_code=500, detail="Failed to create event")

    return {"success": True, "event": event}


# ==================== Drive Endpoints ====================

@router.get("/drive/files")
async def list_drive_files(
    query: str = "",
    max_results: int = Query(10, ge=1, le=100),
    folder_id: Optional[str] = None
):
    """List files in Google Drive."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    files = await google.list_drive_files(query, max_results, folder_id)

    return {"files": files, "count": len(files)}


@router.get("/drive/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Get content of a text-based file."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    content = await google.get_drive_file_content(file_id)

    if content is None:
        raise HTTPException(status_code=404, detail="File not found or cannot read content")

    return {"content": content}


# ==================== Tasks Endpoints ====================

@router.get("/tasks/lists")
async def get_task_lists():
    """Get all task lists."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    lists = await google.get_task_lists()
    return {"task_lists": lists}


@router.get("/tasks")
async def get_tasks(
    task_list_id: str = "@default",
    show_completed: bool = False,
    max_results: int = Query(100, ge=1, le=100)
):
    """Get tasks from a task list."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    tasks = await google.get_tasks(task_list_id, show_completed, max_results)

    return {"tasks": tasks, "count": len(tasks)}


@router.post("/tasks")
async def create_task(request: CreateTaskRequest):
    """Create a new task."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    due = datetime.fromisoformat(request.due) if request.due else None

    task = await google.create_task(
        title=request.title,
        notes=request.notes,
        due=due,
        task_list_id=request.task_list_id
    )

    if not task:
        raise HTTPException(status_code=500, detail="Failed to create task")

    return {"success": True, "task": task}


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, task_list_id: str = "@default"):
    """Mark a task as completed."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    success = await google.complete_task(task_id, task_list_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to complete task")

    return {"success": True}


# ==================== Forms Endpoints ====================

@router.get("/forms")
async def list_forms(max_results: int = Query(20, ge=1, le=100)):
    """List Google Forms from Drive."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    forms = await google.list_forms_in_drive(max_results)
    return {"forms": forms, "count": len(forms)}


@router.get("/forms/{form_id}")
async def get_form(form_id: str):
    """Get a Google Form's structure and questions."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    form = await google.get_form(form_id)

    if not form:
        raise HTTPException(status_code=404, detail="Form not found")

    return form


@router.get("/forms/{form_id}/responses")
async def get_form_responses(form_id: str, max_results: int = Query(100, ge=1, le=1000)):
    """Get responses submitted to a Google Form."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    responses = await google.get_form_responses(form_id, max_results)
    return {"responses": responses, "count": len(responses)}


# ==================== Sheets Endpoints ====================

@router.get("/sheets/{spreadsheet_id}")
async def get_spreadsheet(spreadsheet_id: str, range: str = Query("Sheet1", alias="range")):
    """Get values from a Google Spreadsheet.

    Args:
        spreadsheet_id: The spreadsheet ID (from URL)
        range: A1 notation range (e.g., "Sheet1", "Sheet1!A1:E10")
    """
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    values = await google.get_spreadsheet_values(spreadsheet_id, range)
    return {"values": values, "rows": len(values)}


@router.get("/sheets/{spreadsheet_id}/metadata")
async def get_spreadsheet_metadata(spreadsheet_id: str):
    """Get spreadsheet metadata including sheet names."""
    google = get_google_service()

    if not google.is_authenticated():
        raise HTTPException(status_code=401, detail="Not authenticated with Google")

    metadata = await google.get_spreadsheet_metadata(spreadsheet_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="Spreadsheet not found")

    return metadata
