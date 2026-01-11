"""Google Workspace Integration Service.

Provides OAuth 2.0 authentication and API access for:
- Gmail (read/send emails)
- Google Calendar (read/create events)
- Google Drive (read/upload files)
- Google Tasks (read/create tasks)
- Google Forms (read forms and responses)

Usage:
    from services.google_integration import get_google_service

    google = get_google_service()

    # Check if authenticated
    if not google.is_authenticated():
        auth_url = google.get_auth_url()
        # Direct user to auth_url, get code, then:
        google.exchange_code(code)

    # Use APIs
    emails = await google.get_emails(max_results=10)
    events = await google.get_calendar_events()
"""

from __future__ import annotations

import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List
from urllib.parse import urlencode
import aiohttp
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

# OAuth 2.0 endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# API Scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/tasks.readonly",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/forms.body.readonly",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]

# File paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "config" / "google_credentials.json"
TOKEN_FILE = PROJECT_ROOT / "config" / "google_token.json"


class GoogleIntegrationService:
    """Handles Google Workspace API integration."""

    def __init__(self):
        self._client_id: Optional[str] = None
        self._client_secret: Optional[str] = None
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._load_credentials()
        self._load_token()

    def _load_credentials(self) -> None:
        """Load OAuth credentials from file or environment."""
        # Try environment variables first
        self._client_id = os.getenv("GOOGLE_CLIENT_ID")
        self._client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

        # Fall back to credentials file
        if not self._client_id and CREDENTIALS_FILE.exists():
            try:
                with open(CREDENTIALS_FILE) as f:
                    creds = json.load(f)
                    # Handle both formats: direct or nested under "installed"/"web"
                    if "installed" in creds:
                        creds = creds["installed"]
                    elif "web" in creds:
                        creds = creds["web"]
                    self._client_id = creds.get("client_id")
                    self._client_secret = creds.get("client_secret")
                logger.info("Loaded Google credentials from file")
            except Exception as e:
                logger.error(f"Failed to load Google credentials: {e}")

    def _load_token(self) -> None:
        """Load saved OAuth token."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE) as f:
                    token_data = json.load(f)
                    self._access_token = token_data.get("access_token")
                    self._refresh_token = token_data.get("refresh_token")
                    expiry = token_data.get("expiry")
                    if expiry:
                        self._token_expiry = datetime.fromisoformat(expiry)
                logger.info("Loaded Google token from file")
            except Exception as e:
                logger.error(f"Failed to load Google token: {e}")

    def _save_token(self) -> None:
        """Save OAuth token to file."""
        try:
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            token_data = {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expiry": self._token_expiry.isoformat() if self._token_expiry else None,
            }
            with open(TOKEN_FILE, "w") as f:
                json.dump(token_data, f, indent=2)
            logger.info("Saved Google token to file")
        except Exception as e:
            logger.error(f"Failed to save Google token: {e}")

    def is_configured(self) -> bool:
        """Check if Google credentials are configured."""
        return bool(self._client_id and self._client_secret)

    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        if not self._access_token:
            return False
        if self._token_expiry and datetime.now() >= self._token_expiry:
            return bool(self._refresh_token)  # Can refresh
        return True

    def get_auth_url(self, redirect_uri: str = "http://localhost:8000/api/google/callback") -> str:
        """Generate OAuth authorization URL."""
        if not self.is_configured():
            raise ValueError("Google credentials not configured")

        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "prompt": "consent",  # Force consent to get refresh token
        }
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str = "http://localhost:8000/api/google/callback"
    ) -> bool:
        """Exchange authorization code for tokens."""
        if not self.is_configured():
            raise ValueError("Google credentials not configured")

        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(GOOGLE_TOKEN_URL, data=data) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token exchange failed: {error}")
                    return False

                token_data = await resp.json()
                self._access_token = token_data["access_token"]
                self._refresh_token = token_data.get("refresh_token", self._refresh_token)
                expires_in = token_data.get("expires_in", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
                self._save_token()
                logger.info("Successfully exchanged code for tokens")
                return True

    async def _refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self._refresh_token:
            return False

        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(GOOGLE_TOKEN_URL, data=data) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token refresh failed: {error}")
                    return False

                token_data = await resp.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
                self._save_token()
                logger.info("Successfully refreshed access token")
                return True

    async def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refreshing if needed."""
        if not self._access_token:
            return False

        if self._token_expiry and datetime.now() >= self._token_expiry - timedelta(minutes=5):
            return await self._refresh_access_token()

        return True

    async def _api_request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make an authenticated API request."""
        if not await self._ensure_valid_token():
            logger.error("No valid token for API request")
            return None

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        error = await resp.text()
                        logger.error(f"API request failed: {resp.status} - {error}")
                        return None
            elif method == "POST":
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()
                    else:
                        error = await resp.text()
                        logger.error(f"API request failed: {resp.status} - {error}")
                        return None
            elif method == "PATCH":
                async with session.patch(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        error = await resp.text()
                        logger.error(f"API request failed: {resp.status} - {error}")
                        return None

    # ==================== GMAIL API ====================

    async def get_emails(
        self,
        max_results: int = 10,
        query: str = "",
        label_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get emails from Gmail.

        Args:
            max_results: Maximum number of emails to return
            query: Gmail search query (e.g., "is:unread", "from:someone@email.com")
            label_ids: Filter by label IDs (e.g., ["INBOX", "UNREAD"])

        Returns:
            List of email objects with id, snippet, subject, from, date
        """
        params = {"maxResults": max_results}
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = ",".join(label_ids)

        # Get message list
        result = await self._api_request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            params=params
        )

        if not result or "messages" not in result:
            return []

        # Fetch full message details
        emails = []
        for msg in result["messages"][:max_results]:
            msg_detail = await self._api_request(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}",
                params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]}
            )
            if msg_detail:
                headers = {h["name"]: h["value"] for h in msg_detail.get("payload", {}).get("headers", [])}
                emails.append({
                    "id": msg_detail["id"],
                    "thread_id": msg_detail.get("threadId"),
                    "snippet": msg_detail.get("snippet", ""),
                    "subject": headers.get("Subject", "(no subject)"),
                    "from": headers.get("From", ""),
                    "date": headers.get("Date", ""),
                    "label_ids": msg_detail.get("labelIds", []),
                })

        return emails

    async def get_email_content(self, message_id: str) -> Optional[Dict]:
        """Get full email content including body."""
        result = await self._api_request(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            params={"format": "full"}
        )

        if not result:
            return None

        # Extract body from payload
        def get_body(payload: Dict) -> str:
            if "body" in payload and payload["body"].get("data"):
                return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
            if "parts" in payload:
                for part in payload["parts"]:
                    if part["mimeType"] == "text/plain":
                        if part.get("body", {}).get("data"):
                            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                    elif part["mimeType"] == "text/html":
                        if part.get("body", {}).get("data"):
                            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
            return ""

        headers = {h["name"]: h["value"] for h in result.get("payload", {}).get("headers", [])}

        return {
            "id": result["id"],
            "thread_id": result.get("threadId"),
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "body": get_body(result.get("payload", {})),
            "label_ids": result.get("labelIds", []),
        }

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        html: bool = False
    ) -> Optional[Dict]:
        """Send an email via Gmail.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (plain text or HTML)
            cc: CC recipients
            bcc: BCC recipients
            html: If True, body is treated as HTML

        Returns:
            Sent message object or None if failed
        """
        if html:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body, "html"))
        else:
            message = MIMEText(body)

        message["to"] = to
        message["subject"] = subject
        if cc:
            message["cc"] = cc
        if bcc:
            message["bcc"] = bcc

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        return await self._api_request(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            method="POST",
            data={"raw": raw}
        )

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark an email as read."""
        result = await self._api_request(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/modify",
            method="POST",
            data={"removeLabelIds": ["UNREAD"]}
        )
        return result is not None

    # ==================== CALENDAR API ====================

    async def get_calendars(self) -> List[Dict]:
        """Get list of calendars."""
        result = await self._api_request(
            "https://www.googleapis.com/calendar/v3/users/me/calendarList"
        )
        return result.get("items", []) if result else []

    async def get_calendar_events(
        self,
        calendar_id: str = "primary",
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """Get calendar events.

        Args:
            calendar_id: Calendar ID ("primary" for main calendar)
            time_min: Start of time range (defaults to now)
            time_max: End of time range (defaults to 7 days from now)
            max_results: Maximum number of events

        Returns:
            List of event objects
        """
        if not time_min:
            time_min = datetime.now()
        if not time_max:
            time_max = time_min + timedelta(days=7)

        params = {
            "timeMin": time_min.isoformat() + "Z",
            "timeMax": time_max.isoformat() + "Z",
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        result = await self._api_request(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            params=params
        )

        return result.get("items", []) if result else []

    async def create_calendar_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: str = "",
        location: str = "",
        calendar_id: str = "primary",
        attendees: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """Create a calendar event.

        Args:
            summary: Event title
            start_time: Event start time
            end_time: Event end time
            description: Event description
            location: Event location
            calendar_id: Calendar ID
            attendees: List of attendee email addresses

        Returns:
            Created event object or None
        """
        event = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "America/Los_Angeles",  # TODO: Make configurable
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "America/Los_Angeles",
            },
        }

        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]

        return await self._api_request(
            f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
            method="POST",
            data=event
        )

    # ==================== DRIVE API ====================

    async def list_drive_files(
        self,
        query: str = "",
        max_results: int = 10,
        folder_id: Optional[str] = None
    ) -> List[Dict]:
        """List files in Google Drive.

        Args:
            query: Search query (e.g., "name contains 'report'")
            max_results: Maximum number of files
            folder_id: Filter by folder ID

        Returns:
            List of file objects with id, name, mimeType, etc.
        """
        q_parts = []
        if query:
            q_parts.append(query)
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")
        q_parts.append("trashed = false")

        params = {
            "pageSize": max_results,
            "fields": "files(id, name, mimeType, size, modifiedTime, webViewLink)",
        }
        if q_parts:
            params["q"] = " and ".join(q_parts)

        result = await self._api_request(
            "https://www.googleapis.com/drive/v3/files",
            params=params
        )

        return result.get("files", []) if result else []

    async def get_drive_file_content(self, file_id: str) -> Optional[str]:
        """Get content of a text-based file from Drive."""
        # Export Google Docs as plain text
        result = await self._api_request(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
            params={"mimeType": "text/plain"}
        )

        if result:
            return result

        # For non-Google files, download directly
        if not await self._ensure_valid_token():
            return None

        headers = {"Authorization": f"Bearer {self._access_token}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                headers=headers,
                params={"alt": "media"}
            ) as resp:
                if resp.status == 200:
                    return await resp.text()

        return None

    # ==================== TASKS API ====================

    async def get_task_lists(self) -> List[Dict]:
        """Get all task lists."""
        result = await self._api_request(
            "https://tasks.googleapis.com/tasks/v1/users/@me/lists"
        )
        return result.get("items", []) if result else []

    async def get_tasks(
        self,
        task_list_id: str = "@default",
        show_completed: bool = False,
        max_results: int = 100
    ) -> List[Dict]:
        """Get tasks from a task list.

        Args:
            task_list_id: Task list ID ("@default" for default list)
            show_completed: Include completed tasks
            max_results: Maximum number of tasks

        Returns:
            List of task objects
        """
        params = {
            "maxResults": max_results,
            "showCompleted": str(show_completed).lower(),
        }

        result = await self._api_request(
            f"https://tasks.googleapis.com/tasks/v1/lists/{task_list_id}/tasks",
            params=params
        )

        return result.get("items", []) if result else []

    async def create_task(
        self,
        title: str,
        notes: str = "",
        due: Optional[datetime] = None,
        task_list_id: str = "@default"
    ) -> Optional[Dict]:
        """Create a new task.

        Args:
            title: Task title
            notes: Task notes/description
            due: Due date
            task_list_id: Task list ID

        Returns:
            Created task object or None
        """
        task = {
            "title": title,
            "notes": notes,
        }

        if due:
            task["due"] = due.strftime("%Y-%m-%dT00:00:00.000Z")

        return await self._api_request(
            f"https://tasks.googleapis.com/tasks/v1/lists/{task_list_id}/tasks",
            method="POST",
            data=task
        )

    async def complete_task(self, task_id: str, task_list_id: str = "@default") -> bool:
        """Mark a task as completed."""
        result = await self._api_request(
            f"https://tasks.googleapis.com/tasks/v1/lists/{task_list_id}/tasks/{task_id}",
            method="PATCH",
            data={"status": "completed"}
        )
        return result is not None

    # ==================== FORMS API ====================

    async def get_form(self, form_id: str) -> Optional[Dict]:
        """Get form structure/schema.

        Args:
            form_id: The Google Form ID (from URL)

        Returns:
            Form object with questions, title, etc.
        """
        return await self._api_request(
            f"https://forms.googleapis.com/v1/forms/{form_id}"
        )

    async def get_form_responses(
        self,
        form_id: str,
        page_size: int = 50,
        page_token: Optional[str] = None
    ) -> Optional[Dict]:
        """Get responses submitted to a form.

        Args:
            form_id: The Google Form ID
            page_size: Number of responses per page
            page_token: Token for pagination

        Returns:
            Dict with responses and nextPageToken
        """
        params = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        return await self._api_request(
            f"https://forms.googleapis.com/v1/forms/{form_id}/responses",
            params=params
        )

    async def list_forms_in_drive(self, max_results: int = 20) -> List[Dict]:
        """List Google Forms from Drive.

        Args:
            max_results: Maximum number of forms to return

        Returns:
            List of form files with id, name, webViewLink
        """
        return await self.list_drive_files(
            query="mimeType='application/vnd.google-apps.form'",
            max_results=max_results
        )

    # ==================== SHEETS API ====================

    async def get_spreadsheet_values(
        self,
        spreadsheet_id: str,
        range_name: str = "Sheet1"
    ) -> List[List[str]]:
        """Get values from a Google Spreadsheet.

        Args:
            spreadsheet_id: The spreadsheet ID
            range_name: The A1 notation range (e.g., "Sheet1!A1:E10" or just "Sheet1")

        Returns:
            2D list of cell values
        """
        result = await self._api_request(
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
        )
        return result.get("values", []) if result else []

    async def get_spreadsheet_metadata(self, spreadsheet_id: str) -> Optional[Dict]:
        """Get spreadsheet metadata including sheet names."""
        return await self._api_request(
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
            params={"fields": "properties,sheets.properties"}
        )

    # ==================== PROFILE INFO ====================

    async def get_user_profile(self) -> Optional[Dict]:
        """Get the authenticated user's profile info."""
        result = await self._api_request(
            "https://www.googleapis.com/oauth2/v2/userinfo"
        )
        return result

    def get_status(self) -> Dict:
        """Get current integration status."""
        return {
            "configured": self.is_configured(),
            "authenticated": self.is_authenticated(),
            "has_refresh_token": bool(self._refresh_token),
            "token_expiry": self._token_expiry.isoformat() if self._token_expiry else None,
            "scopes": SCOPES,
        }


# Singleton instance
_google_service: Optional[GoogleIntegrationService] = None


def get_google_service() -> GoogleIntegrationService:
    """Get the Google integration service singleton."""
    global _google_service
    if _google_service is None:
        _google_service = GoogleIntegrationService()
    return _google_service


# Alias for convenience
GoogleIntegration = GoogleIntegrationService
