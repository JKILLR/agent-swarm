"""Chat history persistence management."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


# Global chat history manager singleton
_chat_history: ChatHistoryManager | None = None


def get_chat_history(project_root: Path | None = None) -> ChatHistoryManager:
    """Get or create the chat history manager.

    Args:
        project_root: Path to project root. Only used on first call.

    Returns:
        The global ChatHistoryManager instance.
    """
    global _chat_history
    if _chat_history is None:
        if project_root is None:
            raise ValueError("project_root required on first call to get_chat_history()")
        _chat_history = ChatHistoryManager(project_root)
    return _chat_history
