"""Session manager for persistent Claude sessions.

Maintains session IDs to enable --continue flag, saving 2-3s per agent spawn
by avoiding cold starts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaudeSession:
    """Represents an active Claude CLI session."""
    session_id: str
    chat_id: str
    created_at: datetime
    last_used: datetime


# Singleton instance
_session_manager: Optional["SessionManager"] = None


class SessionManager:
    """Maintain persistent Claude sessions per chat.

    Session continuity provides:
    - Context persistence across messages (no need to resend full history)
    - Faster response times (2-3s saved per agent)
    - Better conversation flow
    """

    def __init__(self):
        self.active_sessions: dict[str, ClaudeSession] = {}
        self._lock = asyncio.Lock()

    async def get_session(self, chat_id: str) -> Optional[str]:
        """Get existing session ID for a chat.

        Args:
            chat_id: The chat/conversation ID

        Returns:
            Claude session ID if one exists, None otherwise
        """
        async with self._lock:
            session = self.active_sessions.get(chat_id)
            if session:
                session.last_used = datetime.now()
                logger.debug(f"Found session {session.session_id} for chat {chat_id}")
                return session.session_id
            return None

    async def register_session(self, chat_id: str, session_id: str):
        """Register a new session from Claude output.

        Args:
            chat_id: The chat/conversation ID
            session_id: The Claude session ID to register
        """
        async with self._lock:
            self.active_sessions[chat_id] = ClaudeSession(
                session_id=session_id,
                chat_id=chat_id,
                created_at=datetime.now(),
                last_used=datetime.now(),
            )
            logger.info(f"Registered session {session_id} for chat {chat_id}")

    async def end_session(self, chat_id: str):
        """End a session and clean up.

        Args:
            chat_id: The chat/conversation ID to end
        """
        async with self._lock:
            if chat_id in self.active_sessions:
                session = self.active_sessions.pop(chat_id)
                logger.info(f"Ended session {session.session_id} for chat {chat_id}")

    def get_continue_flags(self, chat_id: str) -> list[str]:
        """Get --continue flags if session exists (sync for command building).

        Args:
            chat_id: The chat/conversation ID

        Returns:
            ["--continue", "<session_id>"] if session exists, [] otherwise
        """
        session = self.active_sessions.get(chat_id)
        if session:
            return ["--continue", session.session_id]
        return []

    async def cleanup_stale_sessions(self, max_age_hours: int = 24):
        """Clean up sessions that haven't been used recently.

        Args:
            max_age_hours: Maximum hours since last use before cleanup
        """
        async with self._lock:
            now = datetime.now()
            stale = []
            for chat_id, session in self.active_sessions.items():
                age = (now - session.last_used).total_seconds() / 3600
                if age > max_age_hours:
                    stale.append(chat_id)

            for chat_id in stale:
                session = self.active_sessions.pop(chat_id)
                logger.info(f"Cleaned up stale session {session.session_id}")

            if stale:
                logger.info(f"Cleaned up {len(stale)} stale sessions")

    def get_stats(self) -> dict:
        """Get session statistics.

        Returns:
            Dict with active session count and details
        """
        return {
            "active_sessions": len(self.active_sessions),
            "sessions": [
                {
                    "chat_id": s.chat_id,
                    "session_id": s.session_id[:8] + "...",  # Truncate for privacy
                    "created_at": s.created_at.isoformat(),
                    "last_used": s.last_used.isoformat(),
                }
                for s in self.active_sessions.values()
            ]
        }


def get_session_manager() -> SessionManager:
    """Get the singleton session manager.

    Returns:
        The global SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
