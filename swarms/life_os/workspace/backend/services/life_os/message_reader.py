"""iMessage Reader Service - Read messages from macOS Messages app.

Reads from ~/Library/Messages/chat.db (SQLite database).
Requires Full Disk Access permission in System Preferences > Privacy & Security.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class MessageReaderError(Exception):
    """Base exception for message reader errors."""
    pass


class PermissionError(MessageReaderError):
    """Raised when lacking permission to read chat.db."""
    pass


class DatabaseNotFoundError(MessageReaderError):
    """Raised when chat.db doesn't exist."""
    pass


class MessageReader:
    """Read iMessages from macOS chat.db."""

    # macOS stores dates as nanoseconds since 2001-01-01
    APPLE_EPOCH_OFFSET = 978307200  # Seconds between 1970-01-01 and 2001-01-01

    def __init__(self, db_path: Optional[str] = None):
        """Initialize message reader.

        Args:
            db_path: Optional custom path to chat.db. Defaults to ~/Library/Messages/chat.db
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path.home() / "Library" / "Messages" / "chat.db"

        self._connection: Optional[sqlite3.Connection] = None

    def _check_access(self) -> None:
        """Check if we can access the database.

        Raises:
            DatabaseNotFoundError: If chat.db doesn't exist
            PermissionError: If lacking read permission
        """
        if not self.db_path.exists():
            raise DatabaseNotFoundError(
                f"Messages database not found at {self.db_path}. "
                "This may not be a macOS system or Messages hasn't been used."
            )

        if not self.db_path.is_file():
            raise DatabaseNotFoundError(
                f"{self.db_path} exists but is not a file."
            )

        # Try to read to check permissions
        try:
            self.db_path.read_bytes()[:1]
        except IOError as e:
            raise PermissionError(
                f"Cannot read {self.db_path}. "
                "Grant Full Disk Access to this application in "
                "System Preferences > Privacy & Security > Full Disk Access. "
                f"Original error: {e}"
            ) from e

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection.

        Returns:
            SQLite connection to chat.db
        """
        if self._connection is None:
            self._check_access()
            # Read-only connection to avoid any accidental modifications
            self._connection = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _convert_apple_timestamp(self, timestamp: Optional[int]) -> Optional[datetime]:
        """Convert Apple's nanosecond timestamp to datetime.

        Args:
            timestamp: Apple timestamp (nanoseconds since 2001-01-01)

        Returns:
            Python datetime or None if timestamp is None/0
        """
        if not timestamp:
            return None

        # Convert from nanoseconds to seconds, then add epoch offset
        unix_timestamp = (timestamp / 1_000_000_000) + self.APPLE_EPOCH_OFFSET
        return datetime.fromtimestamp(unix_timestamp)

    def search_messages(
        self,
        query: str,
        limit: int = 50,
        chat_id: Optional[str] = None
    ) -> list[dict]:
        """Search messages containing query text.

        Args:
            query: Text to search for (case-insensitive)
            limit: Maximum number of results (default 50)
            chat_id: Optional chat identifier to filter by

        Returns:
            List of message dicts with keys: id, text, date, sender, chat_id
        """
        conn = self._get_connection()

        # Build query - join message with chat and handle for sender info
        sql = """
            SELECT
                m.ROWID as id,
                m.text,
                m.date,
                m.is_from_me,
                h.id as sender_id,
                c.chat_identifier as chat_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text LIKE ?
        """
        params: list = [f"%{query}%"]

        if chat_id:
            sql += " AND c.chat_identifier = ?"
            params.append(chat_id)

        sql += " ORDER BY m.date DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)

        results = []
        for row in cursor:
            # Determine sender
            if row["is_from_me"]:
                sender = "me"
            else:
                sender = row["sender_id"] or "unknown"

            results.append({
                "id": row["id"],
                "text": row["text"],
                "date": self._convert_apple_timestamp(row["date"]),
                "sender": sender,
                "chat_id": row["chat_id"]
            })

        return results

    def get_recent_messages(self, limit: int = 50) -> list[dict]:
        """Get most recent messages.

        Args:
            limit: Maximum number of results (default 50)

        Returns:
            List of message dicts with keys: id, text, date, sender, chat_id
        """
        conn = self._get_connection()

        sql = """
            SELECT
                m.ROWID as id,
                m.text,
                m.date,
                m.is_from_me,
                h.id as sender_id,
                c.chat_identifier as chat_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            LIMIT ?
        """

        cursor = conn.execute(sql, [limit])

        results = []
        for row in cursor:
            if row["is_from_me"]:
                sender = "me"
            else:
                sender = row["sender_id"] or "unknown"

            results.append({
                "id": row["id"],
                "text": row["text"],
                "date": self._convert_apple_timestamp(row["date"]),
                "sender": sender,
                "chat_id": row["chat_id"]
            })

        return results

    def get_chats(self) -> list[dict]:
        """Get list of all chats/conversations.

        Returns:
            List of chat dicts with keys: id, identifier, display_name
        """
        conn = self._get_connection()

        sql = """
            SELECT
                ROWID as id,
                chat_identifier as identifier,
                display_name
            FROM chat
            ORDER BY ROWID DESC
        """

        cursor = conn.execute(sql)

        return [
            {
                "id": row["id"],
                "identifier": row["identifier"],
                "display_name": row["display_name"] or row["identifier"]
            }
            for row in cursor
        ]

    def check_permission(self) -> dict:
        """Check if we have permission to read messages.

        Returns:
            Dict with keys: has_permission, error (if any)
        """
        try:
            self._check_access()
            # Also try to actually query
            conn = self._get_connection()
            conn.execute("SELECT COUNT(*) FROM message LIMIT 1")
            return {"has_permission": True, "error": None}
        except MessageReaderError as e:
            return {"has_permission": False, "error": str(e)}
        except sqlite3.Error as e:
            return {"has_permission": False, "error": f"Database error: {e}"}

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for simple usage
def search_messages(query: str, limit: int = 50) -> list[dict]:
    """Search iMessages for text.

    Args:
        query: Text to search for (case-insensitive)
        limit: Maximum number of results (default 50)

    Returns:
        List of message dicts with keys: id, text, date, sender, chat_id

    Raises:
        MessageReaderError: If database cannot be accessed
    """
    with MessageReader() as reader:
        return reader.search_messages(query, limit)
