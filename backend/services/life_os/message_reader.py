"""iMessage reader for Life OS integration.

Reads messages from macOS Messages database (chat.db).
Requires Full Disk Access permission for the running process.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
import os


class MessageReaderError(Exception):
    """Base exception for message reader errors."""
    pass


class DatabaseNotFoundError(MessageReaderError):
    """Raised when chat.db is not found."""
    pass


class DatabaseAccessError(MessageReaderError):
    """Raised when database cannot be accessed (permissions)."""
    pass


class MessageReader:
    """Read-only access to macOS Messages database."""

    # macOS stores dates as nanoseconds since 2001-01-01
    APPLE_EPOCH_OFFSET = 978307200  # seconds between 1970-01-01 and 2001-01-01

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the message reader.

        Args:
            db_path: Path to chat.db. Defaults to ~/Library/Messages/chat.db
        """
        if db_path is None:
            db_path = os.path.expanduser("~/Library/Messages/chat.db")

        self.db_path = Path(db_path)
        self._validate_database()

    def _validate_database(self) -> None:
        """Validate database exists and is accessible."""
        if not self.db_path.exists():
            raise DatabaseNotFoundError(
                f"Messages database not found at {self.db_path}. "
                "Ensure iMessage is set up on this Mac."
            )

        # Test read access
        try:
            conn = self._get_connection()
            conn.close()
        except sqlite3.OperationalError as e:
            raise DatabaseAccessError(
                f"Cannot access Messages database: {e}. "
                "Ensure Full Disk Access is granted in System Preferences > "
                "Security & Privacy > Privacy > Full Disk Access."
            )

    def _get_connection(self) -> sqlite3.Connection:
        """Get a read-only database connection.

        Uses URI mode with immutable flag to avoid locking issues.
        """
        uri = f"file:{self.db_path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _convert_apple_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Apple's nanosecond timestamp to ISO format string."""
        if timestamp is None:
            return None

        # Apple timestamps are nanoseconds since 2001-01-01
        # Convert to seconds and add offset to get Unix timestamp
        unix_timestamp = (timestamp / 1_000_000_000) + self.APPLE_EPOCH_OFFSET

        try:
            dt = datetime.fromtimestamp(unix_timestamp)
            return dt.isoformat()
        except (OSError, ValueError):
            return None

    def _row_to_message(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a message dict."""
        return {
            "id": row["ROWID"],
            "text": row["text"] or "",
            "date": self._convert_apple_timestamp(row["date"]),
            "is_from_me": bool(row["is_from_me"]),
            "contact_handle": row["handle_id"] or "",
        }

    def get_recent_messages(self, limit: int = 50) -> Generator[dict, None, None]:
        """Get recent messages across all conversations.

        Args:
            limit: Maximum number of messages to return

        Yields:
            Message dicts with id, text, date, is_from_me, contact_handle
        """
        query = """
            SELECT
                m.ROWID,
                m.text,
                m.date,
                m.is_from_me,
                h.id as handle_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            LIMIT ?
        """

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (limit,))

            for row in cursor:
                yield self._row_to_message(row)

            cursor.close()
            conn.close()
        except sqlite3.Error as e:
            raise MessageReaderError(f"Database query failed: {e}")

    def search_messages(self, query: str, limit: int = 20) -> Generator[dict, None, None]:
        """Search messages by text content.

        Args:
            query: Search string (case-insensitive substring match)
            limit: Maximum number of results

        Yields:
            Matching message dicts
        """
        sql = """
            SELECT
                m.ROWID,
                m.text,
                m.date,
                m.is_from_me,
                h.id as handle_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text LIKE ?
            ORDER BY m.date DESC
            LIMIT ?
        """

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (f"%{query}%", limit))

            for row in cursor:
                yield self._row_to_message(row)

            cursor.close()
            conn.close()
        except sqlite3.Error as e:
            raise MessageReaderError(f"Search query failed: {e}")

    def get_conversation(
        self,
        contact_id: str,
        limit: int = 50
    ) -> Generator[dict, None, None]:
        """Get messages from a specific conversation.

        Args:
            contact_id: Phone number or email (handle.id value)
            limit: Maximum number of messages

        Yields:
            Message dicts from the conversation, newest first
        """
        query = """
            SELECT
                m.ROWID,
                m.text,
                m.date,
                m.is_from_me,
                h.id as handle_id
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE h.id = ? AND m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            LIMIT ?
        """

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (contact_id, limit))

            for row in cursor:
                yield self._row_to_message(row)

            cursor.close()
            conn.close()
        except sqlite3.Error as e:
            raise MessageReaderError(f"Conversation query failed: {e}")

    def list_contacts(self, limit: int = 100) -> Generator[dict, None, None]:
        """List contacts with message history.

        Args:
            limit: Maximum number of contacts

        Yields:
            Contact dicts with handle_id and message_count
        """
        query = """
            SELECT
                h.id as handle_id,
                COUNT(m.ROWID) as message_count
            FROM handle h
            JOIN message m ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            GROUP BY h.id
            ORDER BY message_count DESC
            LIMIT ?
        """

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (limit,))

            for row in cursor:
                yield {
                    "handle_id": row["handle_id"],
                    "message_count": row["message_count"],
                }

            cursor.close()
            conn.close()
        except sqlite3.Error as e:
            raise MessageReaderError(f"Contacts query failed: {e}")


# Module-level convenience functions
_reader: Optional[MessageReader] = None


def _get_reader() -> MessageReader:
    """Get or create the singleton reader instance."""
    global _reader
    if _reader is None:
        _reader = MessageReader()
    return _reader


def get_recent_messages(limit: int = 50) -> list[dict]:
    """Get recent messages across all conversations.

    Returns a list (not generator) for easier use.
    """
    return list(_get_reader().get_recent_messages(limit))


def search_messages(query: str, limit: int = 20) -> list[dict]:
    """Search messages by text content."""
    return list(_get_reader().search_messages(query, limit))


def get_conversation(contact_id: str, limit: int = 50) -> list[dict]:
    """Get messages from a specific conversation."""
    return list(_get_reader().get_conversation(contact_id, limit))
