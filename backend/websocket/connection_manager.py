"""WebSocket connection lifecycle management."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from tracking."""
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already removed
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_event(self, websocket: WebSocket, event_type: str, data: dict[str, Any]):
        """Send a structured event to the client.

        Args:
            websocket: The WebSocket to send to
            event_type: The event type string
            data: Event payload data
        """
        try:
            if websocket not in self.active_connections:
                logger.warning(f"Tried to send {event_type} but WebSocket not in active connections")
                return  # Connection already closed
            await websocket.send_json(
                {
                    "type": event_type,
                    **data,
                }
            )
            # Log important events
            if event_type in ("agent_complete", "chat_complete", "agent_start"):
                content_preview = str(data.get("content", ""))[:100] if "content" in data else ""
                logger.info(f"Sent {event_type}: content_len={len(data.get('content', ''))} preview={content_preview}")
        except (RuntimeError, Exception) as e:
            if "close message" in str(e).lower():
                logger.debug(f"Skipped send to closed WebSocket: {e}")
            else:
                logger.error(f"Error sending {event_type} event: {e}")

    async def broadcast(self, event_type: str, data: dict[str, Any]):
        """Broadcast an event to all connected clients.

        Args:
            event_type: The event type string
            data: Event payload data
        """
        for connection in self.active_connections:
            await self.send_event(connection, event_type, data)


# Global singleton instance
manager = ConnectionManager()
