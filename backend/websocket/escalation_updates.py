"""Escalation WebSocket endpoint and event broadcasting.

This module handles real-time updates for escalation events,
allowing the frontend to receive immediate notifications when
escalations are created, updated, or resolved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# WebSocket subscribers for escalation events
escalation_subscribers: list[WebSocket] = []


async def broadcast_escalation_event(
    event_type: str,
    escalation_data: dict,
    manager: "ConnectionManager | None" = None,
):
    """Broadcast escalation events to all subscribers and optionally main chat.

    Event types:
        - escalation_created: New escalation created
        - escalation_updated: Escalation status changed
        - escalation_resolved: Escalation was resolved

    Args:
        event_type: Type of escalation event
        escalation_data: The escalation data (from .to_dict())
        manager: Optional ConnectionManager for broadcasting to chat connections
    """
    event = {
        "type": event_type,
        **escalation_data,
    }

    # Send to dedicated escalation subscribers
    for ws in escalation_subscribers[:]:
        try:
            await ws.send_json(event)
        except Exception:
            try:
                escalation_subscribers.remove(ws)
            except ValueError:
                pass

    # Also notify main chat connections for high priority escalations
    if manager is not None:
        priority = escalation_data.get("priority", "medium")
        to_level = escalation_data.get("to_level", "")

        # Only broadcast to chat for CEO escalations or critical priority
        if to_level == "ceo" or priority == "critical":
            chat_event = {
                "type": "escalation_notification",
                "event_type": event_type,
                "escalation_id": escalation_data.get("id", ""),
                "title": escalation_data.get("title", "Escalation"),
                "priority": priority,
                "to_level": to_level,
            }
            for ws in manager.active_connections[:]:
                try:
                    await ws.send_json(chat_event)
                except Exception:
                    pass


async def websocket_escalations(websocket: WebSocket):
    """WebSocket endpoint for escalation events.

    Clients receive real-time updates when:
    - Escalations are created
    - Escalation status changes
    - Escalations are resolved

    Actions:
        - get_pending: Get pending escalations for a level (coo or ceo)
        - get_blocking: Get escalations blocking work
    """
    from shared.escalation_protocol import (
        get_escalation_manager,
        EscalationLevel,
    )

    await websocket.accept()
    escalation_subscribers.append(websocket)
    logger.info(f"Escalation WebSocket connected. Total subscribers: {len(escalation_subscribers)}")

    try:
        # Send initial summary
        manager = get_escalation_manager()
        coo_pending = manager.get_pending(level=EscalationLevel.COO)
        ceo_pending = manager.get_pending(level=EscalationLevel.CEO)
        blocking = manager.get_blocked_work()

        await websocket.send_json({
            "type": "escalation_status",
            "pending_coo": len(coo_pending),
            "pending_ceo": len(ceo_pending),
            "blocking_work": len(blocking),
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "get_pending":
                    level = data.get("level", "coo")
                    target = EscalationLevel.COO if level == "coo" else EscalationLevel.CEO
                    items = manager.get_pending(level=target)
                    await websocket.send_json({
                        "type": "pending_escalations",
                        "level": level,
                        "escalations": [e.to_dict() for e in items],
                    })

                elif action == "get_blocking":
                    items = manager.get_blocked_work()
                    await websocket.send_json({
                        "type": "blocking_escalations",
                        "escalations": [e.to_dict() for e in items],
                    })

                elif action == "get_status":
                    coo_pending = manager.get_pending(level=EscalationLevel.COO)
                    ceo_pending = manager.get_pending(level=EscalationLevel.CEO)
                    blocking = manager.get_blocked_work()
                    await websocket.send_json({
                        "type": "escalation_status",
                        "pending_coo": len(coo_pending),
                        "pending_ceo": len(ceo_pending),
                        "blocking_work": len(blocking),
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"Escalation WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        try:
            escalation_subscribers.remove(websocket)
        except ValueError:
            pass
        logger.info(f"Escalation WebSocket disconnected. Total subscribers: {len(escalation_subscribers)}")
