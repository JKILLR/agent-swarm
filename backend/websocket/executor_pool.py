"""Executor pool WebSocket endpoint and event broadcasting.

This module handles real-time updates for agent execution status,
including start/complete events and tool usage tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# WebSocket subscribers for executor pool events
executor_pool_subscribers: list[WebSocket] = []


async def broadcast_executor_pool_event(event: dict, manager: "ConnectionManager | None" = None):
    """Broadcast executor pool events to all subscribers and main chat connections.

    Maps executor pool events to chat WebSocket event types for consistency
    with the existing frontend.

    Args:
        event: The executor pool event to broadcast
        manager: Optional ConnectionManager for broadcasting to chat connections
    """
    # Send to dedicated executor pool subscribers
    for ws in executor_pool_subscribers[:]:
        try:
            await ws.send_json(event)
        except Exception:
            try:
                executor_pool_subscribers.remove(ws)
            except ValueError:
                pass

    # Also send to main chat WebSocket connections for parallel agent tracking
    if manager is None:
        return

    event_type = event.get("type", "")
    if event_type == "agent_execution_start":
        # Send as agent_spawn for consistency with existing frontend
        chat_event = {
            "type": "agent_spawn",
            "agent": event.get("agent", "Unknown Agent"),
            "description": f"Executing in {event.get('swarm', 'workspace')}",
            "parentAgent": "COO",
            "executionId": event.get("execution_id", ""),
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass

    elif event_type == "agent_execution_complete":
        # Send as agent_complete_subagent for consistency
        chat_event = {
            "type": "agent_complete_subagent",
            "agent": event.get("agent", "Unknown Agent"),
            "success": event.get("success", False),
            "executionId": event.get("execution_id", ""),
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass

    elif event_type in ("tool_start", "tool_complete"):
        # Pass through tool events with agent attribution
        chat_event = {
            "type": event_type,
            "tool": event.get("tool", "unknown"),
            "description": event.get("description", ""),
            "agentName": event.get("agent", "Unknown Agent"),
            "success": event.get("success", True),
        }
        for ws in manager.active_connections[:]:
            try:
                await ws.send_json(chat_event)
            except Exception:
                pass


async def websocket_executor_pool(websocket: WebSocket):
    """WebSocket endpoint for executor pool events.

    Clients receive real-time updates when:
    - Agents start executing
    - Agents complete (success or failure)
    - Tool usage by agents
    - Progress updates

    Actions:
        - get_status: Get current pool status
    """
    from shared.agent_executor_pool import get_executor_pool

    await websocket.accept()
    executor_pool_subscribers.append(websocket)
    logger.info(f"Executor pool WebSocket connected. Total subscribers: {len(executor_pool_subscribers)}")

    try:
        # Send initial status
        try:
            pool = get_executor_pool()
            await websocket.send_json({
                "type": "executor_pool_status",
                "activeCount": pool.active_count,
                "availableSlots": pool.available_slots,
                "maxConcurrent": pool.max_concurrent,
            })
        except ValueError:
            await websocket.send_json({
                "type": "executor_pool_status",
                "activeCount": 0,
                "availableSlots": 5,
                "maxConcurrent": 5,
                "initialized": False,
            })

        # Keep connection alive and handle any client messages
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "get_status":
                    try:
                        pool = get_executor_pool()
                        await websocket.send_json({
                            "type": "executor_pool_status",
                            "activeCount": pool.active_count,
                            "availableSlots": pool.available_slots,
                            "maxConcurrent": pool.max_concurrent,
                        })
                    except ValueError:
                        await websocket.send_json({
                            "type": "executor_pool_status",
                            "activeCount": 0,
                            "availableSlots": 5,
                            "maxConcurrent": 5,
                            "initialized": False,
                        })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"Executor pool WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        try:
            executor_pool_subscribers.remove(websocket)
        except ValueError:
            pass
        logger.info(f"Executor pool WebSocket disconnected. Total subscribers: {len(executor_pool_subscribers)}")
