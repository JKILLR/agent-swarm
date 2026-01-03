"""Mailbox API endpoints for agent communication.

This module provides endpoints for the Agent Mailbox system including:
- Checking mailboxes
- Sending messages
- Handoffs between agents
- Message threading
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models.requests import MessageSendRequest, HandoffRequest
from shared.agent_mailbox import (
    get_mailbox_manager,
    MessageType,
    MessagePriority,
    HandoffContext,
)

router = APIRouter(prefix="/api/mailbox", tags=["mailbox"])


@router.get("/{agent_name}")
async def check_mailbox(
    agent_name: str,
    unread_only: bool = True,
    message_type: str | None = None,
) -> list[dict]:
    """Check an agent's mailbox for messages.

    Args:
        agent_name: Name of the agent
        unread_only: Only return unread messages
        message_type: Filter by message type

    Returns:
        List of message dictionaries
    """
    mailbox = get_mailbox_manager()

    message_types = None
    if message_type:
        message_types = [MessageType(message_type)]

    messages = mailbox.check_mailbox(
        agent_name=agent_name,
        unread_only=unread_only,
        message_types=message_types,
    )

    return [msg.to_dict() for msg in messages]


@router.get("/{agent_name}/count")
async def get_mailbox_count(agent_name: str) -> dict:
    """Get count of pending messages by priority.

    Args:
        agent_name: Name of the agent

    Returns:
        Dictionary with counts by priority level
    """
    mailbox = get_mailbox_manager()
    return mailbox.get_pending_count(agent_name)


@router.post("/send")
async def send_message(request: MessageSendRequest) -> dict:
    """Send a message to an agent's mailbox.

    Args:
        request: Message send request

    Returns:
        Dictionary with success status and message details
    """
    mailbox = get_mailbox_manager()

    message = mailbox.send(
        from_agent=request.from_agent,
        to_agent=request.to_agent,
        subject=request.subject,
        body=request.body,
        message_type=MessageType(request.message_type),
        priority=MessagePriority[request.priority.upper()],
        swarm_name=request.swarm_name if hasattr(request, 'swarm_name') else None,
        payload=request.metadata,
        reply_to=request.thread_id,
        tags=None,
    )

    return {"success": True, "message_id": message.id, "message": message.to_dict()}


@router.post("/handoff")
async def send_handoff(request: HandoffRequest) -> dict:
    """Send a structured handoff to another agent.

    Args:
        request: Handoff request

    Returns:
        Dictionary with success status and message details
    """
    mailbox = get_mailbox_manager()

    context = HandoffContext(
        work_completed=request.completed_work,
        current_state=request.task_summary,
        next_steps=[request.remaining_work] if request.remaining_work else [],
        files_modified=[],
        blockers=[],
    )

    message = mailbox.handoff(
        from_agent=request.from_agent,
        to_agent=request.to_agent,
        subject=request.task_summary,
        handoff_context=context,
        priority=MessagePriority[request.priority.upper()],
        swarm_name=None,
    )

    return {"success": True, "message_id": message.id, "message": message.to_dict()}


@router.post("/message/{message_id}/read")
async def read_message(message_id: str) -> dict:
    """Mark a message as read.

    Args:
        message_id: The message ID

    Returns:
        Dictionary with success status and message details

    Raises:
        HTTPException: If message not found
    """
    mailbox = get_mailbox_manager()
    message = mailbox.read_message(message_id)

    if not message:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")

    return {"success": True, "message": message.to_dict()}


@router.post("/message/{message_id}/complete")
async def complete_message(message_id: str, archive: bool = True) -> dict:
    """Mark a message as completed.

    Args:
        message_id: The message ID
        archive: Whether to archive the message

    Returns:
        Dictionary with success status and message details

    Raises:
        HTTPException: If message not found
    """
    mailbox = get_mailbox_manager()
    message = mailbox.mark_completed(message_id, archive=archive)

    if not message:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")

    return {"success": True, "message": message.to_dict()}


@router.post("/message/{message_id}/reply")
async def reply_to_message(message_id: str, from_agent: str, body: str, payload: dict | None = None) -> dict:
    """Reply to a message.

    Args:
        message_id: The original message ID
        from_agent: The replying agent
        body: Reply content
        payload: Optional additional data

    Returns:
        Dictionary with success status and reply details

    Raises:
        HTTPException: If original message not found
    """
    mailbox = get_mailbox_manager()
    message = mailbox.reply(message_id, from_agent, body, payload)

    if not message:
        raise HTTPException(status_code=404, detail=f"Original message {message_id} not found")

    return {"success": True, "message": message.to_dict()}


@router.get("/thread/{thread_id}")
async def get_message_thread(thread_id: str) -> list[dict]:
    """Get all messages in a conversation thread.

    Args:
        thread_id: The thread ID

    Returns:
        List of message dictionaries in the thread
    """
    mailbox = get_mailbox_manager()
    messages = mailbox.get_thread(thread_id)
    return [msg.to_dict() for msg in messages]
