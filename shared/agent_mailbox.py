"""Agent Mailbox System for structured agent-to-agent communication.

This module provides a mailbox system that enables structured, persistent,
priority-aware agent communication with support for handoffs, broadcasts,
and thread-based conversations.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Type of message being sent."""

    REQUEST = "request"           # Task request/delegation
    RESPONSE = "response"         # Reply to a request
    NOTIFICATION = "notification" # FYI, no response needed
    HANDOFF = "handoff"          # Structured work handoff with context
    ESCALATION = "escalation"    # Linked to escalation protocol


class MessagePriority(Enum):
    """Priority level for message processing."""

    LOW = 1       # Process when convenient
    NORMAL = 2    # Standard priority
    HIGH = 3      # Process soon
    URGENT = 4    # Process immediately


class MessageStatus(Enum):
    """Status of a message in the mailbox."""

    UNREAD = "unread"          # Not yet read by recipient
    READ = "read"              # Read but not processed
    PROCESSING = "processing"  # Currently being handled
    COMPLETED = "completed"    # Fully processed
    ARCHIVED = "archived"      # Moved to archive
    FAILED = "failed"          # Processing failed


@dataclass
class Message:
    """A message in the agent mailbox system."""

    # Identity
    id: str                           # Unique message ID (UUID)

    # Routing
    from_agent: str                   # Sender agent name (or "system", "user")
    to_agent: str                     # Recipient agent name (or "broadcast:{swarm}")
    swarm_name: str | None            # Swarm context if applicable

    # Message content
    type: MessageType                 # Request, response, notification, handoff
    subject: str                      # Brief summary/title
    body: str                         # Full message content
    payload: dict[str, Any]           # Structured data payload

    # Priority and timing
    priority: MessagePriority         # Processing priority
    created_at: datetime              # When message was created
    expires_at: datetime | None       # Optional expiration time

    # Status tracking
    status: MessageStatus = MessageStatus.UNREAD
    read_at: datetime | None = None
    completed_at: datetime | None = None

    # Threading/conversation
    reply_to: str | None = None       # ID of message this replies to
    thread_id: str | None = None      # Conversation thread ID

    # Context for handoffs
    context: dict[str, Any] = field(default_factory=dict)

    # Metadata
    attachments: list[str] = field(default_factory=list)  # File paths
    tags: list[str] = field(default_factory=list)         # Searchable tags

    def to_dict(self) -> dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "swarm_name": self.swarm_name,
            "type": self.type.value,
            "subject": self.subject,
            "body": self.body,
            "payload": self.payload,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "reply_to": self.reply_to,
            "thread_id": self.thread_id,
            "context": self.context,
            "attachments": self.attachments,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize message from dictionary."""
        return cls(
            id=data["id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            swarm_name=data.get("swarm_name"),
            type=MessageType(data["type"]),
            subject=data["subject"],
            body=data["body"],
            payload=data.get("payload", {}),
            priority=MessagePriority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            status=MessageStatus(data.get("status", "unread")),
            read_at=datetime.fromisoformat(data["read_at"]) if data.get("read_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            reply_to=data.get("reply_to"),
            thread_id=data.get("thread_id"),
            context=data.get("context", {}),
            attachments=data.get("attachments", []),
            tags=data.get("tags", []),
        )

    def to_markdown(self) -> str:
        """Format message as markdown for agent consumption."""
        priority_markers = {
            MessagePriority.LOW: "",
            MessagePriority.NORMAL: "",
            MessagePriority.HIGH: "[HIGH]",
            MessagePriority.URGENT: "[URGENT]",
        }

        lines = [
            f"## {priority_markers.get(self.priority, '')} {self.subject}".strip(),
            "",
            f"**From:** {self.from_agent}",
            f"**Type:** {self.type.value}",
            f"**Priority:** {self.priority.name}",
            f"**Sent:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
        ]

        if self.thread_id:
            lines.append(f"**Thread:** {self.thread_id}")

        if self.swarm_name:
            lines.append(f"**Swarm:** {self.swarm_name}")

        lines.extend(["", "---", "", self.body])

        if self.context:
            lines.extend(["", "### Context", ""])
            for key, value in self.context.items():
                if isinstance(value, list):
                    lines.append(f"- **{key}:**")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"- **{key}:** {value}")

        if self.attachments:
            lines.extend(["", "### Attachments", ""])
            for att in self.attachments:
                lines.append(f"- `{att}`")

        if self.payload:
            lines.extend(["", "### Payload", "", "```json"])
            lines.append(json.dumps(self.payload, indent=2))
            lines.append("```")

        return "\n".join(lines)


@dataclass
class HandoffContext:
    """Structured context for agent-to-agent handoffs."""

    # What was done
    work_completed: str
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)

    # Current state
    current_state: str = ""
    state_file: str | None = None  # e.g., "workspace/STATE.md"

    # What needs to be done next
    next_steps: list[str] = field(default_factory=list)

    # Blockers and dependencies
    blockers: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Related tracking
    related_escalations: list[str] = field(default_factory=list)
    job_id: str | None = None

    # Additional notes
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for message context."""
        return {
            "work_completed": self.work_completed,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "current_state": self.current_state,
            "state_file": self.state_file,
            "next_steps": self.next_steps,
            "blockers": self.blockers,
            "dependencies": self.dependencies,
            "related_escalations": self.related_escalations,
            "job_id": self.job_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandoffContext":
        """Create from dictionary."""
        return cls(
            work_completed=data.get("work_completed", ""),
            files_modified=data.get("files_modified", []),
            files_created=data.get("files_created", []),
            current_state=data.get("current_state", ""),
            state_file=data.get("state_file"),
            next_steps=data.get("next_steps", []),
            blockers=data.get("blockers", []),
            dependencies=data.get("dependencies", []),
            related_escalations=data.get("related_escalations", []),
            job_id=data.get("job_id"),
            notes=data.get("notes", ""),
        )

    def to_markdown(self) -> str:
        """Format as markdown for human-readable output."""
        lines = [
            "## Handoff Context",
            "",
            f"**Work Completed:** {self.work_completed}",
            "",
        ]

        if self.current_state:
            lines.append(f"**Current State:** {self.current_state}")
            lines.append("")

        if self.state_file:
            lines.append(f"**State File:** `{self.state_file}`")
            lines.append("")

        if self.files_modified:
            lines.append("**Files Modified:**")
            for f in self.files_modified:
                lines.append(f"- `{f}`")
            lines.append("")

        if self.files_created:
            lines.append("**Files Created:**")
            for f in self.files_created:
                lines.append(f"- `{f}`")
            lines.append("")

        if self.next_steps:
            lines.append("**Next Steps:**")
            for step in self.next_steps:
                lines.append(f"1. {step}")
            lines.append("")

        if self.blockers:
            lines.append("**Blockers:**")
            for blocker in self.blockers:
                lines.append(f"- {blocker}")
            lines.append("")

        if self.dependencies:
            lines.append("**Dependencies:**")
            for dep in self.dependencies:
                lines.append(f"- {dep}")
            lines.append("")

        if self.related_escalations:
            lines.append("**Related Escalations:**")
            for esc in self.related_escalations:
                lines.append(f"- {esc}")
            lines.append("")

        if self.job_id:
            lines.append(f"**Job ID:** {self.job_id}")
            lines.append("")

        if self.notes:
            lines.append("**Notes:**")
            lines.append(self.notes)

        return "\n".join(lines)


class MailboxManager:
    """Manages agent mailboxes and message routing.

    Thread-safe singleton that handles:
    - Message creation and routing
    - Per-agent mailbox directories
    - Priority queue management
    - Broadcast message distribution
    - Message lifecycle (read, complete, archive)
    """

    def __init__(self, mailboxes_dir: Path | None = None):
        """Initialize the mailbox manager.

        Args:
            mailboxes_dir: Base directory for mailboxes
        """
        self.mailboxes_dir = mailboxes_dir or Path("./workspace/mailboxes")
        self.mailboxes_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookups
        self._messages: dict[str, Message] = {}
        self._lock = threading.RLock()

        # Load existing messages on startup
        self._load_all_messages()

    def _get_agent_mailbox_dir(self, agent_name: str) -> Path:
        """Get or create mailbox directory for an agent."""
        # Normalize agent name (handle swarm/agent format)
        safe_name = agent_name.replace("/", "_").replace("\\", "_")
        mailbox_dir = self.mailboxes_dir / safe_name / "inbox"
        mailbox_dir.mkdir(parents=True, exist_ok=True)
        return mailbox_dir

    def _get_archive_dir(self, agent_name: str) -> Path:
        """Get archive directory for an agent."""
        safe_name = agent_name.replace("/", "_").replace("\\", "_")
        archive_dir = self.mailboxes_dir / safe_name / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        return archive_dir

    def _generate_id(self) -> str:
        """Generate a unique message ID."""
        return f"MSG-{uuid.uuid4().hex[:12]}"

    # ----- Core API Methods -----

    def send(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        body: str,
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        swarm_name: str | None = None,
        reply_to: str | None = None,
        thread_id: str | None = None,
        expires_at: datetime | None = None,
        attachments: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Message:
        """Send a message to an agent's mailbox.

        Args:
            from_agent: Sender name
            to_agent: Recipient name (or "broadcast:{swarm}" for broadcast)
            subject: Brief subject line
            body: Full message body
            message_type: Type of message
            priority: Processing priority
            payload: Structured data payload
            context: Additional context (for handoffs)
            swarm_name: Swarm context
            reply_to: ID of message being replied to
            thread_id: Conversation thread ID
            expires_at: When message expires
            attachments: List of file paths
            tags: Searchable tags

        Returns:
            The created Message
        """
        # Handle thread_id for replies
        if reply_to and not thread_id:
            with self._lock:
                original = self._messages.get(reply_to)
            if original:
                thread_id = original.thread_id or reply_to

        message = Message(
            id=self._generate_id(),
            from_agent=from_agent,
            to_agent=to_agent,
            swarm_name=swarm_name,
            type=message_type,
            subject=subject,
            body=body,
            payload=payload or {},
            priority=priority,
            created_at=datetime.now(),
            expires_at=expires_at,
            reply_to=reply_to,
            thread_id=thread_id,
            context=context or {},
            attachments=attachments or [],
            tags=tags or [],
        )

        with self._lock:
            # Check for broadcast
            if to_agent.startswith("broadcast:"):
                swarm = to_agent.split(":", 1)[1]
                self._broadcast_to_swarm(message, swarm)
            else:
                self._save_message(message, to_agent)

            self._messages[message.id] = message

        logger.info(
            f"Message {message.id} sent: {from_agent} -> {to_agent} "
            f"[{message_type.value}] {subject}"
        )

        return message

    def handoff(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        handoff_context: HandoffContext,
        priority: MessagePriority = MessagePriority.NORMAL,
        swarm_name: str | None = None,
        additional_notes: str = "",
    ) -> Message:
        """Send a structured handoff to another agent.

        This is a convenience method for creating well-structured
        handoff messages with complete context.

        Args:
            from_agent: Agent handing off work
            to_agent: Agent receiving work
            subject: Brief description of handoff
            handoff_context: Structured HandoffContext
            priority: Processing priority
            swarm_name: Swarm context
            additional_notes: Extra notes for the body

        Returns:
            The created Message
        """
        body_parts = [
            f"Work handoff from {from_agent} to {to_agent}.",
            "",
            f"**Work Completed:** {handoff_context.work_completed}",
            "",
            f"**Current State:** {handoff_context.current_state}",
        ]

        if handoff_context.next_steps:
            body_parts.extend(["", "**Next Steps:**"])
            for step in handoff_context.next_steps:
                body_parts.append(f"- {step}")

        if handoff_context.blockers:
            body_parts.extend(["", "**Blockers:**"])
            for blocker in handoff_context.blockers:
                body_parts.append(f"- {blocker}")

        if additional_notes:
            body_parts.extend(["", "**Notes:**", additional_notes])

        return self.send(
            from_agent=from_agent,
            to_agent=to_agent,
            subject=subject,
            body="\n".join(body_parts),
            message_type=MessageType.HANDOFF,
            priority=priority,
            context=handoff_context.to_dict(),
            swarm_name=swarm_name,
            attachments=handoff_context.files_modified + handoff_context.files_created,
            tags=["handoff"],
        )

    def broadcast(
        self,
        from_agent: str,
        swarm_name: str,
        subject: str,
        body: str,
        message_type: MessageType = MessageType.NOTIFICATION,
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude: list[str] | None = None,
        **kwargs,
    ) -> list[Message]:
        """Broadcast a message to all agents in a swarm.

        Args:
            from_agent: Sender name
            swarm_name: Target swarm
            subject: Message subject
            body: Message body
            message_type: Type of message
            priority: Priority level
            exclude: Agent names to exclude
            **kwargs: Additional message parameters

        Returns:
            List of created Messages
        """
        # Get agents in swarm
        agents = self._get_swarm_agents(swarm_name)
        exclude = exclude or []
        exclude.append(from_agent)  # Don't send to self

        messages = []
        for agent in agents:
            if agent not in exclude:
                msg = self.send(
                    from_agent=from_agent,
                    to_agent=agent,
                    subject=subject,
                    body=body,
                    message_type=message_type,
                    priority=priority,
                    swarm_name=swarm_name,
                    tags=["broadcast", f"swarm:{swarm_name}"],
                    **kwargs,
                )
                messages.append(msg)

        logger.info(
            f"Broadcast from {from_agent} to {swarm_name}: "
            f"{len(messages)} messages sent"
        )

        return messages

    def check_mailbox(
        self,
        agent_name: str,
        unread_only: bool = True,
        message_types: list[MessageType] | None = None,
        min_priority: MessagePriority | None = None,
    ) -> list[Message]:
        """Check an agent's mailbox for messages.

        Returns messages sorted by priority (highest first) then by date.

        Args:
            agent_name: Agent whose mailbox to check
            unread_only: Only return unread messages
            message_types: Filter by message types
            min_priority: Minimum priority level

        Returns:
            List of matching messages, priority-sorted
        """
        with self._lock:
            messages = [
                m for m in self._messages.values()
                if m.to_agent == agent_name
            ]

        # Apply filters
        if unread_only:
            messages = [m for m in messages if m.status == MessageStatus.UNREAD]

        if message_types:
            messages = [m for m in messages if m.type in message_types]

        if min_priority:
            messages = [m for m in messages if m.priority.value >= min_priority.value]

        # Filter out expired messages
        now = datetime.now()
        messages = [m for m in messages if not m.expires_at or m.expires_at > now]

        # Sort by priority (descending) then by date (ascending)
        messages.sort(key=lambda m: (-m.priority.value, m.created_at))

        return messages

    def read_message(self, message_id: str) -> Message | None:
        """Mark a message as read and return it.

        Args:
            message_id: ID of message to read

        Returns:
            The message, or None if not found
        """
        with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return None

            if message.status == MessageStatus.UNREAD:
                message.status = MessageStatus.READ
                message.read_at = datetime.now()
                self._save_message(message, message.to_agent)

        return message

    def mark_processing(self, message_id: str) -> Message | None:
        """Mark a message as being processed.

        Args:
            message_id: ID of message

        Returns:
            Updated message, or None if not found
        """
        with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return None

            message.status = MessageStatus.PROCESSING
            self._save_message(message, message.to_agent)

        return message

    def mark_completed(
        self,
        message_id: str,
        archive: bool = True,
    ) -> Message | None:
        """Mark a message as completed.

        Args:
            message_id: ID of message
            archive: Whether to move to archive

        Returns:
            Updated message, or None if not found
        """
        with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return None

            message.status = MessageStatus.COMPLETED
            message.completed_at = datetime.now()

            if archive:
                self._archive_message(message)
            else:
                self._save_message(message, message.to_agent)

        return message

    def reply(
        self,
        original_message_id: str,
        from_agent: str,
        body: str,
        payload: dict[str, Any] | None = None,
    ) -> Message | None:
        """Reply to a message.

        Args:
            original_message_id: ID of message to reply to
            from_agent: Agent sending reply
            body: Reply body
            payload: Optional payload

        Returns:
            The reply message, or None if original not found
        """
        with self._lock:
            original = self._messages.get(original_message_id)

        if not original:
            logger.warning(f"Cannot reply to unknown message: {original_message_id}")
            return None

        return self.send(
            from_agent=from_agent,
            to_agent=original.from_agent,
            subject=f"Re: {original.subject}",
            body=body,
            message_type=MessageType.RESPONSE,
            priority=original.priority,
            payload=payload,
            swarm_name=original.swarm_name,
            reply_to=original_message_id,
            thread_id=original.thread_id or original_message_id,
        )

    def get_thread(self, thread_id: str) -> list[Message]:
        """Get all messages in a conversation thread.

        Args:
            thread_id: Thread ID to retrieve

        Returns:
            Messages in the thread, chronologically sorted
        """
        with self._lock:
            messages = [
                m for m in self._messages.values()
                if m.thread_id == thread_id or m.id == thread_id
            ]

        messages.sort(key=lambda m: m.created_at)
        return messages

    def get_pending_count(self, agent_name: str) -> dict[str, int]:
        """Get count of pending messages by priority.

        Args:
            agent_name: Agent to check

        Returns:
            Dict with priority counts and total
        """
        messages = self.check_mailbox(agent_name, unread_only=True)

        counts = {
            "urgent": sum(1 for m in messages if m.priority == MessagePriority.URGENT),
            "high": sum(1 for m in messages if m.priority == MessagePriority.HIGH),
            "normal": sum(1 for m in messages if m.priority == MessagePriority.NORMAL),
            "low": sum(1 for m in messages if m.priority == MessagePriority.LOW),
            "total": len(messages),
        }

        return counts

    # ----- Private Methods -----

    def _save_message(self, message: Message, agent_name: str) -> None:
        """Save message to disk using atomic write."""
        mailbox_dir = self._get_agent_mailbox_dir(agent_name)
        filepath = mailbox_dir / f"{message.id}.json"
        temp_path = filepath.with_suffix(".json.tmp")

        try:
            with open(temp_path, "w") as f:
                json.dump(message.to_dict(), f, indent=2)
            temp_path.rename(filepath)  # Atomic on POSIX
        except Exception as e:
            logger.error(f"Error saving message {message.id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _archive_message(self, message: Message) -> None:
        """Move message to archive."""
        # Remove from inbox
        inbox_dir = self._get_agent_mailbox_dir(message.to_agent)
        inbox_path = inbox_dir / f"{message.id}.json"
        if inbox_path.exists():
            inbox_path.unlink()

        # Save to archive with atomic write
        archive_dir = self._get_archive_dir(message.to_agent)
        archive_path = archive_dir / f"{message.id}.json"
        temp_path = archive_path.with_suffix(".json.tmp")

        message.status = MessageStatus.ARCHIVED

        try:
            with open(temp_path, "w") as f:
                json.dump(message.to_dict(), f, indent=2)
            temp_path.rename(archive_path)  # Atomic on POSIX
        except Exception as e:
            logger.error(f"Error archiving message {message.id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _broadcast_to_swarm(self, message: Message, swarm_name: str) -> None:
        """Distribute broadcast message to all agents in swarm."""
        agents = self._get_swarm_agents(swarm_name)

        for agent in agents:
            if agent != message.from_agent:
                agent_message = Message(
                    id=self._generate_id(),  # Each copy gets unique ID
                    from_agent=message.from_agent,
                    to_agent=agent,
                    swarm_name=swarm_name,
                    type=message.type,
                    subject=message.subject,
                    body=message.body,
                    payload=message.payload,
                    priority=message.priority,
                    created_at=message.created_at,
                    expires_at=message.expires_at,
                    context=message.context,
                    attachments=message.attachments,
                    tags=message.tags + ["broadcast"],
                )
                self._save_message(agent_message, agent)
                self._messages[agent_message.id] = agent_message

    def _get_swarm_agents(self, swarm_name: str) -> list[str]:
        """Get list of agents in a swarm.

        This integrates with swarm_interface.py to get actual agents.
        Falls back to reading agent definition files from the swarm directory.
        """
        swarm_dir = Path(f"./swarms/{swarm_name}/agents")
        if swarm_dir.exists():
            return [
                f.stem for f in swarm_dir.glob("*.md")
                if not f.name.startswith("_")
            ]
        return []

    def _load_all_messages(self) -> None:
        """Load all messages from disk on startup."""
        if not self.mailboxes_dir.exists():
            return

        for agent_dir in self.mailboxes_dir.iterdir():
            if agent_dir.is_dir():
                inbox = agent_dir / "inbox"
                if inbox.exists():
                    for msg_file in inbox.glob("MSG-*.json"):
                        try:
                            with open(msg_file) as f:
                                data = json.load(f)
                                message = Message.from_dict(data)
                                self._messages[message.id] = message
                        except Exception as e:
                            logger.error(f"Error loading message {msg_file}: {e}")

        logger.info(f"Loaded {len(self._messages)} messages from disk")


# Module-level singleton with thread-safe initialization
_mailbox_manager: MailboxManager | None = None
_singleton_lock = threading.Lock()


def get_mailbox_manager(mailboxes_dir: Path | None = None) -> MailboxManager:
    """Get or create the global mailbox manager.

    Thread-safe singleton pattern with double-checked locking.

    Args:
        mailboxes_dir: Optional directory (used on first call)

    Returns:
        The mailbox manager singleton
    """
    global _mailbox_manager

    if _mailbox_manager is None:
        with _singleton_lock:
            # Double-check after acquiring lock
            if _mailbox_manager is None:
                _mailbox_manager = MailboxManager(mailboxes_dir)

    return _mailbox_manager


# ----- Convenience Functions for Agents -----


def send_message(
    from_agent: str,
    to_agent: str,
    subject: str,
    body: str,
    priority: MessagePriority = MessagePriority.NORMAL,
    **kwargs,
) -> Message:
    """Convenience function to send a message.

    Args:
        from_agent: Sender name
        to_agent: Recipient name
        subject: Message subject
        body: Message body
        priority: Priority level
        **kwargs: Additional message parameters

    Returns:
        Created Message
    """
    manager = get_mailbox_manager()
    return manager.send(
        from_agent=from_agent,
        to_agent=to_agent,
        subject=subject,
        body=body,
        priority=priority,
        **kwargs,
    )


def check_my_mailbox(
    agent_name: str,
    unread_only: bool = True,
) -> list[Message]:
    """Convenience function to check mailbox.

    Args:
        agent_name: Agent whose mailbox to check
        unread_only: Only return unread messages

    Returns:
        List of messages, priority-sorted
    """
    manager = get_mailbox_manager()
    return manager.check_mailbox(agent_name, unread_only=unread_only)


def send_handoff(
    from_agent: str,
    to_agent: str,
    subject: str,
    work_completed: str,
    current_state: str,
    next_steps: list[str],
    files_modified: list[str] | None = None,
    blockers: list[str] | None = None,
    swarm_name: str | None = None,
    priority: MessagePriority = MessagePriority.NORMAL,
) -> Message:
    """Convenience function for structured handoffs.

    Args:
        from_agent: Agent handing off
        to_agent: Agent receiving
        subject: Handoff subject
        work_completed: Summary of completed work
        current_state: Current state description
        next_steps: List of next steps
        files_modified: Files that were modified
        blockers: Any blockers
        swarm_name: Swarm context
        priority: Priority level

    Returns:
        Created handoff Message
    """
    manager = get_mailbox_manager()
    context = HandoffContext(
        work_completed=work_completed,
        current_state=current_state,
        next_steps=next_steps,
        files_modified=files_modified or [],
        blockers=blockers or [],
    )
    return manager.handoff(
        from_agent=from_agent,
        to_agent=to_agent,
        subject=subject,
        handoff_context=context,
        priority=priority,
        swarm_name=swarm_name,
    )


def broadcast_to_swarm(
    from_agent: str,
    swarm_name: str,
    subject: str,
    body: str,
    priority: MessagePriority = MessagePriority.NORMAL,
) -> list[Message]:
    """Convenience function for swarm broadcast.

    Args:
        from_agent: Sender name
        swarm_name: Target swarm
        subject: Message subject
        body: Message body
        priority: Priority level

    Returns:
        List of created Messages
    """
    manager = get_mailbox_manager()
    return manager.broadcast(
        from_agent=from_agent,
        swarm_name=swarm_name,
        subject=subject,
        body=body,
        priority=priority,
    )
