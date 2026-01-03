"""Escalation protocol for agent-swarm hierarchy.

This module defines the escalation levels, reasons, and data structures
for managing escalations between agents, COO, and CEO.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Hierarchy level for escalation."""

    AGENT = "agent"       # Swarm agent level
    COO = "coo"          # Supreme Orchestrator level
    CEO = "ceo"          # Human user level


class EscalationReason(Enum):
    """Reason for escalation."""

    # Agent-to-COO reasons
    BLOCKED = "blocked"              # External dependency blocking progress
    CLARIFICATION = "clarification"  # Ambiguous requirements
    CONFLICT = "conflict"            # Conflicting instructions
    SECURITY = "security"            # Security concern identified
    ARCHITECTURE = "architecture"    # Decision beyond scope
    SCOPE_EXCEEDED = "scope_exceeded"  # Task too large/complex

    # COO-to-CEO reasons
    ARCHITECTURE_MAJOR = "architecture_major"  # System-wide change
    SECURITY_CRITICAL = "security_critical"    # Critical vulnerability
    PRIORITY_CONFLICT = "priority_conflict"    # Business priorities
    COST = "cost"                              # Cost implications
    PERMISSION = "permission"                  # Requires human approval
    BLOCKED_CRITICAL = "blocked_critical"      # Critical path blocked


class EscalationStatus(Enum):
    """Status of an escalation."""

    PENDING = "pending"           # Awaiting response
    IN_PROGRESS = "in_progress"   # Being addressed
    RESOLVED = "resolved"         # Resolution provided
    CANCELLED = "cancelled"       # No longer needed
    DEFERRED = "deferred"         # Postponed for later


class EscalationPriority(Enum):
    """Priority level of escalation."""

    LOW = "low"           # Can wait, not blocking
    MEDIUM = "medium"     # Should address soon
    HIGH = "high"         # Blocking progress
    CRITICAL = "critical" # Immediate attention required


@dataclass
class Escalation:
    """Represents an escalation in the hierarchy."""

    id: str                            # Unique identifier
    from_level: EscalationLevel        # Source level
    to_level: EscalationLevel          # Target level
    reason: EscalationReason           # Why escalating
    priority: EscalationPriority       # Urgency level

    title: str                         # Brief summary
    description: str                   # Detailed description
    context: dict[str, Any]            # Additional context

    created_at: datetime               # When created
    created_by: str                    # Agent/entity name

    status: EscalationStatus = EscalationStatus.PENDING
    resolution: str | None = None      # Resolution when resolved
    resolved_at: datetime | None = None
    resolved_by: str | None = None

    # Related work tracking
    blocked_tasks: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    swarm_name: str | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "from_level": self.from_level.value,
            "to_level": self.to_level.value,
            "reason": self.reason.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "blocked_tasks": self.blocked_tasks,
            "related_files": self.related_files,
            "swarm_name": self.swarm_name,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Escalation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            from_level=EscalationLevel(data["from_level"]),
            to_level=EscalationLevel(data["to_level"]),
            reason=EscalationReason(data["reason"]),
            priority=EscalationPriority(data["priority"]),
            title=data["title"],
            description=data["description"],
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            status=EscalationStatus(data.get("status", "pending")),
            resolution=data.get("resolution"),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolved_by=data.get("resolved_by"),
            blocked_tasks=data.get("blocked_tasks", []),
            related_files=data.get("related_files", []),
            swarm_name=data.get("swarm_name"),
            job_id=data.get("job_id"),
        )

    def to_markdown(self) -> str:
        """Format as markdown for STATE.md logging."""
        priority_emoji = {
            EscalationPriority.LOW: "",
            EscalationPriority.MEDIUM: "[!]",
            EscalationPriority.HIGH: "[!!]",
            EscalationPriority.CRITICAL: "[!!!]",
        }

        status_marker = {
            EscalationStatus.PENDING: "PENDING",
            EscalationStatus.IN_PROGRESS: "IN PROGRESS",
            EscalationStatus.RESOLVED: "RESOLVED",
            EscalationStatus.CANCELLED: "CANCELLED",
            EscalationStatus.DEFERRED: "DEFERRED",
        }

        lines = [
            f"#### {priority_emoji.get(self.priority, '')} {self.title}",
            f"- **ID**: {self.id}",
            f"- **Status**: {status_marker.get(self.status, self.status.value)}",
            f"- **Reason**: {self.reason.value}",
            f"- **From**: {self.from_level.value} -> **To**: {self.to_level.value}",
            f"- **Priority**: {self.priority.value.upper()}",
            f"- **Created**: {self.created_at.strftime('%Y-%m-%d %H:%M')} by {self.created_by}",
            "",
            f"**Description**: {self.description}",
        ]

        if self.blocked_tasks:
            lines.append("")
            lines.append("**Blocked Tasks**:")
            for task in self.blocked_tasks:
                lines.append(f"  - {task}")

        if self.resolution:
            lines.append("")
            lines.append(f"**Resolution**: {self.resolution}")
            if self.resolved_at and self.resolved_by:
                lines.append(f"*Resolved {self.resolved_at.strftime('%Y-%m-%d %H:%M')} by {self.resolved_by}*")

        return "\n".join(lines)


class EscalationManager:
    """Manages escalations and their lifecycle."""

    def __init__(self, logs_dir: Path | None = None):
        """Initialize the escalation manager.

        Args:
            logs_dir: Directory for escalation logs
        """
        self.logs_dir = logs_dir or Path("./logs/escalations")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._escalations: dict[str, Escalation] = {}
        self._id_counter = 0
        self._lock = threading.RLock()  # Thread-safe lock for mutations

    def _generate_id(self) -> str:
        """Generate a unique escalation ID."""
        self._id_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ESC-{timestamp}-{self._id_counter:04d}"

    def create_escalation(
        self,
        from_level: EscalationLevel,
        to_level: EscalationLevel,
        reason: EscalationReason,
        title: str,
        description: str,
        created_by: str,
        priority: EscalationPriority = EscalationPriority.MEDIUM,
        context: dict[str, Any] | None = None,
        blocked_tasks: list[str] | None = None,
        related_files: list[str] | None = None,
        swarm_name: str | None = None,
        job_id: str | None = None,
    ) -> Escalation:
        """Create a new escalation.

        Args:
            from_level: Source hierarchy level
            to_level: Target hierarchy level
            reason: Reason for escalation
            title: Brief summary
            description: Detailed description
            created_by: Agent/entity creating escalation
            priority: Urgency level
            context: Additional context dict
            blocked_tasks: List of tasks blocked by this
            related_files: List of related file paths
            swarm_name: Swarm if applicable
            job_id: Job ID if applicable

        Returns:
            Created Escalation instance

        Raises:
            ValueError: If escalation path is invalid
        """
        # Validate escalation hierarchy path
        valid_paths = {
            (EscalationLevel.AGENT, EscalationLevel.COO),
            (EscalationLevel.COO, EscalationLevel.CEO),
        }
        if (from_level, to_level) not in valid_paths:
            raise ValueError(
                f"Invalid escalation path: {from_level.value} -> {to_level.value}. "
                f"Valid paths: AGENT->COO, COO->CEO"
            )

        with self._lock:
            escalation = Escalation(
                id=self._generate_id(),
                from_level=from_level,
                to_level=to_level,
                reason=reason,
                priority=priority,
                title=title,
                description=description,
                context=context or {},
                created_at=datetime.now(),
                created_by=created_by,
                blocked_tasks=blocked_tasks or [],
                related_files=related_files or [],
                swarm_name=swarm_name,
                job_id=job_id,
            )

            self._escalations[escalation.id] = escalation
            self._save_escalation(escalation)

        logger.info(
            f"Created escalation {escalation.id}: {title} "
            f"({from_level.value} -> {to_level.value})"
        )

        return escalation

    def resolve_escalation(
        self,
        escalation_id: str,
        resolution: str,
        resolved_by: str,
    ) -> Escalation | None:
        """Mark an escalation as resolved.

        Args:
            escalation_id: ID of escalation to resolve
            resolution: Resolution description
            resolved_by: Who resolved it

        Returns:
            Updated Escalation or None if not found
        """
        with self._lock:
            if escalation_id not in self._escalations:
                logger.warning(f"Escalation not found: {escalation_id}")
                return None

            escalation = self._escalations[escalation_id]
            escalation.status = EscalationStatus.RESOLVED
            escalation.resolution = resolution
            escalation.resolved_at = datetime.now()
            escalation.resolved_by = resolved_by

            self._save_escalation(escalation)

        logger.info(f"Resolved escalation {escalation_id}: {resolution[:50]}...")

        return escalation

    def update_status(
        self,
        escalation_id: str,
        status: EscalationStatus,
    ) -> Escalation | None:
        """Update escalation status.

        Args:
            escalation_id: ID of escalation
            status: New status

        Returns:
            Updated Escalation or None if not found
        """
        with self._lock:
            if escalation_id not in self._escalations:
                return None

            escalation = self._escalations[escalation_id]
            escalation.status = status
            self._save_escalation(escalation)

            return escalation

    def get_pending(
        self,
        level: EscalationLevel | None = None,
    ) -> list[Escalation]:
        """Get pending escalations.

        Args:
            level: Filter by target level (optional)

        Returns:
            List of pending escalations
        """
        with self._lock:
            pending = [
                e for e in self._escalations.values()
                if e.status == EscalationStatus.PENDING
            ]

        if level:
            pending = [e for e in pending if e.to_level == level]

        # Sort by priority (critical first) then by creation time
        priority_order = {
            EscalationPriority.CRITICAL: 0,
            EscalationPriority.HIGH: 1,
            EscalationPriority.MEDIUM: 2,
            EscalationPriority.LOW: 3,
        }
        pending.sort(key=lambda e: (priority_order[e.priority], e.created_at))

        return pending

    def get_by_swarm(self, swarm_name: str) -> list[Escalation]:
        """Get all escalations for a swarm."""
        with self._lock:
            return [
                e for e in self._escalations.values()
                if e.swarm_name == swarm_name
            ]

    def get_blocked_work(self) -> list[Escalation]:
        """Get escalations that are blocking work."""
        with self._lock:
            return [
                e for e in self._escalations.values()
                if e.status == EscalationStatus.PENDING and e.blocked_tasks
            ]

    def _save_escalation(self, escalation: Escalation) -> None:
        """Save escalation to log file using atomic write."""
        filename = f"{escalation.id}.json"
        filepath = self.logs_dir / filename
        temp_path = filepath.with_suffix('.json.tmp')

        try:
            with open(temp_path, "w") as f:
                json.dump(escalation.to_dict(), f, indent=2)
            temp_path.rename(filepath)  # Atomic on POSIX
        except Exception as e:
            logger.error(f"Error saving escalation {escalation.id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_from_disk(self) -> None:
        """Load all escalations from disk and restore ID counter."""
        max_counter = 0
        for filepath in self.logs_dir.glob("ESC-*.json"):
            try:
                # Extract counter from ID: ESC-TIMESTAMP-NNNN
                parts = filepath.stem.split('-')
                if len(parts) >= 3:
                    counter = int(parts[-1])
                    max_counter = max(max_counter, counter)
                with open(filepath) as f:
                    data = json.load(f)
                    escalation = Escalation.from_dict(data)
                    self._escalations[escalation.id] = escalation
            except Exception as e:
                logger.error(f"Error loading escalation {filepath}: {e}")
        self._id_counter = max_counter


# Convenience functions for agents

def escalate_to_coo(
    reason: EscalationReason,
    title: str,
    description: str,
    created_by: str,
    priority: EscalationPriority = EscalationPriority.MEDIUM,
    **kwargs,
) -> Escalation:
    """Convenience function for agent-to-COO escalation.

    Args:
        reason: Escalation reason
        title: Brief summary
        description: Detailed description
        created_by: Agent name
        priority: Urgency level
        **kwargs: Additional escalation parameters

    Returns:
        Created Escalation
    """
    manager = get_escalation_manager()
    return manager.create_escalation(
        from_level=EscalationLevel.AGENT,
        to_level=EscalationLevel.COO,
        reason=reason,
        title=title,
        description=description,
        created_by=created_by,
        priority=priority,
        **kwargs,
    )


def escalate_to_ceo(
    reason: EscalationReason,
    title: str,
    description: str,
    priority: EscalationPriority = EscalationPriority.HIGH,
    **kwargs,
) -> Escalation:
    """Convenience function for COO-to-CEO escalation.

    Args:
        reason: Escalation reason
        title: Brief summary
        description: Detailed description
        priority: Urgency level (defaults to HIGH for CEO)
        **kwargs: Additional escalation parameters

    Returns:
        Created Escalation
    """
    manager = get_escalation_manager()
    return manager.create_escalation(
        from_level=EscalationLevel.COO,
        to_level=EscalationLevel.CEO,
        reason=reason,
        title=title,
        description=description,
        created_by="COO",
        priority=priority,
        **kwargs,
    )


# Module-level singleton with thread-safe initialization
_escalation_manager: EscalationManager | None = None
_singleton_lock = threading.Lock()


def get_escalation_manager(logs_dir: Path | None = None) -> EscalationManager:
    """Get or create the global escalation manager.

    Thread-safe singleton pattern with double-checked locking.

    Args:
        logs_dir: Optional logs directory (used on first call)

    Returns:
        The escalation manager singleton
    """
    global _escalation_manager

    if _escalation_manager is None:
        with _singleton_lock:
            # Double-check after acquiring lock
            if _escalation_manager is None:
                _escalation_manager = EscalationManager(logs_dir)
                _escalation_manager.load_from_disk()

    return _escalation_manager
