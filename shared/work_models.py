"""Data models for the Work Ledger system.

This module defines the data structures for persistent work items
that survive agent crashes and restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class WorkStatus(Enum):
    """Status of a work item."""

    PENDING = "pending"           # Created but not started
    IN_PROGRESS = "in_progress"   # Currently being worked on
    BLOCKED = "blocked"           # Waiting on dependencies or external input
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"             # Failed with error
    CANCELLED = "cancelled"       # Manually cancelled


class WorkType(Enum):
    """Type classification for work items."""

    TASK = "task"                 # General task
    FEATURE = "feature"           # Feature implementation
    BUG = "bug"                   # Bug fix
    RESEARCH = "research"         # Research/exploration
    REVIEW = "review"             # Code review
    DESIGN = "design"             # Architecture/design work
    REFACTOR = "refactor"         # Refactoring work
    TEST = "test"                 # Testing work
    DOCUMENTATION = "documentation"  # Documentation
    ESCALATION = "escalation"     # Escalated issue


class WorkPriority(Enum):
    """Priority level for work items."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkHistoryEntry:
    """A single entry in work item history."""

    timestamp: datetime
    action: str                   # "created", "started", "blocked", "completed", etc.
    actor: str                    # Who performed the action
    details: dict[str, Any]       # Additional details
    previous_status: WorkStatus | None
    new_status: WorkStatus | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor": self.actor,
            "details": self.details,
            "previous_status": self.previous_status.value if self.previous_status else None,
            "new_status": self.new_status.value if self.new_status else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkHistoryEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            actor=data["actor"],
            details=data.get("details", {}),
            previous_status=WorkStatus(data["previous_status"]) if data.get("previous_status") else None,
            new_status=WorkStatus(data["new_status"]) if data.get("new_status") else None,
        )


@dataclass
class WorkItem:
    """Represents a unit of work in the ledger."""

    # Identity
    id: str                       # Unique identifier (WRK-YYYYMMDD-NNNN)
    title: str                    # Brief description
    type: WorkType                # Classification
    priority: WorkPriority        # Urgency level

    # State
    status: WorkStatus            # Current status
    owner: str | None             # Agent currently responsible (None = unassigned)

    # Content
    description: str              # Detailed description
    context: dict[str, Any]       # Additional context (files, references, etc.)

    # Hierarchy
    parent_id: str | None         # Parent work item (for subtasks)
    dependencies: list[str]       # IDs of work items this depends on

    # Lifecycle
    created_at: datetime          # When created
    created_by: str               # Who created it (agent name or "CEO")
    updated_at: datetime          # Last update time
    started_at: datetime | None   # When work began
    completed_at: datetime | None # When work finished

    # Results
    result: dict[str, Any] | None # Result data when completed
    error: str | None             # Error message if failed

    # Tracking
    swarm_name: str | None        # Associated swarm
    job_id: str | None            # Associated job (from job system)
    execution_id: str | None      # Current execution ID (from executor pool)

    # History
    history: list[WorkHistoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "owner": self.owner,
            "description": self.description,
            "context": self.context,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "swarm_name": self.swarm_name,
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "history": [h.to_dict() for h in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            type=WorkType(data["type"]),
            priority=WorkPriority(data["priority"]),
            status=WorkStatus(data["status"]),
            owner=data.get("owner"),
            description=data["description"],
            context=data.get("context", {}),
            parent_id=data.get("parent_id"),
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
            swarm_name=data.get("swarm_name"),
            job_id=data.get("job_id"),
            execution_id=data.get("execution_id"),
            history=[WorkHistoryEntry.from_dict(h) for h in data.get("history", [])],
        )


@dataclass
class WorkIndex:
    """Index/manifest of all work items for fast lookup."""

    version: str
    last_updated: datetime
    total_count: int

    # Quick lookups (ID -> filepath)
    items: dict[str, str]         # work_id -> relative file path

    # Status indexes
    by_status: dict[str, list[str]]   # status -> list of work_ids
    by_owner: dict[str, list[str]]    # owner -> list of work_ids
    by_swarm: dict[str, list[str]]    # swarm -> list of work_ids
    by_parent: dict[str, list[str]]   # parent_id -> list of child_ids

    # Counter for ID generation
    id_counter: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "total_count": self.total_count,
            "items": self.items,
            "by_status": self.by_status,
            "by_owner": self.by_owner,
            "by_swarm": self.by_swarm,
            "by_parent": self.by_parent,
            "id_counter": self.id_counter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkIndex":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            total_count=data.get("total_count", 0),
            items=data.get("items", {}),
            by_status=data.get("by_status", {}),
            by_owner=data.get("by_owner", {}),
            by_swarm=data.get("by_swarm", {}),
            by_parent=data.get("by_parent", {}),
            id_counter=data.get("id_counter", 0),
        )

    @classmethod
    def create_empty(cls) -> "WorkIndex":
        """Create an empty index."""
        return cls(
            version="1.0",
            last_updated=datetime.now(),
            total_count=0,
            items={},
            by_status={status.value: [] for status in WorkStatus},
            by_owner={},
            by_swarm={},
            by_parent={},
            id_counter=0,
        )
