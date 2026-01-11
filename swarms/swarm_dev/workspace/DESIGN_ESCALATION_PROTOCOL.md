# Escalation Protocol Design Document

## Agent-to-COO-to-CEO Escalation System

**Author**: Swarm Dev Architect
**Date**: 2026-01-02
**Status**: PROPOSED

---

## 1. Overview

This document defines the escalation protocol for the agent-swarm system. The system has a three-tier hierarchy:

```
+-------------+
|    CEO      |  <- Human User (Business Decisions, Approvals)
+------+------+
       |
       v
+------+------+
|    COO      |  <- Supreme Orchestrator (Coordination, Delegation)
+------+------+
       |
       v
+------+------+
| Swarm Agents|  <- Implementer, Critic, Architect, etc. (Execution)
+-------------+
```

Escalation flows **upward** when agents encounter situations beyond their scope. Each level has defined responsibilities and escalation triggers.

---

## 2. Hierarchy Responsibilities

### 2.1 CEO (Human User)

**Role**: Ultimate authority for business decisions and approvals.

**Responsibilities**:
- Final approval for production deployments
- Authorization of major architectural changes
- Budget and cost decisions
- Security vulnerability response
- Permission grants for sensitive operations
- Conflict resolution between priorities

**When Engaged**:
- COO explicitly escalates
- Security incidents detected
- Cost thresholds exceeded
- Permission requests pending

### 2.2 COO (Supreme Orchestrator)

**Role**: Coordinates swarm agents, makes operational decisions.

**Responsibilities**:
- Task delegation to appropriate agents
- Progress monitoring and reallocation
- Clarifying ambiguous requirements
- Resolving inter-agent conflicts
- Aggregating agent work for CEO review
- First-line triage of escalations

**Authority Limits**:
- Can approve routine changes
- Can request additional context from CEO
- Cannot approve major architectural changes alone
- Cannot bypass security reviews
- Cannot commit to production without approval

### 2.3 Swarm Agents

**Role**: Execute assigned tasks within their scope.

**Responsibilities**:
- Complete assigned work
- Report progress via STATE.md
- Flag blockers immediately
- Request clarification when uncertain
- Raise concerns about requirements
- Follow established patterns

**Authority Limits**:
- Work only within assigned workspace
- Follow permission model per agent type
- Cannot make cross-swarm decisions
- Cannot approve their own code (require reviewer)
- Cannot access credentials directly

---

## 3. Escalation Levels

### 3.1 Agent-to-COO Escalation

Agents should escalate to COO when encountering:

| Reason | Description | Example |
|--------|-------------|---------|
| `BLOCKED` | External dependency preventing progress | Waiting for API key, missing credentials |
| `CLARIFICATION` | Ambiguous requirements | "Should I use REST or GraphQL?" |
| `CONFLICT` | Conflicting instructions | STATE.md says X but request says Y |
| `SECURITY` | Security concern identified | Hardcoded secrets found |
| `ARCHITECTURE` | Decision beyond agent scope | Major pattern change needed |
| `SCOPE_EXCEEDED` | Task too large or complex | Would take >5 days |

### 3.2 COO-to-CEO Escalation

COO should escalate to CEO when encountering:

| Reason | Description | Example |
|--------|-------------|---------|
| `ARCHITECTURE_MAJOR` | Changes affecting entire system | New database, framework change |
| `SECURITY_CRITICAL` | Vulnerability needing immediate attention | Exposed secrets, injection flaw |
| `PRIORITY_CONFLICT` | Business priorities need arbitration | Feature A vs Feature B |
| `COST` | API/infrastructure cost implications | High token usage, new service |
| `PERMISSION` | Operation requiring human approval | Git push, deploy, access grant |
| `BLOCKED_CRITICAL` | Agent blocked and no workaround | Critical path blocked |

---

## 4. Decision Trees

### 4.1 Agent Escalation Decision Tree

```
Agent encounters issue
        |
        v
   Can I resolve this with my current tools/permissions?
        |
   +----+----+
   |         |
  YES        NO
   |         |
   v         v
Proceed    Is this a security concern?
           |
      +----+----+
      |         |
     YES        NO
      |         |
      v         v
   Escalate   Is it a blocking dependency?
   (SECURITY) |
              +----+----+
              |         |
             YES        NO
              |         |
              v         v
           Escalate    Is requirement ambiguous?
           (BLOCKED)   |
                      +----+----+
                      |         |
                     YES        NO
                      |         |
                      v         v
                   Escalate    Do I have conflicting instructions?
                   (CLARIFY)   |
                              +----+----+
                              |         |
                             YES        NO
                              |         |
                              v         v
                           Escalate    Is decision beyond my scope?
                           (CONFLICT)  |
                                      +----+----+
                                      |         |
                                     YES        NO
                                      |         |
                                      v         v
                                   Escalate    Continue with best effort
                                   (ARCH)      and document assumptions
```

### 4.2 COO Escalation Decision Tree

```
COO receives issue from agent or encounters situation
        |
        v
   Is this a security vulnerability (HIGH/CRITICAL)?
        |
   +----+----+
   |         |
  YES        NO
   |         |
   v         v
Escalate   Does this require system-wide architectural change?
(SECURITY) |
           +----+----+
           |         |
          YES        NO
           |         |
           v         v
        Escalate   Are there conflicting business priorities?
        (ARCH)     |
                   +----+----+
                   |         |
                  YES        NO
                   |         |
                   v         v
                Escalate    Does this have cost implications?
                (PRIORITY)  |
                           +----+----+
                           |         |
                          YES        NO
                           |         |
                           v         v
                        Escalate    Does this require permission grant?
                        (COST)      |
                                   +----+----+
                                   |         |
                                  YES        NO
                                   |         |
                                   v         v
                                Escalate    Handle at COO level
                                (PERMISSION)
```

---

## 5. Implementation Specification

### 5.1 Data Structures

**File**: `/Users/jellingson/agent-swarm/shared/escalation_protocol.py`

```python
"""Escalation protocol for agent-swarm hierarchy.

This module defines the escalation levels, reasons, and data structures
for managing escalations between agents, COO, and CEO.
"""

from __future__ import annotations

import json
import logging
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
        """
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
        return [
            e for e in self._escalations.values()
            if e.swarm_name == swarm_name
        ]

    def get_blocked_work(self) -> list[Escalation]:
        """Get escalations that are blocking work."""
        return [
            e for e in self._escalations.values()
            if e.status == EscalationStatus.PENDING and e.blocked_tasks
        ]

    def _save_escalation(self, escalation: Escalation) -> None:
        """Save escalation to log file."""
        filename = f"{escalation.id}.json"
        filepath = self.logs_dir / filename

        with open(filepath, "w") as f:
            json.dump(escalation.to_dict(), f, indent=2)

    def load_from_disk(self) -> None:
        """Load all escalations from disk."""
        for filepath in self.logs_dir.glob("ESC-*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    escalation = Escalation.from_dict(data)
                    self._escalations[escalation.id] = escalation
            except Exception as e:
                logger.error(f"Error loading escalation {filepath}: {e}")


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


# Module-level singleton
_escalation_manager: EscalationManager | None = None


def get_escalation_manager(logs_dir: Path | None = None) -> EscalationManager:
    """Get or create the global escalation manager.

    Args:
        logs_dir: Optional logs directory (used on first call)

    Returns:
        The escalation manager singleton
    """
    global _escalation_manager

    if _escalation_manager is None:
        _escalation_manager = EscalationManager(logs_dir)
        _escalation_manager.load_from_disk()

    return _escalation_manager
```

---

## 6. STATE.md Escalation Logging Format

### 6.1 Escalations Section

Add a dedicated `## Escalations` section to STATE.md with the following format:

```markdown
## Escalations

### Pending

#### [!!] API Key Required for OpenAI Integration
- **ID**: ESC-20260102143022-0001
- **Status**: PENDING
- **Reason**: blocked
- **From**: agent -> **To**: coo
- **Priority**: HIGH
- **Created**: 2026-01-02 14:30 by implementer

**Description**: Cannot proceed with embedding generation - OpenAI API key not configured.

**Blocked Tasks**:
  - Phase 1.3: Vector embedding pipeline
  - Phase 1.4: Semantic search implementation

---

### Resolved

#### Ambiguous Database Schema Requirements
- **ID**: ESC-20260102100512-0001
- **Status**: RESOLVED
- **Reason**: clarification
- **From**: agent -> **To**: coo
- **Priority**: MEDIUM
- **Created**: 2026-01-02 10:05 by architect

**Description**: Unclear whether user_sessions should be in PostgreSQL or Redis.

**Resolution**: Use Redis for active sessions, PostgreSQL for session history.
*Resolved 2026-01-02 10:30 by COO*
```

### 6.2 Quick Reference Table

At the top of the Escalations section, maintain a quick reference:

```markdown
## Escalations

| ID | Status | Priority | Title | Assigned To |
|----|--------|----------|-------|-------------|
| ESC-001 | PENDING | HIGH | API Key Required | COO |
| ESC-002 | IN PROGRESS | CRITICAL | Security Vuln | CEO |
| ESC-003 | RESOLVED | MEDIUM | Schema Clarification | - |
```

---

## 7. Blocked Work Protocol

### 7.1 Marking Work as Blocked

When an agent encounters a blocker:

1. **Log the escalation** using `escalate_to_coo()` or similar
2. **Update STATE.md** with the escalation in the Escalations section
3. **Update the blocked task** in the task list with a `[BLOCKED: ESC-ID]` marker
4. **Continue with unblocked work** - do not stop completely

Example task list update:

```markdown
## Next Steps

1. [ ] **Implementer**: Create vector embedding pipeline [BLOCKED: ESC-001]
   - Waiting on: OpenAI API key
   - ETA after unblock: 2 days

2. [x] **Implementer**: Set up database schema - COMPLETE

3. [ ] **Implementer**: Create REST API endpoints - IN PROGRESS
   - Not blocked, continuing with this
```

### 7.2 Continuing with Other Tasks

Agents should:

1. Check for unblocked tasks in their queue
2. Notify COO of the switch: "Blocked on ESC-001, switching to API endpoints"
3. Track time spent on workaround vs. waiting
4. Re-prioritize when blocker resolves

### 7.3 Unblocking Protocol

When a blocker is resolved:

1. **Resolver** calls `resolve_escalation(id, resolution, resolved_by)`
2. **COO** notifies affected agents via wake message
3. **Agents** resume blocked work, remove `[BLOCKED]` marker
4. **Update STATE.md** - move escalation to Resolved section

Wake message format:
```
ESCALATION RESOLVED: ESC-001 - API Key Provided
Original blocker: OpenAI API key not configured
Resolution: Key added to environment, restart backend to apply
Affected tasks: Phase 1.3, Phase 1.4
Action: Resume work on vector embedding pipeline
```

---

## 8. Examples

### 8.1 Agent-to-COO: Blocked on External Dependency

**Scenario**: Implementer needs AWS credentials to set up S3 integration.

```python
from shared.escalation_protocol import (
    escalate_to_coo,
    EscalationReason,
    EscalationPriority,
)

escalation = escalate_to_coo(
    reason=EscalationReason.BLOCKED,
    title="AWS Credentials Required for S3 Integration",
    description="""
    Phase 2.1 requires S3 bucket access for file uploads.
    Need: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    configured in environment or .env file.

    Without this, cannot proceed with:
    - File upload endpoint
    - Asset storage system
    - CDN integration
    """,
    created_by="implementer",
    priority=EscalationPriority.HIGH,
    blocked_tasks=[
        "Phase 2.1: File upload endpoint",
        "Phase 2.2: Asset storage system",
        "Phase 2.3: CDN integration",
    ],
    swarm_name="swarm_dev",
)

# Agent continues with unblocked work
print(f"Escalation created: {escalation.id}")
print("Continuing with unblocked tasks...")
```

**STATE.md Entry**:
```markdown
#### [!!] AWS Credentials Required for S3 Integration
- **ID**: ESC-20260102150030-0001
- **Status**: PENDING
- **Reason**: blocked
- **From**: agent -> **To**: coo
- **Priority**: HIGH
- **Created**: 2026-01-02 15:00 by implementer

**Description**: Phase 2.1 requires S3 bucket access for file uploads.
Need: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.

**Blocked Tasks**:
  - Phase 2.1: File upload endpoint
  - Phase 2.2: Asset storage system
  - Phase 2.3: CDN integration
```

### 8.2 Agent-to-COO: Clarification Needed

**Scenario**: Architect unsure about authentication approach.

```python
escalation = escalate_to_coo(
    reason=EscalationReason.CLARIFICATION,
    title="Authentication Approach Unclear",
    description="""
    Requirements mention "secure authentication" but don't specify:
    1. Should we use JWT tokens or session cookies?
    2. Do we need OAuth2/social login support?
    3. What's the session timeout policy?

    Need guidance before designing auth system.
    """,
    created_by="architect",
    priority=EscalationPriority.MEDIUM,
    context={
        "options_considered": ["JWT", "Session cookies", "OAuth2"],
        "recommendation": "JWT for API, optional OAuth2 for social",
    },
    related_files=[
        "backend/auth/",
        "docs/requirements.md",
    ],
)
```

### 8.3 Agent-to-COO: Security Concern

**Scenario**: Critic finds hardcoded secret during review.

```python
escalation = escalate_to_coo(
    reason=EscalationReason.SECURITY,
    title="Hardcoded API Key Found in Source",
    description="""
    During code review, found hardcoded OpenAI API key in:
    /backend/services/embedding.py line 42

    This is a security risk - key will be exposed in git history.

    Recommended action:
    1. Rotate the API key immediately
    2. Move to environment variable
    3. Add secret scanning to CI
    """,
    created_by="critic",
    priority=EscalationPriority.CRITICAL,
    related_files=[
        "/backend/services/embedding.py",
    ],
    context={
        "line_number": 42,
        "secret_type": "API Key",
        "exposed_in_git": True,
    },
)
```

### 8.4 COO-to-CEO: Major Architecture Change

**Scenario**: COO needs approval for database migration.

```python
from shared.escalation_protocol import (
    escalate_to_ceo,
    EscalationReason,
    EscalationPriority,
)

escalation = escalate_to_ceo(
    reason=EscalationReason.ARCHITECTURE_MAJOR,
    title="Database Migration: SQLite to PostgreSQL",
    description="""
    Current SQLite database is hitting performance limits:
    - Query latency: 500ms+ on aggregate queries
    - Concurrent connections: limited to 1 writer
    - Data volume: approaching 1GB limit for efficient ops

    Proposal: Migrate to PostgreSQL
    - Estimated effort: 5 developer days
    - Downtime required: 2-4 hours for migration
    - Cost impact: ~$50/month for managed Postgres

    Benefits:
    - 10x query performance improvement
    - Full concurrent access
    - Better tooling and ecosystem

    Request: Approval to proceed with migration planning.
    """,
    priority=EscalationPriority.HIGH,
    context={
        "current_db": "SQLite",
        "proposed_db": "PostgreSQL",
        "estimated_effort": "5 days",
        "estimated_cost": "$50/month",
        "downtime": "2-4 hours",
    },
    blocked_tasks=[
        "Performance optimization initiative",
        "Multi-user support feature",
    ],
)
```

### 8.5 COO-to-CEO: Permission Request

**Scenario**: COO needs approval to push to production branch.

```python
escalation = escalate_to_ceo(
    reason=EscalationReason.PERMISSION,
    title="Deploy v1.2.0 to Production",
    description="""
    Ready to deploy version 1.2.0 to production.

    Changes included:
    - New user dashboard (Phase 3)
    - Performance improvements (15% faster)
    - Bug fixes from v1.1.x

    Verification completed:
    - All tests passing (98% coverage)
    - Security review: APPROVED
    - Performance testing: within SLA
    - Staging validation: 48 hours stable

    Request: Permission to execute production deployment.
    """,
    priority=EscalationPriority.MEDIUM,
    context={
        "version": "1.2.0",
        "test_coverage": "98%",
        "staging_time": "48 hours",
        "rollback_plan": "Automated rollback on error rate >1%",
    },
)
```

---

## 9. Integration Points

### 9.1 WebSocket Events

Add new WebSocket event types for escalation notifications:

```python
# New event types for escalation notifications

{
    "type": "escalation_created",
    "escalation_id": "ESC-20260102150030-0001",
    "from_level": "agent",
    "to_level": "coo",
    "priority": "high",
    "title": "AWS Credentials Required",
    "created_by": "implementer",
}

{
    "type": "escalation_resolved",
    "escalation_id": "ESC-20260102150030-0001",
    "resolution": "Credentials added to environment",
    "resolved_by": "CEO",
}

{
    "type": "escalation_pending_ceo",
    "count": 2,
    "critical_count": 1,
    "escalations": [
        {"id": "ESC-001", "title": "Deploy approval", "priority": "medium"},
        {"id": "ESC-002", "title": "Security vuln", "priority": "critical"},
    ],
}
```

### 9.2 Agent Prompt Integration

Add escalation awareness to agent system prompts:

```markdown
## Escalation Protocol

When you encounter situations beyond your scope:

1. **Blocked**: Cannot proceed due to missing credentials, access, or dependencies
   -> Escalate to COO with reason=BLOCKED

2. **Unclear Requirements**: Ambiguous or conflicting instructions
   -> Escalate to COO with reason=CLARIFICATION

3. **Security Concerns**: Found vulnerabilities, exposed secrets, or risky patterns
   -> Escalate to COO with reason=SECURITY (use CRITICAL priority)

4. **Architecture Decisions**: Changes affecting system design beyond your task
   -> Escalate to COO with reason=ARCHITECTURE

Always:
- Continue with unblocked work after escalating
- Update STATE.md with the escalation
- Mark blocked tasks with [BLOCKED: ESC-ID]
- Do not wait idle - find other work to progress
```

---

## 10. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `shared/escalation_protocol.py` | Escalation data structures and manager |

### Modified Files

| File | Changes |
|------|---------|
| `backend/main.py` | Add escalation WebSocket events |
| `swarms/*/agents/*.md` | Add escalation protocol to prompts |
| `swarms/*/workspace/STATE.md` | Add Escalations section |

---

## 11. Success Criteria

This design is complete when:

- [ ] `shared/escalation_protocol.py` is implemented
- [ ] Agents can create escalations via convenience functions
- [ ] COO receives and can respond to agent escalations
- [ ] CEO receives escalations from COO
- [ ] Escalations are logged to `logs/escalations/`
- [ ] STATE.md shows escalation status
- [ ] Blocked work protocol is followed
- [ ] WebSocket events notify UI of escalations

---

## 12. Open Questions

1. **Escalation Timeout**: Should pending escalations auto-escalate after N hours?
   - Recommendation: Yes, CRITICAL escalations should auto-escalate to CEO after 1 hour

2. **Batch Escalations**: Should related blockers be grouped?
   - Recommendation: Allow linking escalations via `related_escalations` field

3. **Escalation History**: How long to retain resolved escalations?
   - Recommendation: Keep for 30 days, then archive

---

**Document Version**: 1.0
**Last Updated**: 2026-01-02
**Next Review**: After implementation complete
