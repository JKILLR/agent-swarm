# Work Ledger System Design

**Date:** 2026-01-03
**Author:** System Architect
**Status:** PROPOSED

---

## Context

### The Problem

The agent-swarm system currently suffers from work state fragility:

1. **Work state in agent memory**: When agents crash or restart, all in-progress work context is lost
2. **No persistence layer**: Work items exist only in the agent's conversation context
3. **No structured tracking**: No standardized way to track work units across agent lifecycles
4. **No recovery mechanism**: After a crash, there is no way to resume work from where it stopped
5. **No visibility**: External systems cannot query what work is in progress

### Inspiration: Gas Town's "Beads"

Gas Town uses a "Beads" system where:
- Work persists on hooks - survives crashes
- Git-backed ledger of work items
- Each work item has: ID, status, owner, type, dependencies
- Work can be "picked up" by any agent that can see the hook

### Goal

Implement a **Work Ledger** system that provides:
- Persistent work units that survive agent crashes
- Hierarchical task tracking (parent/child relationships)
- Thread-safe access for concurrent agents
- Integration with the existing `agent_executor_pool.py`
- Git-compatible file storage for history/audit

---

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|   Agent/CEO      | --> | Work Ledger API  | --> |  Ledger Storage  |
|   (creates work) |     | (work_ledger.py) |     | (workspace/ledger)|
+------------------+     +------------------+     +------------------+
                                |
                         +------+------+
                         |             |
                         v             v
                  +------------+  +------------+
                  | WorkItem   |  | WorkIndex  |
                  | (per-item) |  | (manifest) |
                  +------------+  +------------+
```

---

## Data Structures

### WorkStatus Enum

```python
class WorkStatus(Enum):
    """Status of a work item."""

    PENDING = "pending"           # Created but not started
    IN_PROGRESS = "in_progress"   # Currently being worked on
    BLOCKED = "blocked"           # Waiting on dependencies or external input
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"             # Failed with error
    CANCELLED = "cancelled"       # Manually cancelled
```

### WorkType Enum

```python
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
```

### WorkPriority Enum

```python
class WorkPriority(Enum):
    """Priority level for work items."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### WorkItem Dataclass

```python
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
```

### WorkHistoryEntry Dataclass

```python
@dataclass
class WorkHistoryEntry:
    """A single entry in work item history."""

    timestamp: datetime
    action: str                   # "created", "started", "blocked", "completed", etc.
    actor: str                    # Who performed the action
    details: dict[str, Any]       # Additional details
    previous_status: WorkStatus | None
    new_status: WorkStatus | None
```

### WorkIndex (Manifest)

```python
@dataclass
class WorkIndex:
    """Index/manifest of all work items for fast lookup."""

    version: str = "1.0"
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
```

---

## API Methods

### WorkLedger Class

```python
class WorkLedger:
    """Manages persistent work items across agent lifecycles."""

    def __init__(
        self,
        ledger_dir: Path | None = None,
        auto_save: bool = True,
    ) -> None:
        """Initialize the work ledger.

        Args:
            ledger_dir: Directory for ledger storage (default: workspace/ledger/)
            auto_save: Automatically persist changes to disk
        """
```

#### Creating Work

```python
def create_work(
    self,
    title: str,
    description: str,
    work_type: WorkType = WorkType.TASK,
    priority: WorkPriority = WorkPriority.MEDIUM,
    created_by: str = "CEO",
    parent_id: str | None = None,
    dependencies: list[str] | None = None,
    context: dict[str, Any] | None = None,
    swarm_name: str | None = None,
    job_id: str | None = None,
) -> WorkItem:
    """Create a new work item.

    Args:
        title: Brief description of the work
        description: Detailed description
        work_type: Type classification
        priority: Urgency level
        created_by: Creator identifier
        parent_id: Parent work item ID (for subtasks)
        dependencies: List of work item IDs this depends on
        context: Additional context dictionary
        swarm_name: Associated swarm name
        job_id: Associated job ID

    Returns:
        Created WorkItem

    Raises:
        ValueError: If parent_id or dependencies reference non-existent items
    """
```

#### Claiming Work

```python
def claim_work(
    self,
    work_id: str,
    owner: str,
    execution_id: str | None = None,
) -> WorkItem | None:
    """Claim a work item for processing.

    Atomically sets the owner and updates status to IN_PROGRESS.
    Fails if work is already owned by another agent.

    Args:
        work_id: ID of work item to claim
        owner: Agent name claiming the work
        execution_id: Optional execution ID for tracking

    Returns:
        Updated WorkItem or None if claim failed
    """
```

```python
def release_work(
    self,
    work_id: str,
    owner: str,
    reason: str = "released",
) -> WorkItem | None:
    """Release a claimed work item back to pending.

    Args:
        work_id: ID of work item to release
        owner: Current owner (must match)
        reason: Reason for release

    Returns:
        Updated WorkItem or None if release failed
    """
```

#### Updating Work Status

```python
def start_work(
    self,
    work_id: str,
    owner: str,
) -> WorkItem | None:
    """Mark work as started (IN_PROGRESS).

    Args:
        work_id: Work item ID
        owner: Agent starting the work

    Returns:
        Updated WorkItem or None
    """
```

```python
def block_work(
    self,
    work_id: str,
    reason: str,
    blocker_id: str | None = None,
) -> WorkItem | None:
    """Mark work as blocked.

    Args:
        work_id: Work item ID
        reason: Why it's blocked
        blocker_id: ID of blocking work item (if applicable)

    Returns:
        Updated WorkItem or None
    """
```

```python
def complete_work(
    self,
    work_id: str,
    owner: str,
    result: dict[str, Any] | None = None,
) -> WorkItem | None:
    """Mark work as completed.

    Args:
        work_id: Work item ID
        owner: Agent completing the work
        result: Result data (files created, metrics, etc.)

    Returns:
        Updated WorkItem or None
    """
```

```python
def fail_work(
    self,
    work_id: str,
    owner: str,
    error: str,
) -> WorkItem | None:
    """Mark work as failed.

    Args:
        work_id: Work item ID
        owner: Agent that encountered the failure
        error: Error message/description

    Returns:
        Updated WorkItem or None
    """
```

#### Querying Work

```python
def get_work(self, work_id: str) -> WorkItem | None:
    """Get a single work item by ID."""
```

```python
def get_pending(
    self,
    owner: str | None = None,
    swarm_name: str | None = None,
    work_type: WorkType | None = None,
) -> list[WorkItem]:
    """Get pending work items.

    Args:
        owner: Filter by owner (None = unassigned)
        swarm_name: Filter by swarm
        work_type: Filter by type

    Returns:
        List of pending work items, sorted by priority then created_at
    """
```

```python
def get_in_progress(
    self,
    owner: str | None = None,
) -> list[WorkItem]:
    """Get work items currently in progress."""
```

```python
def get_blocked(self) -> list[WorkItem]:
    """Get all blocked work items."""
```

```python
def get_children(self, parent_id: str) -> list[WorkItem]:
    """Get all child work items of a parent."""
```

```python
def get_by_swarm(self, swarm_name: str) -> list[WorkItem]:
    """Get all work items for a swarm."""
```

```python
def get_ready_to_start(
    self,
    swarm_name: str | None = None,
) -> list[WorkItem]:
    """Get work items that are ready to start.

    A work item is ready if:
    - Status is PENDING
    - Owner is None (unclaimed)
    - All dependencies are COMPLETED
    """
```

#### Hierarchy Operations

```python
def create_subtask(
    self,
    parent_id: str,
    title: str,
    description: str,
    created_by: str,
    **kwargs,
) -> WorkItem:
    """Create a subtask under a parent work item.

    Automatically inherits swarm_name and some context from parent.
    """
```

```python
def get_progress(self, work_id: str) -> dict[str, Any]:
    """Get progress summary for a work item including children.

    Returns:
        {
            "total_children": int,
            "completed": int,
            "in_progress": int,
            "pending": int,
            "blocked": int,
            "failed": int,
            "percent_complete": float,
        }
    """
```

#### Recovery Operations

```python
def recover_orphaned_work(self) -> list[WorkItem]:
    """Find and reset work items that were abandoned.

    An orphaned work item is IN_PROGRESS but:
    - Has no active execution in the executor pool
    - Was last updated more than timeout ago

    Resets them to PENDING status.

    Returns:
        List of recovered work items
    """
```

```python
def get_stale_work(
    self,
    threshold_minutes: int = 60,
) -> list[WorkItem]:
    """Find work items that haven't been updated recently."""
```

---

## Persistence Strategy

### Directory Structure

```
workspace/
  ledger/
    index.json              # WorkIndex manifest
    active/                 # Currently active work items
      WRK-20260103-0001.json
      WRK-20260103-0002.json
    completed/              # Archived completed work
      2026/
        01/
          WRK-20260103-0001.json
    failed/                 # Archived failed work
      2026/
        01/
          WRK-20260103-0003.json
```

### File Format (WorkItem JSON)

```json
{
  "id": "WRK-20260103-0001",
  "title": "Implement user authentication",
  "type": "feature",
  "priority": "high",
  "status": "in_progress",
  "owner": "swarm_dev/implementer",
  "description": "Implement JWT-based user authentication...",
  "context": {
    "files": ["backend/auth.py", "frontend/lib/auth.ts"],
    "references": ["ADR-003"]
  },
  "parent_id": null,
  "dependencies": [],
  "created_at": "2026-01-03T10:30:00",
  "created_by": "CEO",
  "updated_at": "2026-01-03T11:45:00",
  "started_at": "2026-01-03T11:00:00",
  "completed_at": null,
  "result": null,
  "error": null,
  "swarm_name": "swarm_dev",
  "job_id": "job-abc123",
  "execution_id": "exec-def456",
  "history": [
    {
      "timestamp": "2026-01-03T10:30:00",
      "action": "created",
      "actor": "CEO",
      "details": {},
      "previous_status": null,
      "new_status": "pending"
    },
    {
      "timestamp": "2026-01-03T11:00:00",
      "action": "claimed",
      "actor": "swarm_dev/implementer",
      "details": {"execution_id": "exec-def456"},
      "previous_status": "pending",
      "new_status": "in_progress"
    }
  ]
}
```

### Index File (index.json)

```json
{
  "version": "1.0",
  "last_updated": "2026-01-03T11:45:00",
  "total_count": 15,
  "id_counter": 15,
  "items": {
    "WRK-20260103-0001": "active/WRK-20260103-0001.json",
    "WRK-20260103-0002": "active/WRK-20260103-0002.json"
  },
  "by_status": {
    "pending": ["WRK-20260103-0003", "WRK-20260103-0004"],
    "in_progress": ["WRK-20260103-0001", "WRK-20260103-0002"],
    "blocked": [],
    "completed": ["WRK-20260102-0001"],
    "failed": []
  },
  "by_owner": {
    "swarm_dev/implementer": ["WRK-20260103-0001"],
    "swarm_dev/researcher": ["WRK-20260103-0002"]
  },
  "by_swarm": {
    "swarm_dev": ["WRK-20260103-0001", "WRK-20260103-0002"]
  },
  "by_parent": {
    "WRK-20260103-0001": ["WRK-20260103-0005", "WRK-20260103-0006"]
  }
}
```

### Atomic Write Pattern

Following the pattern from `escalation_protocol.py`:

```python
def _save_work_item(self, work_item: WorkItem) -> None:
    """Save work item using atomic write."""
    filename = f"{work_item.id}.json"

    # Determine directory based on status
    if work_item.status == WorkStatus.COMPLETED:
        subdir = Path("completed") / work_item.completed_at.strftime("%Y/%m")
    elif work_item.status == WorkStatus.FAILED:
        subdir = Path("failed") / work_item.updated_at.strftime("%Y/%m")
    else:
        subdir = Path("active")

    target_dir = self.ledger_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    filepath = target_dir / filename
    temp_path = filepath.with_suffix('.json.tmp')

    try:
        with open(temp_path, "w") as f:
            json.dump(work_item.to_dict(), f, indent=2)
        temp_path.rename(filepath)  # Atomic on POSIX
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise
```

---

## Thread Safety

### Lock Strategy

```python
class WorkLedger:
    def __init__(self, ...):
        self._lock = threading.RLock()  # Reentrant lock
        self._item_locks: dict[str, threading.Lock] = {}  # Per-item locks
```

### Critical Sections

1. **Index operations**: Always hold `_lock`
2. **Work item mutations**: Hold both `_lock` and item-specific lock
3. **Queries**: Hold `_lock` for snapshot, release before returning

```python
def claim_work(self, work_id: str, owner: str, ...) -> WorkItem | None:
    with self._lock:
        # Check if work exists and is claimable
        if work_id not in self._index.items:
            return None

        work_item = self._load_work_item(work_id)

        if work_item.status != WorkStatus.PENDING:
            return None  # Already claimed or not available

        if work_item.owner is not None:
            return None  # Already has an owner

        # Perform the claim
        work_item.owner = owner
        work_item.status = WorkStatus.IN_PROGRESS
        work_item.started_at = datetime.now()
        work_item.updated_at = datetime.now()
        work_item.history.append(WorkHistoryEntry(
            timestamp=datetime.now(),
            action="claimed",
            actor=owner,
            details={},
            previous_status=WorkStatus.PENDING,
            new_status=WorkStatus.IN_PROGRESS,
        ))

        self._save_work_item(work_item)
        self._update_index(work_item)

        return work_item
```

---

## Integration Points

### 1. AgentExecutorPool Integration

Modify `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py`:

```python
class AgentExecutorPool:
    def __init__(
        self,
        max_concurrent: int = 5,
        workspace_manager: WorkspaceManager | None = None,
        work_ledger: WorkLedger | None = None,  # NEW
    ):
        self.work_ledger = work_ledger or get_work_ledger()
```

```python
async def execute(
    self,
    context: AgentExecutionContext,
    prompt: str,
    system_prompt: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    work_id: str | None = None,  # NEW: Associate with work item
) -> AsyncIterator[dict[str, Any]]:

    execution_id = str(uuid.uuid4())

    # Link execution to work item
    if work_id and self.work_ledger:
        self.work_ledger.link_execution(work_id, execution_id)

    try:
        async for event in self._run_agent(...):
            yield event

        # On success, optionally complete work
        if work_id and self.work_ledger:
            self.work_ledger.complete_work(
                work_id,
                context.full_name,
                result={"execution_id": execution_id}
            )

    except Exception as e:
        # On failure, mark work as failed
        if work_id and self.work_ledger:
            self.work_ledger.fail_work(work_id, context.full_name, str(e))
        raise
```

### 2. Backend Main Integration

Modify `/Users/jellingson/agent-swarm/backend/main.py` websocket_chat():

```python
from shared.work_ledger import get_work_ledger, WorkType, WorkPriority

# In websocket_chat(), after receiving user message:
ledger = get_work_ledger()

# Create work item for the request
work_item = ledger.create_work(
    title=user_message[:100],  # First 100 chars as title
    description=user_message,
    work_type=WorkType.TASK,
    priority=WorkPriority.MEDIUM,
    created_by="CEO",
    context={
        "session_id": session_id,
        "message_id": message_id,
    }
)

# Pass work_id to executor
async for event in executor.execute(..., work_id=work_item.id):
    ...
```

### 3. Escalation Protocol Integration

Link escalations to work items:

```python
# In escalation_protocol.py
def create_escalation(..., work_id: str | None = None):
    escalation = Escalation(...)

    # If work_id provided, block that work
    if work_id:
        ledger = get_work_ledger()
        ledger.block_work(
            work_id,
            reason=f"Escalation: {escalation.title}",
            blocker_id=escalation.id,
        )
```

### 4. AgentExecutionContext Extension

Add work_id to context:

```python
# In execution_context.py
@dataclass
class AgentExecutionContext:
    ...
    work_id: str | None = None  # NEW: Associated work item
```

---

## Example Usage

### Creating and Processing Work

```python
from shared.work_ledger import (
    get_work_ledger,
    WorkType,
    WorkPriority
)

ledger = get_work_ledger()

# CEO creates a task
work = ledger.create_work(
    title="Implement user authentication",
    description="Add JWT-based authentication with login/logout endpoints",
    work_type=WorkType.FEATURE,
    priority=WorkPriority.HIGH,
    created_by="CEO",
    swarm_name="swarm_dev",
    context={
        "acceptance_criteria": [
            "JWT tokens with 1-hour expiry",
            "Refresh token support",
            "Login endpoint at /api/auth/login",
        ]
    }
)

print(f"Created work: {work.id}")
```

### Agent Claims and Processes Work

```python
# Agent checks for available work
ready_work = ledger.get_ready_to_start(swarm_name="swarm_dev")

if ready_work:
    # Claim the highest priority item
    work = ready_work[0]
    claimed = ledger.claim_work(work.id, owner="swarm_dev/implementer")

    if claimed:
        # Process the work...
        try:
            result = await process_work(claimed)
            ledger.complete_work(
                claimed.id,
                owner="swarm_dev/implementer",
                result={
                    "files_created": result.files,
                    "tests_passed": result.test_count,
                }
            )
        except Exception as e:
            ledger.fail_work(
                claimed.id,
                owner="swarm_dev/implementer",
                error=str(e)
            )
```

### Breaking Down Work into Subtasks

```python
# Architect breaks down a feature into subtasks
parent = ledger.get_work("WRK-20260103-0001")

subtasks = [
    ("Design auth schema", "Design JWT payload and database schema"),
    ("Implement backend endpoints", "Create login, logout, refresh endpoints"),
    ("Add middleware", "JWT validation middleware"),
    ("Frontend integration", "Add auth context and hooks"),
    ("Write tests", "Unit and integration tests"),
]

for title, description in subtasks:
    ledger.create_subtask(
        parent_id=parent.id,
        title=title,
        description=description,
        created_by="swarm_dev/architect",
    )

# Check progress
progress = ledger.get_progress(parent.id)
print(f"Progress: {progress['percent_complete']:.0f}% complete")
```

### Recovering from Crashes

```python
# On system startup, recover orphaned work
ledger = get_work_ledger()
orphaned = ledger.recover_orphaned_work()

for work in orphaned:
    logger.info(f"Recovered orphaned work: {work.id} - {work.title}")

# Check for stale work
stale = ledger.get_stale_work(threshold_minutes=120)
if stale:
    logger.warning(f"Found {len(stale)} stale work items")
```

### Querying Work Status

```python
# Dashboard view
ledger = get_work_ledger()

# Get all in-progress work
in_progress = ledger.get_in_progress()
print(f"In Progress: {len(in_progress)}")

# Get blocked work
blocked = ledger.get_blocked()
for work in blocked:
    print(f"BLOCKED: {work.id} - {work.title}")
    print(f"  Reason: {work.context.get('block_reason', 'Unknown')}")

# Get work by swarm
swarm_work = ledger.get_by_swarm("swarm_dev")
by_status = {}
for work in swarm_work:
    by_status.setdefault(work.status.value, []).append(work)

print("Swarm Dev Work Summary:")
for status, items in by_status.items():
    print(f"  {status}: {len(items)}")
```

---

## File Structure

```
shared/
  work_ledger.py          # Main WorkLedger class and helpers
  work_models.py          # WorkItem, WorkStatus, etc. dataclasses
  __init__.py             # Update exports

workspace/
  ledger/                 # Created by WorkLedger
    index.json
    active/
    completed/
    failed/
```

---

## Implementation Plan

### Phase 1: Core Data Structures (Day 1)

1. Create `/Users/jellingson/agent-swarm/shared/work_models.py`:
   - WorkStatus, WorkType, WorkPriority enums
   - WorkHistoryEntry dataclass
   - WorkItem dataclass with to_dict/from_dict
   - WorkIndex dataclass

2. Create `/Users/jellingson/agent-swarm/shared/work_ledger.py`:
   - WorkLedger class skeleton
   - Initialization and directory setup
   - ID generation
   - Persistence helpers (atomic write)

### Phase 2: Core Operations (Day 2)

1. Implement create_work()
2. Implement claim_work() and release_work()
3. Implement status transitions (start, block, complete, fail)
4. Implement index management
5. Add thread safety

### Phase 3: Query Operations (Day 3)

1. Implement get_work()
2. Implement filtering queries (get_pending, get_in_progress, etc.)
3. Implement hierarchy queries (get_children, get_progress)
4. Implement recovery operations

### Phase 4: Integration (Day 4)

1. Integrate with agent_executor_pool.py
2. Add work_id to AgentExecutionContext
3. Wire up in backend/main.py
4. Connect with escalation_protocol.py

### Phase 5: Testing and Polish (Day 5)

1. Unit tests for all operations
2. Integration tests with executor pool
3. Stress tests for thread safety
4. Documentation

---

## Trade-offs and Considerations

### Pros

- **Crash resilience**: Work survives agent/system restarts
- **Auditability**: Full history of work items
- **Visibility**: Any component can query work status
- **Recovery**: Orphaned work can be automatically reclaimed
- **Hierarchy**: Complex tasks can be broken into subtasks

### Cons

- **Overhead**: Additional I/O for persistence
- **Complexity**: Another system to maintain
- **Coordination**: Agents must claim/release work properly
- **Storage**: Accumulates files over time (needs archival strategy)

### Mitigations

- Use append-only operations where possible (faster)
- Index caching reduces read overhead
- Periodic cleanup job for old completed work
- Clear documentation for agent developers

---

## Dependencies

**Internal:**
- `shared/execution_context.py` - Add work_id field
- `shared/agent_executor_pool.py` - Integration
- `backend/main.py` - Create work items for requests

**External:**
- None (uses only standard library)

---

## Success Metrics

1. **Crash recovery**: 100% of in-progress work recoverable after restart
2. **Query performance**: Index queries under 10ms
3. **Write performance**: Work item persistence under 50ms
4. **Adoption**: All agent executions linked to work items

---

## Next Steps

1. Review and approve this design
2. Implementer creates Phase 1 (data structures)
3. Critic reviews for thread safety and edge cases
4. Iterate through remaining phases
5. Integration testing with full agent-swarm flow

---

## Appendix: Module Exports

Update `/Users/jellingson/agent-swarm/shared/__init__.py`:

```python
from .work_ledger import (
    WorkLedger,
    get_work_ledger,
)
from .work_models import (
    WorkItem,
    WorkStatus,
    WorkType,
    WorkPriority,
    WorkHistoryEntry,
    WorkIndex,
)

__all__ = [
    # ... existing exports ...

    # Work Ledger
    "WorkLedger",
    "get_work_ledger",
    "WorkItem",
    "WorkStatus",
    "WorkType",
    "WorkPriority",
    "WorkHistoryEntry",
    "WorkIndex",
]
```
