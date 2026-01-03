"""Work Ledger system for persistent work tracking.

This module provides a crash-resilient work item tracking system
that persists work state to disk and survives agent restarts.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .work_models import (
    WorkHistoryEntry,
    WorkIndex,
    WorkItem,
    WorkPriority,
    WorkStatus,
    WorkType,
)

logger = logging.getLogger(__name__)


class WorkLedger:
    """Manages persistent work items across agent lifecycles."""

    def __init__(
        self,
        ledger_dir: Path | None = None,
        auto_save: bool = True,
        on_work_created: Callable[[WorkItem], None] | None = None,
    ) -> None:
        """Initialize the work ledger.

        Args:
            ledger_dir: Directory for ledger storage (default: workspace/ledger/)
            auto_save: Automatically persist changes to disk
            on_work_created: Optional callback triggered when work is created
        """
        self.ledger_dir = ledger_dir or Path("./workspace/ledger")
        self.auto_save = auto_save
        self._on_work_created = on_work_created
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Create directory structure
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        (self.ledger_dir / "active").mkdir(exist_ok=True)
        (self.ledger_dir / "completed").mkdir(exist_ok=True)
        (self.ledger_dir / "failed").mkdir(exist_ok=True)

        # Load or create index
        self._index = self._load_index()

        # In-memory cache of work items
        self._items_cache: dict[str, WorkItem] = {}

    def set_on_work_created(self, callback: Callable[[WorkItem], None] | None) -> None:
        """Set or clear the callback triggered when work is created.

        Args:
            callback: Function to call with newly created WorkItem, or None to disable
        """
        self._on_work_created = callback

    def _notify_work_created(self, work_item: WorkItem) -> None:
        """Notify listener that work was created (for auto-spawn)."""
        if self._on_work_created:
            try:
                self._on_work_created(work_item)
            except Exception as e:
                logger.error(f"Error in on_work_created callback: {e}")

    def _load_index(self) -> WorkIndex:
        """Load index from disk or create empty."""
        index_path = self.ledger_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                return WorkIndex.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
        return WorkIndex.create_empty()

    def _save_index(self) -> None:
        """Save index to disk using atomic write."""
        index_path = self.ledger_dir / "index.json"
        temp_path = index_path.with_suffix('.json.tmp')

        try:
            self._index.last_updated = datetime.now()
            with open(temp_path, "w") as f:
                json.dump(self._index.to_dict(), f, indent=2)
            temp_path.rename(index_path)  # Atomic on POSIX
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _generate_id(self) -> str:
        """Generate a unique work item ID."""
        self._index.id_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"WRK-{timestamp}-{self._index.id_counter:04d}"

    def _get_filepath(self, work_item: WorkItem) -> Path:
        """Get the file path for a work item based on status."""
        filename = f"{work_item.id}.json"

        if work_item.status == WorkStatus.COMPLETED and work_item.completed_at:
            subdir = Path("completed") / work_item.completed_at.strftime("%Y/%m")
        elif work_item.status == WorkStatus.FAILED:
            subdir = Path("failed") / work_item.updated_at.strftime("%Y/%m")
        else:
            subdir = Path("active")

        return self.ledger_dir / subdir / filename

    def _save_work_item(self, work_item: WorkItem) -> None:
        """Save work item using atomic write."""
        filepath = self._get_filepath(work_item)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        temp_path = filepath.with_suffix('.json.tmp')

        try:
            with open(temp_path, "w") as f:
                json.dump(work_item.to_dict(), f, indent=2)
            temp_path.rename(filepath)  # Atomic on POSIX
        except Exception as e:
            logger.error(f"Error saving work item {work_item.id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

        # Update cache
        self._items_cache[work_item.id] = work_item

    def _load_work_item(self, work_id: str) -> WorkItem | None:
        """Load a work item from disk or cache."""
        # Check cache first
        if work_id in self._items_cache:
            return self._items_cache[work_id]

        # Load from disk
        if work_id not in self._index.items:
            return None

        rel_path = self._index.items[work_id]
        filepath = self.ledger_dir / rel_path

        if not filepath.exists():
            logger.warning(f"Work item file not found: {filepath}")
            return None

        try:
            with open(filepath) as f:
                data = json.load(f)
            work_item = WorkItem.from_dict(data)
            self._items_cache[work_id] = work_item
            return work_item
        except Exception as e:
            logger.error(f"Error loading work item {work_id}: {e}")
            return None

    def _update_index(self, work_item: WorkItem, old_status: WorkStatus | None = None) -> None:
        """Update index entries for a work item."""
        work_id = work_item.id

        # Update items mapping
        rel_path = self._get_filepath(work_item).relative_to(self.ledger_dir)
        self._index.items[work_id] = str(rel_path)

        # Update status index
        if old_status and old_status != work_item.status:
            # Remove from old status list
            old_status_key = old_status.value
            if old_status_key in self._index.by_status:
                if work_id in self._index.by_status[old_status_key]:
                    self._index.by_status[old_status_key].remove(work_id)

        # Add to new status list
        status_key = work_item.status.value
        if status_key not in self._index.by_status:
            self._index.by_status[status_key] = []
        if work_id not in self._index.by_status[status_key]:
            self._index.by_status[status_key].append(work_id)

        # Update owner index
        if work_item.owner:
            if work_item.owner not in self._index.by_owner:
                self._index.by_owner[work_item.owner] = []
            if work_id not in self._index.by_owner[work_item.owner]:
                self._index.by_owner[work_item.owner].append(work_id)

        # Update swarm index
        if work_item.swarm_name:
            if work_item.swarm_name not in self._index.by_swarm:
                self._index.by_swarm[work_item.swarm_name] = []
            if work_id not in self._index.by_swarm[work_item.swarm_name]:
                self._index.by_swarm[work_item.swarm_name].append(work_id)

        # Update parent index
        if work_item.parent_id:
            if work_item.parent_id not in self._index.by_parent:
                self._index.by_parent[work_item.parent_id] = []
            if work_id not in self._index.by_parent[work_item.parent_id]:
                self._index.by_parent[work_item.parent_id].append(work_id)

        # Save index
        if self.auto_save:
            self._save_index()

    def _add_history(
        self,
        work_item: WorkItem,
        action: str,
        actor: str,
        details: dict[str, Any] | None = None,
        previous_status: WorkStatus | None = None,
        new_status: WorkStatus | None = None,
    ) -> None:
        """Add a history entry to a work item."""
        entry = WorkHistoryEntry(
            timestamp=datetime.now(),
            action=action,
            actor=actor,
            details=details or {},
            previous_status=previous_status,
            new_status=new_status,
        )
        work_item.history.append(entry)

    # ========== Creating Work ==========

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
        with self._lock:
            # Validate parent_id
            if parent_id and parent_id not in self._index.items:
                raise ValueError(f"Parent work item not found: {parent_id}")

            # Validate dependencies
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in self._index.items:
                        raise ValueError(f"Dependency work item not found: {dep_id}")

            now = datetime.now()
            work_id = self._generate_id()

            work_item = WorkItem(
                id=work_id,
                title=title,
                type=work_type,
                priority=priority,
                status=WorkStatus.PENDING,
                owner=None,
                description=description,
                context=context or {},
                parent_id=parent_id,
                dependencies=dependencies or [],
                created_at=now,
                created_by=created_by,
                updated_at=now,
                started_at=None,
                completed_at=None,
                result=None,
                error=None,
                swarm_name=swarm_name,
                job_id=job_id,
                execution_id=None,
                history=[],
            )

            self._add_history(
                work_item,
                action="created",
                actor=created_by,
                details={"title": title, "type": work_type.value},
                previous_status=None,
                new_status=WorkStatus.PENDING,
            )

            self._index.total_count += 1
            self._save_work_item(work_item)
            self._update_index(work_item)

            logger.info(f"Created work item {work_id}: {title}")

            # Notify listener for auto-spawn (called outside lock to avoid deadlock)
            self._notify_work_created(work_item)

            return work_item

    # ========== Claiming Work ==========

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
        with self._lock:
            work_item = self._load_work_item(work_id)
            if not work_item:
                logger.warning(f"Claim failed - work item not found: {work_id}")
                return None

            if work_item.status != WorkStatus.PENDING:
                logger.warning(f"Claim failed - work item not pending: {work_id} ({work_item.status.value})")
                return None

            if work_item.owner is not None:
                logger.warning(f"Claim failed - work item already owned: {work_id} ({work_item.owner})")
                return None

            old_status = work_item.status
            work_item.owner = owner
            work_item.status = WorkStatus.IN_PROGRESS
            work_item.started_at = datetime.now()
            work_item.updated_at = datetime.now()
            work_item.execution_id = execution_id

            self._add_history(
                work_item,
                action="claimed",
                actor=owner,
                details={"execution_id": execution_id} if execution_id else {},
                previous_status=old_status,
                new_status=WorkStatus.IN_PROGRESS,
            )

            self._save_work_item(work_item)
            self._update_index(work_item, old_status)

            logger.info(f"Work item {work_id} claimed by {owner}")
            return work_item

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
        with self._lock:
            work_item = self._load_work_item(work_id)
            if not work_item:
                return None

            if work_item.owner != owner:
                logger.warning(f"Release failed - owner mismatch: {work_id} ({work_item.owner} != {owner})")
                return None

            old_status = work_item.status
            work_item.owner = None
            work_item.status = WorkStatus.PENDING
            work_item.updated_at = datetime.now()
            work_item.execution_id = None

            # Remove from old owner index
            if owner in self._index.by_owner:
                if work_id in self._index.by_owner[owner]:
                    self._index.by_owner[owner].remove(work_id)

            self._add_history(
                work_item,
                action="released",
                actor=owner,
                details={"reason": reason},
                previous_status=old_status,
                new_status=WorkStatus.PENDING,
            )

            self._save_work_item(work_item)
            self._update_index(work_item, old_status)

            logger.info(f"Work item {work_id} released by {owner}: {reason}")
            return work_item

    # ========== Updating Work Status ==========

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
        with self._lock:
            work_item = self._load_work_item(work_id)
            if not work_item:
                return None

            if work_item.status == WorkStatus.IN_PROGRESS:
                # Already in progress
                return work_item

            old_status = work_item.status
            work_item.status = WorkStatus.IN_PROGRESS
            work_item.owner = owner
            work_item.started_at = work_item.started_at or datetime.now()
            work_item.updated_at = datetime.now()

            self._add_history(
                work_item,
                action="started",
                actor=owner,
                details={},
                previous_status=old_status,
                new_status=WorkStatus.IN_PROGRESS,
            )

            self._save_work_item(work_item)
            self._update_index(work_item, old_status)

            logger.info(f"Work item {work_id} started by {owner}")
            return work_item

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
        with self._lock:
            work_item = self._load_work_item(work_id)
            if not work_item:
                return None

            old_status = work_item.status
            work_item.status = WorkStatus.BLOCKED
            work_item.updated_at = datetime.now()
            work_item.context["block_reason"] = reason
            if blocker_id:
                work_item.context["blocker_id"] = blocker_id

            self._add_history(
                work_item,
                action="blocked",
                actor=work_item.owner or "system",
                details={"reason": reason, "blocker_id": blocker_id},
                previous_status=old_status,
                new_status=WorkStatus.BLOCKED,
            )

            self._save_work_item(work_item)
            self._update_index(work_item, old_status)

            logger.info(f"Work item {work_id} blocked: {reason}")
            return work_item

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
        with self._lock:
            work_item = self._load_work_item(work_id)
            if not work_item:
                return None

            # Remove old file if moving to completed directory
            old_filepath = self._get_filepath(work_item)

            old_status = work_item.status
            work_item.status = WorkStatus.COMPLETED
            work_item.completed_at = datetime.now()
            work_item.updated_at = datetime.now()
            work_item.result = result

            self._add_history(
                work_item,
                action="completed",
                actor=owner,
                details={"result": result} if result else {},
                previous_status=old_status,
                new_status=WorkStatus.COMPLETED,
            )

            # Save to new location
            self._save_work_item(work_item)

            # Remove old file if it's different
            new_filepath = self._get_filepath(work_item)
            if old_filepath != new_filepath and old_filepath.exists():
                old_filepath.unlink()

            self._update_index(work_item, old_status)

            logger.info(f"Work item {work_id} completed by {owner}")
            return work_item

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
        with self._lock:
            work_item = self._load_work_item(work_id)
            if not work_item:
                return None

            # Remove old file if moving to failed directory
            old_filepath = self._get_filepath(work_item)

            old_status = work_item.status
            work_item.status = WorkStatus.FAILED
            work_item.updated_at = datetime.now()
            work_item.error = error

            self._add_history(
                work_item,
                action="failed",
                actor=owner,
                details={"error": error},
                previous_status=old_status,
                new_status=WorkStatus.FAILED,
            )

            # Save to new location
            self._save_work_item(work_item)

            # Remove old file if it's different
            new_filepath = self._get_filepath(work_item)
            if old_filepath != new_filepath and old_filepath.exists():
                old_filepath.unlink()

            self._update_index(work_item, old_status)

            logger.info(f"Work item {work_id} failed: {error[:50]}...")
            return work_item

    # ========== Querying Work ==========

    def get_work(self, work_id: str) -> WorkItem | None:
        """Get a single work item by ID."""
        with self._lock:
            return self._load_work_item(work_id)

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
        with self._lock:
            pending_ids = self._index.by_status.get(WorkStatus.PENDING.value, [])
            items = []

            for work_id in pending_ids:
                work_item = self._load_work_item(work_id)
                if not work_item:
                    continue

                # Apply filters
                if owner is not None and work_item.owner != owner:
                    continue
                if swarm_name and work_item.swarm_name != swarm_name:
                    continue
                if work_type and work_item.type != work_type:
                    continue

                items.append(work_item)

        # Sort by priority (critical first) then by creation time
        priority_order = {
            WorkPriority.CRITICAL: 0,
            WorkPriority.HIGH: 1,
            WorkPriority.MEDIUM: 2,
            WorkPriority.LOW: 3,
        }
        items.sort(key=lambda w: (priority_order[w.priority], w.created_at))

        return items

    def get_in_progress(
        self,
        owner: str | None = None,
    ) -> list[WorkItem]:
        """Get work items currently in progress."""
        with self._lock:
            in_progress_ids = self._index.by_status.get(WorkStatus.IN_PROGRESS.value, [])
            items = []

            for work_id in in_progress_ids:
                work_item = self._load_work_item(work_id)
                if not work_item:
                    continue

                if owner and work_item.owner != owner:
                    continue

                items.append(work_item)

        return items

    def get_blocked(self) -> list[WorkItem]:
        """Get all blocked work items."""
        with self._lock:
            blocked_ids = self._index.by_status.get(WorkStatus.BLOCKED.value, [])
            items = []

            for work_id in blocked_ids:
                work_item = self._load_work_item(work_id)
                if work_item:
                    items.append(work_item)

        return items

    def get_children(self, parent_id: str) -> list[WorkItem]:
        """Get all child work items of a parent."""
        with self._lock:
            child_ids = self._index.by_parent.get(parent_id, [])
            items = []

            for work_id in child_ids:
                work_item = self._load_work_item(work_id)
                if work_item:
                    items.append(work_item)

        return items

    def get_by_swarm(self, swarm_name: str) -> list[WorkItem]:
        """Get all work items for a swarm."""
        with self._lock:
            swarm_ids = self._index.by_swarm.get(swarm_name, [])
            items = []

            for work_id in swarm_ids:
                work_item = self._load_work_item(work_id)
                if work_item:
                    items.append(work_item)

        return items

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
        with self._lock:
            pending_ids = self._index.by_status.get(WorkStatus.PENDING.value, [])
            items = []

            for work_id in pending_ids:
                work_item = self._load_work_item(work_id)
                if not work_item:
                    continue

                # Must be unclaimed
                if work_item.owner is not None:
                    continue

                # Filter by swarm if specified
                if swarm_name and work_item.swarm_name != swarm_name:
                    continue

                # Check dependencies
                if work_item.dependencies:
                    all_complete = True
                    for dep_id in work_item.dependencies:
                        dep = self._load_work_item(dep_id)
                        if not dep or dep.status != WorkStatus.COMPLETED:
                            all_complete = False
                            break
                    if not all_complete:
                        continue

                items.append(work_item)

        # Sort by priority then creation time
        priority_order = {
            WorkPriority.CRITICAL: 0,
            WorkPriority.HIGH: 1,
            WorkPriority.MEDIUM: 2,
            WorkPriority.LOW: 3,
        }
        items.sort(key=lambda w: (priority_order[w.priority], w.created_at))

        return items

    # ========== Hierarchy Operations ==========

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
        with self._lock:
            parent = self._load_work_item(parent_id)
            if not parent:
                raise ValueError(f"Parent work item not found: {parent_id}")

            # Inherit from parent
            swarm_name = kwargs.pop("swarm_name", parent.swarm_name)
            context = kwargs.pop("context", {})
            if parent.context:
                # Merge parent context (subtask context takes precedence)
                merged_context = {**parent.context, **context}
            else:
                merged_context = context

            return self.create_work(
                title=title,
                description=description,
                created_by=created_by,
                parent_id=parent_id,
                swarm_name=swarm_name,
                context=merged_context,
                **kwargs,
            )

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
        children = self.get_children(work_id)

        status_counts = {
            "total_children": len(children),
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "blocked": 0,
            "failed": 0,
            "cancelled": 0,
        }

        for child in children:
            if child.status == WorkStatus.COMPLETED:
                status_counts["completed"] += 1
            elif child.status == WorkStatus.IN_PROGRESS:
                status_counts["in_progress"] += 1
            elif child.status == WorkStatus.PENDING:
                status_counts["pending"] += 1
            elif child.status == WorkStatus.BLOCKED:
                status_counts["blocked"] += 1
            elif child.status == WorkStatus.FAILED:
                status_counts["failed"] += 1
            elif child.status == WorkStatus.CANCELLED:
                status_counts["cancelled"] += 1

        total = status_counts["total_children"]
        if total > 0:
            status_counts["percent_complete"] = (status_counts["completed"] / total) * 100
        else:
            status_counts["percent_complete"] = 0.0

        return status_counts

    # ========== Recovery Operations ==========

    def recover_orphaned_work(
        self,
        timeout_minutes: int = 60,
    ) -> list[WorkItem]:
        """Find and reset work items that were abandoned.

        An orphaned work item is IN_PROGRESS but was last updated
        more than timeout_minutes ago.

        Resets them to PENDING status.

        Args:
            timeout_minutes: Minutes since last update to consider orphaned

        Returns:
            List of recovered work items
        """
        with self._lock:
            in_progress_ids = self._index.by_status.get(WorkStatus.IN_PROGRESS.value, [])
            threshold = datetime.now() - timedelta(minutes=timeout_minutes)
            recovered = []

            for work_id in list(in_progress_ids):  # Copy list to avoid mutation during iteration
                work_item = self._load_work_item(work_id)
                if not work_item:
                    continue

                if work_item.updated_at < threshold:
                    old_owner = work_item.owner
                    old_status = work_item.status

                    work_item.status = WorkStatus.PENDING
                    work_item.owner = None
                    work_item.updated_at = datetime.now()
                    work_item.execution_id = None

                    # Remove from old owner index
                    if old_owner and old_owner in self._index.by_owner:
                        if work_id in self._index.by_owner[old_owner]:
                            self._index.by_owner[old_owner].remove(work_id)

                    self._add_history(
                        work_item,
                        action="recovered",
                        actor="system",
                        details={
                            "previous_owner": old_owner,
                            "timeout_minutes": timeout_minutes,
                        },
                        previous_status=old_status,
                        new_status=WorkStatus.PENDING,
                    )

                    self._save_work_item(work_item)
                    self._update_index(work_item, old_status)
                    recovered.append(work_item)

                    logger.info(f"Recovered orphaned work item: {work_id} (was owned by {old_owner})")

        return recovered

    def get_stale_work(
        self,
        threshold_minutes: int = 60,
    ) -> list[WorkItem]:
        """Find work items that haven't been updated recently."""
        threshold = datetime.now() - timedelta(minutes=threshold_minutes)
        stale = []

        with self._lock:
            # Check in_progress items
            in_progress_ids = self._index.by_status.get(WorkStatus.IN_PROGRESS.value, [])
            for work_id in in_progress_ids:
                work_item = self._load_work_item(work_id)
                if work_item and work_item.updated_at < threshold:
                    stale.append(work_item)

            # Check blocked items
            blocked_ids = self._index.by_status.get(WorkStatus.BLOCKED.value, [])
            for work_id in blocked_ids:
                work_item = self._load_work_item(work_id)
                if work_item and work_item.updated_at < threshold:
                    stale.append(work_item)

        return stale

    def load_from_disk(self) -> None:
        """Load all work items from disk and rebuild index."""
        with self._lock:
            # Reset index
            self._index = WorkIndex.create_empty()
            self._items_cache.clear()

            max_counter = 0

            # Load from active directory
            active_dir = self.ledger_dir / "active"
            if active_dir.exists():
                for filepath in active_dir.glob("WRK-*.json"):
                    self._load_and_index_file(filepath, "active")
                    counter = self._extract_counter(filepath.stem)
                    max_counter = max(max_counter, counter)

            # Load from completed directory (recursively)
            completed_dir = self.ledger_dir / "completed"
            if completed_dir.exists():
                for filepath in completed_dir.rglob("WRK-*.json"):
                    rel_path = filepath.relative_to(self.ledger_dir)
                    self._load_and_index_file(filepath, str(rel_path.parent))
                    counter = self._extract_counter(filepath.stem)
                    max_counter = max(max_counter, counter)

            # Load from failed directory (recursively)
            failed_dir = self.ledger_dir / "failed"
            if failed_dir.exists():
                for filepath in failed_dir.rglob("WRK-*.json"):
                    rel_path = filepath.relative_to(self.ledger_dir)
                    self._load_and_index_file(filepath, str(rel_path.parent))
                    counter = self._extract_counter(filepath.stem)
                    max_counter = max(max_counter, counter)

            self._index.id_counter = max_counter
            self._save_index()

            logger.info(f"Loaded {self._index.total_count} work items from disk")

    def _load_and_index_file(self, filepath: Path, subdir: str) -> None:
        """Load a work item file and add to index."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            work_item = WorkItem.from_dict(data)
            self._items_cache[work_item.id] = work_item

            # Add to index
            rel_path = f"{subdir}/{filepath.name}"
            self._index.items[work_item.id] = rel_path
            self._index.total_count += 1

            # Status index
            status_key = work_item.status.value
            if status_key not in self._index.by_status:
                self._index.by_status[status_key] = []
            self._index.by_status[status_key].append(work_item.id)

            # Owner index
            if work_item.owner:
                if work_item.owner not in self._index.by_owner:
                    self._index.by_owner[work_item.owner] = []
                self._index.by_owner[work_item.owner].append(work_item.id)

            # Swarm index
            if work_item.swarm_name:
                if work_item.swarm_name not in self._index.by_swarm:
                    self._index.by_swarm[work_item.swarm_name] = []
                self._index.by_swarm[work_item.swarm_name].append(work_item.id)

            # Parent index
            if work_item.parent_id:
                if work_item.parent_id not in self._index.by_parent:
                    self._index.by_parent[work_item.parent_id] = []
                self._index.by_parent[work_item.parent_id].append(work_item.id)

        except Exception as e:
            logger.error(f"Error loading work item {filepath}: {e}")

    def _extract_counter(self, work_id: str) -> int:
        """Extract counter from work ID: WRK-YYYYMMDD-NNNN."""
        try:
            parts = work_id.split('-')
            if len(parts) >= 3:
                return int(parts[-1])
        except (ValueError, IndexError):
            pass
        return 0


# Module-level singleton with thread-safe initialization
_work_ledger: WorkLedger | None = None
_singleton_lock = threading.Lock()


def get_work_ledger(ledger_dir: Path | None = None) -> WorkLedger:
    """Get or create the global work ledger.

    Thread-safe singleton pattern with double-checked locking.

    Args:
        ledger_dir: Optional ledger directory (used on first call)

    Returns:
        The work ledger singleton
    """
    global _work_ledger

    if _work_ledger is None:
        with _singleton_lock:
            # Double-check after acquiring lock
            if _work_ledger is None:
                _work_ledger = WorkLedger(ledger_dir)
                _work_ledger.load_from_disk()

    return _work_ledger
