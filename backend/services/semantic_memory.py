"""Semantic Memory - SQLite-backed semantic node storage.

This module provides storage and retrieval for semantic memory:
- Facts, concepts, entities, schemas, frames
- Confidence tracking with Bayesian updates
- Base-level activation (ACT-R style)
- Full-text search via FTS5
- Semantic search via embeddings

Part of the tri-memory cognitive architecture.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from .memory_db import MemoryDatabase, get_memory_db

logger = logging.getLogger(__name__)


class SemanticNodeType(Enum):
    """Types of semantic nodes."""
    CONCEPT = "CONCEPT"       # Abstract idea or topic
    FACT = "FACT"             # Specific piece of information
    ENTITY = "ENTITY"         # Named entity (person, place, thing)
    SCHEMA = "SCHEMA"         # Abstract pattern or template
    FRAME = "FRAME"           # Structured situation representation
    IDENTITY = "IDENTITY"     # Self-knowledge statement
    PREFERENCE = "PREFERENCE" # User preference
    GOAL = "GOAL"             # Active goal or intention
    DECISION = "DECISION"     # Decision that was made
    RELATIONSHIP = "RELATIONSHIP"  # Person or entity relationship
    MEMORY = "MEMORY"         # Episodic memory reference


@dataclass
class SemanticNode:
    """A semantic memory node.

    Represents a piece of knowledge with confidence tracking,
    activation levels, and provenance information.
    """
    id: str
    node_type: SemanticNodeType
    label: str
    description: str = ""
    confidence: float = 0.5
    evidence_count: int = 1
    source_reliability: float = 0.5
    base_level_activation: float = 0.0
    last_access: datetime | None = None
    access_count: int = 0
    derived_from_episodes: list[str] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    color: str = "#8B5CF6"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SemanticNode:
        """Create node from database row."""
        return cls(
            id=row['id'],
            node_type=SemanticNodeType(row['node_type']),
            label=row['label'],
            description=row.get('description') or '',
            confidence=row.get('confidence', 0.5),
            evidence_count=row.get('evidence_count', 1),
            source_reliability=row.get('source_reliability', 0.5),
            base_level_activation=row.get('base_level_activation', 0.0),
            last_access=datetime.fromisoformat(row['last_access']) if row.get('last_access') else None,
            access_count=row.get('access_count', 0),
            derived_from_episodes=json.loads(row['derived_from_episodes']) if row.get('derived_from_episodes') else [],
            slots=json.loads(row['slots']) if row.get('slots') else {},
            source=row.get('source', 'system'),
            provenance=json.loads(row['provenance']) if row.get('provenance') else {},
            metadata=json.loads(row['metadata']) if row.get('metadata') else {},
            color=row.get('color', '#8B5CF6'),
            created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row.get('updated_at') else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': self.id,
            'node_type': self.node_type.value,
            'label': self.label,
            'description': self.description,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'source_reliability': self.source_reliability,
            'base_level_activation': self.base_level_activation,
            'last_access': self.last_access.isoformat() if self.last_access else None,
            'access_count': self.access_count,
            'derived_from_episodes': self.derived_from_episodes,
            'slots': self.slots,
            'source': self.source,
            'provenance': self.provenance,
            'metadata': self.metadata,
            'color': self.color,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class SemanticMemory:
    """SQLite-backed semantic memory store.

    Provides CRUD operations, search, and activation tracking
    for semantic memory nodes.

    Attributes:
        db: The memory database instance
    """

    # ACT-R decay parameter
    DECAY_RATE = 0.5

    def __init__(self, db: MemoryDatabase | None = None):
        """Initialize semantic memory.

        Args:
            db: Memory database instance (uses singleton if None)
        """
        self.db = db or get_memory_db()
        self._lock = threading.Lock()
        logger.info("SemanticMemory initialized")

    def add_node(
        self,
        label: str,
        node_type: SemanticNodeType = SemanticNodeType.CONCEPT,
        description: str = "",
        confidence: float = 0.5,
        source: str = "system",
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        slots: dict[str, Any] | None = None,
        color: str | None = None,
    ) -> SemanticNode:
        """Add a new semantic node.

        Args:
            label: Short label for the node
            node_type: Type of semantic node
            description: Detailed description
            confidence: Initial confidence (0.0-1.0)
            source: Where this node came from
            provenance: Creation context
            metadata: Additional key-value data
            slots: Frame slots for structured data
            color: Display color

        Returns:
            The created node
        """
        node_id = f"sem-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:5]}"
        now = datetime.now()

        # Default color based on type
        if color is None:
            color_map = {
                SemanticNodeType.IDENTITY: "#8B5CF6",
                SemanticNodeType.FACT: "#3B82F6",
                SemanticNodeType.PREFERENCE: "#10B981",
                SemanticNodeType.GOAL: "#F59E0B",
                SemanticNodeType.CONCEPT: "#EC4899",
            }
            color = color_map.get(node_type, "#8B5CF6")

        sql = """
            INSERT INTO semantic_nodes (
                id, node_type, label, description, confidence,
                source, provenance, metadata, slots, color,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.db.execute(sql, (
            node_id,
            node_type.value,
            label,
            description,
            confidence,
            source,
            json.dumps(provenance or {}),
            json.dumps(metadata or {}),
            json.dumps(slots or {}),
            color,
            now.isoformat(),
            now.isoformat(),
        ))

        node = SemanticNode(
            id=node_id,
            node_type=node_type,
            label=label,
            description=description,
            confidence=confidence,
            source=source,
            provenance=provenance or {},
            metadata=metadata or {},
            slots=slots or {},
            color=color,
            created_at=now,
            updated_at=now,
        )

        logger.debug(f"Added semantic node: {label} ({node_type.value})")
        return node

    def get_node(self, node_id: str, record_access: bool = True) -> SemanticNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID
            record_access: Whether to record this access for activation

        Returns:
            The node or None if not found
        """
        row = self.db.fetchone(
            "SELECT * FROM semantic_nodes WHERE id = ?",
            (node_id,)
        )

        if not row:
            return None

        node = SemanticNode.from_row(row)

        if record_access:
            self._record_access(node_id)

        return node

    def update_node(
        self,
        node_id: str,
        label: str | None = None,
        description: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        slots: dict[str, Any] | None = None,
    ) -> SemanticNode | None:
        """Update an existing node.

        Args:
            node_id: ID of node to update
            label: New label (if provided)
            description: New description (if provided)
            confidence: New confidence (if provided)
            metadata: Metadata to merge (if provided)
            slots: Slots to merge (if provided)

        Returns:
            Updated node or None if not found
        """
        current = self.get_node(node_id, record_access=False)
        if not current:
            return None

        updates = []
        params = []

        if label is not None:
            updates.append("label = ?")
            params.append(label)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)

        if metadata is not None:
            merged = {**current.metadata, **metadata}
            updates.append("metadata = ?")
            params.append(json.dumps(merged))

        if slots is not None:
            merged = {**current.slots, **slots}
            updates.append("slots = ?")
            params.append(json.dumps(merged))

        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(node_id)

            sql = f"UPDATE semantic_nodes SET {', '.join(updates)} WHERE id = ?"
            self.db.execute(sql, tuple(params))

        return self.get_node(node_id, record_access=False)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node.

        Args:
            node_id: ID of node to delete

        Returns:
            True if deleted, False if not found
        """
        # Delete edges
        self.db.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))

        # Delete embedding
        self.db.execute("DELETE FROM embeddings WHERE node_id = ?", (node_id,))

        # Delete node
        result = self.db.execute("DELETE FROM semantic_nodes WHERE id = ?", (node_id,))

        deleted = result.rowcount > 0
        if deleted:
            logger.debug(f"Deleted semantic node: {node_id}")

        return deleted

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            weight: Edge weight
            metadata: Edge metadata

        Returns:
            True if created, False if already exists
        """
        try:
            self.db.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, target_id, edge_type, weight, json.dumps(metadata or {}))
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to add edge: {e}")
            return False

    def get_edges(self, node_id: str, direction: str = "both") -> list[dict[str, Any]]:
        """Get edges for a node.

        Args:
            node_id: Node ID
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of edge dicts
        """
        edges = []

        if direction in ("outgoing", "both"):
            rows = self.db.fetchall(
                "SELECT * FROM edges WHERE source_id = ?",
                (node_id,)
            )
            edges.extend(rows)

        if direction in ("incoming", "both"):
            rows = self.db.fetchall(
                "SELECT * FROM edges WHERE target_id = ?",
                (node_id,)
            )
            edges.extend(rows)

        return edges

    def get_children(self, node_id: str) -> list[SemanticNode]:
        """Get child nodes via PARENT edges.

        Args:
            node_id: Parent node ID

        Returns:
            List of child nodes
        """
        rows = self.db.fetchall(
            """SELECT sn.* FROM semantic_nodes sn
               JOIN edges e ON sn.id = e.source_id
               WHERE e.target_id = ? AND e.edge_type = 'PARENT'""",
            (node_id,)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def _record_access(self, node_id: str) -> None:
        """Record node access and update activation.

        Args:
            node_id: ID of accessed node
        """
        now = datetime.now()

        # Update access count and last access
        self.db.execute(
            """UPDATE semantic_nodes
               SET access_count = access_count + 1,
                   last_access = ?,
                   updated_at = ?
               WHERE id = ?""",
            (now.isoformat(), now.isoformat(), node_id)
        )

        # Record in retrieval history
        self.db.execute(
            "INSERT INTO retrieval_history (node_id, retrieved_at) VALUES (?, ?)",
            (node_id, now.isoformat())
        )

        # Update base-level activation using ACT-R formula
        self._update_activation(node_id)

    def _update_activation(self, node_id: str) -> None:
        """Update base-level activation using ACT-R formula.

        B = ln(sum(t_i^-d)) where t_i is time since i-th retrieval

        Args:
            node_id: Node ID to update
        """
        # Get retrieval history
        rows = self.db.fetchall(
            "SELECT retrieved_at FROM retrieval_history WHERE node_id = ? ORDER BY retrieved_at DESC LIMIT 100",
            (node_id,)
        )

        if not rows:
            return

        now = datetime.now()
        activation_sum = 0.0

        for row in rows:
            retrieved_at = datetime.fromisoformat(row['retrieved_at'])
            time_diff = (now - retrieved_at).total_seconds()

            # Minimum time difference to avoid log(0)
            time_diff = max(time_diff, 1.0)

            # ACT-R: t^-d
            activation_sum += time_diff ** (-self.DECAY_RATE)

        # Base-level activation = ln(sum)
        base_level = math.log(activation_sum) if activation_sum > 0 else 0.0

        self.db.execute(
            "UPDATE semantic_nodes SET base_level_activation = ? WHERE id = ?",
            (base_level, node_id)
        )

    def update_confidence(
        self,
        node_id: str,
        new_evidence: bool,
        source_reliability: float = 0.5,
    ) -> float | None:
        """Update confidence using Bayesian update.

        Args:
            node_id: Node to update
            new_evidence: Whether new evidence supports the assertion
            source_reliability: How reliable is the evidence source

        Returns:
            New confidence or None if node not found
        """
        node = self.get_node(node_id, record_access=False)
        if not node:
            return None

        prior = node.confidence

        if new_evidence:
            # P(H|E) = P(E|H)P(H) / P(E)
            # Simplified: increase confidence proportional to source reliability
            posterior = prior + (1 - prior) * source_reliability * 0.3
        else:
            # Decrease confidence
            posterior = prior * (1 - source_reliability * 0.3)

        # Clamp to [0.01, 0.99]
        posterior = max(0.01, min(0.99, posterior))

        self.db.execute(
            """UPDATE semantic_nodes
               SET confidence = ?, evidence_count = evidence_count + 1,
                   updated_at = ?
               WHERE id = ?""",
            (posterior, datetime.now().isoformat(), node_id)
        )

        logger.debug(f"Updated confidence for {node_id}: {prior:.2f} -> {posterior:.2f}")
        return posterior

    # =========================================================================
    # Search Methods
    # =========================================================================

    def search_by_label(self, query: str, limit: int = 10) -> list[SemanticNode]:
        """Search nodes by label (case-insensitive substring).

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching nodes
        """
        rows = self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE label LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def search_by_type(
        self,
        node_type: SemanticNodeType,
        limit: int = 50,
    ) -> list[SemanticNode]:
        """Get nodes by type.

        Args:
            node_type: Type to filter by
            limit: Max results

        Returns:
            List of matching nodes
        """
        rows = self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type = ? LIMIT ?",
            (node_type.value, limit)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def search_fts(self, query: str, limit: int = 10) -> list[SemanticNode]:
        """Full-text search using FTS5.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching nodes ranked by relevance
        """
        rows = self.db.fetchall(
            """SELECT sn.* FROM semantic_nodes sn
               JOIN memory_fts fts ON sn.id = fts.node_id
               WHERE memory_fts MATCH ?
               ORDER BY bm25(memory_fts)
               LIMIT ?""",
            (query, limit)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def get_recent_nodes(self, limit: int = 20) -> list[SemanticNode]:
        """Get recently updated nodes.

        Args:
            limit: Max results

        Returns:
            List of nodes sorted by updated_at desc
        """
        rows = self.db.fetchall(
            "SELECT * FROM semantic_nodes ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def get_high_activation_nodes(self, limit: int = 20) -> list[SemanticNode]:
        """Get nodes with highest activation.

        Args:
            limit: Max results

        Returns:
            List of nodes sorted by activation desc
        """
        rows = self.db.fetchall(
            "SELECT * FROM semantic_nodes ORDER BY base_level_activation DESC LIMIT ?",
            (limit,)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def get_low_confidence_nodes(
        self,
        threshold: float = 0.4,
        limit: int = 20,
    ) -> list[SemanticNode]:
        """Get nodes with low confidence (knowledge gaps).

        Args:
            threshold: Confidence threshold
            limit: Max results

        Returns:
            List of low-confidence nodes
        """
        rows = self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE confidence < ? ORDER BY confidence ASC LIMIT ?",
            (threshold, limit)
        )
        return [SemanticNode.from_row(row) for row in rows]

    def get_all_nodes_generator(
        self,
        batch_size: int = 100,
    ) -> Generator[SemanticNode, None, None]:
        """Iterate all nodes with minimal memory usage.

        Args:
            batch_size: Number of rows per batch

        Yields:
            SemanticNode instances
        """
        for row in self.db.fetchall_generator(
            "SELECT * FROM semantic_nodes ORDER BY id",
            batch_size=batch_size
        ):
            yield SemanticNode.from_row(row)

    def get_stats(self) -> dict[str, Any]:
        """Get semantic memory statistics.

        Returns:
            Statistics dict
        """
        total = self.db.fetchone("SELECT COUNT(*) as count FROM semantic_nodes")

        type_counts = {}
        for node_type in SemanticNodeType:
            count = self.db.fetchone(
                "SELECT COUNT(*) as count FROM semantic_nodes WHERE node_type = ?",
                (node_type.value,)
            )
            type_counts[node_type.value] = count['count'] if count else 0

        avg_confidence = self.db.fetchone(
            "SELECT AVG(confidence) as avg FROM semantic_nodes"
        )

        avg_activation = self.db.fetchone(
            "SELECT AVG(base_level_activation) as avg FROM semantic_nodes"
        )

        return {
            "total_nodes": total['count'] if total else 0,
            "nodes_by_type": type_counts,
            "avg_confidence": avg_confidence['avg'] if avg_confidence and avg_confidence['avg'] else 0.0,
            "avg_activation": avg_activation['avg'] if avg_activation and avg_activation['avg'] else 0.0,
        }


# Singleton
_semantic_memory: SemanticMemory | None = None
_memory_lock = threading.Lock()


def get_semantic_memory(db: MemoryDatabase | None = None) -> SemanticMemory:
    """Get or create the semantic memory singleton.

    Args:
        db: Optional memory database

    Returns:
        The semantic memory singleton
    """
    global _semantic_memory

    if _semantic_memory is None:
        with _memory_lock:
            if _semantic_memory is None:
                _semantic_memory = SemanticMemory(db)

    return _semantic_memory
