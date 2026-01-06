# MEMORY ARCHITECTURE: 8GB Resource-Constrained Implementation

## Practical Implementation for M2 Mac Mini (8GB RAM)

**Version**: 1.0
**Date**: January 2025
**Hardware Target**: M2 Mac Mini with 8GB Unified Memory
**Scope**: Production-ready implementation of the MindGraph cognitive memory system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Storage Architecture](#2-storage-architecture)
3. [Tri-Memory System Implementation](#3-tri-memory-system-implementation)
4. [Working Memory with Bounded Capacity](#4-working-memory-with-bounded-capacity)
5. [Embedding System with Memory Limits](#5-embedding-system-with-memory-limits)
6. [Memory Dynamics and Consolidation](#6-memory-dynamics-and-consolidation)
7. [Spreading Activation (Bounded)](#7-spreading-activation-bounded)
8. [Meta-Cognition Layer](#8-meta-cognition-layer)
9. [Memory Budget Allocations](#9-memory-budget-allocations)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Implementation Priority Order](#11-implementation-priority-order)

---

## 1. Executive Summary

### The Challenge

The visionary memory architecture describes a system that could easily consume gigabytes of RAM:
- In-memory embedding indices
- Full graph structures in memory
- Unbounded spreading activation
- Multiple concurrent inference chains

On an **8GB M2 Mac Mini**, we must assume:
- ~4GB available for our application (OS and other processes take the rest)
- ~2GB safe working memory for the memory system
- Embedding model itself takes ~500MB
- We need headroom for spikes and concurrent operations

### Our Approach: Disk-First, Memory-Cached

| Component | Visionary Approach | 8GB Implementation |
|-----------|-------------------|-------------------|
| **Node Storage** | In-memory dict | SQLite with LRU cache |
| **Embeddings** | In-memory NumPy array | SQLite BLOB + mmap index |
| **Graph Edges** | In-memory adjacency | SQLite adjacency table |
| **Text Search** | Linear scan | SQLite FTS5 |
| **Working Memory** | Unbounded | Hard limit: 100 items |
| **Embedding Cache** | Full dataset | LRU cache: 50MB max |
| **Activation Spread** | Full graph | Bounded: 3 hops, 50 nodes |

### Key Principles

1. **SQLite is the source of truth** - Everything persists to SQLite tables
2. **Lazy loading everywhere** - Never load what you don't need
3. **Strict memory budgets** - Every cache has a hard limit
4. **Batch API calls** - Never embed one item when you can batch
5. **Generator patterns** - Yield results, don't collect them
6. **Cursor-based pagination** - No full dataset scans

---

## 2. Storage Architecture

### 2.1 Database Schema

All memory components share a single SQLite database with FTS5 for full-text search.

```sql
-- memory.db

-- =============================================================================
-- Core Tables
-- =============================================================================

-- Episodic Memory: Conversation episodes and experiences
CREATE TABLE episodic_memories (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    duration_seconds INTEGER,
    summary TEXT NOT NULL,
    compressed_transcript BLOB,  -- gzip compressed

    -- Context
    spatial_context TEXT,  -- project/domain
    social_context TEXT,   -- JSON array of user IDs

    -- Emotional tagging
    emotional_valence REAL DEFAULT 0.0,  -- -1 to +1
    arousal_level REAL DEFAULT 0.0,      -- 0 to 1

    -- Memory strength (Ebbinghaus)
    encoding_strength REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved DATETIME,
    decay_rate REAL DEFAULT 0.1,

    -- Consolidation state
    is_consolidated BOOLEAN DEFAULT FALSE,
    consolidated_at DATETIME,

    -- Temporal links (stored as JSON for flexibility)
    preceded_by TEXT,  -- episode ID
    followed_by TEXT,  -- episode ID

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_episodic_timestamp ON episodic_memories(timestamp DESC);
CREATE INDEX idx_episodic_consolidated ON episodic_memories(is_consolidated);
CREATE INDEX idx_episodic_spatial ON episodic_memories(spatial_context);

-- Semantic Memory: Facts, concepts, entities
CREATE TABLE semantic_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,  -- CONCEPT, FACT, ENTITY, SCHEMA, FRAME
    label TEXT NOT NULL,
    description TEXT,
    formal_definition TEXT,

    -- Epistemics
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    source_reliability REAL DEFAULT 0.5,

    -- ACT-R activation
    base_level_activation REAL DEFAULT 0.0,
    last_access DATETIME,
    access_count INTEGER DEFAULT 0,

    -- Provenance (JSON)
    derived_from_episodes TEXT,  -- JSON array of episode IDs
    source TEXT,  -- 'conversation', 'system', 'inference'
    provenance TEXT,  -- JSON object

    -- Slot values for frames (JSON)
    slots TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_semantic_type ON semantic_nodes(node_type);
CREATE INDEX idx_semantic_confidence ON semantic_nodes(confidence DESC);
CREATE INDEX idx_semantic_activation ON semantic_nodes(base_level_activation DESC);

-- Procedural Memory: Skills and action patterns
CREATE TABLE procedural_skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,

    -- Trigger conditions (JSON)
    trigger_pattern TEXT,
    preconditions TEXT,  -- JSON array
    goal_relevance TEXT, -- JSON array of goal IDs

    -- Execution template (JSON)
    steps TEXT,  -- JSON array of SkillStep
    decision_points TEXT,  -- JSON array
    tool_bindings TEXT,  -- JSON object

    -- Performance metrics
    success_rate REAL DEFAULT 0.5,
    execution_count INTEGER DEFAULT 0,
    avg_execution_time_ms INTEGER,
    failure_modes TEXT,  -- JSON array

    -- Learning
    learned_from TEXT,  -- JSON array of episode IDs
    is_compiled BOOLEAN DEFAULT FALSE,
    chunking_level INTEGER DEFAULT 0,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_procedural_success ON procedural_skills(success_rate DESC);
CREATE INDEX idx_procedural_compiled ON procedural_skills(is_compiled);

-- =============================================================================
-- Graph Edges (Adjacency Table)
-- =============================================================================

CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,  -- ISA, HAS_PART, CAUSES, ENABLES, etc.

    weight REAL DEFAULT 1.0,
    confidence REAL DEFAULT 1.0,

    -- Temporal validity
    valid_from DATETIME,
    valid_to DATETIME,

    metadata TEXT,  -- JSON object

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);

-- =============================================================================
-- Embeddings Table (Binary Storage)
-- =============================================================================

CREATE TABLE embeddings (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,  -- 'episodic', 'semantic', 'procedural'
    embedding BLOB NOT NULL,  -- 384-dim float32 = 1536 bytes
    model_version TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (node_id) REFERENCES semantic_nodes(id) ON DELETE CASCADE
);

-- =============================================================================
-- Full-Text Search (FTS5)
-- =============================================================================

CREATE VIRTUAL TABLE memory_fts USING fts5(
    node_id,
    node_type,
    label,
    description,
    content='semantic_nodes',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER semantic_ai AFTER INSERT ON semantic_nodes BEGIN
    INSERT INTO memory_fts(rowid, node_id, node_type, label, description)
    VALUES (new.rowid, new.id, new.node_type, new.label, new.description);
END;

CREATE TRIGGER semantic_ad AFTER DELETE ON semantic_nodes BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, node_id, node_type, label, description)
    VALUES ('delete', old.rowid, old.id, old.node_type, old.label, old.description);
END;

CREATE TRIGGER semantic_au AFTER UPDATE ON semantic_nodes BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, node_id, node_type, label, description)
    VALUES ('delete', old.rowid, old.id, old.node_type, old.label, old.description);
    INSERT INTO memory_fts(rowid, node_id, node_type, label, description)
    VALUES (new.rowid, new.id, new.node_type, new.label, new.description);
END;

-- =============================================================================
-- Working Memory (Transient, but persisted for crash recovery)
-- =============================================================================

CREATE TABLE working_memory (
    slot_id INTEGER PRIMARY KEY,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    activation REAL NOT NULL,
    loaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Limit enforced at application level
    CHECK (slot_id BETWEEN 0 AND 99)  -- Hard limit: 100 items
);

CREATE INDEX idx_working_activation ON working_memory(activation DESC);

-- =============================================================================
-- Retrieval History (For decay calculations)
-- =============================================================================

CREATE TABLE retrieval_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    retrieved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    context TEXT,  -- What triggered the retrieval

    FOREIGN KEY (node_id) REFERENCES semantic_nodes(id) ON DELETE CASCADE
);

CREATE INDEX idx_retrieval_node ON retrieval_history(node_id);
CREATE INDEX idx_retrieval_time ON retrieval_history(retrieved_at DESC);

-- Cleanup old retrieval history (keep last 1000 per node)
CREATE TRIGGER cleanup_retrieval_history AFTER INSERT ON retrieval_history
BEGIN
    DELETE FROM retrieval_history
    WHERE node_id = new.node_id
    AND id NOT IN (
        SELECT id FROM retrieval_history
        WHERE node_id = new.node_id
        ORDER BY retrieved_at DESC
        LIMIT 1000
    );
END;
```

### 2.2 Database Connection Management

```python
# backend/services/memory_db.py

"""SQLite-backed memory storage with connection pooling."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryDatabase:
    """Thread-safe SQLite connection manager for memory storage.

    Features:
    - Connection pooling per thread
    - WAL mode for concurrent reads
    - Automatic schema initialization
    - Context manager for transactions
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._local.conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn.row_factory = sqlite3.Row

        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for read operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._init_lock:
            if self._initialized:
                return

            schema_path = Path(__file__).parent / "memory_schema.sql"
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = f.read()
            else:
                # Inline schema for portability
                schema = _MEMORY_SCHEMA

            with self.transaction() as conn:
                conn.executescript(schema)

            self._initialized = True
            logger.info(f"Memory database initialized at {self.db_path}")


# Singleton
_db: Optional[MemoryDatabase] = None
_db_lock = threading.Lock()


def get_memory_db(db_path: Optional[Path] = None) -> MemoryDatabase:
    """Get singleton MemoryDatabase instance."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                if db_path is None:
                    db_path = Path("memory/graph/memory.db")
                db_path.parent.mkdir(parents=True, exist_ok=True)
                _db = MemoryDatabase(db_path)
                _db.initialize_schema()
    return _db
```

---

## 3. Tri-Memory System Implementation

### 3.1 Episodic Memory Store

```python
# backend/services/episodic_memory.py

"""Episodic Memory Store - SQLite-backed with lazy loading."""

import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Generator, List, Optional
import sqlite3

from .memory_db import get_memory_db

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """In-memory representation of an episodic memory."""
    id: str
    timestamp: datetime
    summary: str
    duration_seconds: Optional[int] = None

    # Context
    spatial_context: Optional[str] = None
    social_context: List[str] = field(default_factory=list)

    # Emotional
    emotional_valence: float = 0.0
    arousal_level: float = 0.0

    # Memory strength
    encoding_strength: float = 0.5
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    decay_rate: float = 0.1

    # Consolidation
    is_consolidated: bool = False

    # Transcript loaded on demand
    _transcript: Optional[str] = field(default=None, repr=False)


class EpisodicMemoryStore:
    """SQLite-backed episodic memory with lazy loading.

    Memory Constraints:
    - Episodes loaded on-demand, not cached
    - Transcripts loaded only when explicitly requested
    - Results returned as generators for large queries
    - Batch operations for bulk inserts
    """

    BATCH_SIZE = 50  # For bulk operations

    def __init__(self):
        self.db = get_memory_db()

    def store_episode(self, episode: Episode, transcript: Optional[str] = None) -> None:
        """Store an episode. Transcript is compressed before storage."""
        compressed_transcript = None
        if transcript:
            compressed_transcript = gzip.compress(transcript.encode('utf-8'))

        with self.db.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodic_memories (
                    id, timestamp, duration_seconds, summary, compressed_transcript,
                    spatial_context, social_context, emotional_valence, arousal_level,
                    encoding_strength, retrieval_count, last_retrieved, decay_rate,
                    is_consolidated, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                episode.id,
                episode.timestamp.isoformat(),
                episode.duration_seconds,
                episode.summary,
                compressed_transcript,
                episode.spatial_context,
                json.dumps(episode.social_context),
                episode.emotional_valence,
                episode.arousal_level,
                episode.encoding_strength,
                episode.retrieval_count,
                episode.last_retrieved.isoformat() if episode.last_retrieved else None,
                episode.decay_rate,
                episode.is_consolidated,
            ))

    def get_episode(self, episode_id: str, load_transcript: bool = False) -> Optional[Episode]:
        """Load a single episode by ID."""
        with self.db.cursor() as cursor:
            if load_transcript:
                cursor.execute(
                    "SELECT * FROM episodic_memories WHERE id = ?",
                    (episode_id,)
                )
            else:
                cursor.execute("""
                    SELECT id, timestamp, duration_seconds, summary, spatial_context,
                           social_context, emotional_valence, arousal_level,
                           encoding_strength, retrieval_count, last_retrieved,
                           decay_rate, is_consolidated
                    FROM episodic_memories WHERE id = ?
                """, (episode_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_episode(dict(row), load_transcript)

    def get_recent_episodes(
        self,
        limit: int = 20,
        offset: int = 0,
        spatial_context: Optional[str] = None,
    ) -> Generator[Episode, None, None]:
        """Get recent episodes with cursor-based pagination.

        Yields episodes one at a time to avoid loading all into memory.
        """
        with self.db.cursor() as cursor:
            if spatial_context:
                cursor.execute("""
                    SELECT id, timestamp, duration_seconds, summary, spatial_context,
                           social_context, emotional_valence, arousal_level,
                           encoding_strength, retrieval_count, last_retrieved,
                           decay_rate, is_consolidated
                    FROM episodic_memories
                    WHERE spatial_context = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (spatial_context, limit, offset))
            else:
                cursor.execute("""
                    SELECT id, timestamp, duration_seconds, summary, spatial_context,
                           social_context, emotional_valence, arousal_level,
                           encoding_strength, retrieval_count, last_retrieved,
                           decay_rate, is_consolidated
                    FROM episodic_memories
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

            for row in cursor:
                yield self._row_to_episode(dict(row), load_transcript=False)

    def get_unconsolidated_episodes(
        self,
        min_age_hours: int = 24,
        batch_size: int = 50,
    ) -> Generator[Episode, None, None]:
        """Get episodes ready for consolidation.

        Only returns episodes that:
        - Are not yet consolidated
        - Are at least min_age_hours old
        - Have sufficient encoding strength
        """
        cutoff = (datetime.now() - timedelta(hours=min_age_hours)).isoformat()

        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT id, timestamp, duration_seconds, summary, spatial_context,
                       social_context, emotional_valence, arousal_level,
                       encoding_strength, retrieval_count, last_retrieved,
                       decay_rate, is_consolidated
                FROM episodic_memories
                WHERE is_consolidated = FALSE
                  AND timestamp < ?
                  AND encoding_strength > 0.3
                ORDER BY encoding_strength DESC
                LIMIT ?
            """, (cutoff, batch_size))

            for row in cursor:
                yield self._row_to_episode(dict(row), load_transcript=False)

    def record_retrieval(self, episode_id: str) -> None:
        """Record that an episode was retrieved (for decay calculations)."""
        with self.db.transaction() as conn:
            conn.execute("""
                UPDATE episodic_memories
                SET retrieval_count = retrieval_count + 1,
                    last_retrieved = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (episode_id,))

            conn.execute("""
                INSERT INTO retrieval_history (node_id, context)
                VALUES (?, 'episodic_retrieval')
            """, (episode_id,))

    def mark_consolidated(self, episode_id: str) -> None:
        """Mark an episode as consolidated into semantic memory."""
        with self.db.transaction() as conn:
            conn.execute("""
                UPDATE episodic_memories
                SET is_consolidated = TRUE,
                    consolidated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (episode_id,))

    def _row_to_episode(self, row: dict, load_transcript: bool) -> Episode:
        """Convert database row to Episode object."""
        episode = Episode(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            summary=row['summary'],
            duration_seconds=row.get('duration_seconds'),
            spatial_context=row.get('spatial_context'),
            social_context=json.loads(row.get('social_context') or '[]'),
            emotional_valence=row.get('emotional_valence', 0.0),
            arousal_level=row.get('arousal_level', 0.0),
            encoding_strength=row.get('encoding_strength', 0.5),
            retrieval_count=row.get('retrieval_count', 0),
            last_retrieved=datetime.fromisoformat(row['last_retrieved']) if row.get('last_retrieved') else None,
            decay_rate=row.get('decay_rate', 0.1),
            is_consolidated=bool(row.get('is_consolidated', False)),
        )

        if load_transcript and row.get('compressed_transcript'):
            episode._transcript = gzip.decompress(row['compressed_transcript']).decode('utf-8')

        return episode
```

### 3.2 Semantic Memory Store

```python
# backend/services/semantic_memory.py

"""Semantic Memory Store - SQLite-backed with FTS5 search."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Generator, List, Optional, Any
import sqlite3

from .memory_db import get_memory_db

logger = logging.getLogger(__name__)


class SemanticNodeType(str, Enum):
    CONCEPT = "concept"
    FACT = "fact"
    ENTITY = "entity"
    SCHEMA = "schema"
    FRAME = "frame"
    IDENTITY = "identity"
    PREFERENCE = "preference"
    GOAL = "goal"
    DECISION = "decision"
    RELATIONSHIP = "relationship"


@dataclass
class SemanticNode:
    """In-memory representation of a semantic memory node."""
    id: str
    node_type: SemanticNodeType
    label: str
    description: Optional[str] = None
    formal_definition: Optional[str] = None

    # Epistemics
    confidence: float = 0.5
    evidence_count: int = 1
    source_reliability: float = 0.5

    # Activation (ACT-R style)
    base_level_activation: float = 0.0
    last_access: Optional[datetime] = None
    access_count: int = 0

    # Provenance
    derived_from_episodes: List[str] = field(default_factory=list)
    source: str = "conversation"
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Frame slots
    slots: Dict[str, Any] = field(default_factory=dict)


class SemanticMemoryStore:
    """SQLite-backed semantic memory with FTS5 search.

    Memory Constraints:
    - Nodes loaded on-demand
    - FTS5 for text search (no in-memory index)
    - LRU cache for frequently accessed nodes (see WorkingMemory)
    - Batch operations for bulk updates
    """

    def __init__(self):
        self.db = get_memory_db()

    def store_node(self, node: SemanticNode) -> None:
        """Store or update a semantic node."""
        with self.db.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO semantic_nodes (
                    id, node_type, label, description, formal_definition,
                    confidence, evidence_count, source_reliability,
                    base_level_activation, last_access, access_count,
                    derived_from_episodes, source, provenance, slots,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                node.id,
                node.node_type.value,
                node.label,
                node.description,
                node.formal_definition,
                node.confidence,
                node.evidence_count,
                node.source_reliability,
                node.base_level_activation,
                node.last_access.isoformat() if node.last_access else None,
                node.access_count,
                json.dumps(node.derived_from_episodes),
                node.source,
                json.dumps(node.provenance),
                json.dumps(node.slots),
            ))

    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        """Load a single node by ID."""
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM semantic_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_node(dict(row))

    def search_fts(
        self,
        query: str,
        limit: int = 20,
        node_types: Optional[List[SemanticNodeType]] = None,
    ) -> Generator[SemanticNode, None, None]:
        """Full-text search using FTS5.

        Much faster than embedding search for keyword queries.
        """
        with self.db.cursor() as cursor:
            if node_types:
                type_filter = " AND node_type IN ({})".format(
                    ",".join(f"'{t.value}'" for t in node_types)
                )
            else:
                type_filter = ""

            # FTS5 MATCH query
            cursor.execute(f"""
                SELECT s.*
                FROM semantic_nodes s
                JOIN memory_fts f ON s.id = f.node_id
                WHERE memory_fts MATCH ?
                {type_filter}
                ORDER BY rank
                LIMIT ?
            """, (query, limit))

            for row in cursor:
                yield self._row_to_node(dict(row))

    def get_nodes_by_type(
        self,
        node_type: SemanticNodeType,
        limit: int = 100,
        offset: int = 0,
        min_confidence: float = 0.0,
    ) -> Generator[SemanticNode, None, None]:
        """Get nodes of a specific type with pagination."""
        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM semantic_nodes
                WHERE node_type = ?
                  AND confidence >= ?
                ORDER BY base_level_activation DESC
                LIMIT ? OFFSET ?
            """, (node_type.value, min_confidence, limit, offset))

            for row in cursor:
                yield self._row_to_node(dict(row))

    def get_high_activation_nodes(
        self,
        limit: int = 50,
        min_activation: float = 0.0,
    ) -> Generator[SemanticNode, None, None]:
        """Get nodes with highest activation (for working memory priming)."""
        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM semantic_nodes
                WHERE base_level_activation >= ?
                ORDER BY base_level_activation DESC
                LIMIT ?
            """, (min_activation, limit))

            for row in cursor:
                yield self._row_to_node(dict(row))

    def record_access(self, node_id: str) -> None:
        """Record that a node was accessed (updates activation)."""
        with self.db.transaction() as conn:
            conn.execute("""
                UPDATE semantic_nodes
                SET access_count = access_count + 1,
                    last_access = CURRENT_TIMESTAMP,
                    base_level_activation = base_level_activation + 0.1
                WHERE id = ?
            """, (node_id,))

            conn.execute("""
                INSERT INTO retrieval_history (node_id, context)
                VALUES (?, 'semantic_access')
            """, (node_id,))

    def update_confidence(
        self,
        node_id: str,
        new_evidence_strength: float,
        is_supporting: bool = True,
    ) -> None:
        """Bayesian update of node confidence based on new evidence."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT confidence, evidence_count FROM semantic_nodes WHERE id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            current_conf = row[0]
            evidence_count = row[1]

            # Simple Bayesian update
            if is_supporting:
                # Evidence supports the belief
                likelihood = new_evidence_strength
            else:
                # Evidence contradicts the belief
                likelihood = 1 - new_evidence_strength

            # Posterior = prior * likelihood / (prior * likelihood + (1-prior) * (1-likelihood))
            posterior = (current_conf * likelihood) / (
                current_conf * likelihood + (1 - current_conf) * (1 - likelihood)
            )

            conn.execute("""
                UPDATE semantic_nodes
                SET confidence = ?,
                    evidence_count = evidence_count + 1
                WHERE id = ?
            """, (posterior, node_id))

    def decay_activations(self, decay_factor: float = 0.95) -> int:
        """Apply decay to all node activations. Run periodically."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE semantic_nodes
                SET base_level_activation = base_level_activation * ?
                WHERE base_level_activation > 0.01
            """, (decay_factor,))
            return cursor.rowcount

    def _row_to_node(self, row: dict) -> SemanticNode:
        """Convert database row to SemanticNode object."""
        return SemanticNode(
            id=row['id'],
            node_type=SemanticNodeType(row['node_type']),
            label=row['label'],
            description=row.get('description'),
            formal_definition=row.get('formal_definition'),
            confidence=row.get('confidence', 0.5),
            evidence_count=row.get('evidence_count', 1),
            source_reliability=row.get('source_reliability', 0.5),
            base_level_activation=row.get('base_level_activation', 0.0),
            last_access=datetime.fromisoformat(row['last_access']) if row.get('last_access') else None,
            access_count=row.get('access_count', 0),
            derived_from_episodes=json.loads(row.get('derived_from_episodes') or '[]'),
            source=row.get('source', 'unknown'),
            provenance=json.loads(row.get('provenance') or '{}'),
            slots=json.loads(row.get('slots') or '{}'),
        )
```

### 3.3 Procedural Memory Store

```python
# backend/services/procedural_memory.py

"""Procedural Memory Store - Skills and action patterns."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generator, List, Optional, Any
import sqlite3

from .memory_db import get_memory_db

logger = logging.getLogger(__name__)


@dataclass
class SkillStep:
    """A single step in a procedural skill."""
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None
    fallback: Optional[str] = None  # Step to go to on failure


@dataclass
class ProceduralSkill:
    """In-memory representation of a procedural skill."""
    id: str
    name: str
    description: Optional[str] = None

    # Trigger
    trigger_pattern: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    goal_relevance: List[str] = field(default_factory=list)

    # Execution
    steps: List[SkillStep] = field(default_factory=list)
    decision_points: List[Dict] = field(default_factory=list)
    tool_bindings: Dict[str, str] = field(default_factory=dict)

    # Performance
    success_rate: float = 0.5
    execution_count: int = 0
    avg_execution_time_ms: Optional[int] = None
    failure_modes: List[str] = field(default_factory=list)

    # Learning
    learned_from: List[str] = field(default_factory=list)
    is_compiled: bool = False
    chunking_level: int = 0


class ProceduralMemoryStore:
    """SQLite-backed procedural memory for skills and patterns.

    Memory Constraints:
    - Skills loaded on-demand
    - Pattern matching done via SQL LIKE queries
    - No in-memory skill cache (skills are relatively small)
    """

    def __init__(self):
        self.db = get_memory_db()

    def store_skill(self, skill: ProceduralSkill) -> None:
        """Store or update a procedural skill."""
        with self.db.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO procedural_skills (
                    id, name, description, trigger_pattern, preconditions,
                    goal_relevance, steps, decision_points, tool_bindings,
                    success_rate, execution_count, avg_execution_time_ms,
                    failure_modes, learned_from, is_compiled, chunking_level,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                skill.id,
                skill.name,
                skill.description,
                skill.trigger_pattern,
                json.dumps(skill.preconditions),
                json.dumps(skill.goal_relevance),
                json.dumps([{
                    'action': s.action,
                    'parameters': s.parameters,
                    'expected_outcome': s.expected_outcome,
                    'fallback': s.fallback,
                } for s in skill.steps]),
                json.dumps(skill.decision_points),
                json.dumps(skill.tool_bindings),
                skill.success_rate,
                skill.execution_count,
                skill.avg_execution_time_ms,
                json.dumps(skill.failure_modes),
                json.dumps(skill.learned_from),
                skill.is_compiled,
                skill.chunking_level,
            ))

    def get_skill(self, skill_id: str) -> Optional[ProceduralSkill]:
        """Load a single skill by ID."""
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM procedural_skills WHERE id = ?", (skill_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_skill(dict(row))

    def find_matching_skills(
        self,
        context: str,
        goal: Optional[str] = None,
        limit: int = 10,
    ) -> Generator[ProceduralSkill, None, None]:
        """Find skills that match the current context.

        Uses pattern matching on trigger_pattern and goal_relevance.
        """
        with self.db.cursor() as cursor:
            if goal:
                cursor.execute("""
                    SELECT * FROM procedural_skills
                    WHERE (trigger_pattern IS NULL OR ? LIKE '%' || trigger_pattern || '%')
                      AND goal_relevance LIKE ?
                    ORDER BY success_rate DESC, execution_count DESC
                    LIMIT ?
                """, (context, f'%{goal}%', limit))
            else:
                cursor.execute("""
                    SELECT * FROM procedural_skills
                    WHERE trigger_pattern IS NULL OR ? LIKE '%' || trigger_pattern || '%'
                    ORDER BY success_rate DESC, execution_count DESC
                    LIMIT ?
                """, (context, limit))

            for row in cursor:
                yield self._row_to_skill(dict(row))

    def record_execution(
        self,
        skill_id: str,
        success: bool,
        execution_time_ms: int,
        failure_mode: Optional[str] = None,
    ) -> None:
        """Record a skill execution for learning."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT execution_count, success_rate, avg_execution_time_ms, failure_modes FROM procedural_skills WHERE id = ?",
                (skill_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            exec_count = row[0]
            old_success_rate = row[1]
            old_avg_time = row[2] or execution_time_ms
            failure_modes = json.loads(row[3] or '[]')

            # Update running averages
            new_exec_count = exec_count + 1
            new_success_rate = (old_success_rate * exec_count + (1 if success else 0)) / new_exec_count
            new_avg_time = int((old_avg_time * exec_count + execution_time_ms) / new_exec_count)

            # Track failure modes
            if not success and failure_mode and failure_mode not in failure_modes:
                failure_modes.append(failure_mode)
                failure_modes = failure_modes[-10:]  # Keep last 10

            conn.execute("""
                UPDATE procedural_skills
                SET execution_count = ?,
                    success_rate = ?,
                    avg_execution_time_ms = ?,
                    failure_modes = ?
                WHERE id = ?
            """, (new_exec_count, new_success_rate, new_avg_time, json.dumps(failure_modes), skill_id))

    def _row_to_skill(self, row: dict) -> ProceduralSkill:
        """Convert database row to ProceduralSkill object."""
        steps_data = json.loads(row.get('steps') or '[]')
        steps = [
            SkillStep(
                action=s['action'],
                parameters=s.get('parameters', {}),
                expected_outcome=s.get('expected_outcome'),
                fallback=s.get('fallback'),
            )
            for s in steps_data
        ]

        return ProceduralSkill(
            id=row['id'],
            name=row['name'],
            description=row.get('description'),
            trigger_pattern=row.get('trigger_pattern'),
            preconditions=json.loads(row.get('preconditions') or '[]'),
            goal_relevance=json.loads(row.get('goal_relevance') or '[]'),
            steps=steps,
            decision_points=json.loads(row.get('decision_points') or '[]'),
            tool_bindings=json.loads(row.get('tool_bindings') or '{}'),
            success_rate=row.get('success_rate', 0.5),
            execution_count=row.get('execution_count', 0),
            avg_execution_time_ms=row.get('avg_execution_time_ms'),
            failure_modes=json.loads(row.get('failure_modes') or '[]'),
            learned_from=json.loads(row.get('learned_from') or '[]'),
            is_compiled=bool(row.get('is_compiled', False)),
            chunking_level=row.get('chunking_level', 0),
        )
```

---

## 4. Working Memory with Bounded Capacity

```python
# backend/services/working_memory.py

"""Working Memory - Bounded capacity attention buffer.

Implements the 7 +/- 2 cognitive constraint with a hard limit
of 100 items to prevent memory exhaustion.
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

from .memory_db import get_memory_db
from .semantic_memory import SemanticNode, SemanticMemoryStore
from .episodic_memory import Episode, EpisodicMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """An item in working memory with activation level."""
    node_id: str
    node_type: str  # 'semantic', 'episodic', 'procedural'
    activation: float
    loaded_at: datetime

    # Cached node data (not persisted)
    _cached_node: Optional[Union[SemanticNode, Episode]] = None


class WorkingMemory:
    """Bounded working memory with LRU eviction.

    Hard Limits:
    - MAX_ITEMS = 100 (hard ceiling)
    - SOFT_LIMIT = 50 (target capacity)
    - ACTIVATION_THRESHOLD = 0.1 (eviction threshold)

    Eviction Policy:
    - When at MAX_ITEMS, evict lowest activation items
    - Items below ACTIVATION_THRESHOLD are candidates for eviction
    - Recently loaded items get eviction protection (5 seconds)
    """

    MAX_ITEMS = 100
    SOFT_LIMIT = 50
    ACTIVATION_THRESHOLD = 0.1
    EVICTION_PROTECTION_SECONDS = 5

    def __init__(self):
        self.db = get_memory_db()
        self._items: OrderedDict[str, WorkingMemoryItem] = OrderedDict()
        self._lock = threading.Lock()

        # Load from database on startup
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Load persisted working memory state."""
        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT node_id, node_type, activation, loaded_at
                FROM working_memory
                ORDER BY activation DESC
            """)
            for row in cursor:
                item = WorkingMemoryItem(
                    node_id=row['node_id'],
                    node_type=row['node_type'],
                    activation=row['activation'],
                    loaded_at=datetime.fromisoformat(row['loaded_at']),
                )
                self._items[row['node_id']] = item

    def _persist_to_db(self) -> None:
        """Persist working memory state to database."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM working_memory")
            for i, (node_id, item) in enumerate(self._items.items()):
                if i >= self.MAX_ITEMS:
                    break
                conn.execute("""
                    INSERT INTO working_memory (slot_id, node_id, node_type, activation, loaded_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (i, node_id, item.node_type, item.activation, item.loaded_at.isoformat()))

    def activate(self, node_id: str, node_type: str, boost: float = 0.2) -> float:
        """Activate an item in working memory.

        If not present, adds it. If present, boosts activation.
        Returns the new activation level.
        """
        with self._lock:
            now = datetime.now()

            if node_id in self._items:
                # Boost existing item
                item = self._items[node_id]
                item.activation = min(1.0, item.activation + boost)
                # Move to end (most recent)
                self._items.move_to_end(node_id)
                return item.activation

            # Need to add new item - check capacity
            if len(self._items) >= self.MAX_ITEMS:
                self._evict_lowest()

            # Add new item
            item = WorkingMemoryItem(
                node_id=node_id,
                node_type=node_type,
                activation=boost,
                loaded_at=now,
            )
            self._items[node_id] = item

            # Persist periodically (not on every change for performance)
            if len(self._items) % 10 == 0:
                self._persist_to_db()

            return item.activation

    def get_activation(self, node_id: str) -> float:
        """Get current activation level (0 if not in working memory)."""
        with self._lock:
            if node_id in self._items:
                return self._items[node_id].activation
            return 0.0

    def decay_all(self, factor: float = 0.9) -> int:
        """Apply decay to all items. Returns number of items evicted."""
        evicted = 0
        with self._lock:
            to_evict = []
            for node_id, item in self._items.items():
                item.activation *= factor
                if item.activation < self.ACTIVATION_THRESHOLD:
                    # Check eviction protection
                    age = (datetime.now() - item.loaded_at).total_seconds()
                    if age > self.EVICTION_PROTECTION_SECONDS:
                        to_evict.append(node_id)

            for node_id in to_evict:
                del self._items[node_id]
                evicted += 1

            if evicted > 0:
                self._persist_to_db()

        return evicted

    def get_active_items(
        self,
        min_activation: float = 0.0,
        node_type: Optional[str] = None,
    ) -> List[WorkingMemoryItem]:
        """Get items above activation threshold."""
        with self._lock:
            items = []
            for item in self._items.values():
                if item.activation >= min_activation:
                    if node_type is None or item.node_type == node_type:
                        items.append(item)
            return sorted(items, key=lambda x: x.activation, reverse=True)

    def get_top_k(self, k: int = 10) -> List[WorkingMemoryItem]:
        """Get top-k activated items."""
        with self._lock:
            items = list(self._items.values())
            items.sort(key=lambda x: x.activation, reverse=True)
            return items[:k]

    def clear(self) -> None:
        """Clear all working memory."""
        with self._lock:
            self._items.clear()
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM working_memory")

    def _evict_lowest(self) -> None:
        """Evict lowest activation item(s) to make room."""
        # Sort by activation (ascending)
        sorted_items = sorted(
            self._items.items(),
            key=lambda x: x[1].activation
        )

        # Evict enough to get to SOFT_LIMIT
        to_evict = len(self._items) - self.SOFT_LIMIT + 1
        for node_id, _ in sorted_items[:to_evict]:
            del self._items[node_id]

        logger.debug(f"Evicted {to_evict} items from working memory")

    @property
    def size(self) -> int:
        """Current number of items in working memory."""
        return len(self._items)

    @property
    def capacity_remaining(self) -> int:
        """Slots remaining before hard limit."""
        return self.MAX_ITEMS - len(self._items)
```

---

## 5. Embedding System with Memory Limits

### 5.1 SQLite-Backed Embedding Storage

```python
# backend/services/embedding_store.py

"""SQLite-backed embedding storage with memory-mapped search.

This replaces the in-memory embedding storage with a disk-backed
solution that uses memory mapping for efficient similarity search.
"""

import logging
import mmap
import struct
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .memory_db import get_memory_db

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIM = 384
BYTES_PER_FLOAT = 4
EMBEDDING_BYTES = EMBEDDING_DIM * BYTES_PER_FLOAT  # 1536 bytes


class EmbeddingStore:
    """SQLite-backed embedding storage with optional mmap for search.

    Storage Strategy:
    - Embeddings stored as BLOBs in SQLite (source of truth)
    - Optional memory-mapped file for fast similarity search
    - LRU cache for frequently accessed embeddings

    Memory Limits:
    - LRU cache: 50MB max (~32,000 embeddings)
    - Mmap file: read-only, doesn't count against heap
    """

    CACHE_SIZE_MB = 50
    MAX_CACHE_ITEMS = (CACHE_SIZE_MB * 1024 * 1024) // EMBEDDING_BYTES

    def __init__(self, mmap_path: Optional[Path] = None):
        self.db = get_memory_db()
        self.mmap_path = mmap_path or Path("memory/graph/embeddings.mmap")

        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []  # LRU order
        self._lock = threading.Lock()

        # Memory-mapped file (optional, for large-scale search)
        self._mmap: Optional[mmap.mmap] = None
        self._mmap_index: dict[str, int] = {}  # node_id -> offset in mmap

    def store_embedding(self, node_id: str, node_type: str, embedding: np.ndarray) -> None:
        """Store an embedding in SQLite."""
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Expected {EMBEDDING_DIM}-dim embedding, got {embedding.shape}")

        embedding_blob = embedding.astype(np.float32).tobytes()

        with self.db.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings (node_id, node_type, embedding, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (node_id, node_type, embedding_blob))

        # Update cache
        with self._lock:
            self._add_to_cache(node_id, embedding)

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding by node ID."""
        # Check cache first
        with self._lock:
            if node_id in self._cache:
                # Move to end (most recently used)
                self._cache_order.remove(node_id)
                self._cache_order.append(node_id)
                return self._cache[node_id]

        # Load from database
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            embedding = np.frombuffer(row['embedding'], dtype=np.float32)

            # Add to cache
            with self._lock:
                self._add_to_cache(node_id, embedding)

            return embedding

    def delete_embedding(self, node_id: str) -> None:
        """Delete an embedding."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM embeddings WHERE node_id = ?", (node_id,))

        with self._lock:
            if node_id in self._cache:
                self._cache_order.remove(node_id)
                del self._cache[node_id]

    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        min_similarity: float = 0.3,
        node_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings.

        Uses memory-mapped file if available, otherwise loads from DB
        in batches to avoid memory exhaustion.

        Returns list of (node_id, similarity) tuples.
        """
        if self._mmap is not None:
            return self._search_mmap(query_embedding, limit, min_similarity, node_types)
        else:
            return self._search_batched(query_embedding, limit, min_similarity, node_types)

    def _search_batched(
        self,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float,
        node_types: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        """Search by loading embeddings in batches."""
        BATCH_SIZE = 500
        results: List[Tuple[str, float]] = []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        with self.db.cursor() as cursor:
            # Build query with optional type filter
            if node_types:
                type_filter = " WHERE node_type IN ({})".format(
                    ",".join(f"'{t}'" for t in node_types)
                )
            else:
                type_filter = ""

            cursor.execute(f"SELECT COUNT(*) FROM embeddings{type_filter}")
            total = cursor.fetchone()[0]

            for offset in range(0, total, BATCH_SIZE):
                cursor.execute(f"""
                    SELECT node_id, embedding FROM embeddings{type_filter}
                    LIMIT ? OFFSET ?
                """, (BATCH_SIZE, offset))

                batch_ids = []
                batch_embeddings = []

                for row in cursor:
                    batch_ids.append(row['node_id'])
                    emb = np.frombuffer(row['embedding'], dtype=np.float32)
                    batch_embeddings.append(emb)

                if not batch_embeddings:
                    continue

                # Vectorized similarity computation
                embeddings_matrix = np.stack(batch_embeddings)
                norms = np.linalg.norm(embeddings_matrix, axis=1)

                # Avoid division by zero
                valid_mask = norms > 0
                similarities = np.zeros(len(batch_ids))
                similarities[valid_mask] = np.dot(
                    embeddings_matrix[valid_mask],
                    query_normalized
                ) / norms[valid_mask]

                # Collect results above threshold
                for i, (node_id, sim) in enumerate(zip(batch_ids, similarities)):
                    if sim >= min_similarity:
                        results.append((node_id, float(sim)))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _search_mmap(
        self,
        query_embedding: np.ndarray,
        limit: int,
        min_similarity: float,
        node_types: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        """Search using memory-mapped embedding file.

        This is more memory efficient for very large datasets as the
        OS handles paging the mmap file.
        """
        # Implementation for mmap-based search
        # Would require building and maintaining the mmap index
        # For now, fall back to batched search
        return self._search_batched(query_embedding, limit, min_similarity, node_types)

    def _add_to_cache(self, node_id: str, embedding: np.ndarray) -> None:
        """Add embedding to LRU cache, evicting if necessary."""
        if node_id in self._cache:
            return

        # Evict if at capacity
        while len(self._cache) >= self.MAX_CACHE_ITEMS:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[node_id] = embedding
        self._cache_order.append(node_id)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            cache_size_mb = len(self._cache) * EMBEDDING_BYTES / (1024 * 1024)
            return {
                "cache_items": len(self._cache),
                "cache_size_mb": round(cache_size_mb, 2),
                "cache_capacity": self.MAX_CACHE_ITEMS,
                "cache_hit_rate": "N/A",  # Would need to track hits/misses
            }

    def rebuild_mmap_index(self) -> None:
        """Rebuild memory-mapped file from SQLite.

        This creates a contiguous binary file of all embeddings
        for efficient similarity search.
        """
        logger.info("Rebuilding mmap index...")

        with self.db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total = cursor.fetchone()[0]

            if total == 0:
                logger.info("No embeddings to index")
                return

            # Create mmap file
            self.mmap_path.parent.mkdir(parents=True, exist_ok=True)
            file_size = total * EMBEDDING_BYTES

            with open(self.mmap_path, 'wb') as f:
                f.truncate(file_size)

            # Write embeddings
            cursor.execute("SELECT node_id, embedding FROM embeddings ORDER BY node_id")

            offset = 0
            index = {}

            with open(self.mmap_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), file_size)

                for row in cursor:
                    node_id = row['node_id']
                    embedding_bytes = row['embedding']

                    mm[offset:offset + EMBEDDING_BYTES] = embedding_bytes
                    index[node_id] = offset
                    offset += EMBEDDING_BYTES

                mm.close()

            self._mmap_index = index
            logger.info(f"Rebuilt mmap index with {total} embeddings")
```

### 5.2 Batched Embedding Service

```python
# backend/services/embedding_service_batched.py

"""Batched Embedding Service with rate limiting and streaming.

Extends the base embedding service with:
- Batch processing for efficiency
- Rate limiting to prevent API overload
- Streaming results for memory efficiency
"""

import asyncio
import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Tuple
import numpy as np

from .embedding_service import get_embedding_service, EmbeddingService

logger = logging.getLogger(__name__)


class BatchedEmbeddingService:
    """Embedding service with batching and rate limiting.

    Constraints:
    - BATCH_SIZE: 32 items per batch (balance between latency and throughput)
    - RATE_LIMIT: 100 requests per minute (to avoid model overload)
    - QUEUE_SIZE: 1000 pending items max
    """

    BATCH_SIZE = 32
    RATE_LIMIT_PER_MINUTE = 100
    MAX_QUEUE_SIZE = 1000

    def __init__(self):
        self._base_service = get_embedding_service()

        # Rate limiting
        self._request_times: deque[datetime] = deque()
        self._rate_lock = threading.Lock()

        # Pending batch
        self._pending: List[Tuple[str, asyncio.Future]] = []
        self._pending_lock = threading.Lock()

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text with rate limiting."""
        await self._wait_for_rate_limit()
        return await self._base_service.embed_async(text)

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts with rate limiting.

        Splits into sub-batches of BATCH_SIZE for memory efficiency.
        """
        if len(texts) == 0:
            return np.array([])

        results = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            await self._wait_for_rate_limit()
            batch_embeddings = await self._base_service.embed_batch_async(batch)
            results.append(batch_embeddings)

        return np.vstack(results)

    async def embed_stream(
        self,
        texts: List[str],
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """Stream embeddings one at a time for memory efficiency.

        Yields (index, embedding) tuples as they complete.
        """
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            await self._wait_for_rate_limit()
            batch_embeddings = await self._base_service.embed_batch_async(batch)

            for j, embedding in enumerate(batch_embeddings):
                yield (i + j, embedding)

    async def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exceeded."""
        with self._rate_lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)

            # Remove old request times
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()

            # Check rate limit
            if len(self._request_times) >= self.RATE_LIMIT_PER_MINUTE:
                # Calculate wait time
                oldest = self._request_times[0]
                wait_seconds = (oldest + timedelta(minutes=1) - now).total_seconds()
                if wait_seconds > 0:
                    logger.debug(f"Rate limited, waiting {wait_seconds:.1f}s")
                    await asyncio.sleep(wait_seconds)

            self._request_times.append(now)

    def get_rate_limit_status(self) -> dict:
        """Get current rate limit status."""
        with self._rate_lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)

            # Count recent requests
            recent = sum(1 for t in self._request_times if t >= cutoff)

            return {
                "requests_last_minute": recent,
                "limit_per_minute": self.RATE_LIMIT_PER_MINUTE,
                "capacity_remaining": self.RATE_LIMIT_PER_MINUTE - recent,
            }


# Singleton
_batched_service: BatchedEmbeddingService | None = None
_service_lock = threading.Lock()


def get_batched_embedding_service() -> BatchedEmbeddingService:
    """Get singleton BatchedEmbeddingService instance."""
    global _batched_service
    if _batched_service is None:
        with _service_lock:
            if _batched_service is None:
                _batched_service = BatchedEmbeddingService()
    return _batched_service
```

---

## 6. Memory Dynamics and Consolidation

### 6.1 Memory Decay Engine

```python
# backend/services/memory_decay.py

"""Memory Decay Engine - Ebbinghaus forgetting curve implementation.

Implements realistic memory decay with reinforcement through retrieval.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple

from .memory_db import get_memory_db
from .semantic_memory import SemanticMemoryStore

logger = logging.getLogger(__name__)


class MemoryDecayEngine:
    """Implements Ebbinghaus forgetting curve with ACT-R style activation.

    Activation Formula (ACT-R):
        Base-level activation = ln(sum(t_i^(-d)))
        where t_i = time since i-th retrieval, d = decay rate

    This runs as a periodic background task.
    """

    DEFAULT_DECAY_RATE = 0.5  # ACT-R default
    MIN_ACTIVATION = 0.01
    RETRIEVAL_BOOST = 0.2

    def __init__(self):
        self.db = get_memory_db()

    def compute_activation(
        self,
        node_id: str,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> float:
        """Compute current activation level for a node.

        Uses retrieval history from the database.
        """
        with self.db.cursor() as cursor:
            # Get retrieval history
            cursor.execute("""
                SELECT retrieved_at FROM retrieval_history
                WHERE node_id = ?
                ORDER BY retrieved_at DESC
                LIMIT 100
            """, (node_id,))

            now = datetime.now()
            activation_sum = 0.0

            for row in cursor:
                retrieval_time = datetime.fromisoformat(row['retrieved_at'])
                time_since = (now - retrieval_time).total_seconds()

                if time_since > 0:
                    # ACT-R power law decay
                    activation_sum += math.pow(time_since, -decay_rate)

            if activation_sum > 0:
                return math.log(activation_sum)
            return 0.0

    def update_all_activations(self, batch_size: int = 100) -> int:
        """Update activation levels for all semantic nodes.

        Runs in batches to avoid memory issues.
        Returns number of nodes updated.
        """
        updated = 0
        offset = 0

        while True:
            with self.db.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM semantic_nodes
                    ORDER BY id
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))

                node_ids = [row['id'] for row in cursor]

                if not node_ids:
                    break

            # Compute activations for batch
            updates: List[Tuple[float, str]] = []
            for node_id in node_ids:
                activation = self.compute_activation(node_id)
                updates.append((activation, node_id))

            # Batch update
            with self.db.transaction() as conn:
                conn.executemany("""
                    UPDATE semantic_nodes
                    SET base_level_activation = ?
                    WHERE id = ?
                """, updates)

            updated += len(node_ids)
            offset += batch_size

            logger.debug(f"Updated activations for {updated} nodes")

        return updated

    def get_forgetting_candidates(
        self,
        activation_threshold: float = MIN_ACTIVATION,
        min_age_days: int = 30,
        limit: int = 100,
    ) -> List[str]:
        """Get nodes that are candidates for forgetting.

        Criteria:
        - Activation below threshold
        - Not accessed in min_age_days
        - Not marked as important
        """
        cutoff = (datetime.now() - timedelta(days=min_age_days)).isoformat()

        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT id FROM semantic_nodes
                WHERE base_level_activation < ?
                  AND (last_access IS NULL OR last_access < ?)
                  AND confidence < 0.8
                ORDER BY base_level_activation ASC
                LIMIT ?
            """, (activation_threshold, cutoff, limit))

            return [row['id'] for row in cursor]

    def should_forget(self, node_id: str) -> bool:
        """Check if a node should be forgotten.

        Uses activation level and retrieval probability.
        """
        activation = self.compute_activation(node_id)

        # Retrieval probability = 1 / (1 + e^(-activation))
        retrieval_prob = 1.0 / (1.0 + math.exp(-activation))

        return retrieval_prob < 0.1
```

### 6.2 Memory Consolidation Service

```python
# backend/services/consolidation_service.py

"""Memory Consolidation Service - Episodic to Semantic transformation.

Implements the "sleep cycle" that:
1. Reviews recent episodic memories
2. Extracts patterns and abstractions
3. Creates/updates semantic nodes
4. Marks episodes as consolidated
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
import hashlib

from .memory_db import get_memory_db
from .episodic_memory import EpisodicMemoryStore, Episode
from .semantic_memory import SemanticMemoryStore, SemanticNode, SemanticNodeType
from .embedding_store import EmbeddingStore
from .embedding_service_batched import get_batched_embedding_service

logger = logging.getLogger(__name__)


class ConsolidationService:
    """Memory consolidation from episodic to semantic.

    Memory Budget:
    - Process at most 50 episodes per cycle
    - Batch embedding operations
    - Yield control periodically

    Run as background task (e.g., every hour or end of session).
    """

    EPISODES_PER_CYCLE = 50
    MIN_EPISODE_AGE_HOURS = 24
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self):
        self.db = get_memory_db()
        self.episodic = EpisodicMemoryStore()
        self.semantic = SemanticMemoryStore()
        self.embeddings = EmbeddingStore()
        self.embedding_service = get_batched_embedding_service()

    async def run_consolidation_cycle(self) -> dict:
        """Run a full consolidation cycle.

        Returns statistics about the consolidation.
        """
        stats = {
            "episodes_processed": 0,
            "patterns_extracted": 0,
            "nodes_created": 0,
            "nodes_updated": 0,
            "started_at": datetime.now().isoformat(),
        }

        # Phase 1: Get unconsolidated episodes
        episodes = list(self.episodic.get_unconsolidated_episodes(
            min_age_hours=self.MIN_EPISODE_AGE_HOURS,
            batch_size=self.EPISODES_PER_CYCLE,
        ))

        if not episodes:
            logger.info("No episodes to consolidate")
            stats["completed_at"] = datetime.now().isoformat()
            return stats

        logger.info(f"Consolidating {len(episodes)} episodes")
        stats["episodes_processed"] = len(episodes)

        # Phase 2: Extract patterns from similar episodes
        patterns = await self._extract_patterns(episodes)
        stats["patterns_extracted"] = len(patterns)

        # Phase 3: Create/update semantic nodes from patterns
        for pattern in patterns:
            created, updated = await self._process_pattern(pattern)
            stats["nodes_created"] += created
            stats["nodes_updated"] += updated

        # Phase 4: Mark episodes as consolidated
        for episode in episodes:
            self.episodic.mark_consolidated(episode.id)

        stats["completed_at"] = datetime.now().isoformat()
        logger.info(f"Consolidation complete: {stats}")

        return stats

    async def _extract_patterns(self, episodes: List[Episode]) -> List[dict]:
        """Extract common patterns from episodes.

        Groups episodes by semantic similarity and extracts
        common elements.
        """
        if not episodes:
            return []

        # Embed all episode summaries
        summaries = [ep.summary for ep in episodes]
        embeddings = await self.embedding_service.embed_batch(summaries)

        # Cluster by similarity (simple pairwise for small batches)
        clusters: List[List[int]] = []
        assigned = set()

        for i in range(len(episodes)):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j in range(i + 1, len(episodes)):
                if j in assigned:
                    continue

                # Compute similarity
                sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                ))

                if sim >= self.SIMILARITY_THRESHOLD:
                    cluster.append(j)
                    assigned.add(j)

            if len(cluster) >= 2:  # Only clusters with multiple episodes
                clusters.append(cluster)

        # Extract patterns from clusters
        patterns = []
        for cluster in clusters:
            cluster_episodes = [episodes[i] for i in cluster]
            pattern = self._extract_cluster_pattern(cluster_episodes)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _extract_cluster_pattern(self, episodes: List[Episode]) -> Optional[dict]:
        """Extract a common pattern from a cluster of similar episodes.

        Simple implementation - more sophisticated NLP could be used.
        """
        if not episodes:
            return None

        # Find common spatial context
        contexts = [ep.spatial_context for ep in episodes if ep.spatial_context]
        common_context = max(set(contexts), key=contexts.count) if contexts else None

        # Average emotional valence
        avg_valence = sum(ep.emotional_valence for ep in episodes) / len(episodes)

        # Combine summaries (take first as representative)
        representative_summary = episodes[0].summary

        return {
            "type": "abstraction",
            "source_episode_ids": [ep.id for ep in episodes],
            "frequency": len(episodes),
            "representative_summary": representative_summary,
            "spatial_context": common_context,
            "emotional_valence": avg_valence,
        }

    async def _process_pattern(self, pattern: dict) -> tuple[int, int]:
        """Process a pattern into semantic memory.

        Returns (nodes_created, nodes_updated).
        """
        created = 0
        updated = 0

        # Check for existing similar node
        summary = pattern["representative_summary"]
        embedding = await self.embedding_service.embed_single(summary)

        similar = self.embeddings.search_similar(
            embedding,
            limit=1,
            min_similarity=self.SIMILARITY_THRESHOLD,
            node_types=["fact", "concept"],
        )

        if similar:
            # Update existing node
            node_id = similar[0][0]
            node = self.semantic.get_node(node_id)
            if node:
                # Update confidence based on frequency
                self.semantic.update_confidence(
                    node_id,
                    new_evidence_strength=0.8,
                    is_supporting=True,
                )
                updated = 1
        else:
            # Create new semantic node
            node_id = hashlib.sha256(summary.encode()).hexdigest()[:12]

            node = SemanticNode(
                id=node_id,
                node_type=SemanticNodeType.FACT,
                label=summary[:100],  # Truncate for label
                description=summary,
                confidence=0.5 + (pattern["frequency"] * 0.1),
                evidence_count=pattern["frequency"],
                derived_from_episodes=pattern["source_episode_ids"],
                source="consolidation",
            )

            self.semantic.store_node(node)
            self.embeddings.store_embedding(node_id, "semantic", embedding)
            created = 1

        return created, updated


# Import numpy here to avoid circular imports
import numpy as np
```

---

## 7. Spreading Activation (Bounded)

```python
# backend/services/spreading_activation.py

"""Bounded Spreading Activation for semantic network traversal.

Implements ACT-R style spreading activation with hard limits
to prevent memory exhaustion on dense graphs.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .memory_db import get_memory_db
from .working_memory import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class ActivationResult:
    """Result of spreading activation computation."""
    node_id: str
    activation: float
    path_length: int
    source_nodes: List[str]


class BoundedSpreadingActivation:
    """Bounded spreading activation through the semantic network.

    Hard Limits (to prevent memory explosion):
    - MAX_HOPS = 3 (depth of spread)
    - MAX_NODES = 50 (total nodes considered)
    - MAX_EDGES_PER_NODE = 20 (fan-out limit)
    - DECAY_PER_HOP = 0.5 (activation decay)
    - MIN_ACTIVATION = 0.05 (cutoff threshold)
    """

    MAX_HOPS = 3
    MAX_NODES = 50
    MAX_EDGES_PER_NODE = 20
    DECAY_PER_HOP = 0.5
    MIN_ACTIVATION = 0.05

    def __init__(self):
        self.db = get_memory_db()

    def spread_from_sources(
        self,
        source_node_ids: List[str],
        source_activations: Dict[str, float] | None = None,
    ) -> List[ActivationResult]:
        """Spread activation from source nodes.

        Args:
            source_node_ids: Starting nodes for activation spread
            source_activations: Optional activation levels (default 1.0)

        Returns:
            List of ActivationResult sorted by activation (descending)
        """
        if not source_node_ids:
            return []

        # Initialize activations
        activations: Dict[str, float] = {}
        paths: Dict[str, Tuple[int, List[str]]] = {}  # node_id -> (path_length, source_nodes)

        for node_id in source_node_ids:
            activation = 1.0 if source_activations is None else source_activations.get(node_id, 1.0)
            activations[node_id] = activation
            paths[node_id] = (0, [node_id])

        # BFS-style spreading with limits
        visited: Set[str] = set(source_node_ids)
        frontier: Set[str] = set(source_node_ids)

        for hop in range(self.MAX_HOPS):
            if len(visited) >= self.MAX_NODES:
                logger.debug(f"Hit MAX_NODES limit at hop {hop}")
                break

            if not frontier:
                break

            next_frontier: Set[str] = set()
            decay = self.DECAY_PER_HOP ** (hop + 1)

            for node_id in frontier:
                source_activation = activations[node_id]

                # Get edges (limited)
                edges = self._get_edges(node_id, limit=self.MAX_EDGES_PER_NODE)

                for target_id, edge_weight in edges:
                    if target_id in visited:
                        # Already visited - just accumulate activation
                        spread_activation = source_activation * edge_weight * decay
                        if target_id in activations:
                            activations[target_id] += spread_activation
                        continue

                    if len(visited) >= self.MAX_NODES:
                        break

                    # Compute spread activation
                    spread_activation = source_activation * edge_weight * decay

                    if spread_activation < self.MIN_ACTIVATION:
                        continue

                    # Add to results
                    activations[target_id] = activations.get(target_id, 0) + spread_activation

                    # Track path
                    source_path = paths[node_id]
                    paths[target_id] = (hop + 1, source_path[1] + [target_id])

                    visited.add(target_id)
                    next_frontier.add(target_id)

            frontier = next_frontier

        # Build results
        results = []
        for node_id, activation in activations.items():
            if activation >= self.MIN_ACTIVATION:
                path_len, sources = paths.get(node_id, (0, []))
                results.append(ActivationResult(
                    node_id=node_id,
                    activation=activation,
                    path_length=path_len,
                    source_nodes=sources,
                ))

        results.sort(key=lambda r: r.activation, reverse=True)
        return results[:self.MAX_NODES]

    def spread_from_working_memory(
        self,
        working_memory: WorkingMemory,
        top_k: int = 10,
    ) -> List[ActivationResult]:
        """Spread activation from current working memory contents.

        Uses the top-k activated items in working memory as sources.
        """
        top_items = working_memory.get_top_k(top_k)

        source_ids = [item.node_id for item in top_items]
        source_activations = {item.node_id: item.activation for item in top_items}

        return self.spread_from_sources(source_ids, source_activations)

    def _get_edges(
        self,
        node_id: str,
        limit: int = MAX_EDGES_PER_NODE,
    ) -> List[Tuple[str, float]]:
        """Get outgoing edges for a node with limit."""
        edges = []

        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT target_id, weight, confidence
                FROM edges
                WHERE source_id = ?
                ORDER BY weight * confidence DESC
                LIMIT ?
            """, (node_id, limit))

            for row in cursor:
                # Combined weight considers edge weight and confidence
                combined_weight = row['weight'] * row['confidence']
                edges.append((row['target_id'], combined_weight))

        return edges

    def get_activation_path(
        self,
        source_id: str,
        target_id: str,
        max_path_length: int = MAX_HOPS,
    ) -> List[str] | None:
        """Find the activation path between two nodes.

        Returns the shortest path or None if not reachable within limits.
        """
        if source_id == target_id:
            return [source_id]

        # BFS for shortest path
        visited = {source_id}
        frontier = [(source_id, [source_id])]

        for _ in range(max_path_length):
            next_frontier = []

            for node_id, path in frontier:
                edges = self._get_edges(node_id)

                for target, _ in edges:
                    if target == target_id:
                        return path + [target]

                    if target not in visited:
                        visited.add(target)
                        next_frontier.append((target, path + [target]))

            frontier = next_frontier

            if not frontier:
                break

        return None
```

---

## 8. Meta-Cognition Layer

### 8.1 Confidence Tracking

```python
# backend/services/confidence_tracker.py

"""Confidence Tracking - Epistemic status of all assertions."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .memory_db import get_memory_db

logger = logging.getLogger(__name__)


@dataclass
class EpistemicStatus:
    """Full epistemic status of a piece of knowledge."""
    confidence: float  # P(true) 0-1
    confidence_interval: Tuple[float, float]  # Uncertainty bounds
    evidence_count: int
    source_reliability: float
    last_validated: Optional[datetime]
    contradictions: List[str]  # IDs of contradicting nodes

    def overall_certainty(self) -> float:
        """Compute overall certainty score."""
        recency_factor = 1.0
        if self.last_validated:
            days_since = (datetime.now() - self.last_validated).days
            recency_factor = max(0.5, 1.0 - (days_since / 365))

        return (
            self.confidence * 0.4 +
            min(1.0, self.evidence_count / 10) * 0.2 +
            self.source_reliability * 0.2 +
            recency_factor * 0.2
        )


class ConfidenceTracker:
    """Track and update confidence for all knowledge.

    Uses Bayesian updating for new evidence.
    """

    def __init__(self):
        self.db = get_memory_db()

    def get_epistemic_status(self, node_id: str) -> Optional[EpistemicStatus]:
        """Get full epistemic status for a node."""
        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT confidence, evidence_count, source_reliability, updated_at
                FROM semantic_nodes
                WHERE id = ?
            """, (node_id,))

            row = cursor.fetchone()
            if not row:
                return None

            # Get contradictions
            cursor.execute("""
                SELECT source_id FROM edges
                WHERE target_id = ? AND edge_type = 'CONTRADICTS'
            """, (node_id,))
            contradictions = [r['source_id'] for r in cursor]

            confidence = row['confidence']

            return EpistemicStatus(
                confidence=confidence,
                confidence_interval=(
                    max(0, confidence - 0.1),
                    min(1, confidence + 0.1),
                ),
                evidence_count=row['evidence_count'],
                source_reliability=row['source_reliability'],
                last_validated=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                contradictions=contradictions,
            )

    def bayesian_update(
        self,
        node_id: str,
        evidence_strength: float,
        is_supporting: bool,
    ) -> float:
        """Apply Bayesian update to node confidence.

        Returns new confidence value.
        """
        status = self.get_epistemic_status(node_id)
        if not status:
            return 0.0

        prior = status.confidence

        # Likelihood based on evidence direction
        if is_supporting:
            likelihood = evidence_strength
        else:
            likelihood = 1 - evidence_strength

        # Bayes: P(A|E) = P(E|A)P(A) / P(E)
        # Where P(E) = P(E|A)P(A) + P(E|~A)P(~A)
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior)
        )

        # Update database
        with self.db.transaction() as conn:
            conn.execute("""
                UPDATE semantic_nodes
                SET confidence = ?,
                    evidence_count = evidence_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (posterior, node_id))

        return posterior

    def detect_contradictions(self, batch_size: int = 100) -> List[Tuple[str, str]]:
        """Detect contradicting assertions in semantic memory.

        Returns list of (node_id_1, node_id_2) pairs.
        """
        # This would need more sophisticated NLP to detect semantic contradictions
        # For now, check explicit contradiction edges
        contradictions = []

        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT source_id, target_id
                FROM edges
                WHERE edge_type = 'CONTRADICTS'
                LIMIT ?
            """, (batch_size,))

            for row in cursor:
                contradictions.append((row['source_id'], row['target_id']))

        return contradictions
```

### 8.2 Knowledge Gap Detection

```python
# backend/services/gap_detector.py

"""Knowledge Gap Detection - Know what you don't know."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .memory_db import get_memory_db
from .semantic_memory import SemanticMemoryStore, SemanticNodeType

logger = logging.getLogger(__name__)


class GapType(str, Enum):
    MISSING_SLOT = "missing_slot"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"
    NO_EVIDENCE = "no_evidence"
    INCOMPLETE_PATTERN = "incomplete_pattern"


@dataclass
class KnowledgeGap:
    """A detected gap in knowledge."""
    gap_type: GapType
    description: str
    related_node_id: Optional[str]
    importance: float  # 0-1
    suggested_action: str


class KnowledgeGapDetector:
    """Detect what we don't know.

    Scans semantic memory for:
    - Low confidence assertions
    - Missing frame slots
    - Stale information
    - Incomplete patterns
    """

    LOW_CONFIDENCE_THRESHOLD = 0.4
    STALE_DAYS = 90

    def __init__(self):
        self.db = get_memory_db()
        self.semantic = SemanticMemoryStore()

    def detect_gaps(
        self,
        context: Optional[str] = None,
        limit: int = 20,
    ) -> List[KnowledgeGap]:
        """Detect knowledge gaps, optionally filtered by context."""
        gaps = []

        # Detect low confidence assertions
        gaps.extend(self._detect_low_confidence(limit))

        # Detect stale information
        gaps.extend(self._detect_stale(limit))

        # Detect missing frame slots
        gaps.extend(self._detect_missing_slots(limit))

        # Sort by importance
        gaps.sort(key=lambda g: g.importance, reverse=True)

        return gaps[:limit]

    def _detect_low_confidence(self, limit: int) -> List[KnowledgeGap]:
        """Find assertions with low confidence."""
        gaps = []

        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT id, label, confidence, evidence_count
                FROM semantic_nodes
                WHERE confidence < ?
                ORDER BY confidence ASC
                LIMIT ?
            """, (self.LOW_CONFIDENCE_THRESHOLD, limit))

            for row in cursor:
                importance = 1.0 - row['confidence']  # Lower confidence = higher importance

                gaps.append(KnowledgeGap(
                    gap_type=GapType.LOW_CONFIDENCE,
                    description=f"Low confidence ({row['confidence']:.1%}) on: {row['label']}",
                    related_node_id=row['id'],
                    importance=importance,
                    suggested_action="Seek additional evidence or verification",
                ))

        return gaps

    def _detect_stale(self, limit: int) -> List[KnowledgeGap]:
        """Find stale information not accessed recently."""
        gaps = []

        with self.db.cursor() as cursor:
            cursor.execute(f"""
                SELECT id, label, last_access, updated_at
                FROM semantic_nodes
                WHERE last_access < datetime('now', '-{self.STALE_DAYS} days')
                   OR last_access IS NULL
                ORDER BY COALESCE(last_access, updated_at) ASC
                LIMIT ?
            """, (limit,))

            for row in cursor:
                gaps.append(KnowledgeGap(
                    gap_type=GapType.STALE,
                    description=f"Potentially stale: {row['label']}",
                    related_node_id=row['id'],
                    importance=0.6,
                    suggested_action="Verify information is still current",
                ))

        return gaps

    def _detect_missing_slots(self, limit: int) -> List[KnowledgeGap]:
        """Find frames with missing required slots."""
        gaps = []

        with self.db.cursor() as cursor:
            cursor.execute("""
                SELECT id, label, slots
                FROM semantic_nodes
                WHERE node_type = 'frame'
                  AND slots IS NOT NULL
                LIMIT ?
            """, (limit * 2,))  # Get more, filter in code

            import json

            for row in cursor:
                try:
                    slots = json.loads(row['slots'])

                    # Check for null/empty slots
                    for slot_name, slot_value in slots.items():
                        if slot_value is None or slot_value == "":
                            gaps.append(KnowledgeGap(
                                gap_type=GapType.MISSING_SLOT,
                                description=f"Missing slot '{slot_name}' in: {row['label']}",
                                related_node_id=row['id'],
                                importance=0.7,
                                suggested_action=f"Find value for {slot_name}",
                            ))

                            if len(gaps) >= limit:
                                return gaps
                except json.JSONDecodeError:
                    continue

        return gaps
```

---

## 9. Memory Budget Allocations

### Summary Table

| Component | Max RAM | Storage | Notes |
|-----------|---------|---------|-------|
| **Embedding Model** | 500 MB | Loaded once | all-MiniLM-L6-v2 |
| **Embedding Cache** | 50 MB | LRU | ~32K embeddings |
| **Working Memory** | 10 MB | 100 items | Bounded capacity |
| **SQLite Cache** | 64 MB | Page cache | PRAGMA cache_size |
| **Python Heap** | 200 MB | Variable | Generators, temp objects |
| **Batch Buffers** | 50 MB | Peak | During embedding batches |
| **Reserved** | 126 MB | Headroom | For spikes |
| **TOTAL** | ~1 GB | | Leaves ~3GB for OS/other |

### Memory Monitoring

```python
# backend/services/memory_monitor.py

"""Memory monitoring and budget enforcement."""

import logging
import psutil
import threading
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class MemoryBudget:
    """Memory budget for a component."""
    name: str
    max_mb: int
    current_mb: float

    @property
    def utilization(self) -> float:
        return self.current_mb / self.max_mb if self.max_mb > 0 else 0


class MemoryMonitor:
    """Monitor and enforce memory budgets.

    Components register their budgets and report usage.
    Monitor can trigger cleanup when approaching limits.
    """

    TOTAL_BUDGET_MB = 1000  # 1GB for memory system
    WARNING_THRESHOLD = 0.8  # Warn at 80% utilization
    CRITICAL_THRESHOLD = 0.95  # Force cleanup at 95%

    def __init__(self):
        self._budgets: Dict[str, MemoryBudget] = {}
        self._lock = threading.Lock()

        # Pre-define component budgets
        self._budgets = {
            "embedding_model": MemoryBudget("embedding_model", 500, 0),
            "embedding_cache": MemoryBudget("embedding_cache", 50, 0),
            "working_memory": MemoryBudget("working_memory", 10, 0),
            "sqlite_cache": MemoryBudget("sqlite_cache", 64, 0),
            "heap": MemoryBudget("heap", 200, 0),
            "batch_buffers": MemoryBudget("batch_buffers", 50, 0),
            "reserved": MemoryBudget("reserved", 126, 0),
        }

    def report_usage(self, component: str, current_mb: float) -> None:
        """Report current memory usage for a component."""
        with self._lock:
            if component in self._budgets:
                self._budgets[component].current_mb = current_mb

                # Check threshold
                budget = self._budgets[component]
                if budget.utilization >= self.CRITICAL_THRESHOLD:
                    logger.warning(f"CRITICAL: {component} at {budget.utilization:.1%} capacity")
                elif budget.utilization >= self.WARNING_THRESHOLD:
                    logger.info(f"WARNING: {component} at {budget.utilization:.1%} capacity")

    def get_status(self) -> Dict[str, dict]:
        """Get current memory status for all components."""
        with self._lock:
            status = {}
            for name, budget in self._budgets.items():
                status[name] = {
                    "max_mb": budget.max_mb,
                    "current_mb": round(budget.current_mb, 2),
                    "utilization": f"{budget.utilization:.1%}",
                }
            return status

    def get_total_usage(self) -> float:
        """Get total memory usage in MB."""
        with self._lock:
            return sum(b.current_mb for b in self._budgets.values())

    def get_process_memory_mb(self) -> float:
        """Get actual process memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def should_cleanup(self) -> bool:
        """Check if cleanup should be triggered."""
        usage = self.get_total_usage()
        return usage / self.TOTAL_BUDGET_MB >= self.WARNING_THRESHOLD

    def get_cleanup_candidates(self) -> list:
        """Get components that should reduce memory usage."""
        candidates = []
        with self._lock:
            for name, budget in self._budgets.items():
                if budget.utilization >= self.WARNING_THRESHOLD:
                    candidates.append((name, budget.utilization))
        return sorted(candidates, key=lambda x: x[1], reverse=True)


# Singleton
_monitor: MemoryMonitor | None = None


def get_memory_monitor() -> MemoryMonitor:
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor()
    return _monitor
```

---

## 10. Performance Characteristics

### Expected Performance

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Single embedding | 10-50ms | ~2MB peak | Depends on text length |
| Batch embedding (32) | 100-200ms | ~20MB peak | Amortized per item |
| FTS5 search | 1-5ms | ~1MB | Very fast for keyword queries |
| Semantic search (1K nodes) | 50-100ms | ~5MB | Batched similarity |
| Semantic search (10K nodes) | 200-500ms | ~20MB | Linear in batch count |
| Working memory activate | <1ms | Negligible | In-memory operation |
| Spreading activation | 10-50ms | ~5MB | Bounded by MAX_NODES |
| Consolidation cycle | 1-5s | ~50MB peak | Batch embedding |
| Decay update (1K nodes) | 100-500ms | ~5MB | Batch SQL updates |

### Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| SQLite over in-memory | Persistence, low RAM | Disk I/O latency |
| LRU caches | Fast hot-path | Cold cache misses |
| Batched embeddings | Throughput | Latency for single items |
| Bounded activation | Predictable RAM | May miss distant relations |
| FTS5 for text | Very fast keywords | Less semantic than embeddings |

---

## 11. Implementation Priority Order

### Phase 1: Foundation (Week 1-2)
**Goal**: SQLite storage layer working

1. `memory_db.py` - Connection management, schema initialization
2. `semantic_memory.py` - Basic CRUD operations
3. `episodic_memory.py` - Episode storage and retrieval
4. `embedding_store.py` - SQLite BLOB storage for embeddings

**Verification**: Store and retrieve 100 nodes with embeddings

### Phase 2: Search (Week 3)
**Goal**: Both FTS5 and semantic search working

1. Add FTS5 triggers to schema
2. Implement `search_fts()` in semantic memory
3. Implement `search_similar()` in embedding store
4. Add batched embedding service

**Verification**: Search queries return relevant results

### Phase 3: Working Memory (Week 4)
**Goal**: Bounded working memory with activation

1. `working_memory.py` - Full implementation
2. Integration with semantic/episodic stores
3. Decay timer (background task)

**Verification**: 100-item limit enforced, eviction works

### Phase 4: Spreading Activation (Week 5)
**Goal**: Bounded graph traversal

1. `spreading_activation.py` - Full implementation
2. Integration with working memory
3. Test with dense graph sections

**Verification**: MAX_NODES=50 limit enforced

### Phase 5: Memory Dynamics (Week 6-7)
**Goal**: Decay and consolidation working

1. `memory_decay.py` - Ebbinghaus curves
2. `consolidation_service.py` - Basic pattern extraction
3. Background task scheduler

**Verification**: Old memories decay, episodes consolidate

### Phase 6: Meta-Cognition (Week 8)
**Goal**: Confidence and gap detection

1. `confidence_tracker.py` - Bayesian updates
2. `gap_detector.py` - Low confidence, stale detection
3. Integration with retrieval (boost low-confidence warnings)

**Verification**: Low-confidence items flagged

### Phase 7: Integration (Week 9-10)
**Goal**: Full system working together

1. Memory coordinator service
2. API endpoints for all operations
3. Memory monitoring dashboard
4. Performance testing and tuning

**Verification**: End-to-end conversation processing

---

## Appendix: Schema Migration from Current System

If migrating from the current MindGraph JSON storage:

```python
# scripts/migrate_to_sqlite.py

"""Migrate existing MindGraph JSON to SQLite."""

import json
from pathlib import Path
from backend.services.memory_db import get_memory_db
from backend.services.semantic_memory import SemanticMemoryStore, SemanticNode, SemanticNodeType
from backend.services.embedding_store import EmbeddingStore
from backend.services.embedding_service import get_embedding_service
import numpy as np


def migrate_mind_graph(json_path: Path):
    """Migrate mind_graph.json to SQLite."""

    db = get_memory_db()
    semantic = SemanticMemoryStore()
    embeddings = EmbeddingStore()
    embedding_service = get_embedding_service()

    with open(json_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])

    print(f"Migrating {len(nodes)} nodes and {len(edges)} edges")

    # Migrate nodes in batches
    BATCH_SIZE = 50
    node_list = list(nodes.values())

    for i in range(0, len(node_list), BATCH_SIZE):
        batch = node_list[i:i + BATCH_SIZE]

        # Create semantic nodes
        for node_data in batch:
            node_type = node_data.get("type", "concept")
            try:
                node_type_enum = SemanticNodeType(node_type)
            except ValueError:
                node_type_enum = SemanticNodeType.CONCEPT

            node = SemanticNode(
                id=node_data["id"],
                node_type=node_type_enum,
                label=node_data.get("label", ""),
                description=node_data.get("description"),
                provenance=node_data.get("provenance", {}),
            )
            semantic.store_node(node)

        # Batch embed
        texts = [
            f"{n.get('label', '')} {n.get('description', '')}"
            for n in batch
        ]
        emb_array = embedding_service.embed_batch(texts)

        for j, node_data in enumerate(batch):
            embeddings.store_embedding(
                node_data["id"],
                "semantic",
                emb_array[j],
            )

        print(f"Migrated {min(i + BATCH_SIZE, len(node_list))}/{len(node_list)} nodes")

    # Migrate edges
    with db.transaction() as conn:
        for edge in edges:
            conn.execute("""
                INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                edge.get("source"),
                edge.get("target"),
                edge.get("type", "ASSOCIATION"),
                edge.get("weight", 1.0),
                json.dumps(edge.get("metadata", {})),
            ))

    print(f"Migration complete!")


if __name__ == "__main__":
    migrate_mind_graph(Path("memory/graph/mind_graph.json"))
```

---

## References

This implementation draws from:

- **ACT-R Cognitive Architecture** - Activation equations, memory decay
- **Ebbinghaus Forgetting Curve** - Memory strength over time
- **SQLite Documentation** - FTS5, WAL mode, performance tuning
- **Sentence Transformers** - Embedding model selection and usage
- **Working Memory Research** - 7 +/- 2 capacity limits

---

*This document specifies a production-ready memory architecture that respects the 8GB constraint while preserving the cognitive capabilities outlined in the visionary architecture.*
