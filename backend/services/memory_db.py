"""SQLite Memory Database - Foundation for the tri-memory cognitive architecture.

This module provides SQLite connection management with:
- WAL mode for concurrent reads during writes
- Thread-local connections for thread safety
- Connection pooling with context managers
- Schema initialization and migrations
- Memory budget enforcement (64MB page cache)

Designed for 8GB M2 Mac Mini constraints.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Any

logger = logging.getLogger(__name__)

# Memory budget constants (from MYND_MEMORY_ARCHITECTURE.md)
SQLITE_CACHE_SIZE_MB = 64
SQLITE_CACHE_SIZE_PAGES = (SQLITE_CACHE_SIZE_MB * 1024 * 1024) // 4096  # 4KB pages

# Schema version for migrations
SCHEMA_VERSION = 1


class MemoryDatabase:
    """SQLite database manager for the cognitive memory system.

    Provides thread-safe connection management with WAL mode,
    schema initialization, and memory-constrained configuration.

    Attributes:
        db_path: Path to the SQLite database file
    """

    def __init__(self, db_path: Path | str):
        """Initialize the memory database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False

        # Initialize schema on first use
        self._ensure_initialized()

        logger.info(f"MemoryDatabase initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Creates a new connection if one doesn't exist for this thread.
        Connections use WAL mode and row factory for dict-like access.

        Returns:
            SQLite connection for the current thread
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )

            # Configure connection
            conn.row_factory = sqlite3.Row  # Dict-like row access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_SIZE_MB * 1024}")  # Negative = KB
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            conn.execute("PRAGMA foreign_keys=ON")

            self._local.connection = conn
            logger.debug(f"Created new connection for thread {threading.current_thread().name}")

        return self._local.connection

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connection.

        Provides a connection with automatic commit on success
        and rollback on exception.

        Yields:
            SQLite connection
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursor.

        Provides a cursor with automatic commit on success
        and rollback on exception.

        Yields:
            SQLite cursor
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL with automatic commit.

        Args:
            sql: SQL statement to execute
            params: Parameters for the statement

        Returns:
            Cursor with results
        """
        with self.connection() as conn:
            return conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets.

        Args:
            sql: SQL statement to execute
            params_list: List of parameter tuples

        Returns:
            Cursor with results
        """
        with self.connection() as conn:
            return conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        """Execute SQL and fetch one row as dict.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Row as dict or None if no results
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute SQL and fetch all rows as dicts.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List of rows as dicts
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def fetchall_generator(
        self,
        sql: str,
        params: tuple = (),
        batch_size: int = 100,
    ) -> Generator[dict[str, Any], None, None]:
        """Execute SQL and yield rows one at a time.

        Memory-efficient for large result sets.

        Args:
            sql: SQL query
            params: Query parameters
            batch_size: Number of rows to fetch at a time

        Yields:
            Rows as dicts
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield dict(row)

    def _ensure_initialized(self) -> None:
        """Ensure database schema is initialized."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            self._init_schema()
            self._initialized = True

    def _init_schema(self) -> None:
        """Initialize database schema.

        Creates all tables for the tri-memory system:
        - episodic_memories: Conversation episodes
        - semantic_nodes: Facts, concepts, entities
        - procedural_skills: Skills and action patterns
        - edges: Graph relationships
        - embeddings: 384-dim vectors as BLOBs
        - working_memory: Transient attention buffer
        - retrieval_history: For decay calculations
        """
        with self.connection() as conn:
            # Check schema version
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            result = conn.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            ).fetchone()

            current_version = int(result['value']) if result else 0

            if current_version < SCHEMA_VERSION:
                self._create_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                    ('version', str(SCHEMA_VERSION))
                )
                logger.info(f"Database schema initialized to version {SCHEMA_VERSION}")

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all database tables.

        Args:
            conn: Database connection
        """
        # Episodic Memory - stores conversation episodes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                duration_seconds INTEGER,
                summary TEXT NOT NULL,
                compressed_transcript BLOB,
                spatial_context TEXT,
                social_context TEXT,
                emotional_valence REAL DEFAULT 0.0 CHECK (emotional_valence BETWEEN -1.0 AND 1.0),
                arousal_level REAL DEFAULT 0.0 CHECK (arousal_level BETWEEN 0.0 AND 1.0),
                encoding_strength REAL DEFAULT 0.5 CHECK (encoding_strength BETWEEN 0.0 AND 1.0),
                retrieval_count INTEGER DEFAULT 0,
                last_retrieved DATETIME,
                decay_rate REAL DEFAULT 0.1,
                is_consolidated BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Semantic Nodes - facts, concepts, entities
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL CHECK (node_type IN (
                    'CONCEPT', 'FACT', 'ENTITY', 'SCHEMA', 'FRAME',
                    'IDENTITY', 'PREFERENCE', 'GOAL', 'DECISION', 'RELATIONSHIP', 'MEMORY'
                )),
                label TEXT NOT NULL,
                description TEXT,
                confidence REAL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
                evidence_count INTEGER DEFAULT 1,
                source_reliability REAL DEFAULT 0.5,
                base_level_activation REAL DEFAULT 0.0,
                last_access DATETIME,
                access_count INTEGER DEFAULT 0,
                derived_from_episodes TEXT,
                slots TEXT,
                source TEXT DEFAULT 'system',
                provenance TEXT,
                metadata TEXT,
                color TEXT DEFAULT '#8B5CF6',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Procedural Skills
        conn.execute("""
            CREATE TABLE IF NOT EXISTS procedural_skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                trigger_pattern TEXT,
                preconditions TEXT,
                goal_relevance REAL DEFAULT 0.5,
                steps TEXT NOT NULL,
                decision_points TEXT,
                tool_bindings TEXT,
                success_rate REAL DEFAULT 0.5,
                avg_execution_time REAL,
                failure_modes TEXT,
                learned_from TEXT,
                is_compiled BOOLEAN DEFAULT FALSE,
                chunking_level INTEGER DEFAULT 1,
                use_count INTEGER DEFAULT 0,
                last_used DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Graph Edges - relationships between nodes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL CHECK (edge_type IN (
                    'PARENT', 'CHILD', 'ASSOCIATION', 'TEMPORAL',
                    'DERIVED', 'REFERENCE', 'ISA', 'HAS_PART',
                    'CAUSES', 'ENABLES', 'CONTRADICTS', 'SIMILAR_TO'
                )),
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source_id, target_id, edge_type)
            )
        """)

        # Embeddings - 384-dim vectors as BLOBs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_version TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Working Memory - bounded attention buffer
        conn.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                slot_id INTEGER PRIMARY KEY CHECK (slot_id BETWEEN 0 AND 99),
                node_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                activation_level REAL DEFAULT 1.0,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                decay_rate REAL DEFAULT 0.1
            )
        """)

        # Retrieval History - for decay calculations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                retrieved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                success BOOLEAN DEFAULT TRUE
            )
        """)

        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic_memories(is_consolidated)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_nodes(node_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_label ON semantic_nodes(label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_activation ON semantic_nodes(base_level_activation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_working_activation ON working_memory(activation_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_node ON retrieval_history(node_id)")

        # Create FTS5 virtual table for full-text search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                node_id,
                node_type,
                label,
                description,
                content='semantic_nodes',
                content_rowid='rowid'
            )
        """)

        # Triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS semantic_nodes_ai AFTER INSERT ON semantic_nodes BEGIN
                INSERT INTO memory_fts(rowid, node_id, node_type, label, description)
                VALUES (NEW.rowid, NEW.id, NEW.node_type, NEW.label, NEW.description);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS semantic_nodes_ad AFTER DELETE ON semantic_nodes BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, node_id, node_type, label, description)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.node_type, OLD.label, OLD.description);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS semantic_nodes_au AFTER UPDATE ON semantic_nodes BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, node_id, node_type, label, description)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.node_type, OLD.label, OLD.description);
                INSERT INTO memory_fts(rowid, node_id, node_type, label, description)
                VALUES (NEW.rowid, NEW.id, NEW.node_type, NEW.label, NEW.description);
            END
        """)

        logger.info("Created all database tables and indexes")

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with table counts and sizes
        """
        stats = {}

        tables = ['episodic_memories', 'semantic_nodes', 'procedural_skills',
                  'edges', 'embeddings', 'working_memory', 'retrieval_history']

        for table in tables:
            count = self.fetchone(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = count['count'] if count else 0

        # Database file size
        if self.db_path.exists():
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

        return stats

    def vacuum(self) -> None:
        """Reclaim disk space and optimize database."""
        with self.connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug(f"Closed connection for thread {threading.current_thread().name}")


# Singleton instance
_memory_db: MemoryDatabase | None = None
_db_lock = threading.Lock()


def get_memory_db(db_path: Path | str | None = None) -> MemoryDatabase:
    """Get or create the global memory database.

    Args:
        db_path: Path to database file (used on first call)

    Returns:
        The memory database singleton
    """
    global _memory_db

    if _memory_db is None:
        with _db_lock:
            if _memory_db is None:
                if db_path is None:
                    db_path = Path(__file__).parent.parent.parent / "memory" / "cognitive" / "memory.db"
                _memory_db = MemoryDatabase(db_path)

    return _memory_db
