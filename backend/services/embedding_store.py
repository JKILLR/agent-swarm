"""Embedding Store - SQLite BLOB storage for vector embeddings.

This module provides SQLite-backed storage for 384-dim embeddings with:
- BLOB storage (1536 bytes per embedding)
- LRU cache for frequently accessed embeddings
- Batch operations for efficiency
- Semantic similarity search

Memory budget: 50MB LRU cache (~32K embeddings)
Part of the tri-memory cognitive architecture.
"""

from __future__ import annotations

import logging
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator

import numpy as np

from .memory_db import MemoryDatabase, get_memory_db

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIM = 384
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32 = 4 bytes = 1536 bytes
MAX_CACHE_SIZE = 32000  # ~50MB at 1536 bytes each
BATCH_SIZE = 32


@dataclass
class SimilarityResult:
    """Result from similarity search."""
    node_id: str
    similarity: float
    node_type: str


class LRUCache:
    """Simple LRU cache with size limit."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> np.ndarray | None:
        """Get item from cache, moving to end (most recent)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        """Add item to cache, evicting oldest if needed."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def remove(self, key: str) -> None:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class EmbeddingStore:
    """SQLite-backed embedding storage with LRU caching.

    Stores 384-dimensional float32 embeddings as SQLite BLOBs.
    Uses LRU cache for frequently accessed embeddings.

    Attributes:
        db: The memory database instance
    """

    def __init__(self, db: MemoryDatabase | None = None):
        """Initialize embedding store.

        Args:
            db: Memory database instance (uses singleton if None)
        """
        self.db = db or get_memory_db()
        self._cache = LRUCache(MAX_CACHE_SIZE)
        self._lock = threading.Lock()
        logger.info(f"EmbeddingStore initialized (cache size: {MAX_CACHE_SIZE})")

    def _embedding_to_blob(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for SQLite BLOB.

        Args:
            embedding: 384-dim float32 array

        Returns:
            Bytes representation
        """
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        return embedding.tobytes()

    def _blob_to_embedding(self, blob: bytes) -> np.ndarray:
        """Convert SQLite BLOB to numpy array.

        Args:
            blob: Bytes from database

        Returns:
            384-dim float32 array
        """
        return np.frombuffer(blob, dtype=np.float32).copy()

    def store(
        self,
        node_id: str,
        embedding: np.ndarray,
        node_type: str = "CONCEPT",
    ) -> None:
        """Store an embedding.

        Args:
            node_id: ID of the node this embedding belongs to
            embedding: 384-dim embedding vector
            node_type: Type of node
        """
        blob = self._embedding_to_blob(embedding)
        now = datetime.now().isoformat()

        self.db.execute(
            """INSERT OR REPLACE INTO embeddings (node_id, node_type, embedding, updated_at)
               VALUES (?, ?, ?, ?)""",
            (node_id, node_type, blob, now)
        )

        # Update cache
        self._cache.put(node_id, embedding)
        logger.debug(f"Stored embedding for {node_id}")

    def store_batch(
        self,
        items: list[tuple[str, np.ndarray, str]],
    ) -> int:
        """Store multiple embeddings efficiently.

        Args:
            items: List of (node_id, embedding, node_type) tuples

        Returns:
            Number of embeddings stored
        """
        now = datetime.now().isoformat()

        params = []
        for node_id, embedding, node_type in items:
            blob = self._embedding_to_blob(embedding)
            params.append((node_id, node_type, blob, now))
            self._cache.put(node_id, embedding)

        self.db.executemany(
            """INSERT OR REPLACE INTO embeddings (node_id, node_type, embedding, updated_at)
               VALUES (?, ?, ?, ?)""",
            params
        )

        logger.debug(f"Batch stored {len(items)} embeddings")
        return len(items)

    def get(self, node_id: str) -> np.ndarray | None:
        """Get an embedding by node ID.

        Args:
            node_id: Node ID

        Returns:
            Embedding array or None if not found
        """
        # Check cache first
        cached = self._cache.get(node_id)
        if cached is not None:
            return cached

        # Load from database
        row = self.db.fetchone(
            "SELECT embedding FROM embeddings WHERE node_id = ?",
            (node_id,)
        )

        if not row:
            return None

        embedding = self._blob_to_embedding(row['embedding'])
        self._cache.put(node_id, embedding)
        return embedding

    def get_batch(self, node_ids: list[str]) -> dict[str, np.ndarray]:
        """Get multiple embeddings.

        Args:
            node_ids: List of node IDs

        Returns:
            Dict mapping node_id to embedding
        """
        result = {}
        missing = []

        # Check cache first
        for node_id in node_ids:
            cached = self._cache.get(node_id)
            if cached is not None:
                result[node_id] = cached
            else:
                missing.append(node_id)

        # Load missing from database
        if missing:
            placeholders = ",".join("?" * len(missing))
            rows = self.db.fetchall(
                f"SELECT node_id, embedding FROM embeddings WHERE node_id IN ({placeholders})",
                tuple(missing)
            )

            for row in rows:
                embedding = self._blob_to_embedding(row['embedding'])
                self._cache.put(row['node_id'], embedding)
                result[row['node_id']] = embedding

        return result

    def delete(self, node_id: str) -> bool:
        """Delete an embedding.

        Args:
            node_id: Node ID

        Returns:
            True if deleted
        """
        result = self.db.execute(
            "DELETE FROM embeddings WHERE node_id = ?",
            (node_id,)
        )

        self._cache.remove(node_id)
        return result.rowcount > 0

    def exists(self, node_id: str) -> bool:
        """Check if embedding exists.

        Args:
            node_id: Node ID

        Returns:
            True if exists
        """
        if self._cache.get(node_id) is not None:
            return True

        row = self.db.fetchone(
            "SELECT 1 FROM embeddings WHERE node_id = ?",
            (node_id,)
        )
        return row is not None

    # =========================================================================
    # Similarity Search
    # =========================================================================

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        min_similarity: float = 0.3,
        node_types: list[str] | None = None,
    ) -> list[SimilarityResult]:
        """Find similar embeddings using cosine similarity.

        Note: This is a brute-force search. For large datasets (>10K),
        consider using approximate nearest neighbor algorithms.

        Args:
            query_embedding: Query vector
            limit: Max results
            min_similarity: Minimum similarity threshold
            node_types: Filter by node types (None = all)

        Returns:
            List of SimilarityResult sorted by similarity desc
        """
        # Build query
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            sql = f"SELECT node_id, node_type, embedding FROM embeddings WHERE node_type IN ({placeholders})"
            params = tuple(node_types)
        else:
            sql = "SELECT node_id, node_type, embedding FROM embeddings"
            params = ()

        results = []

        # Stream through all embeddings
        for row in self.db.fetchall_generator(sql, params, batch_size=100):
            embedding = self._blob_to_embedding(row['embedding'])
            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= min_similarity:
                results.append(SimilarityResult(
                    node_id=row['node_id'],
                    similarity=similarity,
                    node_type=row['node_type'],
                ))

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def find_similar_nodes(
        self,
        node_id: str,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SimilarityResult]:
        """Find nodes similar to a given node.

        Args:
            node_id: Reference node ID
            limit: Max results
            min_similarity: Minimum similarity

        Returns:
            List of similar nodes (excluding the reference)
        """
        embedding = self.get(node_id)
        if embedding is None:
            return []

        results = self.search_similar(
            embedding,
            limit=limit + 1,  # +1 to account for self
            min_similarity=min_similarity,
        )

        # Remove self from results
        return [r for r in results if r.node_id != node_id][:limit]

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def rebuild_from_nodes(
        self,
        nodes: list[tuple[str, str, str]],  # (node_id, text, node_type)
        embedding_fn: callable,
    ) -> int:
        """Rebuild embeddings for a list of nodes.

        Args:
            nodes: List of (node_id, text, node_type)
            embedding_fn: Function to generate embeddings

        Returns:
            Number of embeddings created
        """
        count = 0

        # Process in batches
        for i in range(0, len(nodes), BATCH_SIZE):
            batch = nodes[i:i + BATCH_SIZE]

            texts = [text for _, text, _ in batch]
            embeddings = embedding_fn(texts)

            items = [
                (node_id, embeddings[j], node_type)
                for j, (node_id, _, node_type) in enumerate(batch)
            ]

            count += self.store_batch(items)

        logger.info(f"Rebuilt {count} embeddings")
        return count

    def get_all_node_ids(self) -> list[str]:
        """Get all node IDs with embeddings.

        Returns:
            List of node IDs
        """
        rows = self.db.fetchall("SELECT node_id FROM embeddings")
        return [row['node_id'] for row in rows]

    def get_all_embeddings_generator(
        self,
        batch_size: int = 100,
    ) -> Generator[tuple[str, np.ndarray, str], None, None]:
        """Iterate all embeddings with minimal memory.

        Args:
            batch_size: Rows per batch

        Yields:
            (node_id, embedding, node_type) tuples
        """
        for row in self.db.fetchall_generator(
            "SELECT node_id, node_type, embedding FROM embeddings",
            batch_size=batch_size
        ):
            embedding = self._blob_to_embedding(row['embedding'])
            yield row['node_id'], embedding, row['node_type']

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get embedding store statistics.

        Returns:
            Statistics dict
        """
        total = self.db.fetchone("SELECT COUNT(*) as count FROM embeddings")

        type_counts = {}
        rows = self.db.fetchall(
            "SELECT node_type, COUNT(*) as count FROM embeddings GROUP BY node_type"
        )
        for row in rows:
            type_counts[row['node_type']] = row['count']

        return {
            "total_embeddings": total['count'] if total else 0,
            "embeddings_by_type": type_counts,
            "cache_size": len(self._cache),
            "cache_max_size": MAX_CACHE_SIZE,
            "embedding_dim": EMBEDDING_DIM,
        }


# Singleton
_embedding_store: EmbeddingStore | None = None
_store_lock = threading.Lock()


def get_embedding_store(db: MemoryDatabase | None = None) -> EmbeddingStore:
    """Get or create the embedding store singleton.

    Args:
        db: Optional memory database

    Returns:
        The embedding store singleton
    """
    global _embedding_store

    if _embedding_store is None:
        with _store_lock:
            if _embedding_store is None:
                _embedding_store = EmbeddingStore(db)

    return _embedding_store
