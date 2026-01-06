"""Semantic Index - Vector storage and similarity search for MindGraph.

This module provides embedding-based semantic search over MindGraph nodes.
Embeddings are stored in memory as dict[node_id -> np.ndarray] and persisted
to disk as NumPy .npz files with JSON metadata.

Key features:
- In-memory dict for fast runtime access
- Persistence via embeddings.npz and embedding_meta.json
- Lazy loading with dirty tracking for efficient saves
- Thread-safe operations
- Async methods for use in async contexts
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.services.mind_graph import MindGraph, MindNode, NodeType

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with similarity score."""
    node: "MindNode"
    similarity: float


class SemanticIndex:
    """Vector index for semantic search over MindGraph nodes.

    Architecture:
    - Embeddings stored in NumPy .npz file (compact, fast load)
    - Metadata maps node_id -> embedding array index
    - In-memory dict for fast access during runtime
    - Persisted to disk on updates

    Attributes:
        graph: The MindGraph instance this index serves
        storage_path: Directory for storing embedding files
        embeddings_file: Path to embeddings.npz
        meta_file: Path to embedding_meta.json
    """

    def __init__(self, graph: "MindGraph", storage_path: Path):
        """Initialize the semantic index.

        Args:
            graph: The MindGraph instance to index
            storage_path: Directory for storing embedding files
        """
        self.graph = graph
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = self.storage_path / "embeddings.npz"
        self.meta_file = self.storage_path / "embedding_meta.json"

        self._embeddings: dict[str, np.ndarray] = {}  # node_id -> embedding
        self._dirty = False
        self._lock = threading.Lock()

        self._load()

    def _load(self) -> None:
        """Load embeddings from disk."""
        if self.embeddings_file.exists() and self.meta_file.exists():
            try:
                with open(self.meta_file, "r") as f:
                    meta = json.load(f)

                data = np.load(self.embeddings_file)
                embeddings_array = data["embeddings"]

                for node_id, idx in meta.get("index", {}).items():
                    if idx < len(embeddings_array):
                        self._embeddings[node_id] = embeddings_array[idx]

                logger.info(f"Loaded {len(self._embeddings)} embeddings from disk")
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.warning(f"Failed to load embeddings: {e}")

    def _save(self) -> None:
        """Persist embeddings to disk."""
        if not self._embeddings:
            return

        try:
            # Import here to get constants
            from backend.services.embedding_service import EmbeddingService

            # Build ordered arrays
            node_ids = list(self._embeddings.keys())
            embeddings_array = np.array([self._embeddings[nid] for nid in node_ids])

            # Save embeddings
            np.savez_compressed(self.embeddings_file, embeddings=embeddings_array)

            # Save metadata
            meta = {
                "version": 1,
                "model": EmbeddingService.MODEL_NAME,
                "dimension": EmbeddingService.EMBEDDING_DIM,
                "count": len(node_ids),
                "index": {nid: i for i, nid in enumerate(node_ids)},
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            self._dirty = False
            logger.info(f"Saved {len(node_ids)} embeddings to disk")
        except IOError as e:
            logger.error(f"Failed to save embeddings: {e}")

    def _build_embedding_text(self, node: "MindNode") -> str:
        """Create rich text representation for embedding.

        Combines multiple fields for better semantic matching:
        - Label (title/name)
        - Description (details)
        - Type (provides context)
        - Parent context (disambiguation)

        Args:
            node: The node to build text for

        Returns:
            Combined text suitable for embedding
        """
        parts = [node.label]

        if node.description:
            parts.append(node.description)

        parts.append(f"Type: {node.node_type.value}")

        # Add parent label for context (helps disambiguate)
        parent_edges = [e for e in node.edges if e.get("type") == "parent"]
        if parent_edges:
            parent = self.graph.get_node(parent_edges[0]["target"])
            if parent:
                parts.append(f"Under: {parent.label}")

        return " | ".join(parts)

    def index_node(self, node: "MindNode") -> None:
        """Add or update embedding for a node (sync).

        Args:
            node: The node to index
        """
        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        text = self._build_embedding_text(node)
        embedding = service.embed(text)

        with self._lock:
            self._embeddings[node.id] = embedding
            self._dirty = True

    async def index_node_async(self, node: "MindNode") -> None:
        """Add or update embedding for a node (async).

        Args:
            node: The node to index
        """
        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        text = self._build_embedding_text(node)
        embedding = await service.embed_async(text)

        with self._lock:
            self._embeddings[node.id] = embedding
            self._dirty = True

    def remove_node(self, node_id: str) -> None:
        """Remove embedding for deleted node.

        Args:
            node_id: ID of the node to remove
        """
        with self._lock:
            if node_id in self._embeddings:
                del self._embeddings[node_id]
                self._dirty = True

    def search(
        self,
        query: str,
        limit: int = 10,
        node_types: list["NodeType"] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SearchResult]:
        """Semantic search over indexed nodes (sync).

        Args:
            query: Natural language query
            limit: Max results to return
            node_types: Filter by node type (None = all types)
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        if not self._embeddings:
            return []

        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        query_embedding = service.embed(query)

        results = []
        with self._lock:
            for node_id, embedding in self._embeddings.items():
                similarity = service.cosine_similarity(query_embedding, embedding)

                if similarity < min_similarity:
                    continue

                node = self.graph.get_node(node_id)
                if not node:
                    continue

                if node_types and node.node_type not in node_types:
                    continue

                results.append(SearchResult(node=node, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    async def search_async(
        self,
        query: str,
        limit: int = 10,
        node_types: list["NodeType"] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SearchResult]:
        """Semantic search over indexed nodes (async).

        Runs embedding in thread pool to avoid blocking event loop.

        Args:
            query: Natural language query
            limit: Max results to return
            node_types: Filter by node type (None = all types)
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        if not self._embeddings:
            return []

        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        query_embedding = await service.embed_async(query)

        results = []
        with self._lock:
            for node_id, embedding in self._embeddings.items():
                similarity = service.cosine_similarity(query_embedding, embedding)

                if similarity < min_similarity:
                    continue

                node = self.graph.get_node(node_id)
                if not node:
                    continue

                if node_types and node.node_type not in node_types:
                    continue

                results.append(SearchResult(node=node, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def find_similar_nodes(
        self,
        node_id: str,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """Find nodes similar to a given node.

        Args:
            node_id: ID of the reference node
            limit: Max results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        results = []

        with self._lock:
            # Check node existence within lock to prevent TOCTOU
            if node_id not in self._embeddings:
                return []

            node_embedding = self._embeddings[node_id]

            for other_id, embedding in self._embeddings.items():
                if other_id == node_id:
                    continue

                similarity = service.cosine_similarity(node_embedding, embedding)
                if similarity < min_similarity:
                    continue

                node = self.graph.get_node(other_id)
                if node:
                    results.append(SearchResult(node=node, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def rebuild_all(self) -> None:
        """Rebuild embeddings for all nodes. Call after bulk import."""
        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        nodes = list(self.graph._nodes.values())

        if not nodes:
            return

        texts = [self._build_embedding_text(n) for n in nodes]
        embeddings = service.embed_batch(texts)

        with self._lock:
            self._embeddings = {
                node.id: embeddings[i]
                for i, node in enumerate(nodes)
            }
            self._dirty = True
            self._save()
        logger.info(f"Rebuilt embeddings for {len(nodes)} nodes")

    async def rebuild_all_async(self) -> None:
        """Rebuild embeddings for all nodes (async). Call after bulk import."""
        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        nodes = list(self.graph._nodes.values())

        if not nodes:
            return

        texts = [self._build_embedding_text(n) for n in nodes]
        embeddings = await service.embed_batch_async(texts)

        with self._lock:
            self._embeddings = {
                node.id: embeddings[i]
                for i, node in enumerate(nodes)
            }
            self._dirty = True
            self._save()
        logger.info(f"Rebuilt embeddings for {len(nodes)} nodes")

    def flush(self) -> None:
        """Save pending changes to disk."""
        with self._lock:
            if self._dirty:
                self._save()

    def get_stats(self) -> dict:
        """Get statistics about the semantic index.

        Returns:
            Dictionary with index statistics
        """
        with self._lock:
            return {
                "indexed_count": len(self._embeddings),
                "dirty": self._dirty,
                "embeddings_file_exists": self.embeddings_file.exists(),
                "meta_file_exists": self.meta_file.exists(),
            }
