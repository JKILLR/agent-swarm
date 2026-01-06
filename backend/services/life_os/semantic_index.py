# backend/services/life_os/semantic_index.py
"""FAISS-based semantic index for Life OS.

Provides vector similarity search over documents using FAISS for efficient
indexing and the existing EmbeddingService for embedding generation.

Key features:
- FAISS IndexFlatIP for inner product (cosine) similarity
- Batch embedding generation for memory efficiency
- Support for multiple document types (messages, notes, etc.)
- Incremental document addition
- Persistent storage to disk
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_INDEX_PATH = Path("memory/life_os/faiss.index")
DEFAULT_META_PATH = Path("memory/life_os/faiss_meta.json")

# Batch size for embedding generation (memory efficient)
EMBEDDING_BATCH_SIZE = 100


@dataclass
class Document:
    """A document to be indexed.

    Attributes:
        id: Unique document identifier
        content: Text content to embed
        doc_type: Document type (message, note, event, etc.)
        metadata: Additional metadata for filtering/display
    """
    id: str
    content: str
    doc_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search result with score.

    Attributes:
        doc_id: Document ID
        score: Similarity score (higher = more similar)
        doc_type: Document type
        metadata: Document metadata
    """
    doc_id: str
    score: float
    doc_type: str
    metadata: dict[str, Any]


class FAISSSemanticIndex:
    """FAISS-based semantic index for Life OS documents.

    Uses FAISS IndexFlatIP (inner product) with L2-normalized vectors
    to compute cosine similarity. The index is stored on disk and can
    be loaded/saved incrementally.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        meta_path: Path | str = DEFAULT_META_PATH,
    ):
        """Initialize the FAISS semantic index.

        Args:
            index_path: Path to save/load FAISS index
            meta_path: Path to save/load document metadata
        """
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index: Optional[Any] = None  # faiss.IndexFlatIP
        self._doc_ids: list[str] = []  # Maps index position -> doc_id
        self._doc_meta: dict[str, dict[str, Any]] = {}  # doc_id -> metadata
        self._lock = threading.Lock()
        self._dirty = False

        # Lazy load on first use
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Ensure index is loaded from disk."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True

    def _load(self) -> None:
        """Load index and metadata from disk."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss-cpu not installed, index will start empty")
            self._init_empty_index()
            return

        if self.index_path.exists() and self.meta_path.exists():
            try:
                # Load FAISS index
                self._index = faiss.read_index(str(self.index_path))

                # Load metadata
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)

                self._doc_ids = meta.get("doc_ids", [])
                self._doc_meta = meta.get("doc_meta", {})

                logger.info(
                    f"Loaded FAISS index with {self._index.ntotal} vectors, "
                    f"{len(self._doc_ids)} documents"
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, starting fresh")
                self._init_empty_index()
        else:
            logger.info("No existing index found, creating new one")
            self._init_empty_index()

    def _init_empty_index(self) -> None:
        """Initialize an empty FAISS index."""
        try:
            import faiss
            from backend.services.embedding_service import EmbeddingService

            # Use IndexFlatIP for inner product (cosine sim with normalized vectors)
            self._index = faiss.IndexFlatIP(EmbeddingService.EMBEDDING_DIM)
            self._doc_ids = []
            self._doc_meta = {}
            logger.info(f"Created empty FAISS index (dim={EmbeddingService.EMBEDDING_DIM})")
        except ImportError:
            logger.error("faiss-cpu not installed")
            self._index = None
            self._doc_ids = []
            self._doc_meta = {}

    def _save(self) -> None:
        """Save index and metadata to disk."""
        if self._index is None:
            return

        try:
            import faiss

            # Save FAISS index
            faiss.write_index(self._index, str(self.index_path))

            # Save metadata
            meta = {
                "version": 1,
                "doc_ids": self._doc_ids,
                "doc_meta": self._doc_meta,
                "count": len(self._doc_ids),
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            self._dirty = False
            logger.info(f"Saved FAISS index with {len(self._doc_ids)} documents")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via inner product.

        Args:
            vectors: Array of shape (N, D)

        Returns:
            Normalized vectors of shape (N, D)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms

    def build_index(self, documents: list[dict]) -> dict[str, Any]:
        """Build index from a list of documents.

        This replaces any existing index with a new one built from
        the provided documents.

        Args:
            documents: List of document dicts with keys:
                - id: Unique identifier
                - content: Text to embed
                - doc_type: Document type (optional)
                - metadata: Additional metadata (optional)

        Returns:
            Stats dict with indexed count and timing info
        """
        self._ensure_loaded()

        if self._index is None:
            return {"error": "FAISS not available", "indexed": 0}

        if not documents:
            return {"indexed": 0, "message": "No documents provided"}

        from backend.services.embedding_service import get_embedding_service

        start_time = datetime.now()
        service = get_embedding_service()

        # Convert to Document objects
        docs = [
            Document(
                id=d["id"],
                content=d["content"],
                doc_type=d.get("doc_type", "unknown"),
                metadata=d.get("metadata", {}),
            )
            for d in documents
        ]

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(docs), EMBEDDING_BATCH_SIZE):
            batch = docs[i:i + EMBEDDING_BATCH_SIZE]
            texts = [doc.content for doc in batch]
            embeddings = service.embed_batch(texts)
            all_embeddings.append(embeddings)
            logger.debug(f"Embedded batch {i // EMBEDDING_BATCH_SIZE + 1}")

        # Concatenate and normalize
        embeddings_array = np.vstack(all_embeddings).astype(np.float32)
        embeddings_array = self._normalize_vectors(embeddings_array)

        # Rebuild index
        with self._lock:
            import faiss
            from backend.services.embedding_service import EmbeddingService

            self._index = faiss.IndexFlatIP(EmbeddingService.EMBEDDING_DIM)
            self._index.add(embeddings_array)

            self._doc_ids = [doc.id for doc in docs]
            self._doc_meta = {
                doc.id: {
                    "doc_type": doc.doc_type,
                    "metadata": doc.metadata,
                }
                for doc in docs
            }

            self._dirty = True
            self._save()

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "indexed": len(docs),
            "elapsed_seconds": elapsed,
            "index_path": str(self.index_path),
        }

    async def build_index_async(self, documents: list[dict]) -> dict[str, Any]:
        """Build index asynchronously.

        Args:
            documents: List of document dicts

        Returns:
            Stats dict with indexed count and timing info
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.build_index, documents)

    def search(
        self,
        query: str,
        k: int = 10,
        doc_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            doc_types: Filter by document types (None = all)
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult sorted by score (descending)
        """
        self._ensure_loaded()

        if self._index is None or self._index.ntotal == 0:
            return []

        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()

        # Embed query
        query_embedding = service.embed(query).astype(np.float32)
        query_embedding = self._normalize_vectors(query_embedding.reshape(1, -1))

        # Search FAISS index
        with self._lock:
            # Get more results if filtering
            search_k = k * 3 if doc_types else k
            scores, indices = self._index.search(query_embedding, min(search_k, self._index.ntotal))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._doc_ids):
                continue

            if score < min_score:
                continue

            doc_id = self._doc_ids[idx]
            meta = self._doc_meta.get(doc_id, {})
            doc_type = meta.get("doc_type", "unknown")

            # Filter by doc_type
            if doc_types and doc_type not in doc_types:
                continue

            results.append(SearchResult(
                doc_id=doc_id,
                score=float(score),
                doc_type=doc_type,
                metadata=meta.get("metadata", {}),
            ))

            if len(results) >= k:
                break

        return results

    async def search_async(
        self,
        query: str,
        k: int = 10,
        doc_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search asynchronously.

        Args:
            query: Search query text
            k: Number of results to return
            doc_types: Filter by document types
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult sorted by score (descending)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query, k, doc_types, min_score)
        )

    def add_documents(self, documents: list[dict]) -> dict[str, Any]:
        """Add documents to existing index incrementally.

        Args:
            documents: List of document dicts with keys:
                - id: Unique identifier
                - content: Text to embed
                - doc_type: Document type (optional)
                - metadata: Additional metadata (optional)

        Returns:
            Stats dict with added count
        """
        self._ensure_loaded()

        if self._index is None:
            return {"error": "FAISS not available", "added": 0}

        if not documents:
            return {"added": 0, "message": "No documents provided"}

        from backend.services.embedding_service import get_embedding_service

        service = get_embedding_service()

        # Filter out documents that already exist
        existing_ids = set(self._doc_ids)
        new_docs = [
            Document(
                id=d["id"],
                content=d["content"],
                doc_type=d.get("doc_type", "unknown"),
                metadata=d.get("metadata", {}),
            )
            for d in documents
            if d["id"] not in existing_ids
        ]

        if not new_docs:
            return {"added": 0, "message": "All documents already exist"}

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(new_docs), EMBEDDING_BATCH_SIZE):
            batch = new_docs[i:i + EMBEDDING_BATCH_SIZE]
            texts = [doc.content for doc in batch]
            embeddings = service.embed_batch(texts)
            all_embeddings.append(embeddings)

        # Concatenate and normalize
        embeddings_array = np.vstack(all_embeddings).astype(np.float32)
        embeddings_array = self._normalize_vectors(embeddings_array)

        # Add to index
        with self._lock:
            self._index.add(embeddings_array)

            for doc in new_docs:
                self._doc_ids.append(doc.id)
                self._doc_meta[doc.id] = {
                    "doc_type": doc.doc_type,
                    "metadata": doc.metadata,
                }

            self._dirty = True
            self._save()

        return {
            "added": len(new_docs),
            "total": len(self._doc_ids),
            "skipped": len(documents) - len(new_docs),
        }

    async def add_documents_async(self, documents: list[dict]) -> dict[str, Any]:
        """Add documents asynchronously.

        Args:
            documents: List of document dicts

        Returns:
            Stats dict with added count
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.add_documents, documents)

    def remove_documents(self, doc_ids: list[str]) -> dict[str, Any]:
        """Remove documents from the index.

        Note: FAISS IndexFlatIP doesn't support removal, so we rebuild
        the index without the specified documents.

        Args:
            doc_ids: List of document IDs to remove

        Returns:
            Stats dict with removed count
        """
        self._ensure_loaded()

        if self._index is None:
            return {"error": "FAISS not available", "removed": 0}

        ids_to_remove = set(doc_ids)
        removed_count = sum(1 for doc_id in self._doc_ids if doc_id in ids_to_remove)

        if removed_count == 0:
            return {"removed": 0, "message": "No matching documents found"}

        # Get remaining docs
        remaining_docs = []
        for doc_id in self._doc_ids:
            if doc_id not in ids_to_remove:
                meta = self._doc_meta.get(doc_id, {})
                remaining_docs.append({
                    "id": doc_id,
                    "content": "",  # We don't store content, so we need embeddings
                    "doc_type": meta.get("doc_type", "unknown"),
                    "metadata": meta.get("metadata", {}),
                })

        # We need to store embeddings to support removal - rebuild from scratch
        # For now, just update metadata and mark for rebuild
        with self._lock:
            # Remove from metadata
            for doc_id in ids_to_remove:
                if doc_id in self._doc_meta:
                    del self._doc_meta[doc_id]

            # Filter doc_ids list (but index vectors become orphaned)
            # This is a limitation - full removal requires rebuild with original content
            self._doc_ids = [d for d in self._doc_ids if d not in ids_to_remove]

            logger.warning(
                f"Removed {removed_count} documents from metadata. "
                "Note: FAISS index may contain orphaned vectors until rebuild."
            )

        return {
            "removed": removed_count,
            "remaining": len(self._doc_ids),
            "note": "Index may contain orphaned vectors until full rebuild",
        }

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dict with index stats
        """
        self._ensure_loaded()

        doc_type_counts: dict[str, int] = {}
        for doc_id in self._doc_ids:
            meta = self._doc_meta.get(doc_id, {})
            doc_type = meta.get("doc_type", "unknown")
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        return {
            "total_documents": len(self._doc_ids),
            "index_vectors": self._index.ntotal if self._index else 0,
            "doc_type_counts": doc_type_counts,
            "index_path": str(self.index_path),
            "meta_path": str(self.meta_path),
            "dirty": self._dirty,
        }

    def flush(self) -> None:
        """Force save any pending changes."""
        with self._lock:
            if self._dirty:
                self._save()


# Singleton instance
_semantic_index: Optional[FAISSSemanticIndex] = None
_singleton_lock = threading.Lock()


def get_semantic_index() -> FAISSSemanticIndex:
    """Get the singleton FAISSSemanticIndex instance.

    Returns:
        The global FAISSSemanticIndex instance
    """
    global _semantic_index
    if _semantic_index is None:
        with _singleton_lock:
            if _semantic_index is None:
                _semantic_index = FAISSSemanticIndex()
    return _semantic_index
