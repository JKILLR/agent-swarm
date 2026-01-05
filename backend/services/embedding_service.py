# backend/services/embedding_service.py

"""Embedding service for semantic search operations.

Provides a thread-safe singleton for managing sentence-transformers
embedding model with lazy loading and batch operations.
"""

import logging
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages embedding model and operations.

    Features:
    - Lazy model loading (only loads when first needed)
    - Configurable model (default: all-MiniLM-L6-v2)
    - Batch embedding for efficiency
    - Thread-safe singleton pattern
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self):
        self._model: "SentenceTransformer | None" = None
        self._lock = threading.Lock()

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load embedding model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(f"Loading embedding model: {self.MODEL_NAME}")
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.MODEL_NAME)
                    logger.info(f"Embedding model loaded successfully")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns 384-dim vector."""
        logger.debug(f"Embedding text of length {len(text)}")
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, 384) array."""
        logger.debug(f"Batch embedding {len(texts)} texts")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 if either vector is zero-length to avoid division by zero.
        """
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Singleton
_embedding_service: Optional[EmbeddingService] = None
_singleton_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """Get the singleton EmbeddingService instance.

    Thread-safe accessor for the global embedding service.
    """
    global _embedding_service
    if _embedding_service is None:
        with _singleton_lock:
            if _embedding_service is None:
                logger.info("Creating EmbeddingService singleton")
                _embedding_service = EmbeddingService()
    return _embedding_service
