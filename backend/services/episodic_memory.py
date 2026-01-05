"""Episodic Memory - SQLite-backed episode storage with temporal context.

This module provides storage and retrieval for episodic memories:
- Conversation episodes with temporal context
- Emotional tagging (valence and arousal)
- Ebbinghaus decay curves for forgetting
- Consolidation tracking for semantic extraction
- Compressed transcript storage

Part of the tri-memory cognitive architecture.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Generator

from .memory_db import MemoryDatabase, get_memory_db

logger = logging.getLogger(__name__)


@dataclass
class KeyMoment:
    """A significant moment within an episode."""
    timestamp: datetime
    content: str
    importance: float = 0.5
    emotional_valence: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass
class Episode:
    """An episodic memory representing a conversation or event.

    Episodic memories are autobiographical and temporal - they represent
    specific experiences with "when" and "what happened" context.
    """
    id: str
    timestamp: datetime
    summary: str
    duration_seconds: int | None = None
    compressed_transcript: bytes | None = None
    spatial_context: str | None = None
    social_context: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    arousal_level: float = 0.0      # 0.0 (calm) to 1.0 (excited)
    encoding_strength: float = 0.5  # Initial memory strength
    retrieval_count: int = 0
    last_retrieved: datetime | None = None
    decay_rate: float = 0.1
    is_consolidated: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Episode:
        """Create episode from database row."""
        return cls(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            summary=row['summary'],
            duration_seconds=row.get('duration_seconds'),
            compressed_transcript=row.get('compressed_transcript'),
            spatial_context=row.get('spatial_context'),
            social_context=json.loads(row['social_context']) if row.get('social_context') else [],
            emotional_valence=row.get('emotional_valence', 0.0),
            arousal_level=row.get('arousal_level', 0.0),
            encoding_strength=row.get('encoding_strength', 0.5),
            retrieval_count=row.get('retrieval_count', 0),
            last_retrieved=datetime.fromisoformat(row['last_retrieved']) if row.get('last_retrieved') else None,
            decay_rate=row.get('decay_rate', 0.1),
            is_consolidated=bool(row.get('is_consolidated', False)),
            created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row.get('updated_at') else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert episode to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'summary': self.summary,
            'duration_seconds': self.duration_seconds,
            'spatial_context': self.spatial_context,
            'social_context': self.social_context,
            'emotional_valence': self.emotional_valence,
            'arousal_level': self.arousal_level,
            'encoding_strength': self.encoding_strength,
            'retrieval_count': self.retrieval_count,
            'last_retrieved': self.last_retrieved.isoformat() if self.last_retrieved else None,
            'decay_rate': self.decay_rate,
            'is_consolidated': self.is_consolidated,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_transcript(self) -> str | None:
        """Decompress and return transcript."""
        if self.compressed_transcript:
            try:
                return gzip.decompress(self.compressed_transcript).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to decompress transcript: {e}")
                return None
        return None

    def memory_strength(self, now: datetime | None = None) -> float:
        """Calculate current memory strength using Ebbinghaus decay.

        R = S * e^(-t/τ) where:
        - R = retention (memory strength)
        - S = encoding strength (initial strength)
        - t = time since encoding
        - τ = time constant (inverse of decay rate)

        Retrievals reinforce the memory (increase encoding strength).

        Returns:
            Current memory strength (0.0 to 1.0)
        """
        if now is None:
            now = datetime.now()

        # Time since encoding (in hours)
        time_since = (now - self.timestamp).total_seconds() / 3600

        # Time constant (higher = slower decay)
        # Retrieval count boosts the time constant
        time_constant = (1 / self.decay_rate) * (1 + 0.1 * self.retrieval_count)

        # Ebbinghaus retention formula
        retention = self.encoding_strength * math.exp(-time_since / time_constant)

        return max(0.0, min(1.0, retention))


class EpisodicMemory:
    """SQLite-backed episodic memory store.

    Provides storage and retrieval for episodic memories with:
    - Emotional tagging
    - Temporal context
    - Decay calculations
    - Consolidation tracking
    """

    # Decay thresholds
    FORGET_THRESHOLD = 0.1  # Below this, memory is essentially forgotten
    CONSOLIDATE_THRESHOLD = 0.3  # Below this but above forget, consolidate

    def __init__(self, db: MemoryDatabase | None = None):
        """Initialize episodic memory.

        Args:
            db: Memory database instance (uses singleton if None)
        """
        self.db = db or get_memory_db()
        self._lock = threading.Lock()
        logger.info("EpisodicMemory initialized")

    def create_episode(
        self,
        summary: str,
        transcript: str | None = None,
        timestamp: datetime | None = None,
        duration_seconds: int | None = None,
        spatial_context: str | None = None,
        social_context: list[str] | None = None,
        emotional_valence: float = 0.0,
        arousal_level: float = 0.0,
        encoding_strength: float = 0.5,
    ) -> Episode:
        """Create a new episodic memory.

        Args:
            summary: Brief summary of the episode
            transcript: Full transcript (will be gzip compressed)
            timestamp: When the episode occurred (defaults to now)
            duration_seconds: How long the episode lasted
            spatial_context: Where the episode occurred
            social_context: Who was involved
            emotional_valence: Emotional tone (-1 to 1)
            arousal_level: Emotional intensity (0 to 1)
            encoding_strength: Initial memory strength

        Returns:
            The created episode
        """
        episode_id = f"ep-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:5]}"
        now = timestamp or datetime.now()

        # Compress transcript if provided
        compressed = None
        if transcript:
            compressed = gzip.compress(transcript.encode('utf-8'))

        sql = """
            INSERT INTO episodic_memories (
                id, timestamp, duration_seconds, summary, compressed_transcript,
                spatial_context, social_context, emotional_valence, arousal_level,
                encoding_strength, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.db.execute(sql, (
            episode_id,
            now.isoformat(),
            duration_seconds,
            summary,
            compressed,
            spatial_context,
            json.dumps(social_context or []),
            emotional_valence,
            arousal_level,
            encoding_strength,
            now.isoformat(),
            now.isoformat(),
        ))

        episode = Episode(
            id=episode_id,
            timestamp=now,
            summary=summary,
            duration_seconds=duration_seconds,
            compressed_transcript=compressed,
            spatial_context=spatial_context,
            social_context=social_context or [],
            emotional_valence=emotional_valence,
            arousal_level=arousal_level,
            encoding_strength=encoding_strength,
            created_at=now,
            updated_at=now,
        )

        logger.debug(f"Created episode: {episode_id} - {summary[:50]}...")
        return episode

    def get_episode(self, episode_id: str, record_retrieval: bool = True) -> Episode | None:
        """Get an episode by ID.

        Args:
            episode_id: The episode ID
            record_retrieval: Whether to record this retrieval

        Returns:
            The episode or None if not found
        """
        row = self.db.fetchone(
            "SELECT * FROM episodic_memories WHERE id = ?",
            (episode_id,)
        )

        if not row:
            return None

        episode = Episode.from_row(row)

        if record_retrieval:
            self._record_retrieval(episode_id)

        return episode

    def _record_retrieval(self, episode_id: str) -> None:
        """Record episode retrieval and reinforce memory.

        Args:
            episode_id: Episode that was retrieved
        """
        now = datetime.now()

        # Update retrieval count and last retrieved
        self.db.execute(
            """UPDATE episodic_memories
               SET retrieval_count = retrieval_count + 1,
                   last_retrieved = ?,
                   encoding_strength = MIN(1.0, encoding_strength + 0.05),
                   updated_at = ?
               WHERE id = ?""",
            (now.isoformat(), now.isoformat(), episode_id)
        )

    def update_episode(
        self,
        episode_id: str,
        summary: str | None = None,
        emotional_valence: float | None = None,
        arousal_level: float | None = None,
        is_consolidated: bool | None = None,
    ) -> Episode | None:
        """Update an episode.

        Args:
            episode_id: Episode to update
            summary: New summary
            emotional_valence: New emotional valence
            arousal_level: New arousal level
            is_consolidated: Mark as consolidated

        Returns:
            Updated episode or None if not found
        """
        updates = []
        params = []

        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)

        if emotional_valence is not None:
            updates.append("emotional_valence = ?")
            params.append(emotional_valence)

        if arousal_level is not None:
            updates.append("arousal_level = ?")
            params.append(arousal_level)

        if is_consolidated is not None:
            updates.append("is_consolidated = ?")
            params.append(is_consolidated)

        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(episode_id)

            sql = f"UPDATE episodic_memories SET {', '.join(updates)} WHERE id = ?"
            self.db.execute(sql, tuple(params))

        return self.get_episode(episode_id, record_retrieval=False)

    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode.

        Args:
            episode_id: Episode to delete

        Returns:
            True if deleted
        """
        result = self.db.execute(
            "DELETE FROM episodic_memories WHERE id = ?",
            (episode_id,)
        )

        deleted = result.rowcount > 0
        if deleted:
            logger.debug(f"Deleted episode: {episode_id}")

        return deleted

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_recent_episodes(self, limit: int = 20) -> list[Episode]:
        """Get most recent episodes.

        Args:
            limit: Max results

        Returns:
            List of recent episodes
        """
        rows = self.db.fetchall(
            "SELECT * FROM episodic_memories ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return [Episode.from_row(row) for row in rows]

    def get_episodes_in_range(
        self,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> list[Episode]:
        """Get episodes within a time range.

        Args:
            start: Start of range
            end: End of range
            limit: Max results

        Returns:
            List of episodes in range
        """
        rows = self.db.fetchall(
            """SELECT * FROM episodic_memories
               WHERE timestamp BETWEEN ? AND ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (start.isoformat(), end.isoformat(), limit)
        )
        return [Episode.from_row(row) for row in rows]

    def get_emotional_episodes(
        self,
        valence_min: float | None = None,
        valence_max: float | None = None,
        arousal_min: float | None = None,
        limit: int = 20,
    ) -> list[Episode]:
        """Get episodes filtered by emotional state.

        Args:
            valence_min: Minimum valence (e.g., 0.5 for positive only)
            valence_max: Maximum valence (e.g., -0.5 for negative only)
            arousal_min: Minimum arousal (e.g., 0.7 for high-arousal)
            limit: Max results

        Returns:
            List of matching episodes
        """
        conditions = []
        params = []

        if valence_min is not None:
            conditions.append("emotional_valence >= ?")
            params.append(valence_min)

        if valence_max is not None:
            conditions.append("emotional_valence <= ?")
            params.append(valence_max)

        if arousal_min is not None:
            conditions.append("arousal_level >= ?")
            params.append(arousal_min)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        rows = self.db.fetchall(
            f"SELECT * FROM episodic_memories WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params)
        )
        return [Episode.from_row(row) for row in rows]

    def get_unconsolidated_episodes(self, limit: int = 50) -> list[Episode]:
        """Get episodes that haven't been consolidated yet.

        Args:
            limit: Max results

        Returns:
            List of unconsolidated episodes
        """
        rows = self.db.fetchall(
            """SELECT * FROM episodic_memories
               WHERE is_consolidated = FALSE
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit,)
        )
        return [Episode.from_row(row) for row in rows]

    def get_episodes_needing_consolidation(
        self,
        strength_threshold: float = 0.3,
        min_age_hours: int = 24,
        limit: int = 50,
    ) -> list[Episode]:
        """Get episodes that should be consolidated before forgetting.

        Finds episodes that:
        - Are not yet consolidated
        - Are old enough (min_age_hours)
        - Have decaying memory strength

        Args:
            strength_threshold: Consolidate if strength below this
            min_age_hours: Minimum age in hours
            limit: Max results

        Returns:
            List of episodes needing consolidation
        """
        cutoff = datetime.now() - timedelta(hours=min_age_hours)

        rows = self.db.fetchall(
            """SELECT * FROM episodic_memories
               WHERE is_consolidated = FALSE
                 AND timestamp < ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (cutoff.isoformat(), limit * 2)  # Get more, then filter by strength
        )

        episodes = [Episode.from_row(row) for row in rows]

        # Filter by memory strength
        now = datetime.now()
        needing_consolidation = [
            ep for ep in episodes
            if ep.memory_strength(now) < strength_threshold
        ]

        return needing_consolidation[:limit]

    def get_fading_episodes(
        self,
        strength_threshold: float = 0.1,
        limit: int = 20,
    ) -> list[Episode]:
        """Get episodes that are about to be forgotten.

        Args:
            strength_threshold: Episodes below this strength
            limit: Max results

        Returns:
            List of fading episodes
        """
        # Get all unconsolidated episodes and filter by strength
        rows = self.db.fetchall(
            """SELECT * FROM episodic_memories
               WHERE is_consolidated = FALSE
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit * 5,)
        )

        episodes = [Episode.from_row(row) for row in rows]
        now = datetime.now()

        fading = [
            ep for ep in episodes
            if ep.memory_strength(now) < strength_threshold
        ]

        return fading[:limit]

    def search_episodes(self, query: str, limit: int = 10) -> list[Episode]:
        """Search episodes by summary.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching episodes
        """
        rows = self.db.fetchall(
            "SELECT * FROM episodic_memories WHERE summary LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit)
        )
        return [Episode.from_row(row) for row in rows]

    def get_all_episodes_generator(
        self,
        batch_size: int = 100,
    ) -> Generator[Episode, None, None]:
        """Iterate all episodes with minimal memory usage.

        Args:
            batch_size: Rows per batch

        Yields:
            Episode instances
        """
        for row in self.db.fetchall_generator(
            "SELECT * FROM episodic_memories ORDER BY timestamp DESC",
            batch_size=batch_size
        ):
            yield Episode.from_row(row)

    # =========================================================================
    # Maintenance Methods
    # =========================================================================

    def cleanup_forgotten_episodes(
        self,
        strength_threshold: float = 0.05,
        keep_consolidated: bool = True,
    ) -> int:
        """Delete episodes that have been essentially forgotten.

        Args:
            strength_threshold: Delete if strength below this
            keep_consolidated: Don't delete consolidated episodes

        Returns:
            Number of deleted episodes
        """
        # Get candidates
        condition = "is_consolidated = FALSE" if keep_consolidated else "1=1"
        rows = self.db.fetchall(
            f"SELECT id, timestamp, encoding_strength, retrieval_count, decay_rate FROM episodic_memories WHERE {condition}"
        )

        now = datetime.now()
        to_delete = []

        for row in rows:
            # Calculate memory strength
            ep = Episode(
                id=row['id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                summary="",
                encoding_strength=row['encoding_strength'],
                retrieval_count=row['retrieval_count'],
                decay_rate=row['decay_rate'],
            )

            if ep.memory_strength(now) < strength_threshold:
                to_delete.append(row['id'])

        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            self.db.execute(
                f"DELETE FROM episodic_memories WHERE id IN ({placeholders})",
                tuple(to_delete)
            )
            logger.info(f"Cleaned up {len(to_delete)} forgotten episodes")

        return len(to_delete)

    def get_stats(self) -> dict[str, Any]:
        """Get episodic memory statistics.

        Returns:
            Statistics dict
        """
        total = self.db.fetchone("SELECT COUNT(*) as count FROM episodic_memories")
        consolidated = self.db.fetchone(
            "SELECT COUNT(*) as count FROM episodic_memories WHERE is_consolidated = TRUE"
        )

        avg_strength = self.db.fetchone(
            "SELECT AVG(encoding_strength) as avg FROM episodic_memories"
        )

        avg_retrievals = self.db.fetchone(
            "SELECT AVG(retrieval_count) as avg FROM episodic_memories"
        )

        # Get episodes by time range
        now = datetime.now()
        today = self.db.fetchone(
            "SELECT COUNT(*) as count FROM episodic_memories WHERE timestamp >= ?",
            ((now - timedelta(days=1)).isoformat(),)
        )
        week = self.db.fetchone(
            "SELECT COUNT(*) as count FROM episodic_memories WHERE timestamp >= ?",
            ((now - timedelta(days=7)).isoformat(),)
        )
        month = self.db.fetchone(
            "SELECT COUNT(*) as count FROM episodic_memories WHERE timestamp >= ?",
            ((now - timedelta(days=30)).isoformat(),)
        )

        return {
            "total_episodes": total['count'] if total else 0,
            "consolidated_episodes": consolidated['count'] if consolidated else 0,
            "avg_encoding_strength": avg_strength['avg'] if avg_strength and avg_strength['avg'] else 0.0,
            "avg_retrieval_count": avg_retrievals['avg'] if avg_retrievals and avg_retrievals['avg'] else 0.0,
            "episodes_today": today['count'] if today else 0,
            "episodes_this_week": week['count'] if week else 0,
            "episodes_this_month": month['count'] if month else 0,
        }


# Singleton
_episodic_memory: EpisodicMemory | None = None
_memory_lock = threading.Lock()


def get_episodic_memory(db: MemoryDatabase | None = None) -> EpisodicMemory:
    """Get or create the episodic memory singleton.

    Args:
        db: Optional memory database

    Returns:
        The episodic memory singleton
    """
    global _episodic_memory

    if _episodic_memory is None:
        with _memory_lock:
            if _episodic_memory is None:
                _episodic_memory = EpisodicMemory(db)

    return _episodic_memory
