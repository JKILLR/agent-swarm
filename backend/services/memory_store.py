"""Persistent memory store for COO cross-session context.

This module provides a simple key-value store for facts that persist
across chat sessions, enabling the COO to remember user information,
preferences, and key context.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistent memory store for cross-session facts.

    Stores facts in a JSON file that persists across sessions.
    Thread-safe with locking for concurrent access.

    Attributes:
        memory_file: Path to the JSON file storing facts
    """

    def __init__(self, memory_dir: Path):
        """Initialize the memory store.

        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "core_facts.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()
        logger.info(f"MemoryStore initialized at {self.memory_file}")

    def _load(self) -> dict[str, Any]:
        """Load memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data.get('facts', {}))} facts from memory")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load memory: {e}")
                return self._default_structure()
        return self._default_structure()

    def _default_structure(self) -> dict[str, Any]:
        """Return default memory structure."""
        return {
            "facts": {},  # key -> {value, updated_at, source}
            "preferences": {},  # preference_name -> value
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": 1,
            },
        }

    def _save(self) -> None:
        """Save memory to disk."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save memory: {e}")

    def set_fact(self, key: str, value: Any, source: str = "user") -> None:
        """Store a fact.

        Args:
            key: Fact key (e.g., "user_name", "preferred_language")
            value: Fact value
            source: Where this fact came from (user, inferred, system)
        """
        with self._lock:
            self._data["facts"][key] = {
                "value": value,
                "updated_at": datetime.now().isoformat(),
                "source": source,
            }
            self._save()
            logger.info(f"Stored fact: {key}={value} (source: {source})")

    def get_fact(self, key: str) -> Any | None:
        """Get a fact value.

        Args:
            key: Fact key

        Returns:
            Fact value or None if not found
        """
        with self._lock:
            fact = self._data["facts"].get(key)
            return fact["value"] if fact else None

    def get_all_facts(self) -> dict[str, Any]:
        """Get all facts as a simple key-value dict.

        Returns:
            Dictionary of fact_key -> value
        """
        with self._lock:
            return {k: v["value"] for k, v in self._data["facts"].items()}

    def get_facts_detailed(self) -> dict[str, dict]:
        """Get all facts with metadata.

        Returns:
            Dictionary of fact_key -> {value, updated_at, source}
        """
        with self._lock:
            return dict(self._data["facts"])

    def delete_fact(self, key: str) -> bool:
        """Delete a fact.

        Args:
            key: Fact key to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._data["facts"]:
                del self._data["facts"][key]
                self._save()
                logger.info(f"Deleted fact: {key}")
                return True
            return False

    def clear_all(self) -> None:
        """Clear all facts."""
        with self._lock:
            self._data = self._default_structure()
            self._save()
            logger.info("Cleared all memory")

    def set_preference(self, name: str, value: Any) -> None:
        """Store a user preference.

        Args:
            name: Preference name
            value: Preference value
        """
        with self._lock:
            self._data["preferences"][name] = {
                "value": value,
                "updated_at": datetime.now().isoformat(),
            }
            self._save()
            logger.info(f"Stored preference: {name}={value}")

    def get_preference(self, name: str) -> Any | None:
        """Get a preference value.

        Args:
            name: Preference name

        Returns:
            Preference value or None if not found
        """
        with self._lock:
            pref = self._data["preferences"].get(name)
            return pref["value"] if pref else None

    def get_all_preferences(self) -> dict[str, Any]:
        """Get all preferences as a simple dict.

        Returns:
            Dictionary of preference_name -> value
        """
        with self._lock:
            return {k: v["value"] for k, v in self._data["preferences"].items()}

    def get_context_for_prompt(self) -> str:
        """Generate context string for injection into system prompt.

        Returns:
            Formatted string with all facts and preferences
        """
        with self._lock:
            lines = []

            # Access _data directly to avoid deadlock (we already hold the lock)
            facts = {k: v["value"] for k, v in self._data["facts"].items()}
            if facts:
                lines.append("### Known Facts About User")
                for key, value in facts.items():
                    # Format key nicely: user_name -> User Name
                    nice_key = key.replace("_", " ").title()
                    lines.append(f"- {nice_key}: {value}")

            prefs = {k: v["value"] for k, v in self._data["preferences"].items()}
            if prefs:
                lines.append("\n### User Preferences")
                for name, value in prefs.items():
                    nice_name = name.replace("_", " ").title()
                    lines.append(f"- {nice_name}: {value}")

            if not lines:
                return ""

            return "\n".join(lines)


# Singleton instance
_memory_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store(memory_dir: Path | str | None = None) -> MemoryStore:
    """Get or create the global memory store.

    Args:
        memory_dir: Directory for memory files (used on first call)

    Returns:
        The memory store singleton
    """
    global _memory_store

    if _memory_store is None:
        with _store_lock:
            if _memory_store is None:
                if memory_dir is None:
                    memory_dir = Path(__file__).parent.parent.parent / "logs" / "memory"
                _memory_store = MemoryStore(Path(memory_dir))

    return _memory_store
