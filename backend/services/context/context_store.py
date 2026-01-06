"""ContextStore: Central registry for all ContextVariables.

Manages registration, lookup, and LRU eviction for memory management.
Supports persistence of context metadata to disk.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Callable, Union
import json
import threading

from .context_variable import ContextVariable, ContextType


@dataclass
class ContextStore:
    """Central registry for all ContextVariables.

    Responsibilities:
    - Register and lookup context variables
    - LRU eviction for memory management
    - Persistence of context metadata
    - Statistics and monitoring

    Memory model:
    - Metadata always in memory (~1KB per context)
    - Content loaded on demand
    - LRU eviction when memory budget exceeded
    """

    storage_root: Path
    max_cached_content_mb: float = 50.0  # Memory budget for cached content

    _registry: OrderedDict = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _current_cache_size: int = field(default=0)  # bytes

    def __post_init__(self):
        self.storage_root = Path(self.storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        # Initialize the registry and lock if not already done
        if not isinstance(self._registry, OrderedDict):
            self._registry = OrderedDict()
        # threading.Lock is a factory function, check if it's a lock object
        if not hasattr(self._lock, 'acquire'):
            self._lock = threading.Lock()
        self._load_registry()

    def register(
        self,
        id: str,
        name: str,
        context_type: ContextType,
        source_path: Path | str,
        metadata: Optional[dict] = None,
        content_loader: Optional[Callable[[], str]] = None
    ) -> ContextVariable:
        """Register a new context variable.

        Args:
            id: Unique identifier
            name: Display name
            context_type: Category
            source_path: File path or URI
            metadata: Additional metadata
            content_loader: Custom loader function

        Returns:
            The registered ContextVariable
        """
        with self._lock:
            cv = ContextVariable(
                id=id,
                name=name,
                context_type=context_type,
                source_path=source_path,
                metadata=metadata or {},
                _content_loader=content_loader
            )

            # Update metadata with file info if path exists
            path = Path(source_path) if isinstance(source_path, str) else source_path
            if isinstance(path, Path) and path.exists() and path.is_file():
                try:
                    stat = path.stat()
                    cv.metadata.update({
                        "size_bytes": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except OSError:
                    pass

            self._registry[id] = cv
            return cv

    def get(self, id: str) -> Optional[ContextVariable]:
        """Get a context variable by ID."""
        with self._lock:
            cv = self._registry.get(id)
            if cv:
                # Move to end for LRU tracking
                self._registry.move_to_end(id)
            return cv

    def remove(self, id: str) -> bool:
        """Remove a context variable from the registry."""
        with self._lock:
            if id in self._registry:
                cv = self._registry[id]
                if cv._content_cache:
                    self._current_cache_size -= cv.get_content_size()
                del self._registry[id]
                return True
            return False

    def list_by_type(self, context_type: ContextType) -> list[ContextVariable]:
        """List all contexts of a given type."""
        with self._lock:
            return [
                cv for cv in self._registry.values()
                if cv.context_type == context_type
            ]

    def list_all(self) -> list[dict]:
        """List all context handles (for agent prompts)."""
        with self._lock:
            return [cv.get_handle_info() for cv in self._registry.values()]

    def search(self, query: str) -> list[ContextVariable]:
        """Search across all contexts using grep.

        Returns contexts that have matches for the query.
        Does NOT load content, just searches.
        """
        results = []
        with self._lock:
            for cv in self._registry.values():
                matches = cv.grep(query)
                if matches:
                    cv.metadata["last_search_matches"] = len(matches)
                    results.append(cv)
        return results

    def evict_lru(self, target_bytes: int) -> int:
        """Evict least recently used cached content.

        Args:
            target_bytes: Target cache size after eviction

        Returns:
            Number of contexts evicted
        """
        evicted = 0
        with self._lock:
            # Create a list of IDs to iterate (avoid modifying during iteration)
            ids_to_check = list(self._registry.keys())

            for oldest_id in ids_to_check:
                if self._current_cache_size <= target_bytes:
                    break

                cv = self._registry.get(oldest_id)
                if cv and cv._content_cache:
                    self._current_cache_size -= cv.get_content_size()
                    cv.invalidate_cache()
                    evicted += 1
                    # Move to end so we try others next time
                    self._registry.move_to_end(oldest_id)

        return evicted

    def update_cache_size(self, cv: ContextVariable) -> None:
        """Update the tracked cache size when a context loads content."""
        with self._lock:
            # Calculate total cache size from all contexts
            total = 0
            for context in self._registry.values():
                total += context.get_content_size()
            self._current_cache_size = total

    def get_stats(self) -> dict:
        """Get store statistics."""
        with self._lock:
            loaded = sum(1 for cv in self._registry.values() if cv._loaded)
            return {
                "total_contexts": len(self._registry),
                "loaded_in_memory": loaded,
                "cache_size_mb": self._current_cache_size / (1024 * 1024),
                "max_cache_mb": self.max_cached_content_mb,
                "by_type": {
                    t.value: sum(1 for cv in self._registry.values() if cv.context_type == t)
                    for t in ContextType
                }
            }

    def _load_registry(self) -> None:
        """Load registry metadata from disk on startup."""
        registry_file = self.storage_root / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)
                for item in data.get("contexts", []):
                    try:
                        self.register(
                            id=item["id"],
                            name=item["name"],
                            context_type=ContextType(item["type"]),
                            source_path=Path(item["source_path"]),
                            metadata=item.get("metadata", {})
                        )
                    except (ValueError, KeyError) as e:
                        # Skip invalid entries
                        continue
            except (json.JSONDecodeError, IOError) as e:
                # Log but don't fail - registry can be rebuilt
                pass

    def save_registry(self) -> None:
        """Persist registry metadata to disk."""
        registry_file = self.storage_root / "registry.json"
        with self._lock:
            data = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "contexts": [
                    {
                        "id": cv.id,
                        "name": cv.name,
                        "type": cv.context_type.value,
                        "source_path": str(cv.source_path),
                        "metadata": {
                            k: v for k, v in cv.metadata.items()
                            if isinstance(v, (str, int, float, bool, list, dict))
                        }
                    }
                    for cv in self._registry.values()
                ]
            }
        try:
            with open(registry_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            # Log error but don't raise
            pass

    def clear(self) -> None:
        """Clear all registered contexts."""
        with self._lock:
            self._registry.clear()
            self._current_cache_size = 0


class MemoryBudgetManager:
    """Enforces memory limits for context system."""

    MAX_CACHE_MB = 50
    MAX_SINGLE_LOAD_MB = 10
    EVICTION_THRESHOLD = 0.8  # Evict when 80% full

    def __init__(self, store: ContextStore):
        self.store = store

    def check_load(self, cv: ContextVariable) -> bool:
        """Check if loading this context would exceed budget."""
        size = cv.metadata.get("size_bytes", 0)

        # Single file too large?
        if size > self.MAX_SINGLE_LOAD_MB * 1024 * 1024:
            return False

        # Would exceed cache budget?
        current = self.store._current_cache_size
        max_bytes = self.MAX_CACHE_MB * 1024 * 1024
        if current + size > max_bytes:
            # Try eviction
            target = int(max_bytes * self.EVICTION_THRESHOLD)
            self.store.evict_lru(target)

        return True

    def get_memory_report(self) -> dict:
        """Current memory usage."""
        max_bytes = self.MAX_CACHE_MB * 1024 * 1024
        return {
            "cache_used_mb": self.store._current_cache_size / (1024 * 1024),
            "cache_max_mb": self.MAX_CACHE_MB,
            "utilization": self.store._current_cache_size / max_bytes if max_bytes > 0 else 0,
            "contexts_loaded": sum(1 for cv in self.store._registry.values() if cv._loaded)
        }
