# RLM-Inspired Context System Architecture for Life OS

## Version: 1.0 | Date: 2026-01-06 | Target: 8GB M2 Mac Mini

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy: Context as Queryable Environment](#2-core-philosophy)
3. [Core Abstractions](#3-core-abstractions)
4. [File/Folder Structure](#4-filefolder-structure)
5. [API Design](#5-api-design)
6. [Integration Points](#6-integration-points)
7. [Memory Budget](#7-memory-budget)
8. [Comparison: RLM vs AI-Corp vs This Design](#8-comparison)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Executive Summary

### The Problem with Traditional Context Systems

Most agent context systems suffer from **context preloading bloat**:
- Load everything into the prompt upfront
- Token limits hit quickly
- No mechanism for agents to explore context strategically
- Static context can't respond to agent queries

### The RLM Insight: Context as Queryable Environment

The Recursive Language Model (RLM) paper introduces a paradigm shift:

> **Context is not data to be loaded, but an ENVIRONMENT to be explored.**

Instead of preloading context into prompts, agents:
1. Receive **handles** to context (ContextVariables)
2. **Write code** to explore context (peek, grep, chunk)
3. **Decide** what to load based on exploration
4. **Recursively spawn** sub-agents for divide-and-conquer

### Key Insights We Incorporate

| RLM Concept | Our Implementation |
|-------------|-------------------|
| ContextVariable | `ContextVariable` class with lazy operations |
| Code-based exploration | Agents call `peek()`, `grep()`, `chunk()` via tools |
| Recursive spawning | Sub-agents via `Task` tool for deep exploration |
| Content on disk | All context stored in files, loaded on demand |
| Emergent strategies | Agents learn to navigate context efficiently |

---

## 2. Core Philosophy

### 2.1 Context as Environment, Not Payload

**Traditional approach** (AI-Corp pattern):
```python
# Preload everything into prompt
context = load_all_relevant_context()
prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuery: {query}"
```

**RLM-inspired approach** (our design):
```python
# Give agent handles to context, let it explore
context_handles = [
    ContextVariable("user_profile", type="identity"),
    ContextVariable("project_langley5", type="project"),
    ContextVariable("recent_emails", type="temporal"),
]
prompt = f"""
You have access to context variables:
{format_handles(context_handles)}

Use peek(), grep(), chunk() to explore before loading.
Only call get_full() when you KNOW you need the content.
"""
```

### 2.2 The Three Laws of Context

1. **Never preload what you can peek**
   - Always preview with `peek()` first (~100 tokens)
   - Full content only when necessary

2. **Search before loading**
   - `grep()` finds relevant sections without loading all
   - Pattern matching is O(log n) not O(n)

3. **Chunk large content**
   - `chunk()` paginates large contexts
   - Agent navigates like a reader, not a buffer

### 2.3 Emergent Navigation Strategies

Agents naturally develop patterns:

```
Strategy: "Quick Survey"
1. peek() all available contexts
2. grep() for query keywords
3. get_full() only matching sections

Strategy: "Deep Dive"
1. peek() to understand structure
2. chunk(0) for first page
3. chunk(n) to navigate to relevant section
4. spawn sub-agent for detailed analysis

Strategy: "Targeted Extraction"
1. grep(pattern) to locate
2. chunk() around matches
3. Extract and summarize
```

---

## 3. Core Abstractions

### 3.1 ContextVariable

The atomic unit of explorable context.

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
import re

class ContextType(Enum):
    """Types of context with different access patterns."""
    IDENTITY = "identity"       # User profile, always available
    PROJECT = "project"         # Project-specific, lazy loaded
    TEMPORAL = "temporal"       # Time-series data (messages, events)
    DOCUMENT = "document"       # External documents
    MEMORY = "memory"           # MindGraph nodes
    TOOL_RESULT = "tool_result" # Cached tool outputs
    WEB = "web"                 # Fetched web content


@dataclass
class ContextVariable:
    """A handle to explorable context content.

    Key insight: Content stays on disk until explicitly requested.
    Agents explore via peek/grep/chunk before deciding to load.

    Attributes:
        id: Unique identifier
        name: Human-readable name for agent display
        context_type: Category of context
        source_path: File path or URI to content
        metadata: Access timestamps, size, etc.
        _content_loader: Function to load full content
        _content_cache: Cached content after first load
    """
    id: str
    name: str
    context_type: ContextType
    source_path: Path | str
    metadata: dict = field(default_factory=dict)

    # Private: lazy loading machinery
    _content_loader: Optional[Callable[[], str]] = field(default=None, repr=False)
    _content_cache: Optional[str] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    # Access tracking
    peek_count: int = field(default=0)
    grep_count: int = field(default=0)
    load_count: int = field(default=0)
    last_accessed: Optional[datetime] = field(default=None)

    def peek(self, lines: int = 10, tokens: int = 100) -> str:
        """Preview content without full loading.

        Returns first N lines or ~100 tokens, whichever is smaller.
        Designed to fit in a single LLM response for quick survey.

        Args:
            lines: Max lines to return
            tokens: Approximate token budget (~4 chars/token)

        Returns:
            Preview string with truncation indicator if needed
        """
        self.peek_count += 1
        self.last_accessed = datetime.now()

        content = self._get_raw_content()
        if not content:
            return "[empty]"

        # Get first N lines
        content_lines = content.split('\n')[:lines]
        preview = '\n'.join(content_lines)

        # Truncate by tokens (~4 chars per token)
        char_limit = tokens * 4
        if len(preview) > char_limit:
            preview = preview[:char_limit] + "... [truncated]"

        total_lines = len(content.split('\n'))
        if total_lines > lines:
            preview += f"\n[{total_lines - lines} more lines]"

        return preview

    def grep(self, pattern: str, context_lines: int = 2) -> list[dict]:
        """Search content for pattern without loading all.

        Returns matches with surrounding context, enabling
        targeted loading of relevant sections.

        Args:
            pattern: Regex pattern to search for
            context_lines: Lines of context around each match

        Returns:
            List of {line_num, match, context} dicts
        """
        self.grep_count += 1
        self.last_accessed = datetime.now()

        content = self._get_raw_content()
        if not content:
            return []

        lines = content.split('\n')
        results = []
        regex = re.compile(pattern, re.IGNORECASE)

        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                results.append({
                    "line_num": i + 1,
                    "match": line.strip(),
                    "context": '\n'.join(lines[start:end]),
                    "pattern": pattern
                })

        return results

    def chunk(self, chunk_index: int = 0, chunk_size: int = 50) -> dict:
        """Get a specific chunk of content.

        Enables pagination through large contexts without
        loading everything at once.

        Args:
            chunk_index: Which chunk to retrieve (0-indexed)
            chunk_size: Lines per chunk

        Returns:
            Dict with chunk content and navigation info
        """
        content = self._get_raw_content()
        if not content:
            return {"content": "", "chunk": 0, "total_chunks": 0}

        lines = content.split('\n')
        total_chunks = (len(lines) + chunk_size - 1) // chunk_size

        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(lines))

        return {
            "content": '\n'.join(lines[start:end]),
            "chunk": chunk_index,
            "total_chunks": total_chunks,
            "lines": f"{start+1}-{end} of {len(lines)}"
        }

    def get_full(self) -> str:
        """Load and return full content.

        Use sparingly! Prefer peek/grep/chunk for exploration.
        Content is cached after first load.

        Returns:
            Full content string
        """
        self.load_count += 1
        self.last_accessed = datetime.now()
        self._loaded = True

        return self._get_raw_content()

    def _get_raw_content(self) -> str:
        """Internal: Get content from cache or loader."""
        if self._content_cache is not None:
            return self._content_cache

        if self._content_loader:
            self._content_cache = self._content_loader()
        elif isinstance(self.source_path, Path) and self.source_path.exists():
            self._content_cache = self.source_path.read_text()
        else:
            self._content_cache = ""

        return self._content_cache

    def invalidate_cache(self) -> None:
        """Clear cached content to force reload on next access."""
        self._content_cache = None
        self._loaded = False

    def get_handle_info(self) -> dict:
        """Return serializable info for agent prompts."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.context_type.value,
            "size_hint": self.metadata.get("size_bytes", "unknown"),
            "last_modified": self.metadata.get("last_modified", "unknown"),
            "access_stats": {
                "peeks": self.peek_count,
                "greps": self.grep_count,
                "full_loads": self.load_count
            }
        }
```

### 3.2 ContextStore

Registry and manager for all context variables.

```python
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
import json
import threading
import yaml


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

    _registry: OrderedDict[str, ContextVariable] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _current_cache_size: int = field(default=0)  # bytes

    def __post_init__(self):
        self.storage_root = Path(self.storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
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
            if isinstance(source_path, Path) and source_path.exists():
                stat = source_path.stat()
                cv.metadata.update({
                    "size_bytes": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

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
            while self._current_cache_size > target_bytes:
                # Get oldest (first in OrderedDict)
                oldest_id = next(iter(self._registry), None)
                if not oldest_id:
                    break

                cv = self._registry[oldest_id]
                if cv._content_cache:
                    self._current_cache_size -= len(cv._content_cache)
                    cv.invalidate_cache()
                    evicted += 1

                # Move to end so we try others next time
                self._registry.move_to_end(oldest_id)

        return evicted

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
                    self.register(
                        id=item["id"],
                        name=item["name"],
                        context_type=ContextType(item["type"]),
                        source_path=Path(item["source_path"]),
                        metadata=item.get("metadata", {})
                    )
            except Exception as e:
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
                        "metadata": cv.metadata
                    }
                    for cv in self._registry.values()
                ]
            }
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)
```

### 3.3 ContextNavigator

Agent-facing tool for context exploration.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextNavigator:
    """Tool interface for agents to explore context.

    This is exposed to agents as a set of tool calls:
    - context_list(): See available contexts
    - context_peek(id): Preview a context
    - context_grep(id, pattern): Search within context
    - context_chunk(id, chunk_num): Paginate through context
    - context_load(id): Load full content

    The navigator tracks access patterns to learn agent strategies.
    """

    store: ContextStore

    # Access pattern tracking
    access_log: list[dict] = field(default_factory=list)

    def list_contexts(
        self,
        context_type: Optional[str] = None
    ) -> list[dict]:
        """List available contexts with handles.

        Agent tool: Shows what contexts exist without loading any.

        Args:
            context_type: Filter by type (identity, project, etc.)

        Returns:
            List of context handles with metadata
        """
        if context_type:
            try:
                ct = ContextType(context_type)
                return [cv.get_handle_info() for cv in self.store.list_by_type(ct)]
            except ValueError:
                return []
        return self.store.list_all()

    def peek(self, context_id: str, lines: int = 10) -> dict:
        """Preview a context.

        Agent tool: Quick look at content structure.

        Args:
            context_id: ID of context to peek
            lines: Number of lines to preview

        Returns:
            {preview, metadata, suggestions}
        """
        cv = self.store.get(context_id)
        if not cv:
            return {"error": f"Context '{context_id}' not found"}

        self._log_access("peek", context_id)

        preview = cv.peek(lines=lines)
        return {
            "preview": preview,
            "metadata": cv.metadata,
            "suggestions": self._suggest_next_action(cv, "peek")
        }

    def grep(
        self,
        context_id: str,
        pattern: str,
        context_lines: int = 2
    ) -> dict:
        """Search within a context.

        Agent tool: Find relevant sections without loading all.

        Args:
            context_id: ID of context to search
            pattern: Regex pattern
            context_lines: Context around matches

        Returns:
            {matches, total_matches, suggestions}
        """
        cv = self.store.get(context_id)
        if not cv:
            return {"error": f"Context '{context_id}' not found"}

        self._log_access("grep", context_id, {"pattern": pattern})

        matches = cv.grep(pattern, context_lines)
        return {
            "matches": matches[:20],  # Limit to prevent token explosion
            "total_matches": len(matches),
            "truncated": len(matches) > 20,
            "suggestions": self._suggest_next_action(cv, "grep", matches)
        }

    def chunk(
        self,
        context_id: str,
        chunk_index: int = 0,
        chunk_size: int = 50
    ) -> dict:
        """Get a chunk of context.

        Agent tool: Paginate through large contexts.

        Args:
            context_id: ID of context
            chunk_index: Which chunk (0-indexed)
            chunk_size: Lines per chunk

        Returns:
            {content, chunk_info, navigation}
        """
        cv = self.store.get(context_id)
        if not cv:
            return {"error": f"Context '{context_id}' not found"}

        self._log_access("chunk", context_id, {"chunk": chunk_index})

        chunk_data = cv.chunk(chunk_index, chunk_size)
        return {
            **chunk_data,
            "navigation": {
                "prev": chunk_index - 1 if chunk_index > 0 else None,
                "next": chunk_index + 1 if chunk_index < chunk_data["total_chunks"] - 1 else None
            }
        }

    def load(self, context_id: str) -> dict:
        """Load full context content.

        Agent tool: Get everything. Use sparingly!

        Args:
            context_id: ID of context to load

        Returns:
            {content, size, warning}
        """
        cv = self.store.get(context_id)
        if not cv:
            return {"error": f"Context '{context_id}' not found"}

        self._log_access("load", context_id)

        content = cv.get_full()
        size = len(content)

        result = {
            "content": content,
            "size_chars": size,
            "size_tokens_approx": size // 4
        }

        # Warn if large
        if size > 10000:
            result["warning"] = f"Large content ({size // 4} tokens). Consider using grep/chunk."

        return result

    def search_all(self, query: str) -> list[dict]:
        """Search across all contexts.

        Agent tool: Find which contexts contain relevant info.

        Args:
            query: Search query

        Returns:
            List of contexts with match counts
        """
        self._log_access("search_all", None, {"query": query})

        matching = self.store.search(query)
        return [
            {
                "context_id": cv.id,
                "name": cv.name,
                "type": cv.context_type.value,
                "match_count": cv.metadata.get("last_search_matches", 0)
            }
            for cv in matching
        ]

    def _log_access(
        self,
        action: str,
        context_id: Optional[str],
        params: Optional[dict] = None
    ) -> None:
        """Log access for pattern learning."""
        self.access_log.append({
            "action": action,
            "context_id": context_id,
            "params": params,
            "timestamp": datetime.now().isoformat()
        })

        # Keep log bounded
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]

    def _suggest_next_action(
        self,
        cv: ContextVariable,
        last_action: str,
        matches: Optional[list] = None
    ) -> list[str]:
        """Suggest next exploration actions."""
        suggestions = []

        if last_action == "peek":
            suggestions.append(f"Use grep('{cv.id}', 'keyword') to search")
            if cv.metadata.get("size_bytes", 0) > 10000:
                suggestions.append(f"Use chunk('{cv.id}', 0) to paginate")

        if last_action == "grep" and matches:
            if len(matches) > 5:
                suggestions.append("Many matches - refine pattern or use chunk()")
            elif len(matches) == 0:
                suggestions.append("No matches - try different keywords")
            else:
                suggestions.append(f"Found {len(matches)} matches - load() if needed")

        return suggestions

    def get_access_summary(self) -> dict:
        """Summarize access patterns for debugging/learning."""
        actions = {}
        for entry in self.access_log:
            action = entry["action"]
            actions[action] = actions.get(action, 0) + 1

        return {
            "total_accesses": len(self.access_log),
            "by_action": actions,
            "load_to_peek_ratio": actions.get("load", 0) / max(actions.get("peek", 1), 1)
        }
```

### 3.4 ContextFactory

Creates and populates ContextVariables from various sources.

```python
from pathlib import Path
from typing import Callable, Optional
import os
import yaml
import json


class ContextFactory:
    """Factory for creating ContextVariables from various sources.

    Supports:
    - File-based contexts (YAML, JSON, Markdown, etc.)
    - Dynamic contexts (function-generated)
    - Memory contexts (MindGraph integration)
    - Temporal contexts (messages, calendar)
    """

    def __init__(self, store: ContextStore):
        self.store = store

    def from_file(
        self,
        path: Path,
        id: Optional[str] = None,
        name: Optional[str] = None,
        context_type: ContextType = ContextType.DOCUMENT
    ) -> ContextVariable:
        """Create context from a file.

        Args:
            path: Path to file
            id: Custom ID (defaults to path stem)
            name: Display name (defaults to filename)
            context_type: Type classification

        Returns:
            Registered ContextVariable
        """
        path = Path(path)
        return self.store.register(
            id=id or path.stem,
            name=name or path.name,
            context_type=context_type,
            source_path=path,
            metadata={"file_type": path.suffix}
        )

    def from_directory(
        self,
        directory: Path,
        pattern: str = "*",
        context_type: ContextType = ContextType.DOCUMENT,
        id_prefix: str = ""
    ) -> list[ContextVariable]:
        """Create contexts from all files in a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files
            context_type: Type for all contexts
            id_prefix: Prefix for IDs

        Returns:
            List of registered ContextVariables
        """
        contexts = []
        directory = Path(directory)

        for path in directory.glob(pattern):
            if path.is_file():
                cv = self.from_file(
                    path,
                    id=f"{id_prefix}{path.stem}" if id_prefix else None,
                    context_type=context_type
                )
                contexts.append(cv)

        return contexts

    def from_function(
        self,
        id: str,
        name: str,
        loader: Callable[[], str],
        context_type: ContextType,
        metadata: Optional[dict] = None
    ) -> ContextVariable:
        """Create context from a dynamic loader function.

        The loader is called lazily when content is accessed.
        Useful for:
        - Database queries
        - API calls
        - Computed content

        Args:
            id: Unique identifier
            name: Display name
            loader: Function that returns content string
            context_type: Type classification
            metadata: Additional metadata

        Returns:
            Registered ContextVariable
        """
        return self.store.register(
            id=id,
            name=name,
            context_type=context_type,
            source_path="<dynamic>",
            metadata=metadata or {},
            content_loader=loader
        )

    def from_yaml_collection(
        self,
        base_path: Path,
        files: list[str],
        context_type: ContextType,
        id_prefix: str = ""
    ) -> list[ContextVariable]:
        """Create contexts from a set of YAML files.

        Commonly used for foundation context:
        - profile.yaml
        - preferences.yaml
        - communication_style.yaml
        """
        contexts = []
        base = Path(base_path)

        for filename in files:
            path = base / filename
            if path.exists():
                cv = self.store.register(
                    id=f"{id_prefix}{path.stem}",
                    name=path.stem.replace("_", " ").title(),
                    context_type=context_type,
                    source_path=path,
                    metadata={"format": "yaml"}
                )
                contexts.append(cv)

        return contexts

    def from_mindgraph_query(
        self,
        id: str,
        name: str,
        query: str,
        graph: "MindGraph"
    ) -> ContextVariable:
        """Create context from MindGraph semantic search.

        Args:
            id: Unique identifier
            name: Display name
            query: Semantic search query
            graph: MindGraph instance

        Returns:
            ContextVariable with search results
        """
        def loader():
            from backend.services.semantic_index import SemanticIndex
            results = graph.semantic_index.search(query, limit=20)
            lines = []
            for r in results:
                lines.append(f"[{r.similarity:.2f}] {r.node.label}")
                if r.node.description:
                    lines.append(f"   {r.node.description}")
            return '\n'.join(lines)

        return self.from_function(
            id=id,
            name=name,
            loader=loader,
            context_type=ContextType.MEMORY,
            metadata={"query": query, "source": "mindgraph"}
        )
```

---

## 4. File/Folder Structure

### 4.1 Implementation Layout

```
backend/
├── services/
│   └── context/                    # NEW: Context system module
│       ├── __init__.py             # Exports main classes
│       ├── context_variable.py     # ContextVariable class
│       ├── context_store.py        # ContextStore registry
│       ├── context_navigator.py    # Agent-facing navigator
│       ├── context_factory.py      # Factory methods
│       ├── context_tools.py        # Tool definitions for agents
│       └── context_loader.py       # Source-specific loaders
├── routes/
│   └── context.py                  # REST endpoints (EXISTING - enhance)

memory/
└── context/                        # Context storage root
    ├── registry.json               # Context registry metadata
    ├── foundation/                 # Identity contexts
    │   ├── profile.yaml
    │   ├── preferences.yaml
    │   └── communication_style.yaml
    ├── projects/                   # Project contexts
    │   └── langley_5/
    │       ├── project.yaml
    │       ├── trades.yaml
    │       └── contacts.yaml
    ├── temporal/                   # Time-series contexts
    │   ├── recent_messages.json
    │   └── upcoming_events.json
    └── cache/                      # Ephemeral cached contexts
        └── tool_results/
```

### 4.2 Integration with Existing Structure

The context system integrates with existing Life OS components:

```
EXISTING:                           INTEGRATION:
backend/services/
├── context_service.py         →    Wraps new ContextStore
├── embedding_service.py       →    Used for semantic grep
├── semantic_index.py          →    MindGraph context queries
├── mind_graph.py              →    Memory context source

memory/
├── context/                   →    Foundation/project contexts
└── graph/                     →    MindGraph semantic memory

swarms/life_os/
├── agents/                    →    Agents use ContextNavigator
└── workspace/                 →    Working context storage
```

---

## 5. API Design

### 5.1 Python API (Internal)

```python
# backend/services/context/__init__.py

from .context_variable import ContextVariable, ContextType
from .context_store import ContextStore
from .context_navigator import ContextNavigator
from .context_factory import ContextFactory

# Singleton access
_store: Optional[ContextStore] = None
_navigator: Optional[ContextNavigator] = None


def get_context_store() -> ContextStore:
    """Get the singleton context store."""
    global _store
    if _store is None:
        from pathlib import Path
        storage_root = Path(__file__).parent.parent.parent.parent / "memory" / "context"
        _store = ContextStore(storage_root=storage_root)
        _initialize_default_contexts(_store)
    return _store


def get_context_navigator() -> ContextNavigator:
    """Get the singleton context navigator."""
    global _navigator
    if _navigator is None:
        _navigator = ContextNavigator(store=get_context_store())
    return _navigator


def _initialize_default_contexts(store: ContextStore) -> None:
    """Register default contexts (foundation, etc.)."""
    factory = ContextFactory(store)
    foundation_path = store.storage_root / "foundation"

    if foundation_path.exists():
        factory.from_yaml_collection(
            base_path=foundation_path,
            files=["profile.yaml", "preferences.yaml", "communication_style.yaml"],
            context_type=ContextType.IDENTITY,
            id_prefix="identity:"
        )
```

### 5.2 Agent Tool Definitions

```python
# backend/services/context/context_tools.py

"""Tool definitions for agents to use context system.

These are registered with the agent executor and exposed
to Claude via the standard tool interface.
"""

CONTEXT_TOOLS = [
    {
        "name": "context_list",
        "description": """List available contexts. Returns handles (not content).

Use this first to see what contexts exist before exploring.

Args:
    type: Optional filter by type (identity, project, temporal, document, memory)

Returns: List of context handles with metadata (id, name, type, size)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["identity", "project", "temporal", "document", "memory"],
                    "description": "Filter by context type"
                }
            }
        }
    },
    {
        "name": "context_peek",
        "description": """Preview a context without loading all content.

Returns first ~100 tokens to understand structure and relevance.
Use this before deciding whether to load full content.

Args:
    context_id: ID of the context to preview
    lines: Number of lines to preview (default: 10)

Returns: Preview text with truncation info""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string",
                    "description": "ID of context to preview"
                },
                "lines": {
                    "type": "integer",
                    "default": 10,
                    "description": "Lines to preview"
                }
            },
            "required": ["context_id"]
        }
    },
    {
        "name": "context_grep",
        "description": """Search within a context for a pattern.

Finds relevant sections without loading all content.
Returns matches with surrounding context lines.

Args:
    context_id: ID of context to search
    pattern: Regex pattern to search for
    context_lines: Lines of context around matches (default: 2)

Returns: List of matches with line numbers and context""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string",
                    "description": "ID of context to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern"
                },
                "context_lines": {
                    "type": "integer",
                    "default": 2
                }
            },
            "required": ["context_id", "pattern"]
        }
    },
    {
        "name": "context_chunk",
        "description": """Get a specific chunk of a context.

Use for paginating through large contexts.
Each chunk is ~50 lines by default.

Args:
    context_id: ID of context
    chunk_index: Which chunk (0-indexed)
    chunk_size: Lines per chunk (default: 50)

Returns: Chunk content with navigation info""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string"
                },
                "chunk_index": {
                    "type": "integer",
                    "default": 0
                },
                "chunk_size": {
                    "type": "integer",
                    "default": 50
                }
            },
            "required": ["context_id"]
        }
    },
    {
        "name": "context_load",
        "description": """Load full content of a context.

WARNING: Use sparingly! Prefer peek/grep/chunk for exploration.
Only load full content when you specifically need it all.

Args:
    context_id: ID of context to load

Returns: Full content (may be large)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string"
                }
            },
            "required": ["context_id"]
        }
    },
    {
        "name": "context_search",
        "description": """Search across ALL contexts for a query.

Finds which contexts contain relevant information.
Use to discover where information lives.

Args:
    query: Search query

Returns: List of contexts with match counts""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
]


def handle_context_tool(
    tool_name: str,
    tool_input: dict,
    navigator: "ContextNavigator"
) -> dict:
    """Route tool calls to navigator methods."""
    handlers = {
        "context_list": lambda: navigator.list_contexts(tool_input.get("type")),
        "context_peek": lambda: navigator.peek(
            tool_input["context_id"],
            tool_input.get("lines", 10)
        ),
        "context_grep": lambda: navigator.grep(
            tool_input["context_id"],
            tool_input["pattern"],
            tool_input.get("context_lines", 2)
        ),
        "context_chunk": lambda: navigator.chunk(
            tool_input["context_id"],
            tool_input.get("chunk_index", 0),
            tool_input.get("chunk_size", 50)
        ),
        "context_load": lambda: navigator.load(tool_input["context_id"]),
        "context_search": lambda: navigator.search_all(tool_input["query"])
    }

    handler = handlers.get(tool_name)
    if handler:
        return handler()
    return {"error": f"Unknown tool: {tool_name}"}
```

### 5.3 REST API Endpoints

```python
# backend/routes/context.py (ENHANCED)

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List

router = APIRouter(prefix="/api/context", tags=["context"])


@router.get("/")
async def list_contexts(
    type: Optional[str] = Query(None, description="Filter by context type")
):
    """List all available context handles.

    Returns metadata only, not content.
    Use /peek, /grep, /chunk to explore.
    """
    from backend.services.context import get_context_navigator
    navigator = get_context_navigator()
    return {"contexts": navigator.list_contexts(type)}


@router.get("/{context_id}/peek")
async def peek_context(
    context_id: str,
    lines: int = Query(10, le=50, description="Lines to preview")
):
    """Preview a context without loading all content."""
    from backend.services.context import get_context_navigator
    navigator = get_context_navigator()
    result = navigator.peek(context_id, lines)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{context_id}/grep")
async def grep_context(
    context_id: str,
    pattern: str = Query(..., description="Regex pattern to search"),
    context_lines: int = Query(2, le=10)
):
    """Search within a context."""
    from backend.services.context import get_context_navigator
    navigator = get_context_navigator()
    result = navigator.grep(context_id, pattern, context_lines)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{context_id}/chunk/{chunk_index}")
async def chunk_context(
    context_id: str,
    chunk_index: int = 0,
    chunk_size: int = Query(50, le=200)
):
    """Get a specific chunk of a context."""
    from backend.services.context import get_context_navigator
    navigator = get_context_navigator()
    result = navigator.chunk(context_id, chunk_index, chunk_size)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{context_id}")
async def load_context(context_id: str):
    """Load full context content.

    Warning: May return large content. Prefer /peek, /grep, /chunk.
    """
    from backend.services.context import get_context_navigator
    navigator = get_context_navigator()
    result = navigator.load(context_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/search/all")
async def search_all_contexts(
    q: str = Query(..., description="Search query")
):
    """Search across all contexts."""
    from backend.services.context import get_context_navigator
    navigator = get_context_navigator()
    return {"results": navigator.search_all(q)}


@router.get("/stats")
async def get_stats():
    """Get context store statistics."""
    from backend.services.context import get_context_store, get_context_navigator
    store = get_context_store()
    navigator = get_context_navigator()
    return {
        "store": store.get_stats(),
        "access_patterns": navigator.get_access_summary()
    }


@router.post("/register")
async def register_context(
    id: str = Body(...),
    name: str = Body(...),
    type: str = Body(...),
    source_path: str = Body(...),
    metadata: Optional[dict] = Body(None)
):
    """Register a new context variable."""
    from backend.services.context import get_context_store, ContextType
    from pathlib import Path

    store = get_context_store()
    try:
        context_type = ContextType(type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid type: {type}")

    cv = store.register(
        id=id,
        name=name,
        context_type=context_type,
        source_path=Path(source_path),
        metadata=metadata
    )
    return {"registered": cv.get_handle_info()}
```

---

## 6. Integration Points

### 6.1 Existing ContextService Integration

Wrap the existing `ContextService` to use the new RLM-style system:

```python
# backend/services/context_service.py (ENHANCED)

from backend.services.context import (
    get_context_store,
    get_context_navigator,
    ContextFactory,
    ContextType
)

class ContextService:
    """Enhanced context service with RLM-style exploration.

    Maintains backward compatibility with existing API while
    adding new exploration capabilities.
    """

    def __init__(self):
        self.store = get_context_store()
        self.navigator = get_context_navigator()
        self.factory = ContextFactory(self.store)
        self._initialize_from_legacy()

    def _initialize_from_legacy(self):
        """Register legacy foundation/project contexts."""
        # Foundation contexts already registered by get_context_store()
        pass

    # Existing methods (backward compatible)
    def get_foundation(self) -> dict:
        """Return foundation context (profile, style, preferences)."""
        result = {}
        for cv in self.store.list_by_type(ContextType.IDENTITY):
            # Use load for foundation (small, always needed)
            content = cv.get_full()
            result[cv.id.replace("identity:", "")] = yaml.safe_load(content) or {}
        return result

    def get_combined_context(self, project_name: Optional[str] = None) -> dict:
        """Get merged context (legacy method)."""
        return {
            "user": self.get_foundation(),
            "working": self._working if hasattr(self, '_working') else {},
            "project": self.get_project(project_name) if project_name else None
        }

    # NEW: RLM-style methods
    def get_navigator(self) -> ContextNavigator:
        """Get the context navigator for exploration."""
        return self.navigator

    def register_project(self, project_name: str, project_path: Path) -> list:
        """Register a project's contexts."""
        return self.factory.from_directory(
            directory=project_path,
            pattern="*.yaml",
            context_type=ContextType.PROJECT,
            id_prefix=f"project:{project_name}:"
        )

    def register_temporal(self, id: str, name: str, loader: Callable) -> None:
        """Register a temporal (dynamic) context."""
        self.factory.from_function(
            id=f"temporal:{id}",
            name=name,
            loader=loader,
            context_type=ContextType.TEMPORAL
        )
```

### 6.2 Embedding Service Integration

For semantic grep across contexts:

```python
# backend/services/context/semantic_grep.py

from backend.services.embedding_service import get_embedding_service
from backend.services.context import ContextVariable
import numpy as np


class SemanticGrep:
    """Semantic search within and across contexts.

    Extends basic regex grep with embedding-based search
    for fuzzy/semantic matching.
    """

    def __init__(self):
        self.embedding_service = get_embedding_service()

    def semantic_grep(
        self,
        cv: ContextVariable,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """Find semantically similar sections in a context.

        Chunks content into paragraphs, embeds each, and finds
        most similar to query.

        Args:
            cv: Context to search
            query: Natural language query
            top_k: Number of matches to return

        Returns:
            List of {chunk, similarity, position} dicts
        """
        content = cv.get_full()
        if not content:
            return []

        # Chunk by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            return []

        # Embed query and paragraphs
        query_embedding = self.embedding_service.embed(query)
        para_embeddings = self.embedding_service.embed_batch(paragraphs)

        # Compute similarities
        results = []
        for i, (para, emb) in enumerate(zip(paragraphs, para_embeddings)):
            sim = self.embedding_service.cosine_similarity(query_embedding, emb)
            results.append({
                "chunk": para[:500],  # Truncate for display
                "similarity": float(sim),
                "position": i,
                "full_length": len(para)
            })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def semantic_search_all(
        self,
        store: "ContextStore",
        query: str,
        top_k: int = 10
    ) -> list[dict]:
        """Search semantically across all contexts.

        Returns top matches across all registered contexts.
        """
        all_results = []

        for cv in store._registry.values():
            matches = self.semantic_grep(cv, query, top_k=3)
            for match in matches:
                match["context_id"] = cv.id
                match["context_name"] = cv.name
                all_results.append(match)

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]
```

### 6.3 MindGraph Integration

Connect MindGraph semantic memory as a context source:

```python
# backend/services/context/mindgraph_context.py

from backend.services.mind_graph import MindGraph, get_mind_graph
from backend.services.context import ContextFactory, ContextStore, ContextType


def register_mindgraph_contexts(store: ContextStore) -> None:
    """Register MindGraph as context sources.

    Creates contexts for:
    - Full graph summary
    - Per-domain subgraphs
    - Semantic search results
    """
    factory = ContextFactory(store)
    graph = get_mind_graph()

    # Full graph overview
    factory.from_function(
        id="memory:graph_overview",
        name="MindGraph Overview",
        loader=lambda: _format_graph_summary(graph),
        context_type=ContextType.MEMORY,
        metadata={"source": "mindgraph", "type": "overview"}
    )

    # Domain-specific views
    domains = graph.get_domains()  # Assuming this method exists
    for domain in domains:
        factory.from_function(
            id=f"memory:domain:{domain}",
            name=f"Memory: {domain.title()}",
            loader=lambda d=domain: _format_domain(graph, d),
            context_type=ContextType.MEMORY,
            metadata={"source": "mindgraph", "domain": domain}
        )


def _format_graph_summary(graph: MindGraph) -> str:
    """Format graph statistics and top nodes."""
    stats = graph.get_stats()
    lines = [
        f"MindGraph Summary",
        f"================",
        f"Total nodes: {stats['node_count']}",
        f"Total edges: {stats['edge_count']}",
        f"",
        "Top 20 nodes by access:"
    ]

    # Add top nodes
    top_nodes = graph.get_top_nodes(20)  # Assuming method
    for node in top_nodes:
        lines.append(f"  - {node.label} ({node.node_type.value})")

    return '\n'.join(lines)


def _format_domain(graph: MindGraph, domain: str) -> str:
    """Format nodes in a specific domain."""
    nodes = graph.get_nodes_by_domain(domain)  # Assuming method
    lines = [f"Domain: {domain}", "=" * 40]

    for node in nodes[:50]:  # Limit output
        lines.append(f"[{node.node_type.value}] {node.label}")
        if node.description:
            lines.append(f"    {node.description[:100]}")

    return '\n'.join(lines)
```

### 6.4 Agent Executor Integration

Add context tools to the agent executor:

```python
# backend/services/agent_executor.py (ENHANCED)

from backend.services.context import get_context_navigator
from backend.services.context.context_tools import CONTEXT_TOOLS, handle_context_tool


class AgentExecutor:
    def __init__(self):
        self.navigator = get_context_navigator()
        # ... existing init

    def get_available_tools(self) -> list[dict]:
        """Get all tools including context tools."""
        tools = self._get_base_tools()
        tools.extend(CONTEXT_TOOLS)
        return tools

    async def handle_tool_call(self, tool_name: str, tool_input: dict) -> dict:
        """Route tool calls including context tools."""
        if tool_name.startswith("context_"):
            return handle_context_tool(tool_name, tool_input, self.navigator)
        return await self._handle_base_tool(tool_name, tool_input)
```

---

## 7. Memory Budget

### 7.1 Component Allocation (1GB Total Budget)

| Component | Max RAM | Notes |
|-----------|---------|-------|
| **Embedding Model** | 500 MB | Shared with existing service |
| **Context Registry** | 10 MB | ~10K contexts @ 1KB metadata each |
| **Content Cache (LRU)** | 50 MB | ~50 loaded contexts @ 1MB each |
| **Navigator State** | 5 MB | Access logs, patterns |
| **Working Buffers** | 35 MB | Chunking, grep results |
| **Reserved Headroom** | 100 MB | Spikes during operations |
| **Total Context System** | **700 MB** | Within 1GB budget |

### 7.2 Memory Management Strategies

```python
# Memory-conscious patterns used throughout

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
        if current + size > self.MAX_CACHE_MB * 1024 * 1024:
            # Try eviction
            self.store.evict_lru(
                int(self.MAX_CACHE_MB * 1024 * 1024 * self.EVICTION_THRESHOLD)
            )

        return True

    def get_memory_report(self) -> dict:
        """Current memory usage."""
        return {
            "cache_used_mb": self.store._current_cache_size / (1024 * 1024),
            "cache_max_mb": self.MAX_CACHE_MB,
            "utilization": self.store._current_cache_size / (self.MAX_CACHE_MB * 1024 * 1024),
            "contexts_loaded": sum(1 for cv in self.store._registry.values() if cv._loaded)
        }
```

### 7.3 Lazy Loading Everywhere

```python
# Key patterns for memory efficiency:

# 1. Content stays on disk until accessed
cv.source_path  # File path stored, not content

# 2. peek() loads minimal data
preview = cv.peek(lines=10)  # ~100 tokens max

# 3. grep() streams through file
for match in cv.grep("pattern"):  # Doesn't load full file
    process(match)

# 4. chunk() paginates large files
for i in range(total_chunks):
    chunk = cv.chunk(i)  # One chunk in memory at a time

# 5. LRU eviction on cache pressure
store.evict_lru(target_bytes)  # Frees oldest content
```

---

## 8. Comparison

### 8.1 RLM (Original Paper) vs Our Implementation

| Aspect | RLM Paper | Our Implementation |
|--------|-----------|-------------------|
| **Focus** | General code-based context exploration | Life OS personal context |
| **Language** | LLM writes arbitrary Python | Structured tool calls (safer) |
| **Storage** | Not specified | SQLite + file-based hybrid |
| **Recursion** | Unlimited spawning | Bounded via Task tool |
| **Memory** | Not constrained | 8GB-optimized |
| **Learning** | Not specified | Access pattern tracking |

### 8.2 AI-Corp Pattern vs Our Implementation

| Aspect | AI-Corp Pattern | Our Implementation |
|--------|-----------------|-------------------|
| **ContextVariable** | Generic data container | Exploration-focused with lazy ops |
| **Operations** | peek, grep, chunk, get_full | Same + semantic grep |
| **ContextTypes** | 7 types | Same + custom types |
| **Loading** | Lazy by type | Lazy by operation |
| **Agent Interface** | Not specified | Explicit tool definitions |
| **Memory Management** | Not specified | LRU eviction, budgets |

### 8.3 Our Improvements

1. **Memory-First Design**: Every component has explicit memory budget
2. **Existing Integration**: Reuses `EmbeddingService`, `SemanticIndex`, `MindGraph`
3. **Tool-Based Exploration**: Structured tool calls instead of arbitrary code
4. **Access Pattern Learning**: Tracks how agents explore to optimize
5. **Semantic Grep**: Embedding-based search, not just regex
6. **Production Ready**: Error handling, caching, persistence

---

## 9. Implementation Roadmap

### Phase 1: Core Abstractions (Week 1)

**Files to create:**
- `backend/services/context/__init__.py`
- `backend/services/context/context_variable.py`
- `backend/services/context/context_store.py`

**Tests:**
- Unit tests for ContextVariable operations
- ContextStore registration and LRU eviction

### Phase 2: Navigator and Tools (Week 2)

**Files to create:**
- `backend/services/context/context_navigator.py`
- `backend/services/context/context_tools.py`

**Tests:**
- Navigator method tests
- Tool handler routing tests

### Phase 3: REST API (Week 3)

**Files to enhance:**
- `backend/routes/context.py` - Add new endpoints

**Tests:**
- API endpoint integration tests
- Error handling tests

### Phase 4: Integration (Week 4)

**Files to enhance:**
- `backend/services/context_service.py` - Wrap new system
- `backend/services/context/mindgraph_context.py` - MindGraph integration

**Tests:**
- End-to-end context exploration tests
- Memory budget enforcement tests

### Phase 5: Semantic Features (Week 5)

**Files to create:**
- `backend/services/context/semantic_grep.py`

**Tests:**
- Semantic search accuracy tests
- Performance benchmarks

### Phase 6: Agent Integration (Week 6)

**Files to enhance:**
- Agent executor tool registration
- Life OS agent prompts to use context tools

**Tests:**
- Agent context exploration flows
- Access pattern learning validation

---

## Appendix A: Example Agent Flow

```
User: "What meetings do I have with John about the Langley project?"

Agent receives context handles:
- identity:profile
- identity:preferences
- project:langley_5:project
- project:langley_5:contacts
- temporal:calendar
- memory:graph_overview

Agent exploration:
1. context_list(type="temporal")
   → [calendar: 2.3KB, upcoming: 1.1KB]

2. context_grep("temporal:calendar", "John")
   → [{line: 45, match: "Meeting with John Smith - Langley review"}]

3. context_peek("project:langley_5:contacts")
   → Preview of contacts file, sees John Smith entry

4. context_load("temporal:calendar")
   → Full calendar, filtered to John+Langley

Response: "You have 3 meetings with John Smith about Langley..."
```

---

## Appendix B: Context Type Reference

| Type | Description | Example Sources | Load Pattern |
|------|-------------|-----------------|--------------|
| IDENTITY | Core user profile | profile.yaml | Always hot |
| PROJECT | Project context | trades.yaml, contacts.yaml | Lazy, cached |
| TEMPORAL | Time-series | calendar, messages | Lazy, refresh |
| DOCUMENT | Static docs | PDFs, SIs | Lazy, rare |
| MEMORY | MindGraph | Semantic memory | Query-based |
| TOOL_RESULT | Cached outputs | API responses | TTL-cached |
| WEB | Fetched content | URLs | Session-only |
