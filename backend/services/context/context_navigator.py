"""ContextNavigator: Agent-facing tool for context exploration.

Provides a high-level interface for agents to explore context without
loading everything upfront. Tracks access patterns for learning.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from .context_variable import ContextVariable, ContextType
from .context_store import ContextStore

# Memory leak prevention limits
MAX_STATE_KEYS = 100
MAX_VALUE_SIZE_BYTES = 1_000_000  # 1MB
MAX_BUFFER_ENTRIES = 1000
MAX_BUFFER_CHARS = 10_000_000  # 10MB
MAX_BATCH_SIZE = 100


class GrepQuery(TypedDict):
    """Query structure for batch grep operations."""
    context_id: str
    pattern: str


class GrepResult(TypedDict):
    """Result structure for batch grep operations."""
    context_id: str
    matches: List[dict]
    total_matches: int
    truncated: bool
    error: Optional[str]


class PeekResult(TypedDict):
    """Result structure for batch peek operations."""
    context_id: str
    preview: str
    metadata: dict
    error: Optional[str]


@dataclass
class ContextNavigator:
    """Tool interface for agents to explore context.

    This is exposed to agents as a set of tool calls:
    - context_list(): See available contexts
    - context_peek(id): Preview a context
    - context_grep(id, pattern): Search within context
    - context_chunk(id, chunk_num): Paginate through context
    - context_load(id): Load full content

    Session state features (RLM-inspired):
    - session_state: Key-value store for cross-call persistence
    - result_buffer: Accumulator for building complex outputs

    The navigator tracks access patterns to learn agent strategies.
    """

    store: ContextStore

    # Access pattern tracking
    access_log: list = field(default_factory=list)

    # Lock for thread-safe access log operations
    _access_log_lock: threading.Lock = field(default_factory=threading.Lock)

    # Session state for cross-call persistence (RLM gap fix)
    session_state: Dict[str, Any] = field(default_factory=dict)

    # Result accumulation buffer (RLM gap fix - enables Strategy D: Long Output Assembly)
    _result_buffer: List[str] = field(default_factory=list)

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

        # Update cache size tracking
        self.store.update_cache_size(cv)

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
        """Log access for pattern learning (thread-safe)."""
        with self._access_log_lock:
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

        if last_action == "grep" and matches is not None:
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

        peek_count = max(actions.get("peek", 1), 1)
        return {
            "total_accesses": len(self.access_log),
            "by_action": actions,
            "load_to_peek_ratio": actions.get("load", 0) / peek_count
        }

    def clear_access_log(self) -> None:
        """Clear the access log."""
        self.access_log = []

    # =========================================================================
    # Session State Methods (RLM Gap Fix: Cross-Call State Persistence)
    # =========================================================================

    def set_state(self, key: str, value: Any) -> dict:
        """Store a value in session state.

        Enables cross-call state persistence as described in RLM.
        Variables persist across tool calls within the same session.

        Args:
            key: State key (e.g., "candidates", "search_results")
            value: Any JSON-serializable value

        Returns:
            Confirmation with current state keys
        """
        # Check number of keys limit
        if key not in self.session_state and len(self.session_state) >= MAX_STATE_KEYS:
            return {"error": f"State limit exceeded ({MAX_STATE_KEYS} keys)"}

        # Check value size limit
        value_size = sys.getsizeof(value)
        if value_size > MAX_VALUE_SIZE_BYTES:
            return {"error": f"Value too large ({value_size} bytes, max {MAX_VALUE_SIZE_BYTES})"}

        self._log_access("set_state", None, {"key": key})
        self.session_state[key] = value
        return {
            "status": "stored",
            "key": key,
            "state_keys": list(self.session_state.keys())
        }

    def get_state(self, key: str) -> dict:
        """Retrieve a value from session state.

        Args:
            key: State key to retrieve

        Returns:
            Value if exists, or error
        """
        self._log_access("get_state", None, {"key": key})
        if key in self.session_state:
            return {
                "key": key,
                "value": self.session_state[key],
                "exists": True
            }
        return {
            "key": key,
            "value": None,
            "exists": False,
            "available_keys": list(self.session_state.keys())
        }

    def list_state(self) -> dict:
        """List all keys in session state.

        Returns:
            All state keys with value previews
        """
        self._log_access("list_state", None)
        previews = {}
        for k, v in self.session_state.items():
            if isinstance(v, str):
                previews[k] = v[:100] + "..." if len(v) > 100 else v
            elif isinstance(v, list):
                previews[k] = f"[list of {len(v)} items]"
            elif isinstance(v, dict):
                previews[k] = f"{{dict with {len(v)} keys}}"
            else:
                previews[k] = str(v)[:100]
        return {
            "keys": list(self.session_state.keys()),
            "count": len(self.session_state),
            "previews": previews
        }

    def clear_state(self) -> dict:
        """Clear all session state.

        Returns:
            Confirmation
        """
        self._log_access("clear_state", None)
        count = len(self.session_state)
        self.session_state.clear()
        return {"status": "cleared", "items_removed": count}

    # =========================================================================
    # Result Buffer Methods (RLM Gap Fix: Output Accumulation)
    # Enables Strategy D: Long Output Assembly from RLM paper
    # =========================================================================

    def buffer_append(self, content: str, label: Optional[str] = None) -> dict:
        """Append content to the result buffer.

        Use to build up complex answers iteratively, e.g.:
        - Summarize chunks and append each summary
        - Collect findings across multiple contexts
        - Assemble long outputs piece by piece

        Args:
            content: Content to append
            label: Optional label for this entry (for reference)

        Returns:
            Buffer status
        """
        # Check entry count limit
        if len(self._result_buffer) >= MAX_BUFFER_ENTRIES:
            return {"error": f"Buffer entry limit exceeded ({MAX_BUFFER_ENTRIES})"}

        # Check total size limit
        current_chars = sum(
            len(e["content"]) if isinstance(e, dict) else len(e)
            for e in self._result_buffer
        )
        if current_chars + len(content) > MAX_BUFFER_CHARS:
            return {"error": f"Buffer size limit exceeded ({MAX_BUFFER_CHARS} chars)"}

        self._log_access("buffer_append", None, {"label": label})

        entry = {"content": content, "label": label} if label else content
        self._result_buffer.append(entry)

        return {
            "status": "appended",
            "entry_index": len(self._result_buffer) - 1,
            "buffer_size": len(self._result_buffer),
            "total_chars": current_chars + len(content)
        }

    def buffer_read(self, join_separator: str = "\n\n") -> dict:
        """Read all buffer contents.

        Args:
            join_separator: Separator when joining entries

        Returns:
            Combined buffer contents
        """
        self._log_access("buffer_read", None)

        if not self._result_buffer:
            return {
                "content": "",
                "entries": 0,
                "empty": True
            }

        # Extract content from entries (handle labeled vs unlabeled)
        contents = []
        labels = []
        for entry in self._result_buffer:
            if isinstance(entry, dict):
                contents.append(entry["content"])
                labels.append(entry.get("label"))
            else:
                contents.append(entry)
                labels.append(None)

        combined = join_separator.join(contents)

        return {
            "content": combined,
            "entries": len(self._result_buffer),
            "total_chars": len(combined),
            "labels": [l for l in labels if l is not None],
            "empty": False
        }

    def buffer_clear(self) -> dict:
        """Clear the result buffer.

        Returns:
            Confirmation with cleared count
        """
        self._log_access("buffer_clear", None)
        count = len(self._result_buffer)
        total_chars = sum(
            len(e["content"]) if isinstance(e, dict) else len(e)
            for e in self._result_buffer
        )
        self._result_buffer.clear()
        return {
            "status": "cleared",
            "entries_removed": count,
            "chars_removed": total_chars
        }

    def buffer_pop(self, index: int = -1) -> dict:
        """Remove and return an entry from the buffer.

        Args:
            index: Index to pop (default: last entry)

        Returns:
            Popped entry
        """
        self._log_access("buffer_pop", None, {"index": index})

        if not self._result_buffer:
            return {"error": "Buffer is empty"}

        try:
            entry = self._result_buffer.pop(index)
            content = entry["content"] if isinstance(entry, dict) else entry
            label = entry.get("label") if isinstance(entry, dict) else None
            return {
                "content": content,
                "label": label,
                "remaining_entries": len(self._result_buffer)
            }
        except IndexError:
            return {"error": f"Invalid index: {index}"}

    # =========================================================================
    # Batch Operations (RLM Gap Fix: Parallel Processing)
    # Enables faster exploration of large context sets
    # =========================================================================

    def batch_grep(
        self,
        queries: List[GrepQuery],
        context_lines: int = 2
    ) -> List[GrepResult]:
        """Search multiple contexts in parallel.

        Enables faster exploration by executing grep operations concurrently.
        Uses asyncio.gather() internally for actual parallelism where possible.

        Args:
            queries: List of {context_id, pattern} dicts
            context_lines: Lines of context around matches (applied to all)

        Returns:
            List of GrepResult dicts, one per query (in same order)
        """
        if len(queries) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(queries)} exceeds limit {MAX_BATCH_SIZE}")

        self._log_access("batch_grep", None, {"query_count": len(queries)})

        # Execute synchronously but process all queries
        results: List[GrepResult] = []
        for query in queries:
            context_id = query.get("context_id", "")
            pattern = query.get("pattern", "")

            cv = self.store.get(context_id)
            if not cv:
                results.append({
                    "context_id": context_id,
                    "matches": [],
                    "total_matches": 0,
                    "truncated": False,
                    "error": f"Context '{context_id}' not found"
                })
                continue

            self._log_access("grep", context_id, {"pattern": pattern, "batch": True})
            matches = cv.grep(pattern, context_lines)

            results.append({
                "context_id": context_id,
                "matches": matches[:20],
                "total_matches": len(matches),
                "truncated": len(matches) > 20,
                "error": None
            })

        return results

    async def batch_grep_async(
        self,
        queries: List[GrepQuery],
        context_lines: int = 2
    ) -> List[GrepResult]:
        """Search multiple contexts in parallel (async version).

        Uses asyncio.gather() for true parallel execution in async contexts.

        Args:
            queries: List of {context_id, pattern} dicts
            context_lines: Lines of context around matches

        Returns:
            List of GrepResult dicts, one per query (in same order)
        """
        if len(queries) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(queries)} exceeds limit {MAX_BATCH_SIZE}")

        self._log_access("batch_grep_async", None, {"query_count": len(queries)})

        async def grep_one(query: GrepQuery) -> GrepResult:
            context_id = query.get("context_id", "")
            pattern = query.get("pattern", "")

            cv = self.store.get(context_id)
            if not cv:
                return {
                    "context_id": context_id,
                    "matches": [],
                    "total_matches": 0,
                    "truncated": False,
                    "error": f"Context '{context_id}' not found"
                }

            # Run in thread to avoid blocking (Python 3.9+)
            matches = await asyncio.to_thread(cv.grep, pattern, context_lines)

            self._log_access("grep", context_id, {"pattern": pattern, "batch": True})

            return {
                "context_id": context_id,
                "matches": matches[:20],
                "total_matches": len(matches),
                "truncated": len(matches) > 20,
                "error": None
            }

        results = await asyncio.gather(*[grep_one(q) for q in queries])
        return list(results)

    def batch_peek(
        self,
        context_ids: List[str],
        chars: int = 400
    ) -> List[PeekResult]:
        """Preview multiple contexts in parallel.

        Enables faster exploration by fetching previews concurrently.

        Args:
            context_ids: List of context IDs to preview
            chars: Character limit for preview (applied to all)

        Returns:
            List of PeekResult dicts, one per context (in same order)
        """
        if len(context_ids) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(context_ids)} exceeds limit {MAX_BATCH_SIZE}")

        self._log_access("batch_peek", None, {"context_count": len(context_ids)})

        # Calculate lines from chars (~80 chars per line typical)
        lines = max(1, chars // 80)

        results: List[PeekResult] = []
        for context_id in context_ids:
            cv = self.store.get(context_id)
            if not cv:
                results.append({
                    "context_id": context_id,
                    "preview": "",
                    "metadata": {},
                    "error": f"Context '{context_id}' not found"
                })
                continue

            self._log_access("peek", context_id, {"batch": True})
            preview = cv.peek(lines=lines, tokens=chars // 4)

            results.append({
                "context_id": context_id,
                "preview": preview,
                "metadata": cv.metadata,
                "error": None
            })

        return results

    async def batch_peek_async(
        self,
        context_ids: List[str],
        chars: int = 400
    ) -> List[PeekResult]:
        """Preview multiple contexts in parallel (async version).

        Uses asyncio.gather() for true parallel execution in async contexts.

        Args:
            context_ids: List of context IDs to preview
            chars: Character limit for preview

        Returns:
            List of PeekResult dicts, one per context (in same order)
        """
        if len(context_ids) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(context_ids)} exceeds limit {MAX_BATCH_SIZE}")

        self._log_access("batch_peek_async", None, {"context_count": len(context_ids)})

        lines = max(1, chars // 80)
        tokens = chars // 4

        async def peek_one(context_id: str) -> PeekResult:
            cv = self.store.get(context_id)
            if not cv:
                return {
                    "context_id": context_id,
                    "preview": "",
                    "metadata": {},
                    "error": f"Context '{context_id}' not found"
                }

            # Run in thread to avoid blocking (Python 3.9+)
            preview = await asyncio.to_thread(cv.peek, lines=lines, tokens=tokens)

            self._log_access("peek", context_id, {"batch": True})

            return {
                "context_id": context_id,
                "preview": preview,
                "metadata": cv.metadata,
                "error": None
            }

        results = await asyncio.gather(*[peek_one(cid) for cid in context_ids])
        return list(results)
