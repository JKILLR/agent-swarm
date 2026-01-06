"""ContextVariable: The atomic unit of explorable context.

Implements lazy loading where content stays on disk until explicitly requested.
Agents explore via peek/grep/chunk before deciding to load full content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
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

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Invalid regex, treat as literal string
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

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
            try:
                self._content_cache = self._content_loader()
            except Exception as e:
                self._content_cache = f"[Error loading content: {e}]"
        elif isinstance(self.source_path, Path) and self.source_path.exists():
            try:
                self._content_cache = self.source_path.read_text()
            except Exception as e:
                self._content_cache = f"[Error reading file: {e}]"
        elif isinstance(self.source_path, str) and self.source_path != "<dynamic>":
            path = Path(self.source_path)
            if path.exists():
                try:
                    self._content_cache = path.read_text()
                except Exception as e:
                    self._content_cache = f"[Error reading file: {e}]"
            else:
                self._content_cache = ""
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

    def get_content_size(self) -> int:
        """Get the size of cached content in bytes."""
        if self._content_cache is not None:
            return len(self._content_cache.encode('utf-8'))
        return 0
