"""Memory management for persistent agent context."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Project root for memory operations
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_ROOT = PROJECT_ROOT / "memory"


class MemoryManager:
    """
    Manages persistent memory for agents.

    Provides hierarchical context loading based on agent role:
    - COO: Broad organizational context
    - VP/Orchestrators: Swarm-level + cross-swarm context
    - Individual agents: Their swarm's context
    """

    def __init__(self, memory_path: Path = MEMORY_ROOT):
        self.memory_path = memory_path
        self._ensure_structure()

    def _ensure_structure(self):
        """Ensure memory directory structure exists."""
        (self.memory_path / "core").mkdir(parents=True, exist_ok=True)
        (self.memory_path / "swarms").mkdir(parents=True, exist_ok=True)
        (self.memory_path / "sessions").mkdir(parents=True, exist_ok=True)

    def _read_file(self, path: Path) -> str:
        """Read a file, return empty string if not found."""
        if path.exists():
            try:
                return path.read_text()
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
        return ""

    def _extract_summary(self, content: str) -> str:
        """Extract the Summary section from a markdown file."""
        # Look for ## Summary section
        match = re.search(r'## Summary\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: first 500 chars
        return content[:500] + "..." if len(content) > 500 else content

    def _extract_recent_entries(self, content: str, n: int = 10) -> str:
        """Extract the last N entries from a log-style markdown file."""
        # Split by ## headers (each entry)
        entries = re.split(r'\n(?=## \d{4}-\d{2}-\d{2})', content)
        # Keep last N
        recent = entries[-n:] if len(entries) > n else entries
        return "\n".join(recent)

    # =========================================================================
    # Context Loading Methods
    # =========================================================================

    def load_coo_context(self) -> str:
        """
        Load full context for the COO (Supreme Orchestrator).

        Returns comprehensive organizational context including:
        - Full vision
        - Full priorities
        - Recent decisions
        - All swarm contexts
        - Cross-swarm dependencies
        """
        sections = []

        # Full vision
        vision = self._read_file(self.memory_path / "core" / "vision.md")
        if vision:
            sections.append("# ORGANIZATIONAL MEMORY\n")
            sections.append(vision)

        # Full priorities
        priorities = self._read_file(self.memory_path / "core" / "priorities.md")
        if priorities:
            sections.append("\n---\n")
            sections.append(priorities)

        # Recent decisions
        decisions = self._read_file(self.memory_path / "core" / "decisions.md")
        if decisions:
            sections.append("\n---\n")
            sections.append(self._extract_recent_entries(decisions, 5))

        # Cross-swarm dependencies
        cross = self._read_file(self.memory_path / "swarms" / "cross_swarm.md")
        if cross:
            sections.append("\n---\n")
            sections.append(cross)

        # Full swarm contexts
        swarms_dir = self.memory_path / "swarms"
        for swarm_dir in sorted(swarms_dir.iterdir()):
            if swarm_dir.is_dir() and swarm_dir.name != "sessions":
                context_file = swarm_dir / "context.md"
                if context_file.exists():
                    content = self._read_file(context_file)
                    sections.append(f"\n---\n## {swarm_dir.name}\n{content}")

        return "\n".join(sections)

    def load_swarm_orchestrator_context(self, swarm_name: str) -> str:
        """
        Load full context for a swarm orchestrator.

        Returns complete swarm context including:
        - Full context file
        - Full progress file
        - Cross-swarm dependencies relevant to this swarm
        """
        sections = []
        swarm_path = self.memory_path / "swarms" / swarm_name

        # Full swarm context
        context = self._read_file(swarm_path / "context.md")
        if context:
            sections.append(f"# {swarm_name} Context\n")
            sections.append(context)

        # Full progress
        progress = self._read_file(swarm_path / "progress.md")
        if progress:
            sections.append("\n---\n")
            sections.append(progress)

        # Cross-swarm dependencies
        cross = self._read_file(self.memory_path / "swarms" / "cross_swarm.md")
        if cross:
            sections.append("\n---\n")
            sections.append(cross)

        return "\n".join(sections)

    def load_agent_context(self, swarm_name: str, agent_name: str) -> str:
        """
        Load context for an individual agent.

        Returns the swarm's context and current progress so agents
        understand their team's goals and current state.
        """
        sections = []
        swarm_path = self.memory_path / "swarms" / swarm_name

        # Full swarm context
        context = self._read_file(swarm_path / "context.md")
        if context:
            sections.append(f"# {swarm_name} Context\n")
            sections.append(context)

        # Current progress
        progress = self._read_file(swarm_path / "progress.md")
        if progress:
            sections.append("\n---\n")
            sections.append(progress)

        return "\n".join(sections)

    def load_vp_context(self) -> str:
        """
        Load context for VP of Operations.

        Returns broad operational context across all swarms.
        """
        sections = []

        # Vision summary
        vision = self._read_file(self.memory_path / "core" / "vision.md")
        if vision:
            sections.append("# OPERATIONAL CONTEXT\n")
            sections.append("## Vision Summary\n")
            sections.append(self._extract_summary(vision))

        # Full priorities
        priorities = self._read_file(self.memory_path / "core" / "priorities.md")
        if priorities:
            sections.append("\n---\n")
            sections.append(priorities)

        # Cross-swarm dependencies
        cross = self._read_file(self.memory_path / "swarms" / "cross_swarm.md")
        if cross:
            sections.append("\n---\n")
            sections.append(cross)

        # All swarms' progress
        swarms_dir = self.memory_path / "swarms"
        progress_sections = []
        for swarm_dir in swarms_dir.iterdir():
            if swarm_dir.is_dir():
                progress = self._read_file(swarm_dir / "progress.md")
                if progress:
                    progress_sections.append(f"### {swarm_dir.name}\n{progress}")

        if progress_sections:
            sections.append("\n---\n## All Swarms Progress\n")
            sections.append("\n\n".join(progress_sections))

        return "\n".join(sections)

    # =========================================================================
    # Memory Update Methods
    # =========================================================================

    def update_progress(self, swarm_name: str, update: str, category: str = "Active Work"):
        """Add an update to a swarm's progress file."""
        swarm_path = self.memory_path / "swarms" / swarm_name
        swarm_path.mkdir(parents=True, exist_ok=True)
        progress_file = swarm_path / "progress.md"

        content = self._read_file(progress_file)
        if not content:
            # Create new progress file
            content = f"# {swarm_name} Progress\n\n## Active Work\n\n## Blockers\n\n## Recently Completed\n\n## Next Up\n"

        # Add update under appropriate category
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"- [{timestamp}] {update}\n"

        # Insert under category
        pattern = rf'(## {category}\s*\n)'
        if re.search(pattern, content):
            content = re.sub(pattern, rf'\1{entry}', content)
        else:
            content += f"\n## {category}\n{entry}"

        progress_file.write_text(content)
        logger.info(f"Updated progress for {swarm_name}: {update[:50]}...")

    def log_decision(self, title: str, context: str, decision: str,
                     rationale: str, impact: str, owner: str = "CEO"):
        """Log a decision to the decisions file."""
        decisions_file = self.memory_path / "core" / "decisions.md"
        content = self._read_file(decisions_file)

        date = datetime.now().strftime("%Y-%m-%d")
        entry = f"""
## {date} - {title}
**Context**: {context}
**Decision**: {decision}
**Rationale**: {rationale}
**Impact**: {impact}
**Owner**: {owner}

---
"""
        # Insert after the header
        if "# " in content:
            # Insert after first line
            lines = content.split('\n', 2)
            content = lines[0] + '\n' + entry + '\n'.join(lines[1:])
        else:
            content = "# Organizational Decision Log\n" + entry + content

        decisions_file.write_text(content)
        logger.info(f"Logged decision: {title}")

    def save_session_summary(self, session_id: str, summary: str, swarm_name: Optional[str] = None):
        """Save a session summary to history."""
        # Save to sessions directory
        session_file = self.memory_path / "sessions" / f"{session_id}.md"
        session_file.write_text(f"# Session {session_id}\n\n{summary}")

        # Also append to swarm history if applicable
        if swarm_name:
            swarm_path = self.memory_path / "swarms" / swarm_name
            swarm_path.mkdir(parents=True, exist_ok=True)
            history_file = swarm_path / "history.md"

            date = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n## {date}\n{summary}\n"

            content = self._read_file(history_file)
            if not content:
                content = f"# {swarm_name} Session History\n"
            content += entry
            history_file.write_text(content)

    def update_swarm_context(self, swarm_name: str, section: str, content: str):
        """Update a specific section in a swarm's context file."""
        swarm_path = self.memory_path / "swarms" / swarm_name
        swarm_path.mkdir(parents=True, exist_ok=True)
        context_file = swarm_path / "context.md"

        current = self._read_file(context_file)
        if not current:
            current = f"# {swarm_name} Context\n"

        # Replace or add section
        pattern = rf'(## {section}\s*\n)(.*?)(?=\n##|\Z)'
        replacement = f'## {section}\n{content}\n'

        if re.search(pattern, current, re.DOTALL):
            current = re.sub(pattern, replacement, current, flags=re.DOTALL)
        else:
            current += f"\n{replacement}"

        context_file.write_text(current)
        logger.info(f"Updated {swarm_name} context: {section}")

    # =========================================================================
    # Session Summarization Methods
    # =========================================================================

    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token average)."""
        return len(text) // 4

    def needs_summarization(self, messages: List[Dict[str, str]], max_tokens: int = 50000) -> bool:
        """
        Check if a conversation needs summarization.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Token threshold before summarization is needed

        Returns:
            True if conversation exceeds token threshold
        """
        total_text = "\n".join(m.get("content", "") for m in messages)
        estimated_tokens = self.estimate_tokens(total_text)
        return estimated_tokens > max_tokens

    def create_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Create a prompt for summarizing conversation history.

        Args:
            messages: List of message dicts to summarize

        Returns:
            Formatted prompt for summarization
        """
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:2000]  # Truncate very long messages
            conversation.append(f"**{role}**: {content}")

        conversation_text = "\n\n".join(conversation)

        return f"""Please summarize the following conversation. Focus on:
1. Key decisions made
2. Important context established
3. Tasks completed or in progress
4. Any blockers or issues identified
5. Current state and next steps

Provide a concise summary (under 500 words) that captures the essential context needed to continue the conversation.

---
CONVERSATION:
{conversation_text}
---

SUMMARY:"""

    def save_conversation_summary(
        self,
        session_id: str,
        summary: str,
        original_message_count: int,
        swarm_name: Optional[str] = None
    ):
        """
        Save a conversation summary for later retrieval.

        Args:
            session_id: The session being summarized
            summary: The generated summary
            original_message_count: Number of messages that were summarized
            swarm_name: Optional swarm association
        """
        summaries_dir = self.memory_path / "sessions" / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)

        summary_file = summaries_dir / f"{session_id}.md"
        date = datetime.now().strftime("%Y-%m-%d %H:%M")

        content = f"""# Session Summary: {session_id}
**Date**: {date}
**Messages Summarized**: {original_message_count}
{"**Swarm**: " + swarm_name if swarm_name else ""}

## Summary
{summary}
"""
        summary_file.write_text(content)
        logger.info(f"Saved summary for session {session_id}")

    def load_session_summary(self, session_id: str) -> Optional[str]:
        """
        Load a previously saved session summary.

        Args:
            session_id: The session to load summary for

        Returns:
            The summary text or None if not found
        """
        summary_file = self.memory_path / "sessions" / "summaries" / f"{session_id}.md"
        if summary_file.exists():
            content = self._read_file(summary_file)
            # Extract just the summary section
            match = re.search(r'## Summary\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return content
        return None

    def get_context_with_summary(
        self,
        session_id: Optional[str],
        recent_messages: List[Dict[str, str]],
        max_recent: int = 10
    ) -> str:
        """
        Build conversation context using summary + recent messages.

        Args:
            session_id: Session to load summary from
            recent_messages: Most recent messages to include in full
            max_recent: Maximum number of recent messages to include

        Returns:
            Formatted context string with summary and recent messages
        """
        sections = []

        # Include summary if available
        if session_id:
            summary = self.load_session_summary(session_id)
            if summary:
                sections.append("## Previous Conversation Summary")
                sections.append(summary)
                sections.append("")

        # Include recent messages
        if recent_messages:
            sections.append("## Recent Messages")
            for msg in recent_messages[-max_recent:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                sections.append(f"**{role}**: {content[:1000]}")
            sections.append("")

        return "\n".join(sections)


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
