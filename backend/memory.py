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

    def load_coo_context(self, max_chars: int = 4000) -> str:
        """
        Load context for the COO (Supreme Orchestrator).

        Returns concise organizational context including:
        - Vision summary
        - Current priorities (condensed)
        - Swarm summaries

        Limited to max_chars to prevent slow responses.
        """
        sections = []

        # Vision - just the summary
        vision = self._read_file(self.memory_path / "core" / "vision.md")
        if vision:
            sections.append("# ORGANIZATIONAL CONTEXT\n")
            sections.append(self._extract_summary(vision))

        # Priorities - condensed
        priorities = self._read_file(self.memory_path / "core" / "priorities.md")
        if priorities:
            sections.append("\n## Current Priorities")
            # Extract just the priority headers and status
            lines = priorities.split('\n')
            priority_lines = [l for l in lines if l.startswith('## Priority') or l.startswith('**')][:10]
            sections.append('\n'.join(priority_lines[:200]))

        # Swarm summaries only
        swarms_dir = self.memory_path / "swarms"
        swarm_summaries = []
        for swarm_dir in sorted(swarms_dir.iterdir()):
            if swarm_dir.is_dir() and swarm_dir.name not in ["sessions", "cross_swarm.md"]:
                context_file = swarm_dir / "context.md"
                if context_file.exists():
                    content = self._read_file(context_file)
                    summary = self._extract_summary(content)
                    if summary:
                        swarm_summaries.append(f"**{swarm_dir.name}**: {summary[:150]}")

        if swarm_summaries:
            sections.append("\n## Swarms\n" + "\n".join(swarm_summaries))

        result = "\n".join(sections)
        # Hard limit
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...[context truncated]"
        return result

    def load_swarm_orchestrator_context(self, swarm_name: str, max_chars: int = 3000) -> str:
        """
        Load context for a swarm orchestrator.
        Limited to max_chars for performance.
        """
        sections = []
        swarm_path = self.memory_path / "swarms" / swarm_name

        # Swarm context summary
        context = self._read_file(swarm_path / "context.md")
        if context:
            sections.append(f"# {swarm_name} Context\n")
            sections.append(self._extract_summary(context))
            # Also include current focus
            focus_match = re.search(r'## Current Focus\s*\n(.*?)(?=\n##|\Z)', context, re.DOTALL)
            if focus_match:
                sections.append("\n## Current Focus\n" + focus_match.group(1).strip()[:300])

        # Progress - just active work
        progress = self._read_file(swarm_path / "progress.md")
        if progress:
            active_match = re.search(r'## Active Work\s*\n(.*?)(?=\n##|\Z)', progress, re.DOTALL)
            if active_match:
                sections.append("\n## Active Work\n" + active_match.group(1).strip()[:400])

        result = "\n".join(sections)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...[truncated]"
        return result

    def load_agent_context(self, swarm_name: str, agent_name: str, max_chars: int = 1500) -> str:
        """
        Load minimal context for an individual agent.
        Very limited to keep agent prompts fast.
        """
        sections = []
        swarm_path = self.memory_path / "swarms" / swarm_name

        context = self._read_file(swarm_path / "context.md")
        if context:
            # Just mission
            mission_match = re.search(r'## Mission\s*\n(.*?)(?=\n##|\Z)', context, re.DOTALL)
            if mission_match:
                sections.append("**Mission**: " + mission_match.group(1).strip()[:200])
            # Current focus
            focus_match = re.search(r'## Current Focus\s*\n(.*?)(?=\n##|\Z)', context, re.DOTALL)
            if focus_match:
                sections.append("**Focus**: " + focus_match.group(1).strip()[:200])

        result = "\n".join(sections)
        return result[:max_chars] if len(result) > max_chars else result

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


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
