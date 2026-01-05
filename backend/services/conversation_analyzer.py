"""Conversation Analyzer for extracting memories from chat conversations.

Pattern-based extraction - no LLM calls in this module.
"""

from dataclasses import dataclass, field
from enum import Enum
import re


class MemoryCategory(Enum):
    """Categories of extractable memories."""
    IDENTITY = "identity"         # Who the user is
    PREFERENCE = "preference"     # What they like/dislike
    FACT = "fact"                 # Facts about user/world
    GOAL = "goal"                 # What they want to achieve
    DECISION = "decision"         # Decisions made
    RELATIONSHIP = "relationship" # People/entities


@dataclass
class ExtractedMemory:
    """A memory extracted from conversation."""
    category: MemoryCategory
    label: str                              # Short title (max 100 chars)
    description: str                        # Full detail
    importance: int                         # 1-5 (5 = core identity)
    source_message: str                     # Original message text
    confidence: float                       # Extraction confidence (0-1)
    related_concepts: list[str] = field(default_factory=list)  # Suggested links


class ConversationAnalyzer:
    """Analyzes conversations to extract memorable information.

    Uses a two-phase approach:
    1. Pattern matching for explicit signals (fast, always runs)
    2. LLM extraction for implicit information (optional, configurable)
    """

    # Patterns that signal explicit memory intent
    # Each tuple: (regex_pattern, category, importance)
    EXPLICIT_PATTERNS = [
        (r"remember(?:\s+that)?:?\s*(.+)", MemoryCategory.FACT, 5),
        (r"note to self:?\s*(.+)", MemoryCategory.FACT, 4),
        (r"my name is\s+(\w+)", MemoryCategory.IDENTITY, 5),
        (r"I am\s+(?:a\s+)?(\w+)", MemoryCategory.IDENTITY, 4),
        (r"I (?:prefer|like|love)\s+(.+)", MemoryCategory.PREFERENCE, 3),
        (r"I (?:hate|dislike|don't like)\s+(.+)", MemoryCategory.PREFERENCE, 3),
        (r"I (?:want to|need to|plan to)\s+(.+)", MemoryCategory.GOAL, 3),
        (r"we decided(?:\s+to)?\s+(.+)", MemoryCategory.DECISION, 4),
        (r"from now on,?\s*(.+)", MemoryCategory.PREFERENCE, 4),
    ]

    def extract_explicit(self, message: str) -> list[ExtractedMemory]:
        """Extract memories from explicit patterns (fast path).

        Args:
            message: The user message to analyze

        Returns:
            List of extracted memories from pattern matches
        """
        memories = []

        for pattern, category, importance in self.EXPLICIT_PATTERNS:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                captured = match.group(1).strip()
                if len(captured) < 3:
                    continue

                memories.append(ExtractedMemory(
                    category=category,
                    label=captured[:100],
                    description=match.group(0),
                    importance=importance,
                    source_message=message,
                    confidence=0.9,
                    related_concepts=[],
                ))

        return memories

    def should_extract_with_llm(
        self,
        messages: list[dict],
        min_messages: int = 3,
    ) -> bool:
        """Decide if LLM extraction is worthwhile.

        Heuristics:
        - At least min_messages exchanged
        - Contains information-dense content
        - Not just small talk or commands

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            min_messages: Minimum number of messages required

        Returns:
            True if LLM extraction should be performed
        """
        if len(messages) < min_messages:
            return False

        # Check for information density signals
        user_messages = [m for m in messages if m.get("role") == "user"]
        total_length = sum(len(m.get("content", "")) for m in user_messages)

        # Skip if mostly short messages
        if total_length < 100:
            return False

        return True
