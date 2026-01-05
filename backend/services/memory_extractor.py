"""LLM-based memory extraction from conversations.

Uses Claude to analyze conversations and extract structured memories
for storage in the Mind Graph.
"""

import json
import logging
from typing import Any, List, Optional

from .conversation_analyzer import ExtractedMemory, MemoryCategory

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analyze this conversation and extract information worth remembering long-term.

<conversation>
{conversation}
</conversation>

Extract meaningful memories in these categories:
- identity: Core facts about who the user is (name, profession, location)
- preference: Likes, dislikes, preferences, opinions
- fact: Important facts about the user's life, projects, or context
- goal: Things the user wants to achieve
- decision: Decisions that were made
- relationship: People or entities mentioned (colleagues, family, companies)

For each memory, provide:
- category: One of the categories above
- label: Short title (3-6 words)
- description: Full detail with context
- importance: 1-5 (5 = fundamental identity, 1 = minor detail)
- related_concepts: List of related topics for linking

Rules:
- Only extract genuinely significant information
- Skip small talk, greetings, or transient requests
- Skip implementation details or code-specific content
- If nothing significant, return an empty array

Return JSON array only:
[{{"category": "...", "label": "...", "description": "...", "importance": N, "related_concepts": [...]}}]
"""


class MemoryExtractor:
    """Uses LLM to extract structured memories from conversations."""

    DEFAULT_MODEL = "claude-3-haiku-20240307"

    def __init__(self, anthropic_client: Any):
        """Initialize the memory extractor.

        Args:
            anthropic_client: An Anthropic async client instance.
        """
        self.client = anthropic_client

    async def extract(
        self,
        messages: List[dict],
        model: Optional[str] = None,
    ) -> List[ExtractedMemory]:
        """Extract memories from conversation using Claude.

        Args:
            messages: Conversation messages [{role, content}]
            model: Model to use (defaults to claude-3-haiku for cost efficiency)

        Returns:
            List of extracted memories
        """
        if model is None:
            model = self.DEFAULT_MODEL

        # Format conversation for the prompt
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
            if m.get("content")
        )

        if not conversation_text.strip():
            return []

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(conversation=conversation_text),
                }],
            )

            # Parse JSON response
            content = response.content[0].text
            parsed_json = self._parse_json_response(content)

            if parsed_json is None:
                return []

            memories = []
            for item in parsed_json:
                try:
                    memory = ExtractedMemory(
                        category=MemoryCategory(item["category"]),
                        label=item["label"][:100],
                        description=item["description"],
                        importance=min(5, max(1, item.get("importance", 3))),
                        source_message=conversation_text[:500],  # Truncate for storage
                        confidence=0.8,
                        related_concepts=item.get("related_concepts", []),
                    )
                    memories.append(memory)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse memory item: {e}")
                    continue

            return memories

        except Exception as e:
            logger.warning(f"Failed to extract memories: {e}")
            return []

    def _parse_json_response(self, content: str) -> Optional[List[dict]]:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            content: Raw response text from the LLM

        Returns:
            Parsed JSON list or None if parsing fails
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0]
                return json.loads(json_str.strip())
            except (IndexError, json.JSONDecodeError):
                pass

        if "```" in content:
            try:
                json_str = content.split("```")[1].split("```")[0]
                return json.loads(json_str.strip())
            except (IndexError, json.JSONDecodeError):
                pass

        # Try parsing the content directly
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None
