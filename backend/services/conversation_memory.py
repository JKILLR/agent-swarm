"""Conversation Memory Service - Orchestrates memory extraction and graph integration.

This service processes conversations to extract memorable information and
integrates it into the MindGraph. It uses both pattern-based extraction
(fast, always runs) and LLM-based extraction (optional, configurable).

Key responsibilities:
- Analyze conversations for memorable content
- Create nodes in MindGraph
- Link nodes via semantic similarity
- Track conversation provenance

Architecture:
- Two-phase extraction: explicit patterns + LLM
- Semantic parent discovery for hierarchical organization
- Concept linking for associative memory
- Episodic memory nodes for conversation tracking
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .mind_graph import MindGraph, MindNode
    from .conversation_analyzer import ConversationAnalyzer, ExtractedMemory
    from .memory_extractor import MemoryExtractor

from .mind_graph import EdgeType, MindNode, NodeType, get_mind_graph
from .semantic_index import SearchResult

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    """Orchestrates memory extraction and graph integration.

    This service is the main entry point for processing conversations
    and creating memory nodes in the MindGraph.

    Attributes:
        graph: The MindGraph instance for storing memories
        analyzer: Pattern-based conversation analyzer
        extractor: LLM-based memory extractor (optional)
        enable_llm: Whether LLM extraction is enabled
    """

    def __init__(
        self,
        graph: "MindGraph",
        anthropic_client: Any = None,
        enable_llm_extraction: bool = True,
    ):
        """Initialize the conversation memory service.

        Args:
            graph: The MindGraph instance to store memories in
            anthropic_client: Anthropic client for LLM extraction (optional)
            enable_llm_extraction: Whether to use LLM for extraction
        """
        self.graph = graph
        self.enable_llm = enable_llm_extraction and anthropic_client is not None

        # Import here to avoid circular dependencies
        from .conversation_analyzer import ConversationAnalyzer
        self.analyzer = ConversationAnalyzer()

        if self.enable_llm:
            from .memory_extractor import MemoryExtractor
            self.extractor = MemoryExtractor(anthropic_client)
        else:
            self.extractor = None

    async def process_conversation(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[MindNode]:
        """Process a conversation and create memory nodes.

        Called after conversation ends or at periodic intervals.
        Uses a three-phase approach:
        1. Explicit pattern extraction (always runs)
        2. LLM extraction (if enabled and worthwhile)
        3. Episodic memory creation (if any nodes were created)

        Args:
            session_id: Chat session ID for provenance tracking
            messages: Conversation messages [{role, content, ...}]

        Returns:
            List of created MindNode instances
        """
        created_nodes: list[MindNode] = []

        # Phase 1: Explicit pattern extraction (always runs, fast)
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    explicit_memories = self.analyzer.extract_explicit(content)
                    for memory in explicit_memories:
                        node = await self._create_node_from_memory(memory, session_id)
                        if node:
                            created_nodes.append(node)

        # Phase 2: LLM extraction (if enabled and conversation is substantial)
        if self.enable_llm and self.extractor and self.analyzer.should_extract_with_llm(messages):
            try:
                llm_memories = await self.extractor.extract(messages)
                for memory in llm_memories:
                    # Skip if too similar to already created node
                    if self._is_duplicate(memory, created_nodes):
                        continue
                    node = await self._create_node_from_memory(memory, session_id)
                    if node:
                        created_nodes.append(node)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        # Phase 3: Create conversation memory node (episodic memory)
        if created_nodes:
            conv_node = self._create_conversation_node(session_id, created_nodes)
            created_nodes.insert(0, conv_node)

        if created_nodes:
            logger.info(
                f"Created {len(created_nodes)} memory nodes from conversation {session_id}"
            )

        return created_nodes

    async def _create_node_from_memory(
        self,
        memory: "ExtractedMemory",
        session_id: str,
    ) -> Optional["MindNode"]:
        """Create a graph node from extracted memory.

        Maps memory categories to node types, finds appropriate parent
        via semantic similarity, and creates the node with full provenance.

        Args:
            memory: The extracted memory to convert to a node
            session_id: Chat session ID for provenance

        Returns:
            Created MindNode or None if creation failed
        """
        from .conversation_analyzer import MemoryCategory

        # Map category to node type
        type_mapping = {
            MemoryCategory.IDENTITY: NodeType.IDENTITY,
            MemoryCategory.PREFERENCE: NodeType.PREFERENCE,
            MemoryCategory.FACT: NodeType.FACT,
            MemoryCategory.GOAL: NodeType.GOAL,
            MemoryCategory.DECISION: NodeType.DECISION,
            MemoryCategory.RELATIONSHIP: NodeType.RELATIONSHIP,
        }

        node_type = type_mapping.get(memory.category, NodeType.MEMORY)

        # Find appropriate parent via semantic similarity (async)
        parent_id = await self._find_parent_async(memory)

        try:
            # Create node with full provenance
            node = self.graph.add_node(
                label=memory.label,
                description=memory.description,
                node_type=node_type,
                parent_id=parent_id,
                source="conversation",
                provenance={
                    "session_id": session_id,
                    "extraction_confidence": memory.confidence,
                    "importance": memory.importance,
                    "source_excerpt": memory.source_message[:200] if memory.source_message else "",
                },
                metadata={
                    "importance": memory.importance,
                },
            )

            # Link to semantically related nodes (async)
            await self._link_related_async(node, memory.related_concepts)

            logger.debug(f"Created memory node: {memory.label} (type: {node_type.value})")
            return node

        except Exception as e:
            logger.error(f"Failed to create node from memory: {e}")
            return None

    async def _find_parent_async(self, memory: "ExtractedMemory") -> Optional[str]:
        """Find appropriate parent node via semantic similarity (async).

        Searches for existing nodes that are semantically similar to the
        memory description. If a highly similar node is found, it becomes
        the parent for hierarchical organization.

        Args:
            memory: The extracted memory to find a parent for

        Returns:
            Node ID of the parent, or None if no suitable parent found
        """
        try:
            # Search for similar nodes using semantic similarity (async)
            results = await self.graph.semantic_index.search_async(
                query=memory.description,
                limit=5,
                min_similarity=0.6,
            )

            if not results:
                return None

            # Use most similar node as parent if high confidence match
            top_result = results[0]
            if top_result.similarity > 0.75:
                logger.debug(
                    f"Found parent for '{memory.label}': '{top_result.node.label}' "
                    f"(similarity: {top_result.similarity:.2f})"
                )
                return top_result.node.id

            return None

        except Exception as e:
            logger.warning(f"Failed to find parent: {e}")
            return None

    async def _link_related_async(self, node: MindNode, concepts: list[str]) -> None:
        """Link node to related concepts via semantic search (async).

        For each concept in the memory's related_concepts list, searches
        for existing nodes that match and creates association edges.

        Args:
            node: The newly created node to link from
            concepts: List of related concept strings to search for
        """
        if not concepts:
            return

        for concept in concepts:
            try:
                # Search for existing node matching concept (async)
                results = await self.graph.semantic_index.search_async(
                    query=concept,
                    limit=1,
                    min_similarity=0.7,
                )

                if results:
                    target_node = results[0].node
                    # Don't link to self
                    if target_node.id != node.id:
                        self.graph.add_association(
                            source_id=node.id,
                            target_id=target_node.id,
                            edge_type=EdgeType.ASSOCIATION,
                            metadata={"concept": concept},
                        )
                        logger.debug(
                            f"Linked '{node.label}' to '{target_node.label}' "
                            f"via concept '{concept}'"
                        )

            except Exception as e:
                logger.warning(f"Failed to link concept '{concept}': {e}")

    def _is_duplicate(
        self,
        memory: "ExtractedMemory",
        existing: list[MindNode],
    ) -> bool:
        """Check if memory is duplicate of existing node.

        Uses simple label substring matching to detect duplicates.
        This prevents the same information from being extracted
        multiple times in the same processing run.

        Args:
            memory: The memory to check for duplicates
            existing: List of already created nodes

        Returns:
            True if memory is a duplicate, False otherwise
        """
        memory_label_lower = memory.label.lower()

        for node in existing:
            node_label_lower = node.label.lower()

            # Check for substring matches in either direction
            if memory_label_lower in node_label_lower:
                return True
            if node_label_lower in memory_label_lower:
                return True

        return False

    def _create_conversation_node(
        self,
        session_id: str,
        extracted_nodes: list[MindNode],
    ) -> MindNode:
        """Create episodic memory node for the conversation itself.

        This creates a "conversation" node that represents the episodic
        memory of the conversation and links to all extracted memory nodes.
        Provides a temporal anchor for when memories were created.

        Args:
            session_id: Chat session ID
            extracted_nodes: List of nodes extracted from the conversation

        Returns:
            The created conversation memory node
        """
        now = datetime.now()

        # Create the episodic memory node
        node = self.graph.add_node(
            label=f"Conversation: {now.strftime('%Y-%m-%d %H:%M')}",
            description=f"Extracted {len(extracted_nodes)} memories from conversation",
            node_type=NodeType.MEMORY,
            source="system",
            provenance={
                "session_id": session_id,
                "extracted_node_ids": [n.id for n in extracted_nodes],
                "extracted_at": now.isoformat(),
            },
            metadata={
                "memory_count": len(extracted_nodes),
            },
        )

        # Link to all extracted nodes with DERIVED edge type
        for extracted in extracted_nodes:
            self.graph.add_association(
                source_id=node.id,
                target_id=extracted.id,
                edge_type=EdgeType.DERIVED,
            )

        logger.info(
            f"Created conversation node linking {len(extracted_nodes)} extracted memories"
        )

        return node


# =============================================================================
# Singleton Access
# =============================================================================

import threading

_conversation_memory_service: Optional[ConversationMemoryService] = None
_singleton_lock = threading.Lock()


def get_conversation_memory_service(
    anthropic_client: Any = None,
    enable_llm_extraction: bool = True,
) -> ConversationMemoryService:
    """Get or create the global conversation memory service.

    Thread-safe singleton using double-checked locking.

    Args:
        anthropic_client: Anthropic client for LLM extraction
        enable_llm_extraction: Whether to use LLM for extraction

    Returns:
        The conversation memory service singleton
    """
    global _conversation_memory_service

    # Fast path: already initialized
    if _conversation_memory_service is not None:
        return _conversation_memory_service

    # Slow path: acquire lock and double-check
    with _singleton_lock:
        if _conversation_memory_service is None:
            graph = get_mind_graph()
            _conversation_memory_service = ConversationMemoryService(
                graph=graph,
                anthropic_client=anthropic_client,
                enable_llm_extraction=enable_llm_extraction,
            )

    return _conversation_memory_service
