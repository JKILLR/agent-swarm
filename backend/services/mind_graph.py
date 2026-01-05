"""Mind Graph - Hierarchical associative memory inspired by MYND.

This module provides a graph-based memory system where:
- Nodes represent concepts, facts, conversations, and memories
- Edges represent relationships (parent-child, associations, temporal)
- Provenance tracks where each node came from (conversation, inference, user)
- Semantic search enables finding related memories

Key features inspired by Axel's MYND system:
- Hierarchical organization (like mind map tree structure)
- Provenance tracking (which conversation created this node)
- Temporal awareness (when was this remembered/updated)
- Association links (concepts can connect to related concepts)
- Efficient traversal for context retrieval
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .semantic_index import SemanticIndex

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the mind graph."""
    CONCEPT = "concept"           # Abstract idea or topic
    FACT = "fact"                 # Specific piece of information
    MEMORY = "memory"             # Episodic memory (conversation, event)
    IDENTITY = "identity"         # Self-knowledge / identity statement
    PREFERENCE = "preference"     # User preference
    GOAL = "goal"                 # Active goal or intention
    DECISION = "decision"         # Decision that was made
    RELATIONSHIP = "relationship" # Person or entity relationship


class EdgeType(Enum):
    """Types of edges connecting nodes."""
    PARENT = "parent"             # Hierarchical parent
    CHILD = "child"               # Hierarchical child
    ASSOCIATION = "association"   # Semantic association
    TEMPORAL = "temporal"         # Time-based sequence
    DERIVED = "derived"           # Inferred from another node
    REFERENCE = "reference"       # References another node


class MindNode:
    """A node in the mind graph.

    Attributes:
        id: Unique identifier
        label: Short display name
        description: Detailed description
        node_type: Type of this node
        color: Optional color for visualization
        source: Where this node came from (user, conversation, inference)
        created_at: When the node was created
        updated_at: When the node was last updated
        provenance: Context about how this node was created
        metadata: Additional key-value data
        children: List of child node IDs
        edges: List of edge dictionaries
    """

    def __init__(
        self,
        label: str,
        description: str = "",
        node_type: NodeType = NodeType.CONCEPT,
        id: str | None = None,
        color: str = "#8B5CF6",  # Default purple like MYND
        source: str = "system",
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id or f"node-{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:5]}"
        self.label = label
        self.description = description
        self.node_type = node_type
        self.color = color
        self.source = source
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.provenance = provenance or {}
        self.metadata = metadata or {}
        self.children: list[str] = []
        self.edges: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "node_type": self.node_type.value,
            "color": self.color,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "children": self.children,
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MindNode:
        """Create node from dictionary."""
        node = cls(
            label=data["label"],
            description=data.get("description", ""),
            node_type=NodeType(data.get("node_type", "concept")),
            id=data["id"],
            color=data.get("color", "#8B5CF6"),
            source=data.get("source", "system"),
            provenance=data.get("provenance", {}),
            metadata=data.get("metadata", {}),
        )
        node.created_at = data.get("created_at", node.created_at)
        node.updated_at = data.get("updated_at", node.updated_at)
        node.children = data.get("children", [])
        node.edges = data.get("edges", [])
        return node

    def add_child(self, child_id: str) -> None:
        """Add a child node ID."""
        if child_id not in self.children:
            self.children.append(child_id)
            self.updated_at = datetime.now().isoformat()

    def add_edge(self, target_id: str, edge_type: EdgeType, metadata: dict[str, Any] | None = None) -> None:
        """Add an edge to another node."""
        edge = {
            "target": target_id,
            "type": edge_type.value,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.edges.append(edge)
        self.updated_at = datetime.now().isoformat()


class MindGraph:
    """The mind graph - a persistent memory structure.

    This is the core data structure for associative memory.
    It stores nodes in a graph with hierarchical and associative relationships.

    The graph has a root node representing "My Mind" from which all
    other concepts branch. Similar to Axel's MYND structure.
    """

    ROOT_ID = "mind-root"

    def __init__(self, storage_path: Path):
        """Initialize the mind graph.

        Args:
            storage_path: Path to store the graph data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_path / "mind_graph.json"
        self._lock = threading.Lock()
        self._nodes: dict[str, MindNode] = {}
        self._semantic_index = None
        self._load()
        self._ensure_root()
        logger.info(f"MindGraph initialized with {len(self._nodes)} nodes")

    @property
    def semantic_index(self) -> "SemanticIndex":
        """Lazy-loaded semantic index for vector search."""
        if self._semantic_index is None:
            from .semantic_index import SemanticIndex
            self._semantic_index = SemanticIndex(self, self.storage_path)
        return self._semantic_index

    def _ensure_root(self) -> None:
        """Ensure the root node exists."""
        if self.ROOT_ID not in self._nodes:
            root = MindNode(
                label="My Mind",
                description="The root of my cognitive architecture - all memories, knowledge, and identity branch from here.",
                node_type=NodeType.IDENTITY,
                id=self.ROOT_ID,
                color="#8B5CF6",
                source="system",
            )
            self._nodes[self.ROOT_ID] = root
            self._save()
            logger.info("Created root mind node")

    def _load(self) -> None:
        """Load graph from disk."""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, "r") as f:
                    data = json.load(f)
                    for node_data in data.get("nodes", []):
                        node = MindNode.from_dict(node_data)
                        self._nodes[node.id] = node
                    logger.info(f"Loaded {len(self._nodes)} nodes from mind graph")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load mind graph: {e}")

    def _save(self) -> None:
        """Save graph to disk."""
        try:
            data = {
                "version": 1,
                "saved_at": datetime.now().isoformat(),
                "nodes": [node.to_dict() for node in self._nodes.values()],
            }
            with open(self.graph_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save mind graph: {e}")

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(
        self,
        label: str,
        description: str = "",
        node_type: NodeType = NodeType.CONCEPT,
        parent_id: str | None = None,
        color: str | None = None,
        source: str = "system",
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MindNode:
        """Add a new node to the graph.

        Args:
            label: Short label for the node
            description: Detailed description
            node_type: Type of node
            parent_id: ID of parent node (defaults to root)
            color: Optional color (inherits from parent if not set)
            source: Where this node came from
            provenance: Creation context (conversation ID, user message, etc.)
            metadata: Additional data

        Returns:
            The created node
        """
        with self._lock:
            # Determine parent
            parent_id = parent_id or self.ROOT_ID
            parent = self._nodes.get(parent_id)

            # Inherit color from parent if not specified
            if color is None and parent:
                color = parent.color
            color = color or "#8B5CF6"

            # Create node
            node = MindNode(
                label=label,
                description=description,
                node_type=node_type,
                color=color,
                source=source,
                provenance=provenance,
                metadata=metadata,
            )

            # Add to graph
            self._nodes[node.id] = node

            # Add parent-child relationship
            if parent:
                parent.add_child(node.id)
                node.add_edge(parent_id, EdgeType.PARENT)

            self._save()

            # Auto-index for semantic search (best-effort)
            try:
                self.semantic_index.index_node(node)
            except Exception as e:
                logger.warning(f"Failed to index node: {e}")

            logger.info(f"Added node: {label} (type: {node_type.value}, parent: {parent_id})")
            return node

    def get_node(self, node_id: str) -> MindNode | None:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def update_node(
        self,
        node_id: str,
        label: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MindNode | None:
        """Update an existing node.

        Args:
            node_id: ID of node to update
            label: New label (if provided)
            description: New description (if provided)
            metadata: Metadata to merge (if provided)

        Returns:
            Updated node or None if not found
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return None

            if label is not None:
                node.label = label
            if description is not None:
                node.description = description
            if metadata is not None:
                node.metadata.update(metadata)

            node.updated_at = datetime.now().isoformat()
            self._save()
            return node

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and update relationships.

        Args:
            node_id: ID of node to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if node_id == self.ROOT_ID:
                logger.warning("Cannot delete root node")
                return False

            if node_id not in self._nodes:
                return False

            node = self._nodes[node_id]

            # Remove from parent's children
            for other_node in self._nodes.values():
                if node_id in other_node.children:
                    other_node.children.remove(node_id)

            # Delete the node
            del self._nodes[node_id]

            try:
                self.semantic_index.remove_node(node_id)
            except Exception as e:
                logger.warning(f"Failed to remove node from index: {e}")

            self._save()
            logger.info(f"Deleted node: {node_id}")
            return True

    def add_association(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.ASSOCIATION,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add an association between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            metadata: Edge metadata

        Returns:
            True if successful
        """
        with self._lock:
            source = self._nodes.get(source_id)
            target = self._nodes.get(target_id)

            if not source or not target:
                return False

            source.add_edge(target_id, edge_type, metadata)
            self._save()
            return True

    # =========================================================================
    # Traversal and Search
    # =========================================================================

    def get_children(self, node_id: str) -> list[MindNode]:
        """Get all child nodes of a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return []
            return [self._nodes[cid] for cid in node.children if cid in self._nodes]

    def get_subtree(self, node_id: str, max_depth: int = 3) -> list[MindNode]:
        """Get all nodes in a subtree up to max_depth."""
        with self._lock:
            result = []
            visited = set()

            def traverse(nid: str, depth: int):
                if depth > max_depth or nid in visited:
                    return
                visited.add(nid)
                node = self._nodes.get(nid)
                if node:
                    result.append(node)
                    for child_id in node.children:
                        traverse(child_id, depth + 1)

            traverse(node_id, 0)
            return result

    def search_by_label(self, query: str, limit: int = 10) -> list[MindNode]:
        """Search nodes by label (case-insensitive substring match)."""
        with self._lock:
            query_lower = query.lower()
            matches = []
            for node in self._nodes.values():
                if query_lower in node.label.lower():
                    matches.append(node)
                    if len(matches) >= limit:
                        break
            return matches

    def search_by_type(self, node_type: NodeType, limit: int = 50) -> list[MindNode]:
        """Get all nodes of a specific type."""
        with self._lock:
            matches = []
            for node in self._nodes.values():
                if node.node_type == node_type:
                    matches.append(node)
                    if len(matches) >= limit:
                        break
            return matches

    def get_recent_nodes(self, limit: int = 20) -> list[MindNode]:
        """Get most recently created/updated nodes."""
        with self._lock:
            sorted_nodes = sorted(
                self._nodes.values(),
                key=lambda n: n.updated_at,
                reverse=True
            )
            return sorted_nodes[:limit]

    def find_related(self, node_id: str) -> list[MindNode]:
        """Find all nodes connected to a node via edges."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return []

            related = set()

            # Add connected via edges
            for edge in node.edges:
                if edge["target"] in self._nodes:
                    related.add(edge["target"])

            # Add nodes that connect to this one
            for other in self._nodes.values():
                for edge in other.edges:
                    if edge["target"] == node_id:
                        related.add(other.id)

            return [self._nodes[nid] for nid in related if nid in self._nodes]

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        node_types: list[NodeType] | None = None,
    ) -> list[tuple[MindNode, float]]:
        """Convenience method for semantic search over nodes.

        Args:
            query: Natural language query
            limit: Max results to return
            node_types: Filter by node type (None = all types)

        Returns:
            List of (node, similarity_score) tuples sorted by similarity
        """
        results = self.semantic_index.search(query, limit, node_types)
        return [(r.node, r.similarity) for r in results]

    # =========================================================================
    # Context Generation
    # =========================================================================

    def get_identity_context(self) -> str:
        """Get all identity nodes as context for prompts."""
        identity_nodes = self.search_by_type(NodeType.IDENTITY)

        if not identity_nodes:
            return ""

        lines = ["### Core Identity"]
        for node in identity_nodes:
            if node.id != self.ROOT_ID:
                lines.append(f"- **{node.label}**: {node.description}")

        return "\n".join(lines)

    def get_facts_context(self) -> str:
        """Get all fact nodes as context for prompts."""
        fact_nodes = self.search_by_type(NodeType.FACT)

        if not fact_nodes:
            return ""

        lines = ["### Known Facts"]
        for node in fact_nodes:
            lines.append(f"- {node.label}: {node.description}")

        return "\n".join(lines)

    def get_recent_context(self, limit: int = 10) -> str:
        """Get recent memory context."""
        recent = self.get_recent_nodes(limit)

        if not recent:
            return ""

        lines = ["### Recent Context"]
        for node in recent:
            if node.id != self.ROOT_ID:
                lines.append(f"- [{node.node_type.value}] {node.label}")

        return "\n".join(lines)

    def get_full_context(self) -> str:
        """Get complete mind graph context for prompts."""
        sections = []

        identity = self.get_identity_context()
        if identity:
            sections.append(identity)

        facts = self.get_facts_context()
        if facts:
            sections.append(facts)

        recent = self.get_recent_context()
        if recent:
            sections.append(recent)

        return "\n\n".join(sections) if sections else ""

    # =========================================================================
    # Import from MYND format
    # =========================================================================

    def import_from_mynd(self, mynd_data: dict[str, Any], parent_id: str | None = None) -> int:
        """Import nodes from MYND export format.

        Args:
            mynd_data: MYND node structure (with children array)
            parent_id: Parent node to attach to (defaults to root)

        Returns:
            Number of nodes imported
        """
        count = 0

        def import_node(node_data: dict, pid: str) -> None:
            nonlocal count

            label = node_data.get("label", "Unnamed")
            description = node_data.get("description", "")
            color = node_data.get("color", "#8B5CF6")
            source = node_data.get("source", "mynd_import")

            # Build provenance from MYND data
            provenance = {}
            if "provenance" in node_data:
                provenance = node_data["provenance"]
            if "createdAt" in node_data:
                provenance["original_created_at"] = node_data["createdAt"]

            # Create the node
            new_node = self.add_node(
                label=label,
                description=description,
                node_type=NodeType.CONCEPT,  # Default, could be smarter
                parent_id=pid,
                color=color,
                source=source,
                provenance=provenance,
            )
            count += 1

            # Recursively import children
            for child in node_data.get("children", []):
                import_node(child, new_node.id)

        # Start import from provided parent or root
        parent_id = parent_id or self.ROOT_ID

        # If mynd_data has a 'map' key, that's the root
        if "map" in mynd_data:
            mynd_data = mynd_data["map"]

        # Import the tree
        for child in mynd_data.get("children", []):
            import_node(child, parent_id)

        logger.info(f"Imported {count} nodes from MYND format")
        return count

    # =========================================================================
    # Export
    # =========================================================================

    def export_to_dict(self) -> dict[str, Any]:
        """Export the entire graph to a dictionary."""
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now().isoformat(),
                "node_count": len(self._nodes),
                "nodes": [node.to_dict() for node in self._nodes.values()],
            }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the mind graph."""
        with self._lock:
            type_counts = {}
            for node in self._nodes.values():
                t = node.node_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

            return {
                "total_nodes": len(self._nodes),
                "nodes_by_type": type_counts,
                "root_children": len(self._nodes.get(self.ROOT_ID, MindNode("")).children),
            }


# =========================================================================
# Singleton Access
# =========================================================================

_mind_graph: MindGraph | None = None
_graph_lock = threading.Lock()


def get_mind_graph(storage_path: Path | str | None = None) -> MindGraph:
    """Get or create the global mind graph.

    Args:
        storage_path: Path for graph storage (used on first call)

    Returns:
        The mind graph singleton
    """
    global _mind_graph

    if _mind_graph is None:
        with _graph_lock:
            if _mind_graph is None:
                if storage_path is None:
                    storage_path = Path(__file__).parent.parent.parent / "memory" / "graph"
                _mind_graph = MindGraph(Path(storage_path))

    return _mind_graph
