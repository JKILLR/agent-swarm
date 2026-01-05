"""API routes for the Mind Graph memory system.

Provides REST API endpoints for:
- Creating, reading, updating, deleting nodes
- Traversing the graph
- Searching and querying
- Importing from MYND format
- Getting context for prompts
"""

import logging
from typing import Optional, List, Dict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.mind_graph import (
    EdgeType,
    MindGraph,
    MindNode,
    NodeType,
    get_mind_graph,
)
from backend.services.conversation_memory import (
    ConversationMemoryService,
    get_conversation_memory_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mind", tags=["mind-graph"])


# =========================================================================
# Request/Response Models
# =========================================================================


class CreateNodeRequest(BaseModel):
    """Request to create a new node."""
    label: str
    description: str = ""
    node_type: str = "concept"
    parent_id: Optional[str] = None
    color: Optional[str] = None
    source: str = "api"
    provenance: Optional[Dict] = None
    metadata: Optional[Dict] = None


class UpdateNodeRequest(BaseModel):
    """Request to update a node."""
    label: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict] = None


class AddEdgeRequest(BaseModel):
    """Request to add an edge between nodes."""
    source_id: str
    target_id: str
    edge_type: str = "association"
    metadata: Optional[Dict] = None


class NodeResponse(BaseModel):
    """Response containing a node."""
    id: str
    label: str
    description: str
    node_type: str
    color: str
    source: str
    created_at: str
    updated_at: str
    provenance: dict
    metadata: dict
    children: List[str]
    edges: List[Dict]


class ImportMyndRequest(BaseModel):
    """Request to import from MYND format."""
    file_path: str
    parent_id: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    query: str
    limit: int = 10
    node_types: Optional[List[str]] = None
    min_similarity: float = 0.3


class SemanticSearchResponse(BaseModel):
    """Response for semantic search results."""
    node: NodeResponse
    similarity: float


class MessageInput(BaseModel):
    """A single message in a conversation."""
    role: str
    content: str


class ExtractConversationRequest(BaseModel):
    """Request to extract memories from a conversation."""
    session_id: str
    messages: List[MessageInput]


class ExtractTextRequest(BaseModel):
    """Request to extract memories from raw text."""
    text: str
    source: str = "manual"


class ExtractResponse(BaseModel):
    """Response containing extracted nodes."""
    success: bool
    session_id: Optional[str] = None
    nodes: List[NodeResponse]
    node_count: int


# =========================================================================
# Helper Functions
# =========================================================================


def node_to_response(node: MindNode) -> NodeResponse:
    """Convert a MindNode to API response."""
    return NodeResponse(
        id=node.id,
        label=node.label,
        description=node.description,
        node_type=node.node_type.value,
        color=node.color,
        source=node.source,
        created_at=node.created_at,
        updated_at=node.updated_at,
        provenance=node.provenance,
        metadata=node.metadata,
        children=node.children,
        edges=node.edges,
    )


# =========================================================================
# Node CRUD Endpoints
# =========================================================================


@router.post("/nodes", response_model=NodeResponse)
async def create_node(request: CreateNodeRequest) -> NodeResponse:
    """Create a new node in the mind graph."""
    graph = get_mind_graph()

    try:
        node_type = NodeType(request.node_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid node type: {request.node_type}")

    node = graph.add_node(
        label=request.label,
        description=request.description,
        node_type=node_type,
        parent_id=request.parent_id,
        color=request.color,
        source=request.source,
        provenance=request.provenance,
        metadata=request.metadata,
    )

    return node_to_response(node)


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str) -> NodeResponse:
    """Get a node by ID."""
    graph = get_mind_graph()
    node = graph.get_node(node_id)

    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    return node_to_response(node)


@router.patch("/nodes/{node_id}", response_model=NodeResponse)
async def update_node(node_id: str, request: UpdateNodeRequest) -> NodeResponse:
    """Update a node."""
    graph = get_mind_graph()

    node = graph.update_node(
        node_id=node_id,
        label=request.label,
        description=request.description,
        metadata=request.metadata,
    )

    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    return node_to_response(node)


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str) -> dict:
    """Delete a node."""
    graph = get_mind_graph()

    if not graph.delete_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node not found or cannot be deleted: {node_id}")

    return {"success": True, "deleted": node_id}


# =========================================================================
# Edge Endpoints
# =========================================================================


@router.post("/edges")
async def add_edge(request: AddEdgeRequest) -> dict:
    """Add an edge between two nodes."""
    graph = get_mind_graph()

    try:
        edge_type = EdgeType(request.edge_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid edge type: {request.edge_type}")

    if not graph.add_association(
        source_id=request.source_id,
        target_id=request.target_id,
        edge_type=edge_type,
        metadata=request.metadata,
    ):
        raise HTTPException(status_code=404, detail="Source or target node not found")

    return {"success": True}


# =========================================================================
# Traversal Endpoints
# =========================================================================


@router.get("/nodes/{node_id}/children", response_model=List[NodeResponse])
async def get_children(node_id: str) -> List[NodeResponse]:
    """Get all children of a node."""
    graph = get_mind_graph()
    children = graph.get_children(node_id)
    return [node_to_response(n) for n in children]


@router.get("/nodes/{node_id}/subtree", response_model=List[NodeResponse])
async def get_subtree(node_id: str, max_depth: int = 3) -> List[NodeResponse]:
    """Get all nodes in a subtree."""
    graph = get_mind_graph()
    nodes = graph.get_subtree(node_id, max_depth=max_depth)
    return [node_to_response(n) for n in nodes]


@router.get("/nodes/{node_id}/related", response_model=List[NodeResponse])
async def get_related(node_id: str) -> List[NodeResponse]:
    """Get all nodes related to a node via edges."""
    graph = get_mind_graph()
    related = graph.find_related(node_id)
    return [node_to_response(n) for n in related]


# =========================================================================
# Search Endpoints
# =========================================================================


@router.get("/search/label", response_model=List[NodeResponse])
async def search_by_label(q: str, limit: int = 10) -> List[NodeResponse]:
    """Search nodes by label."""
    graph = get_mind_graph()
    matches = graph.search_by_label(q, limit=limit)
    return [node_to_response(n) for n in matches]


@router.get("/search/type", response_model=List[NodeResponse])
async def search_by_type(node_type: str, limit: int = 50) -> List[NodeResponse]:
    """Get all nodes of a specific type."""
    graph = get_mind_graph()

    try:
        nt = NodeType(node_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid node type: {node_type}")

    matches = graph.search_by_type(nt, limit=limit)
    return [node_to_response(n) for n in matches]


@router.get("/recent", response_model=List[NodeResponse])
async def get_recent(limit: int = 20) -> List[NodeResponse]:
    """Get recently created/updated nodes."""
    graph = get_mind_graph()
    recent = graph.get_recent_nodes(limit=limit)
    return [node_to_response(n) for n in recent]


# =========================================================================
# Semantic Search Endpoints
# =========================================================================


@router.post("/search/semantic", response_model=List[SemanticSearchResponse])
async def semantic_search(request: SemanticSearchRequest) -> List[SemanticSearchResponse]:
    """Search nodes using semantic similarity."""
    graph = get_mind_graph()

    node_types = None
    if request.node_types:
        try:
            node_types = [NodeType(nt) for nt in request.node_types]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid node type: {e}")

    results = graph.semantic_index.search(
        query=request.query,
        limit=request.limit,
        node_types=node_types,
        min_similarity=request.min_similarity,
    )

    return [
        SemanticSearchResponse(
            node=node_to_response(r.node),
            similarity=r.similarity,
        )
        for r in results
    ]


@router.get("/nodes/{node_id}/similar", response_model=List[SemanticSearchResponse])
async def find_similar(node_id: str, limit: int = 5, min_similarity: float = 0.5) -> List[SemanticSearchResponse]:
    """Find nodes similar to a given node."""
    graph = get_mind_graph()
    results = graph.semantic_index.find_similar_nodes(node_id, limit, min_similarity)
    return [
        SemanticSearchResponse(
            node=node_to_response(r.node),
            similarity=r.similarity,
        )
        for r in results
    ]


@router.post("/index/rebuild")
async def rebuild_index() -> dict:
    """Rebuild the semantic index for all nodes."""
    import asyncio
    graph = get_mind_graph()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, graph.semantic_index.rebuild_all)
    return {"success": True, "indexed": len(graph._nodes)}


@router.get("/index/stats")
async def index_stats() -> dict:
    """Get statistics about the semantic index."""
    graph = get_mind_graph()
    return graph.semantic_index.get_stats()


# =========================================================================
# Context Endpoints
# =========================================================================


@router.get("/context")
async def get_context() -> dict:
    """Get full mind graph context for prompts."""
    graph = get_mind_graph()
    return {
        "identity": graph.get_identity_context(),
        "facts": graph.get_facts_context(),
        "recent": graph.get_recent_context(),
        "full": graph.get_full_context(),
    }


# =========================================================================
# Import/Export Endpoints
# =========================================================================


@router.post("/import/mynd")
async def import_mynd(request: ImportMyndRequest) -> dict:
    """Import nodes from a MYND JSON export file."""
    import json

    graph = get_mind_graph()
    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    try:
        with open(file_path, "r") as f:
            mynd_data = json.load(f)

        count = graph.import_from_mynd(mynd_data, parent_id=request.parent_id)

        return {
            "success": True,
            "imported": count,
            "file": str(file_path),
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")


@router.get("/export")
async def export_graph() -> dict:
    """Export the entire mind graph."""
    graph = get_mind_graph()
    return graph.export_to_dict()


@router.get("/stats")
async def get_stats() -> dict:
    """Get statistics about the mind graph."""
    graph = get_mind_graph()
    return graph.get_stats()


# =========================================================================
# Conversation Extraction Endpoints
# =========================================================================


@router.post("/extract", response_model=ExtractResponse)
async def extract_from_conversation(request: ExtractConversationRequest) -> ExtractResponse:
    """Extract memories from a conversation.

    Processes a list of messages and extracts memorable information,
    creating nodes in the mind graph. Uses both pattern-based extraction
    (fast, always runs) and optionally LLM-based extraction.

    Args:
        request: Contains session_id and messages array

    Returns:
        ExtractResponse with created nodes
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        service = get_conversation_memory_service()

        # Convert MessageInput to dict format expected by the service
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        created_nodes = await service.process_conversation(
            session_id=request.session_id,
            messages=messages,
        )

        return ExtractResponse(
            success=True,
            session_id=request.session_id,
            nodes=[node_to_response(n) for n in created_nodes],
            node_count=len(created_nodes),
        )

    except Exception as e:
        logger.error(f"Failed to extract from conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/extract/text", response_model=ExtractResponse)
async def extract_from_text(request: ExtractTextRequest) -> ExtractResponse:
    """Extract memories from raw text.

    Manually extract memories from a block of text. Useful for importing
    notes, documents, or other unstructured content into the mind graph.

    This endpoint uses pattern-based extraction on the text to find
    explicit memory signals (e.g., "remember that...", "note to self...").

    Args:
        request: Contains text to extract from and optional source tag

    Returns:
        ExtractResponse with created nodes
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        import uuid
        from datetime import datetime

        service = get_conversation_memory_service()

        # Generate a synthetic session ID for provenance tracking
        session_id = f"text-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"

        # Wrap text as a single user message for the analyzer
        messages = [{"role": "user", "content": request.text}]

        created_nodes = await service.process_conversation(
            session_id=session_id,
            messages=messages,
        )

        # Tag nodes with the manual source
        graph = get_mind_graph()
        for node in created_nodes:
            if node.source == "conversation":
                node.source = request.source
                graph.update_node(
                    node_id=node.id,
                    metadata={**node.metadata, "extraction_source": "manual_text"},
                )

        return ExtractResponse(
            success=True,
            session_id=session_id,
            nodes=[node_to_response(n) for n in created_nodes],
            node_count=len(created_nodes),
        )

    except Exception as e:
        logger.error(f"Failed to extract from text: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# =========================================================================
# Convenience Endpoints for Common Operations
# =========================================================================


@router.post("/remember")
async def remember(
    content: str,
    node_type: str = "memory",
    parent_id: Optional[str] = None,
    source: str = "conversation",
) -> NodeResponse:
    """Quick endpoint to add a memory/thought to the graph.

    This is a convenience endpoint for the most common use case:
    adding something to remember from a conversation.
    """
    graph = get_mind_graph()

    # Parse content - first line is label, rest is description
    lines = content.strip().split("\n", 1)
    label = lines[0][:100]  # Max 100 chars for label
    description = lines[1] if len(lines) > 1 else ""

    try:
        nt = NodeType(node_type)
    except ValueError:
        nt = NodeType.MEMORY

    node = graph.add_node(
        label=label,
        description=description,
        node_type=nt,
        parent_id=parent_id,
        source=source,
    )

    return node_to_response(node)


@router.post("/identity")
async def add_identity(statement: str, source: str = "conversation") -> NodeResponse:
    """Add an identity statement about the AI.

    These are core self-knowledge nodes that define who the AI is.
    """
    graph = get_mind_graph()

    node = graph.add_node(
        label=statement[:100],
        description=statement,
        node_type=NodeType.IDENTITY,
        source=source,
    )

    return node_to_response(node)


@router.post("/fact")
async def add_fact(key: str, value: str, source: str = "user") -> NodeResponse:
    """Add a fact about the user or world."""
    graph = get_mind_graph()

    node = graph.add_node(
        label=key,
        description=value,
        node_type=NodeType.FACT,
        source=source,
    )

    return node_to_response(node)
