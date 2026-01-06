"""Life OS API endpoints.

This module provides endpoints for Life OS functionality including:
- iMessage reading and search
- Contact lookup
- Semantic search over personal data
- Index management
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/life", tags=["life"])
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    query: str
    k: int = 10


class SemanticSearchResult(BaseModel):
    """A single semantic search result."""
    doc_id: str
    score: float
    doc_type: str
    metadata: Dict[str, Any]


class SemanticSearchResponse(BaseModel):
    """Response from semantic search."""
    query: str
    results: List[SemanticSearchResult]
    count: int


class IndexBuildResponse(BaseModel):
    """Response from index build."""
    indexed: int
    elapsed_seconds: Optional[float] = None
    index_path: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Message Endpoints
# =============================================================================

@router.get("/messages/recent")
async def get_recent_messages(
    limit: int = Query(default=50, ge=1, le=500)
) -> Dict[str, Any]:
    """Get recent messages across all conversations.

    Args:
        limit: Maximum number of messages to return (default 50, max 500)

    Returns:
        Dictionary with messages list

    Raises:
        HTTPException: If message reading fails
    """
    try:
        from services.life_os.message_reader import get_recent_messages
        messages = get_recent_messages(limit=limit)
        return {
            "success": True,
            "messages": messages,
            "count": len(messages),
        }
    except Exception as e:
        logger.error(f"Failed to get recent messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/search")
async def search_messages(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100)
) -> Dict[str, Any]:
    """Search messages by text content.

    Args:
        q: Search query string
        limit: Maximum number of results (default 20, max 100)

    Returns:
        Dictionary with matching messages

    Raises:
        HTTPException: If search fails
    """
    try:
        from services.life_os.message_reader import search_messages
        messages = search_messages(query=q, limit=limit)
        return {
            "success": True,
            "query": q,
            "messages": messages,
            "count": len(messages),
        }
    except Exception as e:
        logger.error(f"Failed to search messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/conversation/{contact_id:path}")
async def get_conversation(
    contact_id: str,
    limit: int = Query(default=50, ge=1, le=500)
) -> Dict[str, Any]:
    """Get messages from a specific conversation.

    Args:
        contact_id: Phone number or email (handle.id value)
        limit: Maximum number of messages (default 50, max 500)

    Returns:
        Dictionary with conversation messages

    Raises:
        HTTPException: If conversation fetch fails
    """
    try:
        from services.life_os.message_reader import get_conversation
        messages = get_conversation(contact_id=contact_id, limit=limit)
        return {
            "success": True,
            "contact_id": contact_id,
            "messages": messages,
            "count": len(messages),
        }
    except Exception as e:
        logger.error(f"Failed to get conversation for {contact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Contact Endpoints
# =============================================================================

@router.get("/contacts")
async def list_contacts() -> Dict[str, Any]:
    """List all contacts from the address book.

    Returns:
        Dictionary with contacts list

    Raises:
        HTTPException: If contact reading fails
    """
    try:
        from services.life_os.contact_reader import get_all_contacts
        contacts = get_all_contacts()
        return {
            "success": True,
            "contacts": contacts,
            "count": len(contacts),
        }
    except Exception as e:
        logger.error(f"Failed to list contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contacts/search")
async def search_contacts(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100)
) -> Dict[str, Any]:
    """Search contacts by name, email, or phone.

    Args:
        q: Search query string
        limit: Maximum number of results (default 20, max 100)

    Returns:
        Dictionary with matching contacts

    Raises:
        HTTPException: If search fails
    """
    try:
        from services.life_os.contact_reader import search_contacts
        contacts = search_contacts(query=q, limit=limit)
        return {
            "success": True,
            "query": q,
            "contacts": contacts,
            "count": len(contacts),
        }
    except Exception as e:
        logger.error(f"Failed to search contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Semantic Search Endpoints
# =============================================================================

@router.post("/search")
async def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """Perform semantic search over indexed personal data.

    Uses FAISS index to find semantically similar documents.

    Args:
        request: Search request with query and k parameter

    Returns:
        SemanticSearchResponse with matching results

    Raises:
        HTTPException: If search fails
    """
    try:
        from services.life_os.semantic_index import get_semantic_index

        index = get_semantic_index()
        results = await index.search_async(
            query=request.query,
            k=request.k,
        )

        return SemanticSearchResponse(
            query=request.query,
            results=[
                SemanticSearchResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    doc_type=r.doc_type,
                    metadata=r.metadata,
                )
                for r in results
            ],
            count=len(results),
        )
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/build")
async def build_index() -> IndexBuildResponse:
    """Trigger a full index rebuild from messages.

    Reads all messages from iMessage and builds a FAISS index
    for semantic search.

    Returns:
        IndexBuildResponse with build statistics

    Raises:
        HTTPException: If index build fails
    """
    try:
        from services.life_os.message_reader import get_recent_messages
        from services.life_os.semantic_index import get_semantic_index

        # Get all messages (use a large limit for full index)
        messages = get_recent_messages(limit=10000)

        # Convert messages to documents for indexing
        documents = []
        for msg in messages:
            if msg.get("text"):
                documents.append({
                    "id": str(msg["id"]),
                    "content": msg["text"],
                    "doc_type": "message",
                    "metadata": {
                        "date": msg.get("date"),
                        "is_from_me": msg.get("is_from_me"),
                        "contact_handle": msg.get("contact_handle"),
                    },
                })

        # Build the index
        index = get_semantic_index()
        result = await index.build_index_async(documents)

        if "error" in result:
            return IndexBuildResponse(
                indexed=0,
                error=result["error"],
            )

        return IndexBuildResponse(
            indexed=result.get("indexed", 0),
            elapsed_seconds=result.get("elapsed_seconds"),
            index_path=result.get("index_path"),
        )
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats")
async def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the semantic index.

    Returns:
        Dictionary with index statistics

    Raises:
        HTTPException: If stats retrieval fails
    """
    try:
        from services.life_os.semantic_index import get_semantic_index

        index = get_semantic_index()
        stats = index.get_stats()

        return {
            "success": True,
            **stats,
        }
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
