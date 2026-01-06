"""Life OS services for personal data integration."""

from .message_reader import (
    MessageReader,
    get_recent_messages,
    search_messages,
    get_conversation,
)
from .semantic_index import (
    FAISSSemanticIndex,
    Document,
    SearchResult,
    get_semantic_index,
)

__all__ = [
    "MessageReader",
    "get_recent_messages",
    "search_messages",
    "get_conversation",
    "FAISSSemanticIndex",
    "Document",
    "SearchResult",
    "get_semantic_index",
]
