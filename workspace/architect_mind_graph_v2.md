# Mind Graph v2 Architecture
## Semantic Search + Conversation Integration

---

## 1. Executive Summary

This document defines the architecture for extending the Mind Graph with:
1. **Semantic Search** - Embedding-based vector similarity search
2. **Conversation Integration** - Automatic node creation from chat conversations

The design prioritizes:
- Minimal dependencies (local embedding model)
- Non-blocking async operations
- Graceful degradation (works without embeddings)
- Clean integration with existing MindGraph API

---

## 2. Current Architecture Analysis

### Existing Components

```
backend/services/mind_graph.py
├── MindNode              # Node data model with label, description, metadata
├── MindGraph             # Core graph with CRUD, traversal, search_by_label
└── get_mind_graph()      # Singleton accessor

backend/routes/mind_graph.py
├── CRUD endpoints        # /api/mind/nodes, /nodes/{id}
├── Search endpoints      # /search/label, /search/type
├── Context endpoints     # /context, /remember, /fact
└── Import/Export         # /import/mynd, /export
```

### Storage

```
memory/graph/
└── mind_graph.json       # All nodes as JSON array
```

### Existing Chat Infrastructure

```
backend/services/chat_history.py
├── ChatHistoryManager    # Session-based message storage
├── Session structure     # id, title, messages[], timestamps
└── Message structure     # role, content, agent, thinking
```

---

## 3. Semantic Search Architecture

### 3.1 Component Overview

```
backend/services/
├── mind_graph.py              # Existing (unchanged)
├── embedding_service.py       # NEW: Model management + embedding operations
└── semantic_index.py          # NEW: Vector storage + similarity search

memory/graph/
├── mind_graph.json            # Existing node data
├── embeddings.npz             # NEW: NumPy compressed embeddings
└── embedding_meta.json        # NEW: Index metadata (node_id -> array index)
```

### 3.2 Embedding Service

```python
# backend/services/embedding_service.py

class EmbeddingService:
    """Manages embedding model and operations.

    Features:
    - Lazy model loading (only loads when first needed)
    - Configurable model (default: all-MiniLM-L6-v2)
    - Batch embedding for efficiency
    - Thread-safe singleton pattern
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns 384-dim vector."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, 384) array."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Singleton
_embedding_service: EmbeddingService | None = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
```

### 3.3 Semantic Index

```python
# backend/services/semantic_index.py

@dataclass
class SearchResult:
    """A single search result with similarity score."""
    node: MindNode
    similarity: float

class SemanticIndex:
    """Vector index for semantic search over MindGraph nodes.

    Architecture:
    - Embeddings stored in NumPy .npz file (compact, fast load)
    - Metadata maps node_id -> embedding array index
    - In-memory dict for fast access during runtime
    - Persisted to disk on updates
    """

    def __init__(self, graph: MindGraph, storage_path: Path):
        self.graph = graph
        self.storage_path = storage_path
        self.embeddings_file = storage_path / "embeddings.npz"
        self.meta_file = storage_path / "embedding_meta.json"

        self._embeddings: dict[str, np.ndarray] = {}  # node_id -> embedding
        self._dirty = False
        self._lock = threading.Lock()

        self._load()

    def _load(self):
        """Load embeddings from disk."""
        if self.embeddings_file.exists() and self.meta_file.exists():
            with open(self.meta_file) as f:
                meta = json.load(f)

            data = np.load(self.embeddings_file)
            embeddings_array = data["embeddings"]

            for node_id, idx in meta["index"].items():
                self._embeddings[node_id] = embeddings_array[idx]

    def _save(self):
        """Persist embeddings to disk."""
        if not self._embeddings:
            return

        # Build ordered arrays
        node_ids = list(self._embeddings.keys())
        embeddings_array = np.array([self._embeddings[nid] for nid in node_ids])

        # Save embeddings
        np.savez_compressed(self.embeddings_file, embeddings=embeddings_array)

        # Save metadata
        meta = {
            "version": 1,
            "model": EmbeddingService.MODEL_NAME,
            "dimension": EmbeddingService.EMBEDDING_DIM,
            "count": len(node_ids),
            "index": {nid: i for i, nid in enumerate(node_ids)},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        self._dirty = False

    def _build_embedding_text(self, node: MindNode) -> str:
        """Create rich text representation for embedding.

        Combines multiple fields for better semantic matching:
        - Label (title/name)
        - Description (details)
        - Type (provides context)
        - Parent context (disambiguation)
        """
        parts = [node.label]

        if node.description:
            parts.append(node.description)

        parts.append(f"Type: {node.node_type.value}")

        # Add parent label for context (helps disambiguate)
        parent_edges = [e for e in node.edges if e.get("type") == "parent"]
        if parent_edges:
            parent = self.graph.get_node(parent_edges[0]["target"])
            if parent:
                parts.append(f"Under: {parent.label}")

        return " | ".join(parts)

    def index_node(self, node: MindNode):
        """Add or update embedding for a node."""
        service = get_embedding_service()
        text = self._build_embedding_text(node)
        embedding = service.embed(text)

        with self._lock:
            self._embeddings[node.id] = embedding
            self._dirty = True

    def remove_node(self, node_id: str):
        """Remove embedding for deleted node."""
        with self._lock:
            if node_id in self._embeddings:
                del self._embeddings[node_id]
                self._dirty = True

    def search(
        self,
        query: str,
        limit: int = 10,
        node_types: list[NodeType] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SearchResult]:
        """Semantic search over indexed nodes.

        Args:
            query: Natural language query
            limit: Max results to return
            node_types: Filter by node type (None = all types)
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        if not self._embeddings:
            return []

        service = get_embedding_service()
        query_embedding = service.embed(query)

        results = []
        with self._lock:
            for node_id, embedding in self._embeddings.items():
                similarity = service.cosine_similarity(query_embedding, embedding)

                if similarity < min_similarity:
                    continue

                node = self.graph.get_node(node_id)
                if not node:
                    continue

                if node_types and node.node_type not in node_types:
                    continue

                results.append(SearchResult(node=node, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def find_similar_nodes(
        self,
        node_id: str,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """Find nodes similar to a given node."""
        if node_id not in self._embeddings:
            return []

        node_embedding = self._embeddings[node_id]
        service = get_embedding_service()

        results = []
        with self._lock:
            for other_id, embedding in self._embeddings.items():
                if other_id == node_id:
                    continue

                similarity = service.cosine_similarity(node_embedding, embedding)
                if similarity < min_similarity:
                    continue

                node = self.graph.get_node(other_id)
                if node:
                    results.append(SearchResult(node=node, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def rebuild_all(self):
        """Rebuild embeddings for all nodes. Call after bulk import."""
        service = get_embedding_service()
        nodes = list(self.graph._nodes.values())

        if not nodes:
            return

        texts = [self._build_embedding_text(n) for n in nodes]
        embeddings = service.embed_batch(texts)

        with self._lock:
            self._embeddings = {
                node.id: embeddings[i]
                for i, node in enumerate(nodes)
            }
            self._dirty = True

        self._save()

    def flush(self):
        """Save pending changes to disk."""
        if self._dirty:
            self._save()
```

### 3.4 Integration with MindGraph

Modify `MindGraph` to auto-index nodes:

```python
# In backend/services/mind_graph.py

class MindGraph:
    def __init__(self, storage_path: Path):
        # ... existing init ...
        self._semantic_index: SemanticIndex | None = None

    @property
    def semantic_index(self) -> SemanticIndex:
        """Lazy-load semantic index."""
        if self._semantic_index is None:
            from .semantic_index import SemanticIndex
            self._semantic_index = SemanticIndex(self, self.storage_path)
        return self._semantic_index

    def add_node(self, ...):
        # ... existing code ...
        node = MindNode(...)
        self._nodes[node.id] = node
        self._save()

        # Auto-index for semantic search
        try:
            self.semantic_index.index_node(node)
        except Exception as e:
            logger.warning(f"Failed to index node: {e}")

        return node

    def delete_node(self, node_id: str) -> bool:
        # ... existing code ...
        if deleted:
            self.semantic_index.remove_node(node_id)
        return deleted

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        node_types: list[NodeType] | None = None,
    ) -> list[tuple[MindNode, float]]:
        """Semantic search using embeddings."""
        results = self.semantic_index.search(query, limit, node_types)
        return [(r.node, r.similarity) for r in results]
```

### 3.5 API Endpoints

```python
# Add to backend/routes/mind_graph.py

class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 10
    node_types: list[str] | None = None
    min_similarity: float = 0.3

class SemanticSearchResponse(BaseModel):
    node: NodeResponse
    similarity: float

@router.post("/search/semantic", response_model=list[SemanticSearchResponse])
async def semantic_search(request: SemanticSearchRequest) -> list[SemanticSearchResponse]:
    """Search nodes by semantic similarity."""
    graph = get_mind_graph()

    node_types = None
    if request.node_types:
        node_types = [NodeType(nt) for nt in request.node_types]

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

@router.get("/nodes/{node_id}/similar", response_model=list[SemanticSearchResponse])
async def find_similar(node_id: str, limit: int = 5) -> list[SemanticSearchResponse]:
    """Find nodes similar to a given node."""
    graph = get_mind_graph()

    results = graph.semantic_index.find_similar_nodes(node_id, limit)

    return [
        SemanticSearchResponse(
            node=node_to_response(r.node),
            similarity=r.similarity,
        )
        for r in results
    ]

@router.post("/index/rebuild")
async def rebuild_index() -> dict:
    """Rebuild semantic index for all nodes."""
    graph = get_mind_graph()

    # Run in background to not block
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, graph.semantic_index.rebuild_all)

    return {"success": True, "indexed": len(graph._nodes)}
```

---

## 4. Conversation Integration Architecture

### 4.1 Component Overview

```
backend/services/
├── conversation_analyzer.py   # NEW: Extract memories from conversations
├── memory_extractor.py        # NEW: LLM-based extraction
└── mind_graph.py              # Extended with conversation hooks

backend/websocket/
└── chat_handler.py            # Modified to trigger extraction
```

### 4.2 Conversation Analyzer

```python
# backend/services/conversation_analyzer.py

from dataclasses import dataclass
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
    label: str                    # Short title (max 100 chars)
    description: str              # Full detail
    importance: int               # 1-5 (5 = core identity)
    source_message: str           # Original message text
    confidence: float             # Extraction confidence (0-1)
    related_concepts: list[str]   # Suggested links

class ConversationAnalyzer:
    """Analyzes conversations to extract memorable information.

    Uses a two-phase approach:
    1. Pattern matching for explicit signals (fast, always runs)
    2. LLM extraction for implicit information (optional, configurable)
    """

    # Patterns that signal explicit memory intent
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
        """Extract memories from explicit patterns (fast path)."""
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
```

### 4.3 LLM-Based Memory Extractor

```python
# backend/services/memory_extractor.py

import json
from typing import Any

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

    def __init__(self, anthropic_client: Any):
        self.client = anthropic_client

    async def extract(
        self,
        messages: list[dict],
        model: str = "claude-3-haiku-20240307",
    ) -> list[ExtractedMemory]:
        """Extract memories using Claude.

        Args:
            messages: Conversation messages [{role, content}]
            model: Model to use (haiku for cost efficiency)

        Returns:
            List of extracted memories
        """
        # Format conversation
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
            if m.get("content")
        )

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

            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            memories = []
            for item in data:
                memories.append(ExtractedMemory(
                    category=MemoryCategory(item["category"]),
                    label=item["label"][:100],
                    description=item["description"],
                    importance=min(5, max(1, item["importance"])),
                    source_message=conversation_text[:500],  # Truncate for storage
                    confidence=0.8,
                    related_concepts=item.get("related_concepts", []),
                ))

            return memories

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse LLM extraction: {e}")
            return []
```

### 4.4 Conversation Memory Service

```python
# backend/services/conversation_memory.py

class ConversationMemoryService:
    """Orchestrates memory extraction and graph integration.

    Responsibilities:
    - Analyze conversations for memorable content
    - Create nodes in MindGraph
    - Link nodes via semantic similarity
    - Track conversation provenance
    """

    def __init__(
        self,
        graph: MindGraph,
        anthropic_client: Any,
        enable_llm_extraction: bool = True,
    ):
        self.graph = graph
        self.analyzer = ConversationAnalyzer()
        self.extractor = MemoryExtractor(anthropic_client) if enable_llm_extraction else None
        self.enable_llm = enable_llm_extraction

    async def process_conversation(
        self,
        session_id: str,
        messages: list[dict],
    ) -> list[MindNode]:
        """Process a conversation and create memory nodes.

        Called after conversation ends or at periodic intervals.

        Args:
            session_id: Chat session ID for provenance
            messages: Conversation messages

        Returns:
            List of created nodes
        """
        created_nodes = []

        # Phase 1: Explicit pattern extraction (always)
        for msg in messages:
            if msg.get("role") == "user":
                explicit = self.analyzer.extract_explicit(msg.get("content", ""))
                for memory in explicit:
                    node = await self._create_node_from_memory(memory, session_id)
                    if node:
                        created_nodes.append(node)

        # Phase 2: LLM extraction (if enabled and worthwhile)
        if self.enable_llm and self.analyzer.should_extract_with_llm(messages):
            llm_memories = await self.extractor.extract(messages)
            for memory in llm_memories:
                # Skip if too similar to already created node
                if self._is_duplicate(memory, created_nodes):
                    continue
                node = await self._create_node_from_memory(memory, session_id)
                if node:
                    created_nodes.append(node)

        # Phase 3: Create conversation memory node (episodic memory)
        if created_nodes:
            conv_node = self._create_conversation_node(session_id, created_nodes)
            created_nodes.insert(0, conv_node)

        return created_nodes

    async def _create_node_from_memory(
        self,
        memory: ExtractedMemory,
        session_id: str,
    ) -> MindNode | None:
        """Create a graph node from extracted memory."""

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

        # Find appropriate parent via semantic similarity
        parent_id = await self._find_parent(memory)

        # Create node
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
                "source_excerpt": memory.source_message[:200],
            },
            metadata={
                "importance": memory.importance,
            },
        )

        # Link to semantically related nodes
        await self._link_related(node, memory.related_concepts)

        return node

    async def _find_parent(self, memory: ExtractedMemory) -> str | None:
        """Find appropriate parent node via semantic similarity."""
        # Search for similar nodes
        results = self.graph.semantic_index.search(
            query=memory.description,
            limit=5,
            min_similarity=0.6,
        )

        if not results:
            return None

        # Use most similar node as parent if high confidence
        top_result = results[0]
        if top_result.similarity > 0.75:
            return top_result.node.id

        return None

    async def _link_related(self, node: MindNode, concepts: list[str]):
        """Link node to related concepts."""
        for concept in concepts:
            # Search for existing node matching concept
            results = self.graph.semantic_index.search(
                query=concept,
                limit=1,
                min_similarity=0.7,
            )
            if results:
                self.graph.add_association(
                    source_id=node.id,
                    target_id=results[0].node.id,
                    edge_type=EdgeType.ASSOCIATION,
                )

    def _is_duplicate(
        self,
        memory: ExtractedMemory,
        existing: list[MindNode],
    ) -> bool:
        """Check if memory is duplicate of existing node."""
        for node in existing:
            # Simple label similarity check
            if memory.label.lower() in node.label.lower():
                return True
            if node.label.lower() in memory.label.lower():
                return True
        return False

    def _create_conversation_node(
        self,
        session_id: str,
        extracted_nodes: list[MindNode],
    ) -> MindNode:
        """Create episodic memory for the conversation itself."""
        now = datetime.now()

        node = self.graph.add_node(
            label=f"Conversation: {now.strftime('%Y-%m-%d %H:%M')}",
            description=f"Extracted {len(extracted_nodes)} memories from conversation",
            node_type=NodeType.MEMORY,
            source="system",
            provenance={
                "session_id": session_id,
                "extracted_node_ids": [n.id for n in extracted_nodes],
            },
        )

        # Link to all extracted nodes
        for extracted in extracted_nodes:
            self.graph.add_association(
                source_id=node.id,
                target_id=extracted.id,
                edge_type=EdgeType.DERIVED,
            )

        return node
```

### 4.5 Integration Points

#### Chat Handler Integration

```python
# Modify backend/websocket/chat_handler.py

from backend.services.conversation_memory import ConversationMemoryService

# After conversation ends or at end of session:
async def on_conversation_end(session_id: str, messages: list[dict]):
    """Process conversation for memory extraction."""
    memory_service = get_conversation_memory_service()
    created_nodes = await memory_service.process_conversation(session_id, messages)

    if created_nodes:
        logger.info(f"Created {len(created_nodes)} memory nodes from conversation {session_id}")
```

#### Real-Time Processing Option

```python
# For real-time extraction during conversation

class RealtimeMemoryExtractor:
    """Extract memories in real-time as messages arrive."""

    def __init__(self, memory_service: ConversationMemoryService):
        self.service = memory_service
        self.pending_messages: list[dict] = []
        self.extraction_interval = 5  # Extract every 5 messages

    async def on_message(self, session_id: str, message: dict):
        """Process incoming message."""
        self.pending_messages.append(message)

        # Immediate explicit extraction
        if message.get("role") == "user":
            explicit = self.service.analyzer.extract_explicit(message.get("content", ""))
            for memory in explicit:
                await self.service._create_node_from_memory(memory, session_id)

        # Periodic LLM extraction
        if len(self.pending_messages) >= self.extraction_interval:
            await self._do_llm_extraction(session_id)

    async def _do_llm_extraction(self, session_id: str):
        """Run LLM extraction on pending messages."""
        if self.service.enable_llm:
            await self.service.extractor.extract(self.pending_messages)
        self.pending_messages = []
```

---

## 5. API Summary

### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mind/search/semantic` | POST | Semantic search with embeddings |
| `/api/mind/nodes/{id}/similar` | GET | Find similar nodes |
| `/api/mind/index/rebuild` | POST | Rebuild embedding index |
| `/api/mind/extract` | POST | Manual memory extraction from text |

### Request/Response Examples

#### Semantic Search

```http
POST /api/mind/search/semantic
Content-Type: application/json

{
  "query": "what do I like about Python?",
  "limit": 10,
  "node_types": ["preference", "fact"],
  "min_similarity": 0.4
}
```

Response:
```json
[
  {
    "node": {
      "id": "node-123",
      "label": "Prefers Python for scripting",
      "description": "I like Python because it's readable and fast to write",
      "node_type": "preference",
      ...
    },
    "similarity": 0.82
  }
]
```

---

## 6. Data Flow Diagrams

### Semantic Search Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ EmbeddingService│
│   embed(query)  │
└────────┬────────┘
         │ 384-dim vector
         ▼
┌─────────────────┐
│  SemanticIndex  │
│  cosine_sim()   │
│  for all nodes  │
└────────┬────────┘
         │ sorted results
         ▼
┌─────────────────┐
│   Filter by     │
│  type/threshold │
└────────┬────────┘
         │
         ▼
   Search Results
```

### Conversation Integration Flow

```
Chat Messages
    │
    ├──────────────────────────┐
    ▼                          ▼
┌────────────────┐    ┌─────────────────┐
│ Explicit Regex │    │  LLM Extraction │
│   (always)     │    │  (if enabled)   │
└───────┬────────┘    └────────┬────────┘
        │                      │
        └──────────┬───────────┘
                   ▼
           ExtractedMemory[]
                   │
                   ▼
         ┌─────────────────┐
         │ Semantic Parent │
         │   Discovery     │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Create Nodes   │
         │  in MindGraph   │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Link Related   │
         │    Concepts     │
         └────────┬────────┘
                  │
                  ▼
         Conversation Memory
              (episodic)
```

---

## 7. Dependencies

### Required

```toml
# pyproject.toml additions

[project.dependencies]
sentence-transformers = "^2.2.0"  # Local embedding model
numpy = "^1.24.0"                 # Already likely present
```

### Optional (Future)

```toml
# For production scale
chromadb = "^0.4.0"              # Vector database
qdrant-client = "^1.7.0"         # Alternative vector DB
```

---

## 8. Configuration

```python
# backend/config.py

class MindGraphConfig:
    # Embedding settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Semantic search defaults
    DEFAULT_SEARCH_LIMIT: int = 10
    DEFAULT_MIN_SIMILARITY: float = 0.3

    # Conversation extraction
    ENABLE_LLM_EXTRACTION: bool = True
    LLM_EXTRACTION_MODEL: str = "claude-3-haiku-20240307"
    MIN_MESSAGES_FOR_LLM: int = 3

    # Auto-extraction triggers
    EXTRACT_ON_SESSION_END: bool = True
    EXTRACT_INTERVAL_MESSAGES: int = 10
```

---

## 9. Migration Plan

### Phase 1: Semantic Search

1. Add `sentence-transformers` dependency
2. Create `EmbeddingService` class
3. Create `SemanticIndex` class
4. Add `semantic_search()` to MindGraph
5. Add API endpoints
6. Background job to index existing nodes

### Phase 2: Conversation Integration

1. Create `ConversationAnalyzer` (pattern-based)
2. Create `MemoryExtractor` (LLM-based)
3. Create `ConversationMemoryService`
4. Add hooks in chat handler
5. Test with real conversations

### Phase 3: Refinement

1. Tune similarity thresholds based on feedback
2. Add duplicate detection improvements
3. Implement memory importance decay
4. Add user feedback mechanism

---

## 10. Testing Strategy

```python
# tests/test_semantic_search.py

def test_embedding_consistency():
    """Same text should produce same embedding."""
    service = EmbeddingService()
    e1 = service.embed("Hello world")
    e2 = service.embed("Hello world")
    assert np.allclose(e1, e2)

def test_similarity_ordering():
    """Similar texts should have higher similarity."""
    service = EmbeddingService()
    base = service.embed("Python programming language")
    similar = service.embed("Python coding and development")
    different = service.embed("Cooking recipes and ingredients")

    sim_similar = service.cosine_similarity(base, similar)
    sim_different = service.cosine_similarity(base, different)

    assert sim_similar > sim_different

def test_semantic_search_returns_relevant():
    """Search should return semantically relevant nodes."""
    graph = create_test_graph()
    graph.add_node(label="Python is great", description="I love Python programming")
    graph.add_node(label="Java experience", description="I worked with Java")

    results = graph.semantic_search("What do I think about Python?")

    assert len(results) > 0
    assert "Python" in results[0][0].label
```

---

## 11. Security Considerations

1. **Sensitive Data**: Add sensitivity classification to nodes
2. **No API Keys**: Never store credentials in memories
3. **Encryption**: Consider encrypting sensitive node descriptions
4. **Rate Limiting**: Limit embedding API calls if using external service
5. **User Consent**: Prompt before storing personal information

---

## 12. Future Enhancements

1. **Vector Database**: Migrate to ChromaDB/Qdrant for scale
2. **Hybrid Search**: Combine keyword + semantic search
3. **Memory Decay**: Implement forgetting curves
4. **Multi-Agent Memories**: Per-agent memory isolation
5. **Semantic Neighborhoods**: Pre-compute similar nodes
6. **Memory Compression**: Summarize old memories
