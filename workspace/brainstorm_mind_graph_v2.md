# Mind Graph v2 Brainstorm
## Semantic Search + Conversation Integration

---

## Current State Analysis

The existing `MindGraph` implementation has:
- Node types: CONCEPT, FACT, MEMORY, IDENTITY, PREFERENCE, GOAL, DECISION, RELATIONSHIP
- Edge types: PARENT, CHILD, ASSOCIATION, TEMPORAL, DERIVED, REFERENCE
- Basic label substring search
- Provenance tracking (source, metadata)
- Hierarchical tree structure with root node
- Thread-safe singleton pattern
- JSON file persistence

**What's missing:**
- Semantic/meaning-based search (only substring matching on labels)
- No embeddings infrastructure
- No automatic node creation from conversations
- No importance detection or salience scoring

---

## Part 1: Semantic Search via Embeddings

### Embedding Model Options

| Approach | Pros | Cons |
|----------|------|------|
| **Local: sentence-transformers** | Privacy, no API costs, fast after load | ~400MB model, initial load time, CPU-bound |
| **Local: ONNX-optimized models** | Faster inference, smaller footprint | More setup complexity |
| **API: OpenAI text-embedding-3-small** | High quality, simple | Costs ~$0.02/1M tokens, latency, dependency |
| **API: Voyage AI / Cohere** | Competitive quality | Same API concerns |
| **Hybrid: Local for batch, API for quality** | Best of both | Complexity of managing two systems |

**Recommendation:** Start with `sentence-transformers/all-MiniLM-L6-v2` (local)
- 384-dimensional embeddings
- ~80MB model
- Fast enough for real-time
- Good quality for semantic similarity
- No external dependencies

### Creative Embedding Strategies

#### 1. Multi-Field Embeddings
Don't just embed `label` - create a rich text representation:
```python
def get_embedding_text(node: MindNode) -> str:
    parts = [
        f"Title: {node.label}",
        f"Description: {node.description}" if node.description else "",
        f"Type: {node.node_type.value}",
        # Include parent context for hierarchy awareness
        f"Context: {parent_labels_chain}" if parent else "",
    ]
    return " | ".join(filter(None, parts))
```

#### 2. Contextual Embeddings
Embed nodes **with their neighborhood**:
- Include parent/child labels in embedding text
- This makes "Python programming language" different from "Python snake"

#### 3. Temporal Decay Weighting
When searching, apply recency boost:
```python
recency_score = 1.0 / (1 + days_since_update * 0.1)
final_score = semantic_similarity * (0.7 + 0.3 * recency_score)
```

#### 4. Type-Aware Search
Weight results by node type relevance to query intent:
- User asks about "what do I think about X" → boost PREFERENCE, IDENTITY
- User asks "when did I learn X" → boost MEMORY nodes
- User asks "what is X" → boost CONCEPT, FACT nodes

### Embedding Storage Architecture

#### Option A: Sidecar File
```
memory/graph/
  mind_graph.json       # Node data
  embeddings.npy        # NumPy array of embeddings
  embedding_index.json  # Maps node_id -> array index
```
**Pros:** Simple, fast bulk load
**Cons:** Index management complexity

#### Option B: Embedded in Node
```python
class MindNode:
    embedding: list[float] | None = None  # Store directly in node
```
**Pros:** Single source of truth
**Cons:** JSON bloat, slower saves

#### Option C: Vector Database (Future)
Use ChromaDB, Qdrant, or FAISS for production scale.
**Pros:** Scalable, optimized ANN search
**Cons:** Additional dependency

**Recommendation:** Start with Option A (sidecar file) - simple and performant.

### Semantic Search Implementation Sketch

```python
class SemanticIndex:
    """Manages embeddings for semantic search."""

    def __init__(self, graph: MindGraph):
        self.graph = graph
        self.model = None  # Lazy load
        self.embeddings: dict[str, np.ndarray] = {}
        self._load_or_build()

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model

    def embed_node(self, node: MindNode) -> np.ndarray:
        text = self._build_embedding_text(node)
        return self._get_model().encode(text)

    def search(
        self,
        query: str,
        limit: int = 10,
        node_types: list[NodeType] | None = None,
        min_similarity: float = 0.3
    ) -> list[tuple[MindNode, float]]:
        """Semantic search with optional type filtering."""
        query_embedding = self._get_model().encode(query)

        results = []
        for node_id, embedding in self.embeddings.items():
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity >= min_similarity:
                node = self.graph.get_node(node_id)
                if node and (node_types is None or node.node_type in node_types):
                    results.append((node, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
```

### Advanced: Hierarchical Search

Leverage the tree structure for smarter retrieval:

1. **Search down from context:** If user is talking about "Python", search within the Python subtree first
2. **Bubble up for disambiguation:** If query matches multiple branches, return the common ancestor
3. **Path-aware ranking:** Score higher if query matches multiple nodes in same branch

---

## Part 2: Conversation Integration

### The Challenge
How do we detect which parts of a conversation should become permanent memories?

### Importance Detection Strategies

#### 1. Explicit Signals
Look for phrases that indicate importance:
```python
IMPORTANCE_PATTERNS = [
    r"remember that",
    r"important:",
    r"note to self",
    r"I (am|prefer|like|hate|want)",
    r"my (name|email|preference|goal) is",
    r"always|never do",
    r"from now on",
]
```

#### 2. LLM-Based Extraction (Structured Output)
After each conversation turn, ask Claude to extract memories:

```python
EXTRACTION_PROMPT = """
Analyze this conversation and extract any information worth remembering:

{conversation}

For each memory, provide:
1. category: identity | preference | fact | decision | goal
2. label: Short title (3-6 words)
3. description: Full detail
4. importance: 1-5 (5 = fundamental identity, 1 = minor preference)
5. related_concepts: List of related topics for linking

Return JSON array of memories, or empty array if nothing significant.
"""
```

**This is the most robust approach** - use the LLM's understanding rather than regex.

#### 3. Information Density Heuristics
Score messages by information density:
- Named entities (people, places, dates)
- Numbers and specific values
- Future intentions ("I will", "I plan to")
- Past events ("I did", "we decided")

#### 4. Emotional Salience
Detect emotionally significant moments:
- Strong sentiment (very positive/negative)
- Exclamation marks, caps (excitement)
- Phrases like "I love", "I hate", "this is crucial"

### Linking New Nodes to Context

When creating a node from conversation, we need to place it correctly in the graph.

#### Strategy 1: Topic Clustering
1. Embed the new memory
2. Find top-3 similar existing nodes
3. If similarity > 0.7, link as ASSOCIATION
4. If similarity > 0.85, make it a child of most similar

#### Strategy 2: LLM Classification
Ask Claude to categorize where it belongs:
```python
PLACEMENT_PROMPT = """
Given this new memory:
{new_memory}

And these existing categories in my mind:
{top_level_categories}

Which category does this belong under?
Should it be a subcategory of an existing node, or a new top-level category?
"""
```

#### Strategy 3: Conversation Context Anchoring
Track the "active context" during conversation:
```python
class ConversationContext:
    active_topics: list[str]  # Node IDs currently being discussed

    def on_new_memory(self, node: MindNode):
        # Auto-link to recently discussed topics
        for topic_id in self.active_topics[-3:]:
            node.add_edge(topic_id, EdgeType.REFERENCE)
```

### Conversation Session as Memory

Create a MEMORY node for each significant conversation:
```python
def create_conversation_memory(session_id: str, summary: str, extracted_nodes: list[str]):
    """Create episodic memory of a conversation."""
    memory = graph.add_node(
        label=f"Conversation: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        description=summary,
        node_type=NodeType.MEMORY,
        provenance={
            "session_id": session_id,
            "extracted_nodes": extracted_nodes,
        }
    )

    # Link to all nodes extracted from this conversation
    for node_id in extracted_nodes:
        graph.add_association(memory.id, node_id, EdgeType.DERIVED)

    return memory
```

This creates a bidirectional link:
- Memory → derived nodes (what we learned)
- Nodes → provenance has session_id (where we learned it)

---

## Part 3: Edge Cases and Challenges

### 1. Duplicate/Conflicting Memories
**Problem:** User says "I prefer dark mode" then later "I prefer light mode"

**Solutions:**
- **Temporal precedence:** Latest wins, but keep history
- **Explicit update:** "Update my preference for..." triggers update vs create
- **Conflict detection:** Before creating, search for conflicting nodes
  ```python
  def check_conflicts(new_node: MindNode) -> list[MindNode]:
      similar = semantic_search(new_node.description, node_types=[new_node.node_type])
      conflicts = [n for n in similar if n.label_similarity(new_node) > 0.8]
      return conflicts
  ```

### 2. Memory Decay / Forgetting
**Problem:** Graph grows forever, old irrelevant memories clutter search

**Solutions:**
- **Access-based scoring:** Track when nodes are retrieved, decay unused ones
- **Explicit archival:** Move old memories to "archive" subtree
- **Confidence decay:** Reduce importance score over time without access
  ```python
  def calculate_relevance(node: MindNode) -> float:
      days_since_access = (now - node.last_accessed).days
      base_importance = node.metadata.get("importance", 3)
      decay = 0.95 ** (days_since_access / 30)  # 5% decay per month
      return base_importance * decay
  ```

### 3. Privacy / Sensitivity
**Problem:** User shares sensitive info (passwords, health, relationships)

**Solutions:**
- **Sensitivity classification:** Tag nodes as `sensitive: true`
- **No-store patterns:** Detect and refuse to store passwords, API keys
- **User confirmation:** "I'm about to remember your medical condition. OK?"
- **Encryption at rest:** Encrypt sensitive node descriptions

### 4. Context Window Limits
**Problem:** Graph grows, but we can only inject limited context into prompts

**Solutions:**
- **Relevance-based retrieval:** Only inject semantically relevant nodes
- **Hierarchical summarization:** Summarize branches, expand on demand
- **Importance filtering:** Only include nodes above importance threshold
- **Token budgeting:**
  ```python
  def get_context_within_budget(query: str, max_tokens: int = 2000) -> str:
      nodes = semantic_search(query, limit=50)
      context = []
      token_count = 0
      for node, score in nodes:
          node_text = format_node(node)
          node_tokens = count_tokens(node_text)
          if token_count + node_tokens > max_tokens:
              break
          context.append(node_text)
          token_count += node_tokens
      return "\n".join(context)
  ```

### 5. Multi-User / Agent Scenarios
**Problem:** Multiple agents or users sharing a mind graph

**Solutions:**
- **Source attribution:** Always track `source` field
- **Per-agent views:** Filter by source for agent-specific context
- **Merge conflicts:** Implement conflict resolution for simultaneous edits
- **Namespacing:** `agent_123/preferences/` style node paths

### 6. Hallucinated Memories
**Problem:** LLM extracts something that wasn't actually said

**Solutions:**
- **Source anchoring:** Always store the original message that triggered extraction
- **Confidence scores:** LLM rates its own extraction confidence
- **User verification:** Periodic "Is this still true?" prompts
- **Audit trail:** Provenance includes exact quote and timestamp

---

## Part 4: Implementation Roadmap

### Phase 1: Semantic Search Foundation
1. Add `sentence-transformers` dependency
2. Create `SemanticIndex` class with lazy model loading
3. Implement sidecar embedding storage
4. Add `semantic_search()` method to `MindGraph`
5. Auto-embed new nodes on creation
6. Background task to embed existing nodes

### Phase 2: Basic Conversation Integration
1. Create `ConversationAnalyzer` class
2. Implement explicit signal detection (regex patterns)
3. Add `extract_memories()` method using LLM
4. Create conversation memory nodes
5. Basic topic linking via embedding similarity

### Phase 3: Smart Placement
1. Implement hierarchical search
2. Add LLM-based placement classification
3. Track conversation context for auto-linking
4. Conflict detection and resolution

### Phase 4: Production Hardening
1. Sensitivity detection and handling
2. Memory decay and archival
3. Token-budgeted context retrieval
4. Performance optimization (caching, batching)

---

## Part 5: Wild Ideas

### 1. Dream Mode
Periodic background process that:
- Reviews all memories
- Finds unexpected connections
- Creates new DERIVED nodes linking distant concepts
- "I just realized X is related to Y because..."

### 2. Forgetting Curves
Implement Ebbinghaus forgetting curves:
- Memories decay if not accessed
- "Spaced repetition" for important facts
- Surface forgotten but relevant memories in conversation

### 3. Emotional Color Mapping
Assign emotional valence to nodes:
- Positive experiences → warm colors
- Stressful topics → red tones
- Visualize the emotional landscape of memory

### 4. Counterfactual Memories
Store not just what happened, but what was considered:
- "We considered using Redis but chose PostgreSQL"
- DECISION nodes with `alternatives` metadata
- Enables "why did we decide X?" queries

### 5. Memory Perspectives
Same event from multiple viewpoints:
- User's perspective vs Agent's perspective
- Different agents have different "memories" of same conversation
- Enables "how did X perceive this?"

### 6. Semantic Neighborhoods
Pre-compute "neighborhoods" for fast retrieval:
- Each node stores its top-10 semantic neighbors
- Updated incrementally as graph changes
- Enables instant "related memories" without search

### 7. Memory Compression
As conversations age:
- Merge similar nodes into summaries
- "Week 1: Discussed project architecture" vs detailed nodes
- Hierarchical compression: day → week → month → year

---

## Appendix: Code Sketches

### Embedding Service

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

class EmbeddingService:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._model = None
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_cache()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _load_cache(self):
        embeddings_file = self.cache_dir / "embeddings.npz"
        if embeddings_file.exists():
            data = np.load(embeddings_file, allow_pickle=True)
            self._embeddings = dict(data.items())

    def _save_cache(self):
        np.savez(self.cache_dir / "embeddings.npz", **self._embeddings)
```

### Conversation Analyzer

```python
import re
from dataclasses import dataclass

@dataclass
class ExtractedMemory:
    category: str
    label: str
    description: str
    importance: int
    source_message: str
    related_concepts: list[str]

class ConversationAnalyzer:
    EXPLICIT_PATTERNS = [
        (r"remember:?\s*(.+)", "explicit"),
        (r"my name is\s+(\w+)", "identity"),
        (r"I (?:prefer|like|love)\s+(.+)", "preference"),
        (r"I (?:hate|dislike)\s+(.+)", "anti-preference"),
        (r"I am\s+(?:a\s+)?(.+)", "identity"),
        (r"note to self:?\s*(.+)", "note"),
    ]

    def extract_explicit(self, message: str) -> list[ExtractedMemory]:
        """Extract memories from explicit patterns."""
        memories = []
        for pattern, category in self.EXPLICIT_PATTERNS:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                memories.append(ExtractedMemory(
                    category=category,
                    label=match.group(1)[:50],
                    description=match.group(0),
                    importance=4,
                    source_message=message,
                    related_concepts=[],
                ))
        return memories

    async def extract_with_llm(
        self,
        conversation: list[dict],
        client: Any,
    ) -> list[ExtractedMemory]:
        """Use LLM to extract meaningful memories."""
        # Format conversation for prompt
        conv_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation
        )

        response = await client.messages.create(
            model="claude-3-haiku",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Analyze this conversation and extract information worth remembering.

Conversation:
{conv_text}

Extract memories as JSON array:
[{{
  "category": "identity|preference|fact|decision|goal",
  "label": "Short title (3-6 words)",
  "description": "Full detail",
  "importance": 1-5,
  "related_concepts": ["topic1", "topic2"]
}}]

Only extract genuinely important information. Return [] if nothing significant."""
            }]
        )

        # Parse response and return ExtractedMemory objects
        ...
```

---

## Summary

The Mind Graph v2 needs two main capabilities:

1. **Semantic Search:** Embeddings enable meaning-based recall. Use local `sentence-transformers` for privacy and speed. Embed rich node representations (label + description + context). Store embeddings in sidecar files.

2. **Conversation Integration:** Auto-create nodes from chats. Use LLM extraction for robust importance detection. Link new nodes via embedding similarity. Track provenance to source conversations.

Key challenges: duplicate detection, memory decay, privacy, context limits. Start simple, iterate based on real usage patterns.

The mind graph becomes a true associative memory - not just storage, but an extension of cognitive architecture.
