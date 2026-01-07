# Connection Engine Design Specification

**Version**: 1.0
**Date**: 2026-01-07
**Status**: Final Design - Ready for Implementation
**Constraint**: 8GB RAM Mac Mini (SQLite + JSON foundation)

---

## 1. Executive Summary

### What This System Does

The Connection Engine transforms captured thoughts into an **active thinking partner**. It's not a filing cabinet—it's a cognitive extension that:

- **Captures** thoughts from voice, text, screenshots, and documents with zero friction
- **Connects** memories automatically through semantic relationships and learned associations
- **Surfaces** contradictions, patterns, and insights at moments when you're ready to receive them
- **Evolves** with use—strengthening valuable connections, letting irrelevant ones fade

### The Core Experience

```
User: "What have I been thinking about authentication?"

System: "You've explored authentication 8 times over 3 weeks:

📁 Security Architecture (5 memories)
   → Started with 'JWT is overkill' (Dec 15)
   → Evolved to 'need stateless auth for scaling' (Dec 28)

⚠️ Potential Tension:
   'JWT is overkill' vs 'need stateless auth' — JWT IS stateless.
   These might be compatible; want to reconcile?

🔗 Connected to: API Performance theme (3 related memories)"
```

### Design Philosophy

**"The best technology disappears."**

The Connection Engine succeeds when users forget there's an engine at all. They experience a thoughtful friend who remembers everything—not a database they query.

---

## 2. Core Architecture

### 2.1 System Layers

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         CONNECTION ENGINE                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     USER INTERFACE LAYER                             │   │
│  │                                                                      │   │
│  │  capture(content, source) → instant confirmation                     │   │
│  │  query(question) → results + insight badge count                     │   │
│  │  get_insights() → pending insights (contradictions, patterns)        │   │
│  │  feedback(insight_id, rating) → trains the system                    │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                   SURFACING CONTROLLER                               │   │
│  │                                                                      │   │
│  │  UX Timing Rules:                                                    │   │
│  │  • Never interrupt typing or flow state                              │   │
│  │  • Contradictions: surface at next pause if high confidence          │   │
│  │  • Related work: badge only, never proactive                         │   │
│  │  • Patterns: session-end or weekly digest                            │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                    COGNITIVE ENGINE                                  │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐     │   │
│  │  │ ACTIVATION      │  │ CHUNK/SCHEMA    │  │ CONSOLIDATION    │     │   │
│  │  │ LAYER           │  │ MANAGER         │  │ ENGINE           │     │   │
│  │  │                 │  │                 │  │                  │     │   │
│  │  │ • Spreading     │  │ • Auto-chunking │  │ • Competitive    │     │   │
│  │  │   activation    │  │ • Schema fit    │  │   consolidation  │     │   │
│  │  │ • Context       │  │ • Inference     │  │ • Hebbian update │     │   │
│  │  │   priming       │  │                 │  │ • Smart pruning  │     │   │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────┘     │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                    PROCESSING PIPELINE                               │   │
│  │                                                                      │   │
│  │  Input → Capture → Embed → Connect → Analyze → Complete              │   │
│  │    │       │         │        │         │                            │   │
│  │  instant  <50ms    async    async     batch                          │   │
│  │                                                                      │   │
│  │  Graceful degradation: later stages can fail without blocking        │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                    PERSISTENCE LAYER                                 │   │
│  │                    (SQLite + sqlite-vec)                             │   │
│  │                                                                      │   │
│  │  memories | edges | chunks | insights | feedback | archive           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why This Architecture

**Rationale from all three lenses:**

| Component | Architecture Rationale | Cognitive Rationale | UX Rationale |
|-----------|----------------------|---------------------|--------------|
| SQLite + sqlite-vec | Single file, portable, ACID, sufficient for 100K+ memories | N/A | Easy backup, works offline, no server setup |
| Activation Layer | In-memory for speed | Mirrors spreading activation in human memory | Enables instant badge updates |
| Processing Pipeline | Async/staged for throughput | Matches encoding → consolidation → retrieval | Sub-second capture, background enrichment |
| Surfacing Controller | Decouples detection from display | Respects cognitive load | Never interrupts, timing is UX decision |
| Consolidation Engine | Batch processing efficiency | Mimics sleep consolidation | Weekly digest feels curated |

---

## 3. Data Model

### 3.1 Memory Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,  -- numpy array as bytes (384 dimensions)

    -- Source provenance
    source_type TEXT NOT NULL,  -- 'voice', 'note', 'screenshot', 'document', 'conversation'
    source_uri TEXT,            -- Original file path or identifier

    -- Temporal
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    occurred_at TIMESTAMP,      -- When the event happened (vs when captured)

    -- Cognitive metadata
    encoding_strength REAL DEFAULT 0.5,  -- Initial encoding quality (0-1)
    importance REAL DEFAULT 0.5,         -- User/system importance (0-1)
    confidence REAL DEFAULT 0.7,         -- How certain are we this is accurate?

    -- Emotional (optional, from voice/text analysis)
    emotional_valence REAL,     -- -1 (negative) to +1 (positive)
    emotional_arousal REAL,     -- 0 (calm) to 1 (excited)

    -- Processing status
    status TEXT DEFAULT 'pending',  -- 'pending', 'embedded', 'connected', 'complete'

    -- ACT-R activation tracking
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP
);

CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_source ON memories(source_type);
```

### 3.2 Edge Schema (Connections)

```sql
CREATE TABLE edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories(id),
    target_id TEXT NOT NULL REFERENCES memories(id),

    -- Edge type (simplified from cognitive complexity)
    edge_type TEXT NOT NULL,  -- 'semantic', 'temporal', 'contradiction', 'evolution'

    -- Strength (Hebbian learning)
    base_weight REAL DEFAULT 0.5,       -- Initial semantic similarity
    coactivation_count INTEGER DEFAULT 0,
    coactivation_recency TIMESTAMP,
    hebbian_boost REAL DEFAULT 0.0,

    -- Lifecycle
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_valid BOOLEAN DEFAULT TRUE,

    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);
```

### 3.3 Chunk Schema (Grouped Memories)

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,              -- LLM-generated name: "API Security"
    member_ids JSON NOT NULL,         -- List of memory IDs
    centroid_embedding BLOB,          -- Average embedding of members

    coherence_score REAL,             -- How tightly grouped (0-1)
    stability REAL DEFAULT 0.5,       -- How often this grouping persists

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 3.4 Insight Schema

```sql
CREATE TABLE insights (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,  -- 'contradiction', 'pattern', 'connection', 'evolution'
    confidence REAL NOT NULL,

    -- Payload varies by type
    payload JSON NOT NULL,

    -- Provenance (for "Explain" button)
    source_memories JSON NOT NULL,     -- Memory IDs that contributed
    contributing_edges JSON,           -- Edge IDs (if applicable)
    reasoning_trace JSON NOT NULL,     -- Human-readable explanation steps
    detection_method TEXT,             -- 'semantic_filter_llm', 'clustering', etc.
    detection_scores JSON,             -- {'semantic_similarity': 0.82, ...}

    -- Lifecycle
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    surfaced_at TIMESTAMP,             -- When shown to user (NULL if not yet)
    user_response TEXT,                -- 'useful', 'not_helpful', 'wrong', NULL
    dismissed_at TIMESTAMP
);

CREATE INDEX idx_insights_type ON insights(type);
CREATE INDEX idx_insights_surfaced ON insights(surfaced_at);
```

### 3.5 Connection Types

The system uses four primary edge types (simplified from cognitive complexity for UX clarity):

| Edge Type | Internal Meaning | User Display |
|-----------|------------------|--------------|
| `semantic` | High embedding similarity (>0.7) | "Related" |
| `temporal` | Occurred within 24 hours + some similarity | "Same timeframe" |
| `contradiction` | High similarity + incompatible assertions | "In tension" |
| `evolution` | Temporal sequence showing stance change | "Evolved into" |

---

## 4. Key Algorithms

### 4.1 Spreading Activation

**Purpose**: Find contextually relevant memories by spreading activation through the network, not just exact matching.

```python
def spread_activation(seed_node_ids: list[str], depth: int = 3, decay: float = 0.7) -> dict[str, float]:
    """
    Given seed nodes (from current context/query), spread activation outward.

    Cognitive basis: ACT-R spreading activation (Anderson, 1983)
    A_i = B_i + Σ_j (W_ji * A_j)

    Returns: {node_id: activation_level} for all nodes above threshold
    """
    ACTIVATION_THRESHOLD = 0.1

    # Initialize with seed nodes at full activation
    activated = {node_id: 1.0 for node_id in seed_node_ids}
    frontier = list(seed_node_ids)

    for hop in range(depth):
        next_frontier = []

        for node_id in frontier:
            current_activation = activated[node_id]

            # Get edges from this node
            edges = get_edges_from(node_id)

            for edge in edges:
                neighbor_id = edge.target_id

                # Calculate spreading activation
                edge_weight = effective_weight(edge)
                spread_activation = current_activation * decay * edge_weight

                if spread_activation > ACTIVATION_THRESHOLD:
                    # Max of existing and new (don't reduce activation)
                    activated[neighbor_id] = max(
                        activated.get(neighbor_id, 0),
                        spread_activation
                    )
                    next_frontier.append(neighbor_id)

        frontier = list(set(next_frontier))  # Dedupe

    return activated


def effective_weight(edge: Edge) -> float:
    """
    Calculate effective edge weight including Hebbian learning.

    Cognitive basis: "Neurons that fire together wire together"
    """
    # Recency decay on Hebbian boost
    if edge.coactivation_recency:
        hours_since = (now() - edge.coactivation_recency).total_seconds() / 3600
        recency_factor = math.exp(-hours_since / 168)  # 1-week half-life
    else:
        recency_factor = 0

    # Frequency boost (log scale to prevent runaway)
    frequency_boost = math.log(1 + edge.coactivation_count) * 0.05

    return edge.base_weight + (edge.hebbian_boost * recency_factor) + frequency_boost
```

### 4.2 Contradiction Detection

**Purpose**: Find memories that are semantically related but make incompatible assertions.

```python
async def detect_contradictions(memory: Memory) -> list[Contradiction]:
    """
    Two-stage contradiction detection:
    1. Semantic filter: Find memories about the same topic (vector similarity)
    2. Stance analysis: Check if assertions are incompatible (LLM analysis)

    Cognitive basis: High relatedness + incompatible assertions = cognitive dissonance
    Note: Contradictions are NOT semantic opposites—they're similar topics with
    conflicting stances.
    """

    # Stage 1: Semantic filter (fast)
    similar_memories = vector_search(memory.embedding, k=30)
    candidates = [
        m for m in similar_memories
        if cosine_similarity(memory.embedding, m.embedding) > 0.7
    ]

    if not candidates:
        return []

    # Stage 2: Stance analysis (LLM, expensive)
    contradictions = []

    for candidate in candidates:
        # Skip if too close in time (might be same thought stream)
        if abs(memory.created_at - candidate.created_at).days < 1:
            continue

        # LLM stance analysis
        analysis = await analyze_stance_compatibility(memory, candidate)

        if analysis.compatibility_score < 0.3:  # Incompatible assertions
            contradiction = Contradiction(
                memory_a=memory,
                memory_b=candidate,
                confidence=analysis.confidence,
                topic=analysis.shared_topic,
                nature=analysis.incompatibility_reason,
                reasoning_trace=[
                    f"Both discuss '{analysis.shared_topic}'",
                    f"Memory A asserts: {analysis.stance_a}",
                    f"Memory B asserts: {analysis.stance_b}",
                    f"Incompatibility: {analysis.incompatibility_reason}"
                ]
            )
            contradictions.append(contradiction)

    return contradictions


async def analyze_stance_compatibility(memory_a: Memory, memory_b: Memory) -> StanceAnalysis:
    """
    Use LLM to analyze whether two memories make compatible assertions.
    """
    prompt = f"""
    Analyze whether these two statements are compatible:

    Statement A (from {memory_a.created_at.date()}):
    "{memory_a.content}"

    Statement B (from {memory_b.created_at.date()}):
    "{memory_b.content}"

    Determine:
    1. What topic do both discuss? (If different topics, they cannot contradict)
    2. What stance does A take on this topic?
    3. What stance does B take on this topic?
    4. Are these stances compatible? (0.0 = incompatible, 1.0 = fully compatible)
    5. If incompatible, explain why.

    Note: Evolution of thinking (A was earlier belief, B is updated belief)
    is different from contradiction. Identify which this is.
    """

    return await llm.analyze(prompt, response_model=StanceAnalysis)
```

### 4.3 Connection Discovery Pipeline

**Purpose**: Automatically create edges between related memories.

```python
async def discover_connections(memory: Memory) -> list[Edge]:
    """
    Run on each new memory to create initial connections.

    Connection types discovered:
    1. Semantic: High embedding similarity
    2. Temporal: Close in time + moderate similarity
    3. Evolution: Same topic, temporal sequence, stance shift
    """
    edges = []

    # 1. Semantic connections
    similar = vector_search(memory.embedding, k=20)
    for m, similarity in similar:
        if similarity > 0.75:  # High threshold for auto-creation
            edges.append(Edge(
                source_id=memory.id,
                target_id=m.id,
                edge_type='semantic',
                base_weight=similarity
            ))

    # 2. Temporal connections
    recent = get_memories_in_window(
        start=memory.created_at - timedelta(hours=24),
        end=memory.created_at
    )
    for m in recent:
        similarity = cosine_similarity(memory.embedding, m.embedding)
        if similarity > 0.5:  # Lower threshold for temporal
            edges.append(Edge(
                source_id=memory.id,
                target_id=m.id,
                edge_type='temporal',
                base_weight=similarity * 0.8  # Slight discount
            ))

    # 3. Evolution detection (run if we found similar memories)
    if similar:
        older_similar = [m for m, s in similar if m.created_at < memory.created_at - timedelta(days=7)]
        for m in older_similar[:5]:  # Check top 5
            evolution = await detect_stance_evolution(m, memory)
            if evolution.is_evolution:
                edges.append(Edge(
                    source_id=m.id,  # Older → newer
                    target_id=memory.id,
                    edge_type='evolution',
                    base_weight=0.8
                ))

    return edges
```

### 4.4 Consolidation Cycle (Nightly)

**Purpose**: Simulate sleep consolidation—strengthen winners, weaken losers.

```python
async def nightly_consolidation():
    """
    Run during idle time (typically overnight).

    Cognitive basis: Complementary Learning Systems theory (McClelland et al., 1995)
    - Memories compete for consolidation
    - Winners get strengthened and integrated with schemas
    - Losers decay faster
    - Weak connections get pruned
    """

    # 1. Score all recent memories for consolidation priority
    recent_memories = get_memories_since(days=7)

    scored = []
    for memory in recent_memories:
        priority = compute_consolidation_priority(memory)
        scored.append((memory, priority))

    # Sort by priority
    scored.sort(key=lambda x: x[1], reverse=True)

    # 2. Top N get strengthened (winners)
    CONSOLIDATION_SLOTS = min(50, len(scored) // 2)
    winners = scored[:CONSOLIDATION_SLOTS]

    for memory, _ in winners:
        memory.encoding_strength *= 1.15  # 15% boost
        memory.encoding_strength = min(1.0, memory.encoding_strength)

        # Strengthen edges to/from this memory
        edges = get_edges_involving(memory.id)
        for edge in edges:
            edge.hebbian_boost += 0.05

    # 3. Bottom N get accelerated decay (losers)
    losers = scored[-CONSOLIDATION_SLOTS:]

    for memory, _ in losers:
        memory.encoding_strength *= 0.85  # 15% decay

    # 4. Prune weak edges
    all_edges = get_all_edges()
    for edge in all_edges:
        weight = effective_weight(edge)
        if weight < 0.2:  # Below threshold
            if edge.coactivation_count < 2:  # Never really used together
                soft_delete_edge(edge)  # Mark invalid, don't hard delete

    # 5. Move very weak memories to archive
    all_memories = get_all_memories()
    for memory in all_memories:
        if memory.encoding_strength < 0.1 and memory.access_count < 2:
            archive_memory(memory)

    await save_all_changes()


def compute_consolidation_priority(memory: Memory) -> float:
    """
    Score memory for consolidation competition.

    Higher scores = more likely to be consolidated.
    """
    factors = {
        # Emotional salience (strong emotions = better remembered)
        'emotional': (abs(memory.emotional_valence or 0) *
                     (memory.emotional_arousal or 0.5)),

        # Recent activation (used memories survive)
        'activation': memory.access_count / max(1, days_since(memory.created_at)),

        # Connection richness (well-connected memories are valuable)
        'connections': len(get_edges_involving(memory.id)) / 10,

        # User importance signals
        'importance': memory.importance,
    }

    # Weighted sum
    weights = {
        'emotional': 0.25,
        'activation': 0.35,
        'connections': 0.20,
        'importance': 0.20
    }

    return sum(factors[k] * weights[k] for k in factors)
```

### 4.5 Chunk Formation

**Purpose**: Group related memories into coherent themes for easier retrieval.

```python
async def detect_chunks():
    """
    Find groups of memories that naturally cluster together.

    Cognitive basis: Miller's chunking (1956) - grouping reduces cognitive load.
    """

    # Get all memory embeddings
    memories = get_all_memories()
    embeddings = np.array([m.embedding for m in memories])

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.3,  # Cosine distance threshold
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Group memories by cluster
    clusters = defaultdict(list)
    for memory, label in zip(memories, labels):
        clusters[label].append(memory)

    # Create/update chunks for clusters with 3+ members
    chunks = []
    for label, members in clusters.items():
        if len(members) >= 3:
            # Calculate centroid
            centroid = np.mean([m.embedding for m in members], axis=0)

            # Generate label via LLM
            chunk_label = await generate_chunk_label(members)

            # Calculate coherence (average pairwise similarity)
            coherence = calculate_coherence(members)

            chunk = Chunk(
                id=generate_id(),
                label=chunk_label,
                member_ids=[m.id for m in members],
                centroid_embedding=centroid,
                coherence_score=coherence
            )
            chunks.append(chunk)

    return chunks


async def generate_chunk_label(members: list[Memory]) -> str:
    """
    Use LLM to generate a concise label for a group of memories.
    """
    sample = random.sample(members, min(5, len(members)))
    contents = [m.content for m in sample]

    prompt = f"""
    These memories belong together thematically:

    {chr(10).join(f'- {c}' for c in contents)}

    Generate a concise label (2-4 words) that captures their common theme.
    Examples: "API Security", "Morning Routine", "Product Strategy"
    """

    return await llm.generate(prompt)
```

---

## 5. UX Integration

### 5.1 When to Surface Insights

The Surfacing Controller applies strict timing rules:

```python
class SurfacingController:
    """
    Cognitive engine produces insights continuously.
    This controller decides WHEN to show them.

    Principle: Better to show less than to annoy.
    """

    CONFIDENCE_GATES = {
        'contradiction': 0.8,   # High bar—accusing of contradiction is sensitive
        'pattern': 0.6,         # Medium bar
        'connection': 0.5,      # Lower bar for "might be related"
    }

    def should_surface_now(self, insight: Insight, user_state: UserState) -> str:
        """
        Returns: 'now', 'badge', 'batch', or 'suppress'
        """

        # Veto rules (never interrupt these states)
        if user_state.is_actively_typing:
            return 'badge'
        if user_state.is_in_flow_state:
            return 'batch'
        if user_state.seconds_since_last_interaction < 3:
            return 'badge'  # Wait for pause

        # Confidence gate
        threshold = self.CONFIDENCE_GATES.get(insight.type, 0.7)
        if insight.confidence < threshold:
            return 'suppress'

        # Type-specific rules
        if insight.type == 'contradiction':
            if insight.confidence >= 0.85:
                return 'now' if user_state.at_natural_pause else 'badge'
            return 'badge'

        if insight.type == 'pattern':
            return 'batch'  # Patterns go in weekly digest

        if insight.type == 'connection':
            return 'badge'  # Never proactive, always badge

        return 'badge'  # Default to non-intrusive
```

### 5.2 The Badge-First Display Model

```
┌─────────────────────────────────────────────────────────────┐
│  Your current work: Authentication system redesign           │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  [🔗 4 related] [⚠️ 1 tension]          ← Subtle badges    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Click "🔗 4 related" expands:**

```
┌─────────────────────────────────────────────────────────────┐
│  Related Memories                                    [−]     │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  📁 Authentication Discussions (4)                          │
│     • "JWT is overkill for our use case" - Dec 15           │
│     • "Need stateless auth for scaling" - Dec 28            │
│     • "Session management approaches" - Jan 2                │
│     • "OAuth consideration" - Jan 5                          │
│                                                              │
│  [ℹ️ Why these?]                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Click "ℹ️ Why these?" shows explanation:**

```
┌─────────────────────────────────────────────────────────────┐
│  Connection Explanation                                      │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  These memories are connected because:                       │
│                                                              │
│  • All discuss user authentication (semantic similarity: 82%)│
│  • Three occurred in the same week (temporal clustering)     │
│  • You've accessed these together before (learned pattern)   │
│                                                              │
│  Confidence: High                                            │
│                                                              │
│  [👍 Useful]  [👎 Not helpful]  [Report issue]              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Contradiction Surfacing

When surfaced, contradictions use this interface:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️ POTENTIAL TENSION                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  THEN (Dec 15):                    NOW (Jan 5):                     │
│  ┌─────────────────────────┐      ┌─────────────────────────┐      │
│  │ "SQLite is enough for   │      │ "We need PostgreSQL for │      │
│  │  our memory needs"      │      │  the memory system"     │      │
│  └─────────────────────────┘      └─────────────────────────┘      │
│                                                                     │
│  These appear to be about the same decision but reach different     │
│  conclusions. This could be:                                        │
│                                                                     │
│  ○ Thinking evolved (new information since December)                │
│  ○ Context-dependent (SQLite for MVP, Postgres for scale)           │
│  ○ Genuine inconsistency (needs resolution)                         │
│  ○ Different use cases (talking about different systems)            │
│                                                                     │
│  [Record Resolution]   [Need More Context]   [Dismiss]              │
│                                                                     │
│  [ℹ️ Why was this flagged?]                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Feedback Loops

**Implicit feedback (automatic, zero effort):**

| Signal | Meaning | System Response |
|--------|---------|-----------------|
| User clicked connection | Relevant | +1 access count, strengthen edge |
| User ignored badge | Low value | Decay factor applied |
| User followed up in session | High value | +3 access count |
| User copied content | Very high value | Mark as "gold", boost importance |

**Explicit feedback (one click):**

```python
async def handle_feedback(insight_id: str, feedback: str):
    insight = await load_insight(insight_id)

    if feedback == 'useful':
        # Strengthen contributing connections
        for edge_id in insight.contributing_edges:
            edge = await load_edge(edge_id)
            edge.hebbian_boost += 0.1

        # Boost source memory importance
        for memory_id in insight.source_memories:
            memory = await load_memory(memory_id)
            memory.importance = min(1.0, memory.importance + 0.05)

    elif feedback == 'not_helpful':
        # Weaken connections (don't break)
        for edge_id in insight.contributing_edges:
            edge = await load_edge(edge_id)
            edge.hebbian_boost -= 0.05

    elif feedback == 'wrong':
        # Mark connections as invalid for review
        for edge_id in insight.contributing_edges:
            await mark_edge_for_review(edge_id)

        # Log for pattern detection (are we making systematic errors?)
        await log_wrong_insight(insight)

    insight.user_response = feedback
    await save_insight(insight)
```

### 5.5 Weekly Digest

Generated from nightly consolidation, surfaced once per week:

```
┌─────────────────────────────────────────────────────────────────────┐
│  📬 This Week in Your Thinking (Jan 1-7)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 DOMINANT THEME: System Architecture                             │
│     You returned to architecture decisions 8 times.                 │
│     First: "Need to design memory system" (Mon)                     │
│     Latest: "SQLite vs Postgres decision" (Sat)                     │
│     Trajectory: Increasing complexity concerns                      │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  ⚠️ 2 TENSIONS DETECTED                                            │
│     1. SQLite vs PostgreSQL (Dec 15 ↔ Jan 5)                       │
│     2. "Start simple" vs "Plan for scale" (multiple mentions)       │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  🔗 NEW CONNECTIONS                                                 │
│     • "API Performance" now linked to "Memory System" (via caching) │
│     • "User Trust" theme emerged (4 related memories)               │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  💡 INSIGHT                                                         │
│     Your concerns about scaling appear earlier in topics that       │
│     later become "critical priority." Pattern: worry → work.        │
│                                                                     │
│  [View Full Details]  [Adjust Preferences]  [Dismiss Until Next Week]│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Phases

### Phase 1: MVP — Trust Foundation (2-3 weeks)

**Goal**: Reliable capture and basic retrieval that earns trust.

**Deliverables**:

1. **SQLite database with sqlite-vec extension**
   - Memory table with embedding column
   - Basic edge table
   - Simple insight table

2. **Capture pipeline**
   - Text input → immediate store
   - Voice input → Whisper transcription → store
   - Screenshot → OCR → store
   - Status: pending → embedded → complete

3. **Embedding service**
   - all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
   - Async embedding after capture

4. **Basic vector search**
   - sqlite-vec k-NN queries
   - Simple relevance ranking

5. **Minimal UI**
   - Capture widget (text/voice/image)
   - Search box
   - Results list with source attribution

**Success criteria**:
- Capture-to-confirmed: <2 seconds
- Search returns relevant results
- No data loss

### Phase 2: V1 — Connection Intelligence (3-4 weeks)

**Goal**: Automatic connections and contradiction detection.

**Deliverables**:

1. **Auto-connection pipeline**
   - Semantic edges (>0.75 similarity)
   - Temporal edges (within 24h + similarity)
   - Runs on each new memory

2. **Spreading activation retrieval**
   - Context-aware search results
   - "Related via X" explanations

3. **Contradiction detection**
   - Semantic filter + LLM verification
   - Confidence scoring
   - Storage in insights table

4. **Badge-based surfacing**
   - "🔗 N related" badge
   - "⚠️ N tensions" badge
   - Click to expand
   - Basic "Why this connection?" explanation

5. **Feedback collection**
   - Implicit: clicks, ignores, copies
   - Explicit: useful/not helpful/wrong buttons
   - Feedback updates edge weights

**Success criteria**:
- Connections feel relevant (>70% useful feedback)
- Contradictions are accurate (>80% confidence = true positive)
- Zero false confident contradictions

### Phase 3: V2 — Thinking Partner (4-6 weeks)

**Goal**: Proactive insights and evolved cognitive model.

**Deliverables**:

1. **Chunk formation**
   - Nightly clustering
   - LLM-generated labels
   - Chunk-based retrieval ("📁 API Security (5)")

2. **Consolidation cycle**
   - Nightly competitive consolidation
   - Hebbian edge updates
   - Smart pruning (archive, don't delete)

3. **Weekly digest**
   - Theme detection
   - Contradiction summary
   - Thinking evolution narrative

4. **Advanced surfacing**
   - Cognitive load estimation
   - Context-appropriate timing
   - Progressive feature unlock (based on memory count)

5. **Enhanced explanations**
   - Cognitive framing ("These are in the same mental neighborhood")
   - Confidence indicators in words, not numbers
   - Full provenance chain

**Success criteria**:
- Weekly digest completion rate >30%
- "Aha moment" reports >2/week
- Trust survey score >7/10

---

## 7. Success Metrics

### User Experience Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Capture-to-confirmed time | <2 seconds | Time from submit to UI confirmation |
| Connection click-through rate | >20% | Clicks on badges / badges shown |
| Insight dismiss rate | <50% | Dismissed / surfaced (high = annoying) |
| Daily active usage | >1 session | Sessions per day per user |
| Weekly digest engagement | >30% | Opened and scrolled / sent |
| "Useful" feedback rate | >60% | Useful / (useful + not helpful) |
| Trust survey score | >7/10 | Quarterly survey |

### System Health Metrics

| Metric | Target | Danger Zone |
|--------|--------|-------------|
| Graph density | 0.01 - 0.05 | >0.1 (hairball) or <0.005 (disconnected) |
| Avg edges per memory | 3-8 | >15 (over-connected) or <2 (isolated) |
| Contradiction accuracy | >80% | <60% (false positives destroy trust) |
| Embedding latency | <500ms | >2s (user waits too long) |
| Nightly consolidation time | <30 min | >2hr (won't finish before morning) |
| Archive rate | 5-15% monthly | >30% (over-forgetting) or <2% (never pruning) |

### Cognitive Model Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Chunk stability | Do groupings persist? | >70% month-over-month |
| Edge Hebbian correlation | Do co-activated edges strengthen? | r > 0.5 |
| Retrieval improvement | Does spreading activation help? | +15% relevance vs pure vector |
| Consolidation signal | Do high-priority memories persist? | Top 20% survive 3x longer |

---

## 8. Open Questions for Future Research

### 8.1 Schema Learning

**Question**: How does the system learn user schemas? Pure clustering, or guided by user feedback?

**Considerations**:
- Clustering can find structure but may miss conceptual boundaries
- User feedback is gold but requires effort
- Hybrid: cluster first, then ask user to name/merge/split

**Research needed**: A/B test pure clustering vs user-guided schema formation

### 8.2 Interference vs Completeness

**Question**: How do we balance showing all relevant memories vs avoiding retrieval competition noise?

**Considerations**:
- Cognitive science says similar memories interfere
- But users might want completeness ("show me everything about X")
- Current design: chunk-based display reduces noise while preserving access

**Research needed**: User studies on optimal retrieval set size

### 8.3 Cognitive Style Adaptation

**Question**: Should the system adapt to different cognitive styles?

**Considerations**:
- Some users think verbally, others visually
- Some prefer structured output, others prefer narrative
- Could detect from input patterns and adapt presentation

**Research needed**: Identify markers of cognitive style from interaction patterns

### 8.4 Multi-Agent Memory

**Question**: If this becomes shared across AI agents, whose cognitive model applies?

**Considerations**:
- Current design assumes single human user
- Multi-agent would need namespacing or merged models
- Contradictions between agents are different from within-user contradictions

**Deferred**: Design for single user first, extend to multi-agent in future version

### 8.5 Optimal Embedding Model

**Question**: What's the right embedding model for personal memory?

**Current choice**: all-MiniLM-L6-v2 (fast, good quality, 384d)

**Alternatives to research**:
- Domain-specific fine-tuning on user's vocabulary
- Multimodal (CLIP) for better screenshot handling
- Larger models (768d+) for better semantic capture

**Research needed**: Benchmark on personal memory retrieval task

---

## 9. Technical Notes

### 9.1 8GB RAM Constraint

**Memory budget allocation**:
- SQLite: ~100MB (handles 100K+ memories)
- sqlite-vec index: ~150MB (384d * 100K * 4 bytes)
- Embedding model: ~100MB (MiniLM)
- Activation cache: ~50MB (session state)
- Application overhead: ~100MB
- **Total**: ~500MB (well within 8GB)

**Large operations**:
- Nightly clustering: Stream memories, don't load all into RAM
- Embedding: Batch in chunks of 100

### 9.2 Offline-First

The system works fully offline:
- SQLite is local
- Embedding model runs locally
- LLM calls (for contradiction analysis) can queue for when online
- Sync to cloud is optional and user-controlled

### 9.3 Privacy

All data stays local by default:
- No telemetry without explicit opt-in
- No cloud sync without explicit setup
- Export in standard formats (JSON, markdown)

---

## 10. Appendix: Key Cognitive Science References

1. **Anderson, J. R. (1983)**: ACT-R spreading activation — foundation for our retrieval model
2. **Miller, G. A. (1956)**: Chunking — why we group memories
3. **Ebbinghaus, H. (1885)**: Forgetting curve — why memories decay
4. **McClelland et al. (1995)**: Complementary Learning Systems — why we consolidate
5. **Bowden & Jung-Beeman (2003)**: Insight — why incubation matters
6. **Roediger & Karpicke (2006)**: Testing effect — why retrieval strengthens memory
7. **McGaugh, J. L. (2004)**: Emotional memory modulation — why emotional salience matters

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-07 | Synthesis Agent | Initial design spec from Round 1-2 brainstorms |

---

*"The art of memory is the art of attention."* — Samuel Johnson

*"The best technology disappears."* — Design Philosophy

**This document is the deliverable. Implementation begins now.**
