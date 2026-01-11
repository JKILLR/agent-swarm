# Connection Engine Architecture Brainstorm
## Round 1 - Architect Lens

**Date**: 2026-01-07
**Focus**: Divergent architectural thinking for a memory system that thinks with you

---

## The Vision

A **Connection Engine** isn't just retrieval - it's a cognitive partner that:
- Sees patterns you haven't noticed yet
- Surfaces contradictions that sharpen your thinking
- Builds emergent understanding from scattered inputs
- Turns fragments into insights

The goal: **From storage to synthesis.**

---

## Part 1: Graph vs Vector vs Hybrid - The Structure Wars

### Approach A: Pure Graph (Neo4j/Neptune Pattern)

```
┌─────────────────────────────────────────────────────────┐
│                    GRAPH-FIRST                          │
│                                                         │
│  ┌─────┐   SUPPORTS   ┌─────────┐   CONTRADICTS        │
│  │Idea │─────────────▶│Argument │◀──────────────┐      │
│  └─────┘              └─────────┘               │      │
│     │                      │                    │      │
│     │ EVOLVED_FROM         │ LEADS_TO      ┌───┴───┐  │
│     ▼                      ▼               │Counter│  │
│  ┌─────┐              ┌─────────┐          │ Point │  │
│  │Root │              │Decision │          └───────┘  │
│  │Idea │              └─────────┘                     │
│  └─────┘                                              │
└─────────────────────────────────────────────────────────┘
```

**Connection Discovery**: Graph traversal, pathfinding, community detection
- Cypher queries: `MATCH (a)-[:RELATES_TO*1..3]-(b) WHERE a.id = $id`
- PageRank for importance scoring
- Louvain clustering for theme detection

**Strengths**:
- Explicit relationship types (CONTRADICTS, SUPPORTS, EVOLVED_FROM)
- Multi-hop reasoning is native
- "Why are these connected?" is answerable
- Provenance chains are first-class

**Weaknesses**:
- Relationships must be pre-defined or LLM-extracted
- Cold start: empty graph knows nothing
- Can't find connections that weren't explicitly created
- Rigid schema fights emergent patterns

**Wild Variation**: **Hypergraph**
- Edges connect N nodes, not just 2
- "This idea emerged from the intersection of A, B, and C"
- Models collaborative insight more naturally

---

### Approach B: Pure Vector (Embedding Space Pattern)

```
┌─────────────────────────────────────────────────────────┐
│              EMBEDDING SPACE                             │
│                                                         │
│                    *  idea_3                            │
│           idea_1 *                                      │
│                        * idea_4                         │
│      * idea_2                                           │
│                             ┌─────────────────────────┐ │
│           * query ──────────▶ k-NN Search            │ │
│                             │ Returns: idea_3, idea_4│ │
│    * idea_5                 └─────────────────────────┘ │
│                  * idea_6                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Connection Discovery**: Semantic similarity in high-dimensional space
- ANN search (HNSW, IVF)
- Clustering (K-means, DBSCAN)
- Dimensionality reduction for visualization (UMAP)

**Strengths**:
- Discovers implicit connections (semantic neighbors)
- Works from day one (pre-trained embeddings)
- No schema required - everything embeddable is connected
- Scales beautifully

**Weaknesses**:
- "Why are these connected?" → "Because math" (opaque)
- Can't represent relationship types
- Temporal relationships lost (unless encoded)
- Contradictions look like similarities (both about same topic)

**Wild Variation**: **Multi-Vector Representations**
- Each fact has 3 embeddings: content, context, contradiction-space
- Search in different spaces for different purposes
- "What's similar?" vs "What challenges this?"

---

### Approach C: Hybrid Architecture (The Obvious Answer)

```
┌───────────────────────────────────────────────────────────────┐
│                    HYBRID: GRAPH + VECTORS                    │
│                                                               │
│  ┌─────────────┐           ┌─────────────────────────────────┐│
│  │ GRAPH LAYER │           │        VECTOR LAYER              ││
│  │             │           │                                  ││
│  │  Explicit   │◀─────────▶│  Implicit Semantic Space        ││
│  │  Relations  │  sync     │                                  ││
│  │             │           │  * * *  *   *                    ││
│  │  A──▶B──▶C  │           │    *  *    *                     ││
│  │      │      │           │  *     *                         ││
│  │      ▼      │           │                                  ││
│  │      D      │           │                                  ││
│  └─────────────┘           └─────────────────────────────────┘│
│                                                               │
│  Query Flow:                                                  │
│  1. Vector search finds semantically similar                  │
│  2. Graph traversal finds explicitly connected                │
│  3. Merge, dedupe, rank by combined score                     │
└───────────────────────────────────────────────────────────────┘
```

**But here's the interesting question**: How do they interact?

### Hybrid Interaction Patterns

#### Pattern H1: Vector-First, Graph-Augment
```
query → vector_search(k=20) → for each result: graph_neighbors(depth=2) → rerank
```
Use vectors to find the ballpark, graphs to explore the neighborhood.

#### Pattern H2: Graph-First, Vector-Expand
```
query → exact_match_or_recent() → graph_traverse() → vector_expand(similar_to_each) → merge
```
Start from known context, expand semantically.

#### Pattern H3: Parallel Merge
```
query ─┬─▶ vector_search() ───┐
       │                      ├─▶ fusion_rank() → results
       └─▶ graph_search() ────┘
```
Race both, combine results with learned weights.

#### Pattern H4: Vector-Inferred Graph (Most Interesting)
```
periodic_job:
  for each node:
    neighbors = vector_search(node.embedding, k=10)
    for n in neighbors:
      if similarity > 0.8 and no_edge_exists(node, n):
        create_edge(node, n, type=SEMANTIC_SIMILARITY, strength=similarity)
```
**The graph grows itself from vector space.** Emergent structure.

---

### Approach D: The Unconventional - Attention-Based Memory

```
┌─────────────────────────────────────────────────────────┐
│            MEMORY AS TRANSFORMER ATTENTION               │
│                                                         │
│  Query: "What contradicts my belief about X?"           │
│                      │                                  │
│                      ▼                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Cross-Attention Layer                     │  │
│  │                                                   │  │
│  │  Q: query_embedding                               │  │
│  │  K: all_memory_embeddings                         │  │
│  │  V: all_memory_content                            │  │
│  │                                                   │  │
│  │  attention_weights = softmax(QK^T / √d)           │  │
│  │  output = weighted_sum(V, attention_weights)      │  │
│  └──────────────────────────────────────────────────┘  │
│                      │                                  │
│                      ▼                                  │
│  Most attended memories surface naturally               │
└─────────────────────────────────────────────────────────┘
```

**The Twist**: Train a small transformer to do memory retrieval
- Input: query + memory bank
- Output: relevance-weighted synthesis
- Learns complex patterns like "contradiction" implicitly

**Inspired by**: Google Titans, MemGPT internal architecture

**Wildest version**: The memory system itself is a fine-tuned LLM
- No explicit storage at all
- "Remembers" by having been trained on your data
- Retrieval = generation with the right prompt

---

## Part 2: Connection Discovery Mechanisms

### How do connections get found?

#### Mechanism 1: Semantic Neighbors (Passive)
```python
def find_connections(node):
    # Already connected in vector space
    return vector_search(node.embedding, k=10)
```
**Pro**: Zero effort, always works
**Con**: Shallow, obvious connections only

#### Mechanism 2: Graph Traversal (Structural)
```python
def find_connections(node):
    # Follow explicit edges
    direct = graph.neighbors(node, depth=1)
    indirect = graph.neighbors(node, depth=2)
    return direct + indirect
```
**Pro**: Explainable paths
**Con**: Only finds what was explicitly linked

#### Mechanism 3: LLM Discovery (Active)
```python
async def discover_connections(node, all_nodes_sample):
    prompt = f"""
    Given this idea: {node.content}

    And these other ideas:
    {format_nodes(all_nodes_sample)}

    Find:
    1. Ideas that SUPPORT this (provide evidence)
    2. Ideas that CONTRADICT this (create tension)
    3. Ideas that EXTEND this (build upon)
    4. Ideas from DIFFERENT DOMAINS that surprisingly connect

    Explain each connection.
    """
    return await llm.analyze(prompt)
```
**Pro**: Deep, novel connections; understands nuance
**Con**: Expensive, slow, requires sampling strategy

#### Mechanism 4: Pattern Mining (Batch)
```python
def mine_patterns():
    # Run periodically on full corpus

    # Co-occurrence: What's mentioned together?
    cooccurrence_matrix = build_cooccurrence(all_memories)

    # Temporal: What follows what?
    sequences = extract_temporal_sequences(timestamped_memories)

    # Clustering: What naturally groups?
    clusters = cluster_embeddings(all_embeddings)

    # Association rules: If X then often Y
    rules = apriori(memory_tags)

    return synthesize_patterns(cooccurrence, sequences, clusters, rules)
```
**Pro**: Finds structural patterns humans miss
**Con**: Requires sufficient data, batch processing

#### Mechanism 5: Contrastive Discovery (Adversarial)
```python
def find_contradictions(node):
    # Generate the OPPOSITE embedding
    anti_embedding = negate_embedding(node.embedding)

    # Or: ask LLM to generate contradicting statement
    contradiction = await llm.generate(f"What would contradict: {node.content}")
    anti_embedding = embed(contradiction)

    # Search for things close to the anti-embedding
    contradicting_nodes = vector_search(anti_embedding, k=5)
    return contradicting_nodes
```
**Pro**: Explicitly finds tension
**Con**: "Negation in embedding space" is tricky

#### Mechanism 6: Random Walks + Serendipity
```python
def serendipitous_discovery():
    # Start from random node
    current = random.choice(all_nodes)
    path = [current]

    for _ in range(5):
        # Mix of semantic + random jumps
        if random.random() < 0.7:
            current = random_semantic_neighbor(current)
        else:
            current = random.choice(all_nodes)  # wild jump
        path.append(current)

    # Ask: Is there a surprising connection in this path?
    return llm.analyze_path(path)
```
**Pro**: Finds unexpected bridges
**Con**: Mostly noise, gems are rare

---

## Part 3: Contradiction & Tension Detection

This is the hard problem. Similar topics ≠ contradictions.

### Strategy T1: Explicit Contradiction Markers

```python
CONTRADICTION_SIGNALS = [
    "but actually",
    "I changed my mind",
    "this contradicts",
    "on the other hand",
    "I used to think... now I think",
    "the opposite is true",
]
```
Store metadata: `is_contradiction_of: [node_ids]`

### Strategy T2: Stance Detection
```python
async def detect_stance(memory1, memory2, topic):
    """
    Are these memories pro/anti/neutral on the same topic?
    """
    prompt = f"""
    Topic: {topic}

    Statement A: {memory1.content}
    Statement B: {memory2.content}

    Analysis:
    1. What is A's stance on the topic? (positive/negative/neutral)
    2. What is B's stance on the topic? (positive/negative/neutral)
    3. Do they contradict? (yes/no/partially)
    4. Explain the tension if any.
    """
    return await llm.analyze(prompt)
```

### Strategy T3: Temporal Contradiction
```python
def find_temporal_contradictions(topic_memories):
    """
    Did stance change over time?
    """
    sorted_by_time = sorted(topic_memories, key=lambda m: m.timestamp)

    contradictions = []
    for i in range(len(sorted_by_time) - 1):
        earlier = sorted_by_time[i]
        later = sorted_by_time[i + 1]

        if stance_changed(earlier, later):
            contradictions.append({
                "earlier": earlier,
                "later": later,
                "type": "evolution" if gradual else "reversal"
            })

    return contradictions
```

### Strategy T4: Belief Consistency Graph
```
┌────────────────────────────────────────────────────────┐
│           BELIEF CONSISTENCY GRAPH                      │
│                                                         │
│   "AI is beneficial"                                    │
│         │                                               │
│         │ IMPLIES                                       │
│         ▼                                               │
│   "We should invest in AI"                             │
│         │                                               │
│         │ CONFLICTS_WITH                                │
│         ▼                                               │
│   "AI jobs will destroy livelihoods"  ← TENSION!       │
│         │                                               │
│         │ IMPLIES                                       │
│         ▼                                               │
│   "We should slow AI development"                       │
│                                                         │
│   Path: A implies B conflicts_with C implies D          │
│   A and D are in logical tension                        │
└────────────────────────────────────────────────────────┘
```

Maintain a graph of logical implications. Contradictions emerge from cycles.

### Strategy T5: Devil's Advocate Agent
```python
async def devils_advocate(belief):
    """
    Actively search for counterarguments.
    """
    # 1. Generate strongest counter-position
    counter = await llm.generate(f"Argue against: {belief}")

    # 2. Search memory for evidence supporting counter
    supporting_evidence = vector_search(embed(counter), k=10)

    # 3. Synthesize
    return {
        "original": belief,
        "counter_argument": counter,
        "evidence_in_memory": supporting_evidence,
        "tension_score": calculate_tension(belief, supporting_evidence)
    }
```

---

## Part 4: Pattern Building - Recurring Themes

How does the system recognize that it's seen this before?

### Pattern P1: Frequency Counting
```python
def extract_themes():
    # Simple but effective
    all_tags = flatten([m.tags for m in memories])
    theme_counts = Counter(all_tags)

    # Also: n-gram analysis on content
    content_ngrams = extract_ngrams(all_content, n=3)
    recurring_phrases = filter_by_frequency(content_ngrams, min_count=3)

    return merge(theme_counts, recurring_phrases)
```

### Pattern P2: Clustering Evolution
```python
def track_theme_evolution():
    """
    How have clusters changed over time?
    """
    for time_window in weekly_windows:
        memories_in_window = filter_by_time(memories, time_window)
        clusters = cluster(memories_in_window)

        # Compare to previous window
        new_themes = clusters - previous_clusters
        dying_themes = previous_clusters - clusters
        growing_themes = [c for c in clusters if size(c) > previous_size(c)]

        yield {
            "window": time_window,
            "emerging": new_themes,
            "fading": dying_themes,
            "growing": growing_themes
        }
```

### Pattern P3: Semantic Centroid Tracking
```python
class Theme:
    centroid: np.ndarray  # Average embedding of members
    members: list[Memory]
    birth_time: datetime

    def update(self, new_memory):
        self.members.append(new_memory)
        # Moving average centroid
        self.centroid = 0.9 * self.centroid + 0.1 * new_memory.embedding

    def drift_from_origin(self) -> float:
        """How much has this theme evolved from where it started?"""
        return cosine_distance(self.centroid, self.original_centroid)
```

### Pattern P4: Narrative Arc Detection
```python
async def detect_narrative_arcs():
    """
    Find story-like patterns in memories.

    Arcs:
    - Problem → Exploration → Solution
    - Hypothesis → Test → Conclusion
    - Question → Search → Answer
    - Belief → Challenge → Evolution
    """
    arc_templates = load_narrative_templates()

    for memory_sequence in temporal_sequences:
        for template in arc_templates:
            match_score = match_to_template(memory_sequence, template)
            if match_score > 0.7:
                yield {
                    "arc_type": template.name,
                    "memories": memory_sequence,
                    "score": match_score
                }
```

### Pattern P5: Concept Hierarchies (Auto-Generated)
```python
def build_concept_hierarchy():
    """
    Induce IS-A / PART-OF relationships from content.
    """
    # 1. Extract all noun phrases
    concepts = extract_noun_phrases(all_content)

    # 2. Ask LLM to organize into hierarchy
    hierarchy = await llm.organize(f"""
    Organize these concepts into a taxonomy:
    {concepts}

    Format:
    - Top-level category
      - Subcategory
        - Specific concept
    """)

    # 3. Convert to graph edges
    return hierarchy_to_graph(hierarchy)
```

---

## Part 5: Multi-Source Ingestion Architecture

### The Challenge
- Voice: Messy, conversational, fragmented
- Notes: Structured, intentional, condensed
- Screenshots: Visual, often without context
- Docs: Long-form, formal, comprehensive

### Unified Memory Schema

```python
@dataclass
class Memory:
    id: str
    content: str  # Always text (transcribed, OCR'd, extracted)
    embedding: np.ndarray

    # Provenance
    source_type: Literal["voice", "note", "screenshot", "doc", "conversation"]
    source_uri: str  # Original file/URL
    raw_content: Optional[bytes]  # Original media

    # Temporal
    created_at: datetime
    occurred_at: Optional[datetime]  # When the event happened (vs captured)

    # Cognitive metadata
    importance: float  # 0-1, Ebbinghaus decay applies
    activation: float  # ACT-R activation level
    confidence: float  # Bayesian confidence
    emotional_valence: float  # -1 to 1
    emotional_arousal: float  # 0 to 1

    # Structural
    tags: list[str]
    entities: list[Entity]  # Extracted named entities
    relations: list[Relation]  # Extracted relationships

    # Connections
    explicit_links: list[str]  # Manual links to other memories
    auto_links: list[tuple[str, float]]  # (memory_id, similarity_score)
```

### Source-Specific Pipelines

```
┌──────────────────────────────────────────────────────────────────┐
│                    MULTI-SOURCE INGESTION                         │
│                                                                   │
│  ┌─────────┐    ┌──────────────────────────────────────────────┐ │
│  │  VOICE  │───▶│ Whisper → Diarization → Chunk → Clean       │ │
│  └─────────┘    │ → Entity Extraction → Embed → Store          │ │
│                 └──────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────┐    ┌──────────────────────────────────────────────┐ │
│  │  NOTES  │───▶│ Parse (MD/Notion/etc) → Chunk by section    │ │
│  └─────────┘    │ → Preserve structure → Embed → Store         │ │
│                 └──────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────┐    ┌──────────────────────────────────────────────┐ │
│  │SCREENSHOT│──▶│ OCR + CLIP Vision → Caption → Context       │ │
│  └─────────┘    │ → Merge text+visual embed → Store            │ │
│                 └──────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────┐    ┌──────────────────────────────────────────────┐ │
│  │  DOCS   │───▶│ Parse → RAPTOR-style hierarchy → Chunk      │ │
│  └─────────┘    │ → Multi-level summaries → Embed each → Store │ │
│                 └──────────────────────────────────────────────┘ │
│                                                                   │
│                         │                                         │
│                         ▼                                         │
│                 ┌───────────────┐                                │
│                 │ UNIFIED STORE │                                │
│                 │               │                                │
│                 │  Graph + Vec  │                                │
│                 └───────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### Source Fusion Strategies

#### Strategy F1: Late Fusion
Each source → independent pipeline → merge at query time
```python
def query(q):
    voice_results = search_voice_index(q)
    note_results = search_note_index(q)
    screenshot_results = search_screenshot_index(q)
    doc_results = search_doc_index(q)

    return rerank(merge(voice_results, note_results, screenshot_results, doc_results))
```

#### Strategy F2: Early Fusion (Unified Embedding)
All sources → same embedding space
```python
def embed_memory(memory):
    if memory.source_type == "screenshot":
        # Use multimodal embedding
        return clip_embed(memory.image, memory.extracted_text)
    else:
        return text_embed(memory.content)
```

#### Strategy F3: Cross-Modal Linking
After ingestion, find connections between modalities
```python
def cross_modal_link():
    for voice_memory in voice_memories:
        # Find notes about same topic
        related_notes = search(voice_memory.content, source_type="note")

        for note in related_notes:
            create_link(voice_memory, note, type="cross_modal_reference")
```

---

## Part 6: Real-Time vs Batch Processing

### The Spectrum

```
REAL-TIME ◀───────────────────────────────────────▶ BATCH

Instant:                                          Nightly:
- Embedding new memory                            - Full re-clustering
- K-NN for immediate retrieval                    - Contradiction discovery
- Decay/activation updates                        - Pattern mining
- Explicit link creation                          - Hierarchy induction
                                                  - Embedding re-computation
                                                  - Graph optimization
```

### Hybrid Architecture

```python
class ConnectionEngine:
    # REAL-TIME: Sub-second operations
    async def on_new_memory(self, memory: Memory):
        # 1. Embed immediately
        memory.embedding = await self.embed(memory.content)

        # 2. Find immediate neighbors
        neighbors = await self.vector_search(memory.embedding, k=10)

        # 3. Create high-confidence auto-links
        for n, score in neighbors:
            if score > 0.85:
                await self.create_link(memory, n, type="semantic", score=score)

        # 4. Update activation levels of touched memories
        for n, _ in neighbors:
            await self.boost_activation(n)

        # 5. Store
        await self.store(memory)

        # 6. Queue for deeper analysis
        await self.deep_analysis_queue.put(memory)

    # BATCH: Minutes to hours
    async def nightly_consolidation(self):
        # 1. Re-cluster all memories
        clusters = await self.full_clustering()

        # 2. Mine patterns
        patterns = await self.mine_patterns()

        # 3. Discover contradictions
        contradictions = await self.find_all_contradictions()

        # 4. Build/update concept hierarchies
        hierarchy = await self.induce_hierarchy()

        # 5. Apply decay to all memories
        await self.apply_ebbinghaus_decay()

        # 6. Generate insights report
        insights = await self.synthesize_insights(clusters, patterns, contradictions)

        return insights
```

### Stream Processing Option
```python
# Kafka/Redis Streams for continuous processing
async def connection_stream_processor():
    async for memory in memory_stream:
        # Micro-batch: Process every 100ms
        await asyncio.gather(
            update_activation_model(memory),
            trigger_decay_check(memory),
            check_contradiction_candidates(memory),
            update_theme_centroids(memory),
        )
```

---

## Part 7: Wild Architectures

### W1: Memory as Dialogue
```
The memory system is itself an agent you can talk to:

User: "What have I been obsessing about lately?"
Memory Agent: "Based on the last 2 weeks, you've returned to
             'AI alignment' 12 times, each time with more concern.
             Want me to summarize the evolution?"

User: "Show me where I've contradicted myself"
Memory Agent: "On Tuesday you said X. On Thursday you said Y.
             These seem in tension. Want to resolve this?"
```

### W2: Metabolic Memory
```python
class MetabolicMemory:
    """
    Memories have energy budgets.
    Active memories consume energy.
    Energy comes from attention.
    Starved memories fade.
    """

    def tick(self):
        for memory in self.memories:
            # Consume energy (decay)
            memory.energy -= BASE_METABOLIC_COST

            # Energy from connections (popular memories stay alive)
            memory.energy += LINK_BONUS * len(memory.incoming_links)

            # Energy from access (used memories stay alive)
            if memory.recently_accessed:
                memory.energy += ACCESS_BONUS

            # Death
            if memory.energy <= 0:
                self.archive(memory)  # Don't delete, archive
```

### W3: Dream Mode (Consolidation)
```python
async def dream():
    """
    Run during idle time. Simulates sleep consolidation.
    """
    # 1. Replay recent memories
    recent = get_recent_memories(days=1)

    # 2. Find surprising connections to old memories
    for memory in recent:
        old_connections = search_archive(memory.embedding, min_age_days=30)
        for old in old_connections:
            surprise = measure_surprise(memory, old)
            if surprise > THRESHOLD:
                # "Dream insight" - unexpected connection
                await create_insight(memory, old, type="dream_connection")

    # 3. Consolidate fragmented memories into coherent narratives
    fragments = find_incomplete_narratives()
    for fragment_set in fragments:
        narrative = await llm.synthesize_narrative(fragment_set)
        await store_as_consolidated_memory(narrative)
```

### W4: Adversarial Memory
```python
class AdversarialMemory:
    """
    Two memory systems: one that affirms, one that challenges.
    """

    def query(self, q):
        # System 1: What supports this query?
        supporting = self.supporter.search(q)

        # System 2: What challenges this query?
        challenging = self.challenger.search(q)

        return {
            "evidence_for": supporting,
            "evidence_against": challenging,
            "synthesis": self.synthesize(supporting, challenging)
        }
```

### W5: Holographic Memory
```
Every piece contains the whole (at low resolution).

Each memory stores:
- Full detail of itself
- Compressed representation of nearby cluster
- Ultra-compressed representation of full memory space

Query at different "zoom levels":
- High res: Exact match
- Medium res: Cluster summary
- Low res: Full memory essence
```

### W6: Temporal Braiding
```
Don't store linear time. Store braided time.

Past memories can "retroactively" link to future memories
when new information recontextualizes old.

"What I thought in January makes more sense given what
happened in March" → bidirectional temporal link
```

---

## Part 8: Connection Strength & Decay

### The Math Behind Connections

#### Ebbinghaus Decay (Memory Strength)
```python
def memory_strength(memory, now):
    """
    Retention = e^(-t/S) where S is stability
    Stability increases with each successful recall
    """
    hours_since_creation = (now - memory.created_at).total_seconds() / 3600
    stability = memory.stability  # Increases with recalls

    return math.exp(-hours_since_creation / stability)
```

#### ACT-R Activation (Retrieval Probability)
```python
def activation(memory, context):
    """
    A = B + sum(W_j * S_ji) + noise
    B = base-level activation (frequency + recency)
    S_ji = strength of association from context j to memory i
    """
    # Base-level: log of recency-weighted frequency
    base = math.log(sum(
        (now - access_time).total_seconds() ** -0.5
        for access_time in memory.access_times
    ))

    # Spreading activation from context
    spread = sum(
        context_weight * association_strength(context_node, memory)
        for context_node, context_weight in context.items()
    )

    return base + spread + random.gauss(0, NOISE_STD)
```

#### Bayesian Confidence Updates
```python
def update_confidence(memory, evidence):
    """
    P(memory|evidence) ∝ P(evidence|memory) * P(memory)
    """
    prior = memory.confidence

    if evidence.supports:
        likelihood = 0.9  # Evidence given memory is true
    elif evidence.contradicts:
        likelihood = 0.1  # Evidence given memory is true
    else:
        likelihood = 0.5  # Neutral

    # Base rate: how likely is any memory to be supported?
    base_rate = 0.5

    # Bayes
    posterior = (likelihood * prior) / (
        likelihood * prior + base_rate * (1 - prior)
    )

    memory.confidence = posterior
```

#### Connection Strength Evolution
```python
def evolve_connection(link):
    """
    Connections strengthen with co-activation (Hebbian)
    Connections weaken with disuse
    """
    # Hebbian: "Neurons that fire together wire together"
    if link.source.recently_activated and link.target.recently_activated:
        link.strength = min(1.0, link.strength * 1.1)

    # Decay
    hours_since_use = (now - link.last_used).total_seconds() / 3600
    link.strength *= math.exp(-hours_since_use / LINK_STABILITY)

    # Prune weak connections
    if link.strength < MIN_STRENGTH:
        delete(link)
```

---

## Part 9: Storage Architecture Options

### Option S1: File-Based (Simple)
```
memory/
  memories/
    {uuid}.json           # Individual memory files
  embeddings/
    embeddings.npy        # All embeddings as numpy array
    index.json            # UUID → array index
  graph/
    edges.json            # All graph edges
  indexes/
    by_time.json          # Time-ordered memory IDs
    by_source.json        # Source → memory IDs
```
**Pro**: Simple, portable, version-controllable
**Con**: Doesn't scale past ~10K memories

### Option S2: SQLite + Extensions
```sql
-- Core storage
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB,  -- numpy tobytes()
    metadata JSON,
    created_at TIMESTAMP
);

-- Graph edges
CREATE TABLE edges (
    source_id TEXT,
    target_id TEXT,
    edge_type TEXT,
    strength REAL,
    created_at TIMESTAMP
);

-- Use sqlite-vec for vector search
-- Use sqlite FTS5 for text search
```
**Pro**: Single file, ACID, good tooling
**Con**: sqlite-vec is new, may have limits

### Option S3: Postgres + pgvector
```sql
CREATE EXTENSION vector;

CREATE TABLE memories (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMPTZ
);

CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
```
**Pro**: Production-grade, full-featured
**Con**: Requires running Postgres

### Option S4: Specialized Stack
```
┌───────────────────────────────────────┐
│           SPECIALIZED STACK           │
│                                       │
│  ┌─────────────┐  ┌─────────────────┐│
│  │   Qdrant    │  │     Neo4j       ││
│  │  (vectors)  │  │    (graph)      ││
│  └─────────────┘  └─────────────────┘│
│          │                 │          │
│          └────────┬────────┘          │
│                   │                   │
│          ┌────────▼────────┐          │
│          │   Coordinator   │          │
│          │    Service      │          │
│          └─────────────────┘          │
└───────────────────────────────────────┘
```
**Pro**: Best-in-class for each capability
**Con**: Operational complexity, data sync

### Option S5: Event-Sourced
```python
# All mutations are events
events = [
    {"type": "MEMORY_CREATED", "memory": {...}, "timestamp": "..."},
    {"type": "LINK_CREATED", "source": "...", "target": "...", "timestamp": "..."},
    {"type": "ACTIVATION_UPDATED", "memory_id": "...", "new_value": 0.7},
    ...
]

# Current state is computed from events
def get_current_state():
    state = initial_state()
    for event in events:
        state = apply(state, event)
    return state
```
**Pro**: Full history, time travel, recomputation
**Con**: Read performance requires snapshots

---

## Part 10: Query Interface Design

### Natural Language Interface
```python
async def query(q: str):
    """
    "What did I think about AI in January?"
    "Show me contradictions in my beliefs about work"
    "What themes keep recurring?"
    "Connect these two ideas: X and Y"
    """
    # Parse intent
    intent = await llm.classify_intent(q)

    if intent == "retrieval":
        return await retrieve(q)
    elif intent == "contradiction_check":
        topic = await llm.extract_topic(q)
        return await find_contradictions_for(topic)
    elif intent == "theme_analysis":
        return await analyze_themes()
    elif intent == "connect":
        ideas = await llm.extract_ideas(q)
        return await find_connections_between(ideas)
```

### Structured Query Interface
```python
# For programmatic use
engine.query({
    "type": "semantic_search",
    "query": "machine learning",
    "filters": {
        "source_type": ["note", "doc"],
        "created_after": "2025-01-01",
        "min_confidence": 0.7
    },
    "limit": 10,
    "include_connections": True
})

engine.query({
    "type": "find_contradictions",
    "topic": "productivity",
    "time_window": "last_30_days"
})

engine.query({
    "type": "trace_evolution",
    "concept": "my views on remote work",
    "from": "2024-01-01",
    "to": "2025-01-01"
})
```

### Streaming Results
```python
async def stream_query(q: str):
    """
    Stream results as they're found, with increasing quality.
    """
    # Fast: Vector search
    async for result in vector_search_stream(q):
        yield {"phase": "vector", "result": result}

    # Medium: Graph expansion
    async for result in graph_expansion_stream(seen_results):
        yield {"phase": "graph", "result": result}

    # Slow: LLM analysis
    analysis = await llm.analyze(q, all_results)
    yield {"phase": "analysis", "result": analysis}
```

---

## Summary: Architecture Recommendations

### For MVP (Start Here)
1. **SQLite + sqlite-vec** for storage
2. **Hybrid: vectors + explicit graph edges**
3. **Real-time embedding + K-NN on ingest**
4. **Nightly batch job for contradiction discovery + pattern mining**
5. **LLM-powered connection discovery (sampled)**
6. **Simple file-based multi-source ingestion**

### For Scale
1. **Postgres + pgvector** or **Qdrant + Neo4j**
2. **Stream processing for continuous updates**
3. **Learned retrieval (fine-tuned reranker)**
4. **Distributed embedding service**

### Wild Cards Worth Exploring
1. **Memory as agent dialogue** - Conversational interface to your memories
2. **Dream mode consolidation** - Background insight generation
3. **Adversarial memory** - Always show counter-evidence
4. **Temporal braiding** - Retroactive connections as understanding evolves

---

## Open Questions

1. **How aggressive should auto-linking be?** Too much = noise. Too little = missed connections.

2. **When does a "pattern" become worth surfacing?** Frequency threshold? Surprise score?

3. **How to handle contradictions?** Surface as problems? Track as evolution? Flag for resolution?

4. **Multi-user/agent memory boundaries?** Shared pool vs isolated namespaces vs hybrid?

5. **What's the right embedding model?** General (all-MiniLM) vs domain-specific vs multimodal (CLIP)?

6. **Decay vs archival?** Do memories die or just get harder to find?

---

*This is divergent brainstorming. No bad ideas yet. The real architecture will emerge from synthesis.*
