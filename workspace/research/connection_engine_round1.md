# Connection Engine Design - Brainstorm Round 1

**Date**: 2026-01-07
**Focus**: Architectural approaches for a memory system that LEARNS, not just stores
**Constraint**: 8GB Mac Mini, integrate with existing Agent Swarm memory

---

## The Problem with Standard RAG

Standard RAG (Retrieval-Augmented Generation) is fundamentally **reactive**:
1. User asks question
2. System embeds question
3. System finds similar documents
4. LLM generates answer

This misses what makes human memory powerful:
- **Spontaneous connections** - "This reminds me of..."
- **Pattern recognition over time** - "You keep coming back to X"
- **Contradiction detection** - "But last week you said..."
- **Synthesis** - Combining disparate ideas into new insights
- **Refinement** - Ideas evolving through challenge and reinforcement

A **Connection Engine** should actively BUILD understanding, not just retrieve it.

---

## Approach 1: Temporal Resonance Network (TRN)

### Core Philosophy
Ideas that co-occur in time are related. Ideas that resurface across temporal gaps have deeper significance. The brain doesn't just store facts - it tracks *what thinking patterns repeat*.

### Core Data Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPORAL RESONANCE NETWORK                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  THOUGHT FRAGMENT                                                │
│  ├── id: uuid                                                    │
│  ├── content: str (raw text)                                     │
│  ├── embedding: float[384]                                       │
│  ├── timestamp: datetime                                         │
│  ├── source_type: enum (voice, note, screenshot, doc)           │
│  └── session_id: str (which thinking session)                   │
│                                                                  │
│  RESONANCE PATTERN                                               │
│  ├── id: uuid                                                    │
│  ├── signature: str (what this pattern represents)              │
│  ├── fragment_ids: list[uuid]                                   │
│  ├── first_seen: datetime                                        │
│  ├── last_seen: datetime                                         │
│  ├── occurrence_count: int                                       │
│  ├── temporal_gaps: list[timedelta] (time between occurrences) │
│  └── resonance_score: float (increases with gaps + repetition) │
│                                                                  │
│  SESSION (thinking window)                                       │
│  ├── id: str                                                     │
│  ├── start_time: datetime                                        │
│  ├── fragments: list[uuid]                                       │
│  └── co_occurrence_matrix: sparse_matrix                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How Connections Are Discovered

**Phase 1: Real-time Co-occurrence**
```python
# Within a session, track what appears together
def on_fragment_added(fragment, session):
    window = get_recent_fragments(session, window_size=5)
    for other in window:
        strength = 1.0 / (position_distance + 1)
        session.co_occurrence_matrix.update(fragment.id, other.id, strength)
```

**Phase 2: Cross-Session Pattern Detection**
```python
# Background job: find patterns that repeat across sessions
def detect_resonance_patterns():
    for session_pair in recent_session_pairs():
        # Find fragments with high embedding similarity across sessions
        cross_matches = find_cross_session_matches(
            session_a, session_b,
            similarity_threshold=0.75
        )

        for match in cross_matches:
            temporal_gap = session_b.start - session_a.end

            pattern = get_or_create_pattern(match.signature)
            pattern.add_occurrence(match, temporal_gap)

            # KEY: Longer gaps = stronger resonance
            # Ideas that persist across days are more fundamental
            pattern.resonance_score = calculate_resonance(
                occurrence_count=pattern.occurrence_count,
                avg_gap=mean(pattern.temporal_gaps),
                gap_variance=var(pattern.temporal_gaps)
            )
```

**Phase 3: Resonance-Based Surfacing**
```python
# Proactively surface high-resonance patterns
def generate_proactive_insights():
    strong_patterns = get_patterns_by_resonance(min_score=0.7)

    for pattern in strong_patterns:
        if not recently_surfaced(pattern):
            insight = synthesize_pattern(pattern)
            # "You've been thinking about X across 5 sessions over 2 weeks.
            #  Here's what seems to be crystallizing..."
            yield ProactiveInsight(pattern, insight)
```

### Key Differentiator from Vector Search

| Vector Search | Temporal Resonance Network |
|---------------|---------------------------|
| Finds similar content | Finds *recurring* patterns |
| Static similarity | Dynamic resonance that grows |
| Query-driven | Pattern-driven surfacing |
| Point-in-time | Tracks evolution over time |
| "What's similar?" | "What keeps coming back?" |

### Integration Points

1. **Ingestion Hook** (`backend/services/memory_extractor.py`)
   - Every extraction becomes a ThoughtFragment
   - Tag with source_type and session_id

2. **Session Wrapper** (new)
   - Group interactions into thinking sessions
   - Track co-occurrence within sessions

3. **Background Worker** (new)
   - Cron job for cross-session pattern detection
   - Update resonance scores hourly

4. **Proactive Surface Hook** (`backend/websocket/chat_handler.py`)
   - Check for unsurfaced high-resonance patterns on session start
   - Inject as system context

### Strengths & Weaknesses

**Strengths:**
- Naturally finds what matters (persistence = importance)
- Captures evolution of ideas over time
- Low computational overhead (sparse matrices, incremental updates)
- Biologically inspired (spaced repetition in reverse)

**Weaknesses:**
- Cold start problem (needs time to build patterns)
- Session boundary detection is non-trivial
- May miss connections that don't repeat

---

## Approach 2: Semantic Constellation Graph (SCG)

### Core Philosophy
Knowledge isn't a flat embedding space - it's a multi-dimensional constellation where ideas cluster, repel, and bridge. Instead of one embedding, create multiple "lenses" that reveal different connection types.

### Core Data Model

```
┌─────────────────────────────────────────────────────────────────┐
│                  SEMANTIC CONSTELLATION GRAPH                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CONCEPT NODE                                                    │
│  ├── id: uuid                                                    │
│  ├── label: str                                                  │
│  ├── description: str                                            │
│  ├── embeddings: Dict[lens_type, float[384]]                    │
│  │   ├── semantic: "what is it about"                           │
│  │   ├── functional: "what does it enable"                      │
│  │   ├── emotional: "how does it feel"                          │
│  │   └── temporal: "when is it relevant"                        │
│  ├── constellation_id: str (which cluster)                      │
│  └── bridge_score: float (connects constellations?)             │
│                                                                  │
│  CONSTELLATION (cluster of related concepts)                    │
│  ├── id: str                                                     │
│  ├── centroid: Dict[lens_type, float[384]]                      │
│  ├── members: list[uuid]                                         │
│  ├── name: str (LLM-generated)                                  │
│  ├── description: str (LLM-generated)                           │
│  └── tension_edges: list[TensionEdge] (contradictions)          │
│                                                                  │
│  BRIDGE (connection between constellations)                     │
│  ├── source_constellation: str                                   │
│  ├── target_constellation: str                                   │
│  ├── bridge_concepts: list[uuid] (concepts that span both)      │
│  ├── bridge_type: enum (enables, contradicts, evolves, combines)│
│  └── strength: float                                             │
│                                                                  │
│  TENSION EDGE (internal contradiction)                          │
│  ├── concept_a: uuid                                             │
│  ├── concept_b: uuid                                             │
│  ├── tension_type: str (contradiction, unresolved, evolved)     │
│  └── resolution_status: enum (open, resolved, accepted)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How Connections Are Discovered

**Multi-Lens Embedding**
```python
# Each piece of knowledge gets 4 embeddings
def create_multi_lens_embedding(text: str, context: dict) -> dict:
    base_embedding = embed(text)

    # Functional lens: "What does this enable? What problems does it solve?"
    functional_prompt = f"Describe the practical applications of: {text}"
    functional_embedding = embed(llm_transform(functional_prompt))

    # Emotional lens: "How does this make you feel? What values does it connect to?"
    emotional_prompt = f"Describe the emotional significance of: {text}"
    emotional_embedding = embed(llm_transform(emotional_prompt))

    # Temporal lens: "When is this relevant? What triggers thinking about this?"
    temporal_prompt = f"Describe when this becomes important: {text}"
    temporal_embedding = embed(llm_transform(temporal_prompt))

    return {
        "semantic": base_embedding,
        "functional": functional_embedding,
        "emotional": emotional_embedding,
        "temporal": temporal_embedding
    }
```

**Constellation Formation**
```python
# Periodic clustering across all lenses
def update_constellations():
    # Weighted combination of lens distances
    combined_distance = lambda a, b: (
        0.4 * cosine_dist(a.semantic, b.semantic) +
        0.3 * cosine_dist(a.functional, b.functional) +
        0.2 * cosine_dist(a.emotional, b.emotional) +
        0.1 * cosine_dist(a.temporal, b.temporal)
    )

    clusters = hdbscan_cluster(all_concepts, distance_fn=combined_distance)

    for cluster in clusters:
        constellation = Constellation(
            members=cluster.members,
            centroid=compute_multi_lens_centroid(cluster),
            name=llm_name_cluster(cluster.members),
            description=llm_describe_cluster(cluster.members)
        )
        save(constellation)
```

**Bridge Detection**
```python
# Find concepts that span multiple constellations
def detect_bridges():
    for concept in all_concepts:
        distances_to_constellations = [
            (c, multi_lens_distance(concept, c.centroid))
            for c in all_constellations
        ]

        # Concepts close to 2+ constellations are bridges
        close_constellations = [
            c for c, d in distances_to_constellations if d < threshold
        ]

        if len(close_constellations) >= 2:
            concept.bridge_score = calculate_bridge_strength(
                concept, close_constellations
            )
            create_bridge_edges(concept, close_constellations)
```

**Tension Detection**
```python
# Find contradictions within constellations
def detect_tensions():
    for constellation in all_constellations:
        for pair in combinations(constellation.members, 2):
            # High semantic similarity but different emotional/functional?
            semantic_sim = cosine_sim(pair[0].semantic, pair[1].semantic)
            functional_sim = cosine_sim(pair[0].functional, pair[1].functional)

            if semantic_sim > 0.8 and functional_sim < 0.3:
                # Tension: similar topic, different purpose
                create_tension_edge(
                    pair[0], pair[1],
                    tension_type="functional_divergence"
                )
```

### Key Differentiator from Vector Search

| Vector Search | Semantic Constellation Graph |
|---------------|------------------------------|
| Single embedding | Multi-lens embeddings |
| Flat similarity | Hierarchical clusters |
| No structure | Constellations + Bridges |
| Ignores contradictions | Explicitly models tension |
| "Find similar" | "Find connections AND contradictions" |

### Integration Points

1. **Enhanced Extraction** (`backend/services/memory_extractor.py`)
   - Generate 4 embeddings per concept
   - Store in existing embedding_store with lens prefix

2. **Constellation Service** (new)
   - Background job for clustering
   - LLM calls for naming/describing clusters

3. **Query Enhancement**
   - "Which lens should I search?" based on query type
   - Bridge concepts get boosted in cross-domain queries

4. **Proactive Surfacing**
   - Surface unresolved tensions: "You have conflicting ideas about X"
   - Surface bridges: "This connects to your other work on Y"

### Strengths & Weaknesses

**Strengths:**
- Rich, nuanced understanding of relationships
- Explicitly models contradictions (valuable for refinement)
- Bridge detection reveals non-obvious connections
- LLM-generated cluster names are interpretable

**Weaknesses:**
- 4x embedding cost (could be expensive)
- Clustering is computationally heavier
- LLM calls for lens transformation add latency
- More complex to implement and debug

---

## Approach 3: Dialogue-Driven Knowledge Crystallization (DDKC)

### Core Philosophy
Knowledge doesn't crystallize in isolation - it forms through **dialogue**. Instead of passive storage, the Connection Engine actively engages with new information, questioning, connecting, and refining. Every piece of knowledge is processed through an internal "debate."

### Core Data Model

```
┌─────────────────────────────────────────────────────────────────┐
│            DIALOGUE-DRIVEN KNOWLEDGE CRYSTALLIZATION            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RAW INGESTION                                                   │
│  ├── id: uuid                                                    │
│  ├── content: str                                                │
│  ├── source: enum (voice, note, screenshot, doc)                │
│  ├── timestamp: datetime                                         │
│  └── processed: bool                                             │
│                                                                  │
│  DIALOGUE TURN                                                   │
│  ├── id: uuid                                                    │
│  ├── ingestion_id: uuid                                          │
│  ├── agent: enum (extractor, connector, challenger, synthesizer)│
│  ├── content: str                                                │
│  ├── references: list[uuid] (existing knowledge referenced)     │
│  └── timestamp: datetime                                         │
│                                                                  │
│  CRYSTALLIZED KNOWLEDGE                                          │
│  ├── id: uuid                                                    │
│  ├── label: str                                                  │
│  ├── description: str                                            │
│  ├── confidence: float (0-1, from dialogue consensus)           │
│  ├── maturity: enum (seed, developing, crystallized, validated) │
│  ├── dialogue_transcript: list[DialogueTurn]                    │
│  ├── supporting_evidence: list[uuid]                            │
│  ├── challenging_evidence: list[uuid]                           │
│  ├── connections: list[Connection]                              │
│  └── last_challenged: datetime                                   │
│                                                                  │
│  CONNECTION                                                      │
│  ├── source_id: uuid                                             │
│  ├── target_id: uuid                                             │
│  ├── connection_type: enum (supports, contradicts, enables,     │
│  │                          requires, evolved_from, related_to) │
│  ├── rationale: str (why this connection exists)                │
│  └── created_by_dialogue: uuid (which dialogue created this)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How Connections Are Discovered

**The Internal Dialogue**

Every new piece of information triggers a 4-agent internal dialogue:

```python
async def process_ingestion(raw: RawIngestion) -> CrystallizedKnowledge:
    dialogue = Dialogue(ingestion_id=raw.id)

    # AGENT 1: EXTRACTOR
    # "What is the core claim or insight here?"
    extraction = await extractor_agent.process(
        raw.content,
        prompt="Extract the key claims, insights, or information. Be precise."
    )
    dialogue.add_turn(agent="extractor", content=extraction)

    # AGENT 2: CONNECTOR
    # "What does this relate to in existing knowledge?"
    related = search_existing_knowledge(extraction, limit=10)
    connections = await connector_agent.process(
        new_content=extraction,
        existing_knowledge=related,
        prompt="""
        Identify connections between this new information and existing knowledge.
        For each connection, specify:
        - Which existing knowledge it connects to
        - The type of connection (supports, contradicts, enables, etc.)
        - A brief rationale
        """
    )
    dialogue.add_turn(agent="connector", content=connections)

    # AGENT 3: CHALLENGER
    # "What's wrong with this? What assumptions are being made?"
    challenge = await challenger_agent.process(
        extraction=extraction,
        connections=connections,
        prompt="""
        Challenge this information:
        - What assumptions is it making?
        - What contradictions exist with known facts?
        - What evidence would be needed to validate this?
        - Rate your confidence (0-1) that this is accurate.
        """
    )
    dialogue.add_turn(agent="challenger", content=challenge)

    # AGENT 4: SYNTHESIZER
    # "Given all perspectives, what should we crystallize?"
    synthesis = await synthesizer_agent.process(
        dialogue_so_far=dialogue.turns,
        prompt="""
        Synthesize the dialogue into crystallized knowledge:
        - Final label and description
        - Confidence score (accounting for challenges)
        - Maturity level (seed if uncertain, crystallized if well-supported)
        - Final list of connections with rationales
        """
    )
    dialogue.add_turn(agent="synthesizer", content=synthesis)

    # Create the crystallized knowledge
    return CrystallizedKnowledge.from_synthesis(synthesis, dialogue)
```

**Periodic Re-Challenge**
```python
# Knowledge should be periodically re-examined
async def rechallenge_old_knowledge():
    # Find knowledge that hasn't been challenged in a while
    stale = get_knowledge_needing_challenge(days_since_challenge=30)

    for knowledge in stale:
        # Has new evidence emerged?
        new_related = search_new_knowledge_since(
            knowledge.last_challenged,
            query=knowledge.label
        )

        if new_related:
            # Re-run challenger with new context
            new_challenge = await challenger_agent.process(
                existing=knowledge,
                new_evidence=new_related,
                prompt="Given this new information, should we revise our confidence?"
            )

            if new_challenge.suggests_revision:
                # Update or mark for human review
                update_or_flag(knowledge, new_challenge)
```

**Proactive Synthesis**
```python
# Periodically synthesize across multiple knowledge pieces
async def generate_meta_insights():
    # Find clusters of related knowledge
    clusters = cluster_knowledge_by_embedding(threshold=0.7)

    for cluster in clusters:
        if len(cluster) >= 3 and not has_meta_synthesis(cluster):
            synthesis = await synthesizer_agent.process(
                knowledge_items=cluster,
                prompt="""
                These pieces of knowledge seem related.
                Can you articulate an overarching insight that connects them?
                What pattern or principle emerges from considering them together?
                """
            )

            if synthesis.has_novel_insight:
                # Create new meta-knowledge
                create_meta_knowledge(cluster, synthesis)
                # Notify user
                yield ProactiveInsight(
                    type="synthesis",
                    content=synthesis.insight,
                    supporting_knowledge=cluster
                )
```

### Key Differentiator from Vector Search

| Vector Search | Dialogue-Driven Crystallization |
|---------------|--------------------------------|
| Passive storage | Active processing |
| No validation | Built-in challenging |
| No rationale | Every connection has WHY |
| Static confidence | Evolving confidence |
| "Store and retrieve" | "Debate and crystallize" |

### Integration Points

1. **Ingestion Pipeline** (new)
   - Voice/screenshot/note all become RawIngestion
   - Queue for async dialogue processing

2. **Agent Pool** (leverage existing swarm)
   - 4 specialized agents: extractor, connector, challenger, synthesizer
   - Could use Claude Haiku for speed, Sonnet for quality

3. **Existing Memory Bridge**
   - CrystallizedKnowledge maps to SemanticNode
   - Connections map to existing edge types
   - Dialogue transcript stored in provenance

4. **User Interaction**
   - User can view dialogue transcripts ("Why do you think X?")
   - User can trigger re-challenge ("Are you sure about Y?")
   - User feedback reinforces/weakens connections

### Strengths & Weaknesses

**Strengths:**
- Every piece of knowledge has a "paper trail"
- Built-in mechanism for refinement and challenge
- Connections have explicit rationales
- Naturally handles contradictions
- Aligns with how humans actually form beliefs

**Weaknesses:**
- High LLM token cost (4 agents per ingestion)
- Latency for real-time use
- Agent quality is critical
- Complex debugging when dialogues go wrong

---

## Comparison Matrix

| Dimension | TRN | SCG | DDKC |
|-----------|-----|-----|------|
| **Connection Discovery** | Time-based patterns | Multi-lens clustering | LLM dialogue |
| **Proactive Surfacing** | Resonance-based | Bridge + tension based | Synthesis-based |
| **Challenge/Refine** | Implicit (decay) | Explicit (tensions) | Explicit (challenger agent) |
| **Computational Cost** | Low | Medium-High | High |
| **Latency** | Low | Medium | High (async) |
| **Interpretability** | Medium | High (cluster names) | Very High (transcripts) |
| **Cold Start** | Needs time | Works immediately | Works immediately |
| **RAM Usage** | ~20MB (sparse) | ~50MB (4x embeddings) | ~30MB + LLM calls |

---

## Recommendation: Hybrid Architecture

The strongest system combines elements from all three:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID CONNECTION ENGINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER 1: Fast Path (TRN-inspired)                              │
│  ├── Real-time co-occurrence tracking                           │
│  ├── Session-based temporal patterns                            │
│  └── Low overhead, always-on                                    │
│                                                                  │
│  LAYER 2: Structure Path (SCG-inspired)                         │
│  ├── Multi-lens embedding (defer functional/emotional)          │
│  ├── Periodic constellation clustering                          │
│  └── Bridge detection for cross-domain insights                 │
│                                                                  │
│  LAYER 3: Deep Path (DDKC-inspired)                             │
│  ├── On-demand dialogue for important ingestions                │
│  ├── Periodic re-challenge of crystallized knowledge            │
│  └── Human-in-the-loop for high-uncertainty items               │
│                                                                  │
│  ROUTER: Decides which path based on:                           │
│  ├── Source importance (voice note = Layer 3)                   │
│  ├── Novelty (high = Layer 3, low = Layer 1)                   │
│  └── User preference                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Priority (if building hybrid)

1. **Week 1**: Layer 1 (TRN basics)
   - Session tracking
   - Co-occurrence matrix
   - Basic resonance scoring

2. **Week 2**: Layer 2 (SCG basics)
   - Semantic embedding (single lens first)
   - Simple clustering (k-means on schedule)
   - Bridge score calculation

3. **Week 3**: Layer 3 (DDKC basics)
   - Challenger agent only (skip full dialogue)
   - Periodic re-challenge job
   - Connection rationale storage

4. **Week 4**: Integration
   - Router logic
   - Proactive surfacing hooks
   - User feedback loop

---

## Open Questions for Discussion

1. **Session Boundaries**: How do we detect when a "thinking session" starts/ends? Time gaps? Topic shifts?

2. **Challenge Frequency**: How often should crystallized knowledge be re-challenged? Too often = wasted compute. Too rare = stale beliefs.

3. **Multi-source Weighting**: Should voice notes get heavier weighting than screenshots? User preference?

4. **Contradiction Resolution**: When challenger finds contradiction, who decides? Auto-resolve, flag for human, or hold both?

5. **Privacy/Deletability**: How do we handle "forget this" requests when knowledge is deeply connected?

---

*Round 1 Complete - Ready for synthesis and selection*
