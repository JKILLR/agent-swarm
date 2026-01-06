# MAXIMALLY INTELLIGENT MEMORY SYSTEM - Implementation Plan

**Version**: 1.0
**Created**: 2026-01-05
**Target**: External execution via Claude Code CLI
**Reference**: docs/MAXIMALLY_INTELLIGENT_MEMORY_ARCHITECTURE.md

---

## Executive Summary

This plan transforms the current MindGraph memory system into a cognitive architecture with:
- **Tri-Memory System**: Episodic, Semantic, and Procedural memory stores
- **Memory Dynamics**: Decay, reinforcement, and consolidation
- **Reasoning Capabilities**: Inference chains and spreading activation
- **Temporal Intelligence**: Time-aware memory with prediction
- **Meta-Cognition**: Self-reflection and confidence tracking

### Current State Analysis

| Component | Location | Status |
|-----------|----------|--------|
| MindGraph | `backend/services/mind_graph.py` | Core graph storage with 8 node types |
| SemanticIndex | `backend/services/semantic_index.py` | 384-dim embeddings, cosine similarity |
| EmbeddingService | `backend/services/embedding_service.py` | all-MiniLM-L6-v2 model |
| MemoryExtractor | `backend/services/memory_extractor.py` | LLM-based extraction (Haiku) |
| ConversationMemory | `backend/services/conversation_memory.py` | Orchestrates extraction |
| ConversationAnalyzer | `backend/services/conversation_analyzer.py` | Pattern-based extraction |

### Key Limitations to Address

1. **Single memory type** - No distinction between episodic/semantic/procedural
2. **No memory decay** - Everything persists equally forever
3. **No confidence tracking** - All facts treated as equally certain
4. **O(n) similarity search** - Won't scale past ~10K nodes
5. **No consolidation** - Memories never evolve or abstract
6. **No inference** - Graph is static, can't derive new knowledge
7. **No procedural memory** - Can't learn "how to do things"

---

## Phase 1: Foundation
**Goal**: Separate memory types, add decay and confidence

### 1.1 Data Model Extensions

#### Files to Modify

**`backend/services/mind_graph.py`**

Add new node types to `NodeType` enum:
```python
class NodeType(Enum):
    # Existing types (keep for backward compatibility)
    CONCEPT = "concept"
    FACT = "fact"
    MEMORY = "memory"          # Becomes EPISODIC
    IDENTITY = "identity"
    PREFERENCE = "preference"
    GOAL = "goal"
    DECISION = "decision"
    RELATIONSHIP = "relationship"

    # New types for tri-memory system
    EPISODIC = "episodic"      # Time-bound experiences
    SEMANTIC = "semantic"       # Abstracted facts
    PROCEDURAL = "procedural"   # Skills and procedures
    SCHEMA = "schema"           # Abstract patterns
```

Add new fields to `MindNode` class:
```python
class MindNode:
    # ... existing fields ...

    # Memory dynamics (new)
    confidence: float = 1.0           # 0.0-1.0 certainty
    decay_rate: float = 0.1           # Ebbinghaus decay parameter
    last_accessed: str = None         # ISO timestamp
    access_count: int = 0             # Retrieval count
    encoding_strength: float = 1.0    # Initial encoding strength
    emotional_valence: float = 0.0    # -1.0 to +1.0
    temporal_validity: dict = None    # {valid_from, valid_to}
    derived_from: list[str] = []      # Source node IDs
    consensus_count: int = 1          # Supporting evidence count
```

#### Files to Create

**`backend/services/memory_types.py`** (NEW)
```
Purpose: Defines the three memory type classes
- EpisodicMemory: Time-bound experiences with context
- SemanticMemory: Abstracted facts with confidence
- ProceduralMemory: Skill templates with success metrics
```

**`backend/services/memory_dynamics.py`** (NEW)
```
Purpose: Implements decay, activation, and reinforcement
- MemoryDynamics class with:
  - compute_activation(node) -> float
  - should_forget(node) -> bool
  - reinforce(node) -> None
  - apply_decay() -> List[node_id]  # Returns nodes to prune
```

### 1.2 Migration Script

**`scripts/migrate_to_v2.py`** (NEW)
```
Purpose: Migrate existing mind_graph.json to new format
Steps:
1. Load current mind_graph.json
2. Add default values for new fields:
   - confidence: 0.8 (default for existing)
   - decay_rate: 0.1
   - last_accessed: created_at
   - access_count: 1
   - encoding_strength: importance / 5.0
3. Reclassify MEMORY nodes as EPISODIC
4. Preserve all existing data
5. Create backup before migration
```

### 1.3 Decay and Activation System

**Implementation in `memory_dynamics.py`**:

```python
# Ebbinghaus forgetting curve
# R = e^(-t/S) where:
#   R = retention (0-1)
#   t = time since last access
#   S = memory strength

def compute_activation(node: MindNode) -> float:
    """ACT-R style activation computation."""
    base_level = compute_base_level_activation(node)
    importance_boost = node.metadata.get('importance', 3) * 0.1
    emotional_boost = abs(node.emotional_valence) * 0.3
    return base_level + importance_boost + emotional_boost

def compute_base_level_activation(node: MindNode) -> float:
    """Base-level = ln(sum of t_i^(-d))"""
    if node.access_count == 0:
        return 0.0

    time_since = (now - parse(node.last_accessed)).total_seconds()
    decay = node.decay_rate

    # Simplified: use time since last access
    if time_since <= 0:
        return 1.0
    return math.log(time_since ** (-decay) + 1)
```

### 1.4 Confidence Tracking

**`backend/services/confidence_tracker.py`** (NEW)
```
Purpose: Track and update confidence scores

Key methods:
- update_confidence(node_id, evidence) -> float
- get_confidence_interval(node_id) -> Tuple[float, float]
- propagate_confidence(source_id, target_id, relation_type)

Bayesian update formula:
P(A|E) = P(E|A) * P(A) / P(E)
```

### 1.5 Success Criteria - Phase 1

| Criteria | Test Method |
|----------|-------------|
| Migration completes without data loss | Compare node counts pre/post |
| New fields persist correctly | Create node, reload, verify fields |
| Decay calculation is correct | Unit test with known values |
| Activation updates on access | Access node, verify access_count++ |
| Backward compatibility | Existing API calls still work |

### 1.6 Rollback Strategy - Phase 1

1. Backup `memory/graph/mind_graph.json` before migration
2. Backup `memory/graph/embeddings.npz` and `embedding_meta.json`
3. Migration script creates `mind_graph.v1.backup.json`
4. Rollback: `cp mind_graph.v1.backup.json mind_graph.json`
5. New fields are optional with defaults - old code continues working

### 1.7 CLI Commands - Phase 1

```bash
# Migrate to v2 format
claude --print "Run migration: python scripts/migrate_to_v2.py"

# Verify migration
claude --print "Check node count and new fields in mind_graph.json"

# Run unit tests
claude --print "pytest backend/tests/test_memory_dynamics.py -v"
```

---

## Phase 2: Memory Consolidation
**Goal**: Implement sleep cycles where episodic memories consolidate into semantic

### 2.1 Consolidation Engine

**`backend/services/consolidation_engine.py`** (NEW)

```
Purpose: Background process that consolidates memories

Key classes:
- ConsolidationEngine
  - run_consolidation_cycle() -> ConsolidationReport
  - select_candidates() -> List[EpisodicMemory]
  - replay_and_extract(episodes) -> List[Pattern]
  - abstract_to_semantic(pattern) -> SemanticNode
  - prune_weak_memories() -> List[str]  # Deleted node IDs
  - merge_duplicates() -> int  # Merged count
```

### 2.2 Pattern Extraction

**Logic for extracting patterns from episodic memories**:

1. **Clustering**: Group similar episodic memories
   - Use embedding similarity (>0.7 threshold)
   - Minimum 2 episodes to form a pattern

2. **Common Element Extraction**:
   - Find shared entities across episodes
   - Find shared relationships
   - Find shared temporal patterns

3. **Abstraction**:
   - Create semantic node with generalized description
   - Set confidence based on episode count
   - Link back to source episodes

### 2.3 Procedural Learning

**Logic for skill extraction**:

1. **Action Pattern Detection**:
   - Analyze user requests and successful outcomes
   - Track tool usage sequences
   - Record decision points

2. **Skill Template Creation**:
   ```python
   class ProceduralSkill:
       skill_id: str
       name: str
       trigger_pattern: str  # Regex or semantic pattern
       steps: List[SkillStep]
       success_rate: float
       avg_execution_time: float
       learned_from: List[str]  # Episode IDs
   ```

3. **Chunking**:
   - Compile frequently used sequences
   - Higher chunking_level = more automatic

### 2.4 Strategic Forgetting

**Implementation**:

```python
def prune_weak_memories(self) -> List[str]:
    """Delete memories below activation threshold."""
    pruned = []

    for node in self.graph.get_all_episodic():
        activation = self.dynamics.compute_activation(node)
        retrieval_prob = 1 / (1 + math.exp(-activation))

        if retrieval_prob < 0.05:  # 5% threshold
            # Check if consolidated
            if node.metadata.get('consolidated', False):
                self.graph.delete_node(node.id)
                pruned.append(node.id)
            else:
                # Move to cold storage instead
                self.archive_node(node)

    return pruned
```

### 2.5 Background Job Setup

**`backend/services/scheduler.py`** (NEW or MODIFY)

```python
# Add consolidation to scheduler
schedule.every().day.at("03:00").do(run_consolidation)  # Night cycle
schedule.every(4).hours.do(run_mini_consolidation)      # Mini cycles

async def run_consolidation():
    engine = ConsolidationEngine(get_mind_graph())
    report = await engine.run_consolidation_cycle()
    logger.info(f"Consolidation: {report}")
```

### 2.6 Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/services/consolidation_engine.py` | CREATE | Main consolidation logic |
| `backend/services/pattern_extractor.py` | CREATE | Extract patterns from episodes |
| `backend/services/procedural_memory.py` | CREATE | Skill learning and storage |
| `backend/services/memory_archive.py` | CREATE | Cold storage for weak memories |
| `backend/services/scheduler.py` | MODIFY | Add consolidation jobs |

### 2.7 Success Criteria - Phase 2

| Criteria | Test Method |
|----------|-------------|
| Episodes cluster correctly | Create 3 similar episodes, verify pattern |
| Semantic nodes created from patterns | Check node_type='semantic' in graph |
| Skills extracted from actions | Verify ProceduralSkill creation |
| Weak memories pruned | Check deleted_count > 0 after cycle |
| No data corruption | Full graph validation after cycle |

### 2.8 Rollback Strategy - Phase 2

1. Consolidation creates snapshot before each run
2. Each deleted node logged with full content
3. `memory/archive/` stores cold memories
4. Rollback: Restore from snapshot + re-import archive

---

## Phase 3: Reasoning Engine
**Goal**: Enable inference and knowledge derivation

### 3.1 Inference Chain Manager

**`backend/services/inference_engine.py`** (NEW)

```
Purpose: Multi-strategy reasoning over knowledge graph

Key classes:
- InferenceChainManager
  - build_inference_chain(query, max_depth=5) -> InferenceChain
  - explain_chain(chain) -> str

- DeductiveReasoner
  - modus_ponens(premise, antecedent) -> Conclusion
  - transitive_inference(a_rel_b, b_rel_c) -> a_rel_c
  - inheritance_inference(entity, hierarchy) -> Properties

- AnalogicalReasoner
  - find_analogies(target, sources, min_similarity) -> List[Analogy]
  - structure_map(source, target) -> StructuralMapping
  - project_inferences(mapping) -> List[CandidateInference]
```

### 3.2 Spreading Activation

**Modify `backend/services/semantic_index.py`**:

```python
def spread_activation(
    self,
    seed_nodes: List[str],
    iterations: int = 3,
    decay: float = 0.8
) -> Dict[str, float]:
    """
    Spread activation through the semantic network.
    Returns activation levels for all affected nodes.
    """
    activations = {nid: 1.0 for nid in seed_nodes}

    for _ in range(iterations):
        new_activations = {}
        for node_id, activation in activations.items():
            # Get connected nodes via edges
            node = self.graph.get_node(node_id)
            for edge in node.edges:
                target = edge['target']
                spread = activation * decay
                new_activations[target] = max(
                    new_activations.get(target, 0),
                    spread
                )

        # Merge, keeping max
        for nid, act in new_activations.items():
            if act > 0.1:  # Threshold
                activations[nid] = max(activations.get(nid, 0), act)

    return activations
```

### 3.3 Deductive Inference

**Key inference rules to implement**:

1. **IS-A Inheritance**:
   - If `dog IS-A animal` and `animal HAS-PART heart`
   - Then infer `dog HAS-PART heart`

2. **Transitive Relations**:
   - If `A parent-of B` and `B parent-of C`
   - Then infer `A grandparent-of C`

3. **Causal Chains**:
   - If `A causes B` and `B causes C`
   - Then infer `A indirectly-causes C` (with reduced confidence)

### 3.4 Analogical Reasoning

**Structure Mapping Implementation**:

```python
def structure_map(source: ConceptualGraph, target: ConceptualGraph) -> StructuralMapping:
    """
    Gentner's Structure Mapping Theory implementation.
    1. Find local matches (same relation types)
    2. Build global interpretation (consistent mappings)
    3. Compute systematicity (prefer deep structure)
    """
    # Phase 1: Local matches
    local_matches = []
    for s_edge in source.edges:
        for t_edge in target.edges:
            if s_edge.type == t_edge.type:
                local_matches.append((s_edge, t_edge))

    # Phase 2: Global consistency
    # Use constraint satisfaction to find consistent mapping
    mapping = find_consistent_mapping(local_matches)

    # Phase 3: Systematicity scoring
    # Higher-order relations score higher
    systematicity = compute_systematicity(mapping)

    return StructuralMapping(
        correspondences=mapping,
        systematicity_score=systematicity,
        unmapped_source=get_unmapped(source, mapping)
    )
```

### 3.5 Files to Create

| File | Purpose |
|------|---------|
| `backend/services/inference_engine.py` | Main inference orchestration |
| `backend/services/deductive_reasoner.py` | Logical inference rules |
| `backend/services/analogical_reasoner.py` | Structure mapping |
| `backend/services/inference_chain.py` | Chain tracking and explanation |

### 3.6 Success Criteria - Phase 3

| Criteria | Test Method |
|----------|-------------|
| Transitive inference works | Create A→B→C, query A→C |
| Inheritance works | Create hierarchy, query properties |
| Analogies found | Create parallel structures, verify mapping |
| Explanations generated | Check natural language output |
| Confidence propagates | Verify reduced confidence in inferences |

### 3.7 Rollback Strategy - Phase 3

- Inference is read-only on existing data
- Derived nodes marked with `source='inference'`
- Rollback: Delete all nodes where source='inference'

---

## Phase 4: Temporal Intelligence
**Goal**: Time-aware memory with prediction

### 4.1 Temporal Knowledge Graph

**`backend/services/temporal_graph.py`** (NEW)

```
Purpose: Store and query time-valid facts

Key structures:
- TemporalFact
  - subject, predicate, object
  - valid_from: datetime
  - valid_to: Optional[datetime]  # None = still valid
  - temporal_type: POINT | INTERVAL | RECURRING
  - recurrence_pattern: Optional[str]

- TemporalQuery
  - is_valid_at(timestamp) -> bool
  - overlaps(other) -> bool
  - get_temporal_relation(other) -> AllenRelation
```

### 4.2 Allen's Interval Algebra

**13 temporal relations to implement**:

```python
class TemporalRelation(Enum):
    BEFORE = "before"         # A ends before B starts
    AFTER = "after"           # A starts after B ends
    MEETS = "meets"           # A ends exactly when B starts
    MET_BY = "met_by"
    OVERLAPS = "overlaps"     # A starts before B, ends during B
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"         # A starts with B, ends earlier
    STARTED_BY = "started_by"
    DURING = "during"         # A fully contained in B
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"
```

### 4.3 Event Sequence Mining

**`backend/services/event_sequence_miner.py`** (NEW)

```python
class EventSequenceMiner:
    """Discover patterns in temporal event sequences."""

    def mine_patterns(
        self,
        episodes: List[EpisodicMemory],
        min_support: float = 0.3,
        max_gap: timedelta = timedelta(days=7)
    ) -> List[TemporalPattern]:
        """
        Uses modified PrefixSpan algorithm for sequences.
        """
        sequences = [self.extract_events(ep) for ep in episodes]
        patterns = self.prefix_span(sequences, min_support)
        return self.add_temporal_constraints(patterns, sequences)

    def predict_next_events(
        self,
        current_context: List[Event],
        horizon: timedelta = timedelta(days=7)
    ) -> List[PredictedEvent]:
        """Predict likely future events based on patterns."""
        pass
```

### 4.4 Future Prediction

**Capabilities**:

1. **Trend Extrapolation**:
   - If pattern shows weekly occurrence, predict next occurrence

2. **Goal Progress ETA**:
   - Based on similar goal completion patterns

3. **Risk Anticipation**:
   - Identify patterns that led to negative outcomes

### 4.5 Files to Create

| File | Purpose |
|------|---------|
| `backend/services/temporal_graph.py` | Temporal fact storage |
| `backend/services/temporal_relations.py` | Allen's interval algebra |
| `backend/services/event_sequence_miner.py` | Pattern mining |
| `backend/services/temporal_predictor.py` | Future prediction |

### 4.6 Success Criteria - Phase 4

| Criteria | Test Method |
|----------|-------------|
| Temporal validity works | Create fact, query at different times |
| Allen relations correct | Unit tests for all 13 relations |
| Patterns discovered | Create sequence, verify pattern detection |
| Predictions reasonable | Create test pattern, verify prediction |

---

## Phase 5: Meta-Cognition
**Goal**: System knows what it knows

### 5.1 Epistemic Status Tracking

**`backend/services/epistemic_tracker.py`** (NEW)

```python
@dataclass
class EpistemicStatus:
    """Full epistemic status of knowledge."""
    confidence: float                    # P(true)
    confidence_interval: Tuple[float, float]
    evidence_strength: float
    evidence_count: int
    source_reliability: float
    source_diversity: int
    temporal_decay: float
    last_validated: datetime
    consistency_score: float
    contradictions: List[str]

    def overall_certainty(self) -> float:
        return (
            self.confidence * 0.4 +
            self.evidence_strength * 0.2 +
            self.source_reliability * 0.2 +
            (1 - self.temporal_decay) * 0.1 +
            self.consistency_score * 0.1
        )
```

### 5.2 Knowledge Gap Detection

**`backend/services/gap_detector.py`** (NEW)

```python
class KnowledgeGapDetector:
    """Identify what we don't know."""

    def detect_gaps(
        self,
        query: Query,
        retrieved: List[SemanticNode]
    ) -> List[KnowledgeGap]:
        gaps = []

        # Check for missing required slots
        for node in retrieved:
            gaps.extend(self.find_missing_slots(node))

        # Check for low-confidence assertions
        for node in retrieved:
            if node.confidence < 0.5:
                gaps.append(KnowledgeGap(
                    type=GapType.LOW_CONFIDENCE,
                    node_id=node.id,
                    description=f"Low confidence: {node.label}"
                ))

        # Check for stale knowledge
        for node in retrieved:
            if self.is_stale(node):
                gaps.append(KnowledgeGap(
                    type=GapType.STALE,
                    node_id=node.id
                ))

        return gaps
```

### 5.3 Self-Reflection Engine

**`backend/services/self_reflection.py`** (NEW)

```python
class SelfReflectionEngine:
    """Periodic self-examination."""

    def run_reflection_cycle(self) -> ReflectionReport:
        """Run at end of session or periodically."""
        report = ReflectionReport()

        # 1. Find contradictions
        report.contradictions = self.find_contradictions()

        # 2. Resolve contradictions
        for contradiction in report.contradictions:
            resolution = self.resolve_contradiction(contradiction)
            report.revisions.append(resolution)

        # 3. Recalibrate confidence
        for stale in self.find_stale_beliefs():
            report.confidence_updates.append(
                self.recalibrate(stale)
            )

        # 4. Learning insights
        report.learning_velocity = self.compute_learning_velocity()
        report.recommended_focus = self.recommend_learning_focus()

        return report
```

### 5.4 Learning Monitor

**Capabilities**:

1. **Knowledge Velocity**:
   - Nodes created per day
   - Quality of new knowledge
   - Integration ratio (edges per node)

2. **Learning Recommendations**:
   - Based on goals vs knowledge gaps
   - Priority = relevance / acquisition_cost

### 5.5 Files to Create

| File | Purpose |
|------|---------|
| `backend/services/epistemic_tracker.py` | Epistemic status tracking |
| `backend/services/gap_detector.py` | Knowledge gap detection |
| `backend/services/self_reflection.py` | Self-reflection engine |
| `backend/services/learning_monitor.py` | Learning metrics and recommendations |
| `backend/services/contradiction_resolver.py` | Belief revision logic |

### 5.6 Success Criteria - Phase 5

| Criteria | Test Method |
|----------|-------------|
| Gaps detected correctly | Create incomplete data, verify detection |
| Contradictions found | Create conflicting facts, verify detection |
| Contradictions resolved | Verify correct belief is retained |
| Learning velocity calculated | Create nodes over time, check metrics |
| Recommendations relevant | Verify recommendation matches gaps |

---

## Integration Testing Strategy

### End-to-End Test Scenarios

**Scenario 1: Memory Lifecycle**
```
1. Create episodic memory from conversation
2. Wait for consolidation cycle
3. Verify semantic node created
4. Access original episode
5. Verify access_count incremented
6. Simulate time passage
7. Verify decay applied
```

**Scenario 2: Inference Chain**
```
1. Create knowledge: Dog IS-A Animal, Animal HAS Heart
2. Query: "Does Dog have Heart?"
3. Verify inference chain created
4. Verify explanation generated
5. Verify confidence < 1.0
```

**Scenario 3: Temporal Reasoning**
```
1. Create facts with temporal validity
2. Query at different timestamps
3. Verify correct facts returned
4. Create temporal pattern
5. Verify prediction works
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data loss during migration | Medium | High | Automated backups, rollback scripts |
| Consolidation corrupts graph | Low | High | Snapshot before each cycle |
| Inference creates invalid knowledge | Medium | Medium | Confidence tracking, human review |
| Performance degradation at scale | Medium | Medium | Indexing, lazy evaluation |
| Breaking existing API | Low | High | Backward-compatible defaults |

---

## Dependency Order

```
Phase 1 (Foundation)
    ↓
Phase 2 (Consolidation) ←──── requires memory types, decay
    ↓
Phase 3 (Reasoning) ←──────── requires semantic memory, confidence
    ↓
Phase 4 (Temporal) ←───────── requires reasoning for prediction
    ↓
Phase 5 (Meta-Cognition) ←─── requires all above for self-awareness
```

Each phase must be completed and tested before the next begins.

---

## File Summary by Phase

### Phase 1 Files
- MODIFY: `backend/services/mind_graph.py`
- CREATE: `backend/services/memory_types.py`
- CREATE: `backend/services/memory_dynamics.py`
- CREATE: `backend/services/confidence_tracker.py`
- CREATE: `scripts/migrate_to_v2.py`
- CREATE: `backend/tests/test_memory_dynamics.py`

### Phase 2 Files
- CREATE: `backend/services/consolidation_engine.py`
- CREATE: `backend/services/pattern_extractor.py`
- CREATE: `backend/services/procedural_memory.py`
- CREATE: `backend/services/memory_archive.py`
- MODIFY: `backend/services/scheduler.py` (if exists, else CREATE)
- CREATE: `backend/tests/test_consolidation.py`

### Phase 3 Files
- CREATE: `backend/services/inference_engine.py`
- CREATE: `backend/services/deductive_reasoner.py`
- CREATE: `backend/services/analogical_reasoner.py`
- CREATE: `backend/services/inference_chain.py`
- MODIFY: `backend/services/semantic_index.py` (add spreading activation)
- CREATE: `backend/tests/test_inference.py`

### Phase 4 Files
- CREATE: `backend/services/temporal_graph.py`
- CREATE: `backend/services/temporal_relations.py`
- CREATE: `backend/services/event_sequence_miner.py`
- CREATE: `backend/services/temporal_predictor.py`
- CREATE: `backend/tests/test_temporal.py`

### Phase 5 Files
- CREATE: `backend/services/epistemic_tracker.py`
- CREATE: `backend/services/gap_detector.py`
- CREATE: `backend/services/self_reflection.py`
- CREATE: `backend/services/learning_monitor.py`
- CREATE: `backend/services/contradiction_resolver.py`
- CREATE: `backend/tests/test_metacognition.py`

---

## CLI Execution Commands

### Phase 1
```bash
# Create new files
claude --print "Create backend/services/memory_types.py with EpisodicMemory, SemanticMemory, ProceduralMemory classes"

claude --print "Create backend/services/memory_dynamics.py with MemoryDynamics class implementing Ebbinghaus decay"

claude --print "Create backend/services/confidence_tracker.py with Bayesian confidence updates"

# Modify existing
claude --print "Add new fields to MindNode in backend/services/mind_graph.py: confidence, decay_rate, last_accessed, access_count, encoding_strength, emotional_valence, temporal_validity, derived_from, consensus_count"

# Create migration
claude --print "Create scripts/migrate_to_v2.py to add new fields to existing mind_graph.json with safe defaults"

# Run migration
claude --print "Run python scripts/migrate_to_v2.py with backup"

# Test
claude --print "Create and run backend/tests/test_memory_dynamics.py"
```

### Phase 2
```bash
claude --print "Create backend/services/consolidation_engine.py with ConsolidationEngine class"

claude --print "Create backend/services/pattern_extractor.py for extracting patterns from episodic memories"

claude --print "Create backend/services/procedural_memory.py with ProceduralSkill class"

claude --print "Set up consolidation background job in scheduler"

claude --print "Create and run backend/tests/test_consolidation.py"
```

### Phase 3
```bash
claude --print "Create backend/services/inference_engine.py with InferenceChainManager"

claude --print "Create backend/services/deductive_reasoner.py with transitive and inheritance inference"

claude --print "Add spreading_activation method to backend/services/semantic_index.py"

claude --print "Create backend/services/analogical_reasoner.py with structure mapping"

claude --print "Create and run backend/tests/test_inference.py"
```

### Phase 4
```bash
claude --print "Create backend/services/temporal_graph.py with TemporalFact class"

claude --print "Create backend/services/temporal_relations.py implementing Allen's 13 interval relations"

claude --print "Create backend/services/event_sequence_miner.py with PrefixSpan algorithm"

claude --print "Create backend/services/temporal_predictor.py for future event prediction"

claude --print "Create and run backend/tests/test_temporal.py"
```

### Phase 5
```bash
claude --print "Create backend/services/epistemic_tracker.py with EpistemicStatus class"

claude --print "Create backend/services/gap_detector.py for knowledge gap detection"

claude --print "Create backend/services/self_reflection.py with SelfReflectionEngine"

claude --print "Create backend/services/learning_monitor.py for knowledge velocity tracking"

claude --print "Create and run backend/tests/test_metacognition.py"
```

---

## Version Control Strategy

```bash
# Before each phase
git checkout -b feature/memory-system-phase-N

# After each phase completion
git add .
git commit -m "feat(memory): Implement Phase N - [Description]"
git push origin feature/memory-system-phase-N

# Create PR for review before merging
gh pr create --title "Memory System Phase N" --body "..."
```

---

## Monitoring and Observability

Add logging throughout:
```python
import logging
logger = logging.getLogger(__name__)

# Key events to log:
logger.info(f"Consolidation cycle: {nodes_consolidated} nodes, {patterns_found} patterns")
logger.info(f"Inference created: {inference.type} with confidence {confidence:.2f}")
logger.warning(f"Contradiction detected: {node_a.id} vs {node_b.id}")
logger.info(f"Memory pruned: {node.id} (activation: {activation:.4f})")
```

---

*This implementation plan is designed for external execution via Claude Code CLI. Each phase is independently deployable and testable.*
