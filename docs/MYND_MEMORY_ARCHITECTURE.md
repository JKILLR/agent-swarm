# MYND Memory Architecture
## Complete Design Specification for 8GB M2 Mac Mini

**Version**: 1.0
**Date**: January 2025
**Hardware Target**: M2 Mac Mini with 8GB Unified Memory

---

## Table of Contents

1. [Executive Vision](#1-executive-vision)
2. [Hardware Constraints & Design Principles](#2-hardware-constraints--design-principles)
3. [Core Architecture: Tri-Memory System](#3-core-architecture-tri-memory-system)
4. [Storage Layer: SQLite-First Design](#4-storage-layer-sqlite-first-design)
5. [Knowledge Representation: The Semantic Lattice](#5-knowledge-representation-the-semantic-lattice)
6. [Active Reasoning Engine](#6-active-reasoning-engine)
7. [Temporal Intelligence](#7-temporal-intelligence)
8. [Meta-Cognition Layer](#8-meta-cognition-layer)
9. [Memory Budget Allocations](#9-memory-budget-allocations)
10. [Implementation Priority Order](#10-implementation-priority-order)

---

## 1. Executive Vision

### The Goal: A Mind, Not Just Memory

The current MindGraph is a **storage system that remembers**. We will transform it into a **cognitive system that thinks**.

**Current State**: Graph storage + semantic search + extraction
**Target State**: Living cognitive architecture with episodic replay, semantic consolidation, procedural learning, temporal reasoning, analogical inference, and metacognitive self-awareness

### Core Principle: Memory IS Intelligence

The most intelligent memory system doesn't just store and retrieve—it:
- **Dreams**: Consolidates experiences into abstract knowledge during "sleep" cycles
- **Forgets strategically**: Uses decay curves to maintain relevance
- **Infers**: Derives new knowledge from existing relationships
- **Predicts**: Projects future states from temporal patterns
- **Reflects**: Knows what it knows and doesn't know
- **Evolves**: Self-organizes into increasingly coherent structures

---

## 2. Hardware Constraints & Design Principles

### The Challenge: 8GB M2 Mac Mini

On an **8GB M2 Mac Mini**, we must assume:
- ~4GB available for our application (OS and other processes take the rest)
- ~2GB safe working memory for the memory system
- Embedding model itself takes ~500MB
- We need headroom for spikes and concurrent operations

### What We CAN'T Do on 8GB
- Large in-memory graphs (millions of nodes)
- Heavy embedding models running locally
- Multiple concurrent ML processes
- Keep everything loaded at once

### Our Approach: Disk-First, Memory-Cached

| Component | Visionary Approach | 8GB Implementation |
|-----------|-------------------|-------------------|
| **Node Storage** | In-memory dict | SQLite with LRU cache |
| **Embeddings** | In-memory NumPy array | SQLite BLOB + mmap index |
| **Graph Edges** | In-memory adjacency | SQLite adjacency table |
| **Text Search** | Linear scan | SQLite FTS5 |
| **Working Memory** | Unbounded | Hard limit: 100 items |
| **Embedding Cache** | Full dataset | LRU cache: 50MB max |
| **Activation Spread** | Full graph | Bounded: 3 hops, 50 nodes |

### Key Design Principles

1. **SQLite is the source of truth** - Everything persists to SQLite tables
2. **Lazy loading everywhere** - Never load what you don't need
3. **Strict memory budgets** - Every cache has a hard limit
4. **Batch API calls** - Never embed one item when you can batch
5. **Generator patterns** - Yield results, don't collect them
6. **Cursor-based pagination** - No full dataset scans

---

## 3. Core Architecture: Tri-Memory System

### Inspired by Human Cognition + ACT-R + SOAR

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WORKING MEMORY                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Current Context Buffer (limited capacity ~7±2 concepts)          │   │
│  │ Active Goals Stack • Current Reasoning Chain • Attention Focus  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲ ▼                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                       LONG-TERM MEMORY                                   │
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────────┐ │
│  │  EPISODIC MEMORY  │ │  SEMANTIC MEMORY  │ │  PROCEDURAL MEMORY    │ │
│  │                   │ │                   │ │                       │ │
│  │ "What happened"   │ │ "What I know"     │ │ "How to do things"    │ │
│  │                   │ │                   │ │                       │ │
│  │ • Conversation    │ │ • Facts           │ │ • Skill templates     │ │
│  │   episodes        │ │ • Concepts        │ │ • Tool use patterns   │ │
│  │ • Event sequences │ │ • Identities      │ │ • Reasoning chains    │ │
│  │ • Temporal tags   │ │ • Relationships   │ │ • Error corrections   │ │
│  │ • Emotional       │ │ • Schemas         │ │ • Successful          │ │
│  │   salience        │ │ • Abstractions    │ │   strategies          │ │
│  │                   │ │                   │ │                       │ │
│  │ Decays over time  │ │ Consolidated      │ │ Reinforced by use     │ │
│  │ unless reinforced │ │ from episodes     │ │ Compiled for speed    │ │
│  └───────────────────┘ └───────────────────┘ └───────────────────────┘ │
│                              ▲                                          │
│                     CONSOLIDATION                                       │
│                     (sleep cycles)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Episodic Memory

**Purpose**: Store autobiographical experiences with temporal context

Core attributes:
- `episode_id`, `timestamp`, `duration`
- `summary`, `compressed_transcript`, `key_moments`
- Context binding: `spatial_context`, `social_context`
- Emotional tags: `emotional_valence` (-1 to +1), `arousal_level` (0 to 1)
- Memory strength: `encoding_strength`, `retrieval_count`, `decay_rate`
- Consolidation state: `is_consolidated`, `extracted_facts`

**Key Operations**:
- **Encoding**: Automatic capture of conversation episodes with emotional tagging
- **Retrieval**: Context-dependent recall with spreading activation
- **Replay**: Mental simulation during consolidation
- **Decay**: Exponential forgetting (Ebbinghaus curve) unless reinforced

### 3.2 Semantic Memory

**Purpose**: Store factual knowledge abstracted from episodes

Core attributes:
- Identity: `node_id`, `node_type` (CONCEPT, FACT, ENTITY, SCHEMA, FRAME)
- Content: `label`, `description`, `formal_definition`
- Relationships: `isa`, `has_part`, `causes`, `enables`, `contradicts`, `similar_to`
- Provenance: `derived_from_episodes`, `confidence`, `consensus_count`
- Activation: `base_level_activation`, `spreading_activation`
- Frame slots: `slots`, `constraints`

**Key Operations**:
- **Abstraction**: Generalize from specific episodes to general facts
- **Integration**: Merge new information with existing knowledge
- **Spreading Activation**: Activate related concepts based on context
- **Inheritance**: Derive properties through IS-A hierarchies

### 3.3 Procedural Memory

**Purpose**: Store "how-to" knowledge as executable skills

Core attributes:
- Identity: `skill_id`, `name`, `description`
- Trigger: `trigger_pattern`, `preconditions`, `goal_relevance`
- Execution: `steps`, `decision_points`, `tool_bindings`
- Performance: `success_rate`, `avg_execution_time`, `failure_modes`
- Learning: `learned_from`, `is_compiled`, `chunking_level`

**Key Operations**:
- **Learning**: Extract action patterns from successful episodes
- **Chunking**: Compile multi-step sequences into single units
- **Generalization**: Abstract skills to apply in new contexts
- **Error-driven learning**: Refine based on failures

---

## 4. Storage Layer: SQLite-First Design

### Database Schema Overview

All memory components share a single SQLite database with FTS5 for full-text search.

```sql
-- Core Tables
episodic_memories      -- Conversation episodes
semantic_nodes         -- Facts, concepts, entities
procedural_skills      -- Skills and action patterns
edges                  -- Graph relationships
embeddings             -- 384-dim vectors as BLOBs
working_memory         -- Transient attention buffer
retrieval_history      -- For decay calculations
memory_fts             -- FTS5 virtual table for text search
```

### Key Table: episodic_memories

```sql
CREATE TABLE episodic_memories (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    duration_seconds INTEGER,
    summary TEXT NOT NULL,
    compressed_transcript BLOB,  -- gzip compressed
    spatial_context TEXT,
    social_context TEXT,   -- JSON array
    emotional_valence REAL DEFAULT 0.0,
    arousal_level REAL DEFAULT 0.0,
    encoding_strength REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved DATETIME,
    decay_rate REAL DEFAULT 0.1,
    is_consolidated BOOLEAN DEFAULT FALSE
);
```

### Key Table: semantic_nodes

```sql
CREATE TABLE semantic_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,  -- CONCEPT, FACT, ENTITY, SCHEMA, FRAME
    label TEXT NOT NULL,
    description TEXT,
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    source_reliability REAL DEFAULT 0.5,
    base_level_activation REAL DEFAULT 0.0,
    last_access DATETIME,
    access_count INTEGER DEFAULT 0,
    derived_from_episodes TEXT,  -- JSON array
    slots TEXT  -- JSON for frame semantics
);
```

### Key Table: embeddings

```sql
CREATE TABLE embeddings (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- 384-dim float32 = 1536 bytes
    model_version TEXT DEFAULT 'all-MiniLM-L6-v2'
);
```

### Full-Text Search (FTS5)

```sql
CREATE VIRTUAL TABLE memory_fts USING fts5(
    node_id, node_type, label, description,
    content='semantic_nodes'
);
-- With triggers to keep FTS in sync
```

### Connection Management

- WAL mode for concurrent reads
- Thread-local connections with pooling
- 64MB page cache
- Context managers for transactions

---

## 5. Knowledge Representation: The Semantic Lattice

### Multi-Layer Representation

```
Layer 5: ABSTRACT SCHEMAS
  • Meta-patterns (causation, containment, sequence)
  • Conceptual metaphors (argument is war, time is money)
  • Reasoning templates (if-then, compare-contrast)
              ▲
Layer 4: SCRIPTS & FRAMES
  • Restaurant script (enter, seat, order, eat, pay, leave)
  • Debug script (reproduce, isolate, hypothesize, fix, verify)
  • Project frame (owner, stack, goals, constraints)
              ▲
Layer 3: CONCEPTUAL GRAPHS
  • Typed relationships with semantic roles
  • First-order logic compatible
  • Supports inference and unification
              ▲
Layer 2: ENTITIES & RELATIONS
  • Named entities (John, Python, React)
  • Explicit relationships (knows, uses, creates)
  • Properties and attributes
              ▲
Layer 1: RAW FACTS
  • Atomic propositions
  • Direct observations
  • Unstructured memories
```

### Frame System

Minsky-style frames for structured situation representation:
- Slots with defaults and constraints
- Inheritance hierarchy
- Procedural attachments (if_needed, if_added, if_removed)

### Script System

Schank-style scripts for stereotypical event sequences:
- Entry conditions and roles
- Scene sequences with branches
- Expected outcomes and exceptions

---

## 6. Active Reasoning Engine

### Beyond Retrieval: Generative Knowledge

```
                    INFERENCE ORCHESTRATOR
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │  DEDUCTIVE  │   │  INDUCTIVE  │   │  ABDUCTIVE  │
   │  REASONING  │   │  REASONING  │   │  REASONING  │
   │             │   │             │   │             │
   │ From general│   │ From specific│  │ Best        │
   │ to specific │   │ to general  │   │ explanation │
   └─────────────┘   └─────────────┘   └─────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                 ANALOGICAL REASONER
                 (Structure Mapping)
                           │
                           ▼
               COUNTERFACTUAL SIMULATOR
               ("What if X instead?")
```

### Deductive Reasoning
- Modus ponens, transitive inference
- Inheritance through class hierarchies
- Syllogistic reasoning

### Analogical Reasoning (Structure Mapping)
- Find structural correspondences between domains
- Transfer inferences across domains
- Based on Gentner's Structure Mapping Theory

### Counterfactual Reasoning
- Causal model manipulation
- "What would have happened if X?"
- Pearl's do-calculus semantics

### Inference Chain Management
- Multi-step reasoning with backtracking
- Beam search with multiple strategies
- Natural language explanation generation

---

## 7. Temporal Intelligence

### Time-Aware Memory and Prediction

```
           TEMPORAL GRAPH LAYER
     (Facts with validity intervals)
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
PAST UNDERSTANDING            FUTURE PROJECTION
• Event sequence mining       • Trend extrapolation
• Causal chain discovery      • Goal achievement ETA
• Pattern recognition         • Risk anticipation
    │                               │
    └───────────────┬───────────────┘
                    ▼
          DECAY & REINFORCEMENT
   Memory Strength = base × Σ(recency × importance)
   Ebbinghaus decay: R = e^(-t/S)
```

### Allen's 13 Interval Relations
BEFORE, AFTER, MEETS, OVERLAPS, STARTS, DURING, FINISHES, EQUALS, etc.

### Event Sequence Mining
- Extract frequent temporal patterns from episodes
- Modified PrefixSpan algorithm for temporal sequences
- Predict likely future events based on patterns

### Memory Decay (Ebbinghaus + ACT-R)

```python
# ACT-R style activation
Base-level activation = ln(Σ t_i^(-d))
# where t_i = time since i-th retrieval, d = decay rate

# Retrieval probability
P(retrieval) = 1 / (1 + e^(-activation))

# Forget if retrieval probability < 0.1
```

---

## 8. Meta-Cognition Layer

### The System That Thinks About Thinking

```
        CONFIDENCE TRACKING
  (Every assertion: P(true), sources, evidence)
                    │
        KNOWLEDGE GAP DETECTION
  ("I don't know", partial knowledge, missing slots)
                    │
        SOURCE RELIABILITY
  (Track accuracy over time, calibrate trust)
                    │
        SELF-REFLECTION ENGINE
  (Consistency check, contradiction detection)
                    │
        LEARNING MONITOR
  (What am I learning? What should I learn?)
```

### Epistemic Status Tracking

Every assertion carries:
- `confidence`: P(true) = 0.0 to 1.0
- `confidence_interval`: Uncertainty bounds
- `evidence_strength`: How strong is supporting evidence
- `evidence_count`: Number of supporting observations
- `source_reliability`: Historical accuracy of source
- `temporal_decay`: How much has confidence decayed
- `consistency_score`: Coherence with other beliefs
- `contradictions`: Known contradicting assertions

### Knowledge Gap Detection

Detect:
- Low confidence assertions (< 40%)
- Missing required frame slots
- Stale information (not accessed in 90+ days)
- No evidence (assertions without supporting data)

### Self-Reflection Cycles

Run periodically (end of session, nightly):
1. Find contradictions
2. Resolve via belief revision
3. Recalibrate stale confidences
4. Consolidate episodes → semantic
5. Refine procedural skills
6. Extract meta-learning insights

---

## 9. Memory Budget Allocations

### Summary Table

| Component | Max RAM | Purpose |
|-----------|---------|---------|
| **Embedding Model** | 500 MB | all-MiniLM-L6-v2, loaded once |
| **Embedding Cache** | 50 MB | LRU cache (~32K embeddings) |
| **Working Memory** | 10 MB | 100 items hard limit |
| **SQLite Cache** | 64 MB | Page cache (PRAGMA cache_size) |
| **Python Heap** | 200 MB | Generators, temp objects |
| **Batch Buffers** | 50 MB | Peak during embedding batches |
| **Reserved** | 126 MB | Headroom for spikes |
| **TOTAL** | ~1 GB | Leaves ~3GB for OS/other |

### Hard Limits Enforced

| System | Limit | Enforced By |
|--------|-------|-------------|
| Working Memory | 100 items | `CHECK (slot_id BETWEEN 0 AND 99)` |
| Embedding Cache | 50MB / 32K items | LRU eviction |
| Spreading Activation | 3 hops, 50 nodes max | BFS with counters |
| Batch Size | 32 items | Service constant |
| Rate Limit | 100 req/min | Token bucket |

### Expected Performance

| Operation | Time | Memory Peak |
|-----------|------|-------------|
| Single embedding | 10-50ms | ~2MB |
| Batch embedding (32) | 100-200ms | ~20MB |
| FTS5 search | 1-5ms | ~1MB |
| Semantic search (1K) | 50-100ms | ~5MB |
| Semantic search (10K) | 200-500ms | ~20MB |
| Working memory activate | <1ms | Negligible |
| Spreading activation | 10-50ms | ~5MB |
| Consolidation cycle | 1-5s | ~50MB |

---

## 10. Implementation Priority Order

### Phase 1: Foundation (Week 1-2)
**Goal**: SQLite storage layer working

1. `memory_db.py` - Connection management, schema initialization
2. `semantic_memory.py` - Basic CRUD operations
3. `episodic_memory.py` - Episode storage and retrieval
4. `embedding_store.py` - SQLite BLOB storage for embeddings

**Verification**: Store and retrieve 100 nodes with embeddings

### Phase 2: Search (Week 3)
**Goal**: Both FTS5 and semantic search working

1. Add FTS5 triggers to schema
2. Implement `search_fts()` in semantic memory
3. Implement `search_similar()` in embedding store
4. Add batched embedding service

**Verification**: Search queries return relevant results

### Phase 3: Working Memory (Week 4)
**Goal**: Bounded working memory with activation

1. `working_memory.py` - Full implementation
2. Integration with semantic/episodic stores
3. Decay timer (background task)

**Verification**: 100-item limit enforced, eviction works

### Phase 4: Spreading Activation (Week 5)
**Goal**: Bounded graph traversal

1. `spreading_activation.py` - Full implementation
2. Integration with working memory
3. Test with dense graph sections

**Verification**: MAX_NODES=50 limit enforced

### Phase 5: Memory Dynamics (Week 6-7)
**Goal**: Decay and consolidation working

1. `memory_decay.py` - Ebbinghaus curves
2. `consolidation_service.py` - Basic pattern extraction
3. Background task scheduler

**Verification**: Old memories decay, episodes consolidate

### Phase 6: Meta-Cognition (Week 8)
**Goal**: Confidence and gap detection

1. `confidence_tracker.py` - Bayesian updates
2. `gap_detector.py` - Low confidence, stale detection
3. Integration with retrieval (boost low-confidence warnings)

**Verification**: Low-confidence items flagged

### Phase 7: Integration (Week 9-10)
**Goal**: Full system working together

1. Memory coordinator service
2. API endpoints for all operations
3. Memory monitoring dashboard
4. Performance testing and tuning

**Verification**: End-to-end conversation processing

---

## Summary: The Vision

| Capability | Current State | Target State |
|------------|--------------|--------------|
| **Memory Types** | Single "memory" | Episodic + Semantic + Procedural |
| **Knowledge Rep** | Simple graph | Frames + Scripts + Conceptual Graphs |
| **Inference** | None | Deductive + Inductive + Analogical |
| **Temporal** | Timestamps only | Temporal graph + prediction |
| **Confidence** | None | Full uncertainty quantification |
| **Evolution** | Static | Consolidation + decay + learning |
| **Meta-cognition** | None | Gap detection + self-reflection |
| **Scale** | ~100 nodes | 100K+ nodes (with disk-backed storage) |

**The result**: Not just a memory that stores—but a **mind that thinks**.

---

## References

### Cognitive Architectures
- ACT-R Cognitive Architecture
- SOAR Architecture
- Common Model of Cognition

### Memory Systems
- A-Mem: Agentic Memory
- Mem0: Production-Ready Memory
- Memory in Agentic AI Systems

### Reasoning & Knowledge Graphs
- Gentner's Structure Mapping Theory
- Pearl's Causal Inference
- Allen's Temporal Interval Algebra
