# Brainstorm Round 2: MVP Memory Architecture for 8GB Mac Mini

**Date**: 2026-01-06
**Based on**: Round 1 Architecture + Risk Analysis
**Constraint**: 8GB RAM Mac Mini - HARD LIMIT

---

## Executive Summary

Round 1 proposed an ideal architecture with graph databases, vector search, MCP protocols, and 5-tier memory. Round 2 strips this down to what actually works on 8GB RAM.

**REMOVED from Round 1:**
- Neo4j / Graph databases (4GB+ RAM minimum)
- Large embedding models (1B+ parameters)
- ColBERT/BGE reranking (GPU required)
- Full MCP inter-agent protocol (premature)
- RAPTOR recursive summarization (too expensive)
- Dedicated vector database (ChromaDB overkill)

**KEPT and REFINED:**
- SQLite FTS5 (already have it in memory_db.py)
- Filesystem-based tiers (already have memory/ structure)
- Existing Memory API (memory.py, memory_store.py, memory_db.py)
- Simple 3-tier memory (not 5)
- Python-native solutions only

---

## The 90% Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               MINIMAL VIABLE MEMORY ARCHITECTURE                                │
│                     (8GB Mac Mini Constrained)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                                   ┌───────────────────┐
                                   │   Chat Handler    │
                                   │   (WebSocket)     │
                                   └─────────┬─────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MEMORY COORDINATOR                                      │
│                                                                                  │
│     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐           │
│     │ Context Loader │     │ Memory Writer  │     │ Query Router   │           │
│     │ (load_*_context│     │ (update_*,     │     │ (BM25 or       │           │
│     │  methods)      │     │  log_decision) │     │  recency)      │           │
│     └────────────────┘     └────────────────┘     └────────────────┘           │
│                                                                                  │
│                         memory.py + memory_store.py                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                             │
           ┌─────────────────────────────────┼─────────────────────────────────┐
           │                                 │                                 │
           ▼                                 ▼                                 ▼
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│   TIER 1: CORE      │       │   TIER 2: WORKING   │       │   TIER 3: ARCHIVAL  │
│   (Always Loaded)   │       │   (Session Scoped)  │       │   (On-Demand)       │
├─────────────────────┤       ├─────────────────────┤       ├─────────────────────┤
│                     │       │                     │       │                     │
│  memory/core/       │       │  memory/sessions/   │       │  SQLite FTS5        │
│   ├─ vision.md      │       │   ├─ {id}.md        │       │   memory.db         │
│   ├─ priorities.md  │       │   └─ summaries/     │       │                     │
│   └─ decisions.md   │       │                     │       │  memory/swarms/     │
│                     │       │  logs/memory/       │       │   └─ {swarm}/       │
│  logs/memory/       │       │   └─ core_facts.json│       │       └─ history.md │
│   └─ core_facts.json│       │                     │       │                     │
│                     │       │                     │       │                     │
│  ~2K tokens         │       │  ~8K tokens         │       │  Unbounded          │
│  Updates: Rare      │       │  Updates: Per turn  │       │  Access: Search     │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
         │                             │                             │
         └─────────────────────────────┴─────────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │      SQLite FTS5        │
                          │  (Full-Text Search)     │
                          ├─────────────────────────┤
                          │  memory_fts table       │
                          │  - node_id              │
                          │  - node_type            │
                          │  - label                │
                          │  - description          │
                          │                         │
                          │  BM25 ranking built-in  │
                          │  ~10MB overhead         │
                          └─────────────────────────┘
```

---

## Memory Tier Details

### Tier 1: Core Memory (Filesystem)

**Purpose**: Agent identity, user facts, organizational context
**Storage**: Markdown files + JSON
**Size Budget**: ~2K tokens (~8KB)
**Update Frequency**: Rare (role changes, new preferences)

```
memory/
├── core/
│   ├── vision.md          # Org vision (CEO sets this)
│   ├── priorities.md      # Current priorities
│   └── decisions.md       # Decision log (append-only)
└── ...

logs/memory/
└── core_facts.json        # User facts (MemoryStore)
```

**Implementation**: Already exists in `memory.py` and `memory_store.py`

```python
# Existing API - no changes needed
memory_manager.load_coo_context()    # Full org context
memory_store.get_context_for_prompt() # User facts
```

### Tier 2: Working Memory (Session-Scoped)

**Purpose**: Current conversation context, active state
**Storage**: JSON files per session
**Size Budget**: ~8K tokens (~32KB)
**Update Frequency**: Every turn

```
memory/
├── sessions/
│   ├── {session_id}.md       # Session transcript
│   └── summaries/
│       └── {session_id}.md   # Compressed summary
└── ...
```

**Implementation**: Already exists in `memory.py`

```python
# Existing API - minimal changes
memory_manager.save_session_summary(session_id, summary)
memory_manager.load_session_summary(session_id)
memory_manager.get_context_with_summary(session_id, recent_messages)
```

### Tier 3: Archival Memory (SQLite FTS5)

**Purpose**: Searchable history, facts, episodes
**Storage**: SQLite database with FTS5 virtual table
**Size**: Unbounded (but ~10MB overhead for FTS)
**Access**: On-demand via BM25 search

**Implementation**: Already exists in `memory_db.py`

```python
# Existing schema in memory_db.py
# FTS5 table: memory_fts
# Columns: node_id, node_type, label, description

# Search API (to add)
def search_memory(query: str, limit: int = 5) -> list[dict]:
    """BM25 search across all semantic nodes."""
    return db.fetchall("""
        SELECT node_id, node_type, label, description,
               bm25(memory_fts) as rank
        FROM memory_fts
        WHERE memory_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query, limit))
```

---

## Retrieval Strategy

### Simple Two-Path Retrieval

```
Query ─────┬─────────────────────────────────────┐
           │                                     │
           ▼                                     ▼
    ┌─────────────┐                      ┌─────────────┐
    │  RECENCY    │                      │   BM25      │
    │  (Last N)   │                      │  (FTS5)     │
    └──────┬──────┘                      └──────┬──────┘
           │                                     │
           └─────────────┬───────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Interleave Results │
              │  - Recent first     │
              │  - Then by BM25     │
              └─────────────────────┘
                         │
                         ▼
                   Top-K Results
```

**Why no vectors?**
- Embedding models add 80MB+ RAM
- BM25 covers 80% of retrieval needs
- Can add vectors later if proven necessary

**Implementation**:

```python
def retrieve_context(query: str, session_id: str) -> str:
    """Retrieve relevant context for a query.

    Strategy:
    1. Always include core context (Tier 1)
    2. Include last 10 messages from session (Tier 2)
    3. BM25 search for relevant facts (Tier 3)
    """
    sections = []

    # Tier 1: Core (always)
    core = memory_store.get_context_for_prompt()
    if core:
        sections.append(core)

    # Tier 2: Recent messages (always)
    summary = memory_manager.load_session_summary(session_id)
    if summary:
        sections.append(f"## Previous Context\n{summary}")

    # Tier 3: Search if query looks like recall
    if is_recall_query(query):
        results = search_memory(query, limit=3)
        if results:
            sections.append("## Relevant Knowledge")
            for r in results:
                sections.append(f"- {r['label']}: {r['description']}")

    return "\n\n".join(sections)

def is_recall_query(query: str) -> bool:
    """Detect if query needs memory retrieval."""
    recall_patterns = [
        "what was", "what did", "remember when",
        "previous", "earlier", "last time",
        "you said", "we discussed", "decided"
    ]
    return any(p in query.lower() for p in recall_patterns)
```

---

## Memory Operations

### Write Path (Append-Only)

```
User Message ──▶ Extract Facts ──▶ Store to appropriate tier

Tier 1 (Core):
  - User preferences → memory_store.set_fact()
  - Decisions → memory_manager.log_decision()

Tier 2 (Working):
  - Message → append to session buffer
  - On threshold → summarize and archive

Tier 3 (Archival):
  - Facts → memory_db semantic_nodes table
  - FTS triggers auto-update memory_fts
```

### Read Path

```
Load Context Request
         │
         ├──▶ Core: memory_manager.load_*_context()
         │
         ├──▶ Working: memory_manager.get_context_with_summary()
         │
         └──▶ Archival: search_memory() via FTS5
                   │
                   ▼
              Merge & Return
```

---

## Resource Budget

| Component | RAM | Disk | Notes |
|-----------|-----|------|-------|
| SQLite engine | ~20 MB | - | Shared with OS |
| FTS5 index | ~10 MB | ~50 MB | Scales with content |
| Core memory files | ~1 MB | ~100 KB | Markdown + JSON |
| Session files | ~5 MB | ~1 MB | Per active session |
| Python memory.py | ~10 MB | - | Already running |
| **TOTAL** | **~46 MB** | **~51 MB** | Well under budget |

Compare to Round 1 proposal:
- ChromaDB alone: 500 MB - 2 GB RAM
- Neo4j: 4+ GB RAM
- Large embeddings: 2+ GB RAM

**Savings: ~2-4 GB RAM**

---

## What We're NOT Building

### Removed: Graph Database

**Round 1 proposed**: Neo4j-style entity relationships
**Round 2 removes**: Too expensive, cold start problem

**Alternative**: Simple edges table in SQLite (already in memory_db.py)

```sql
-- Already exists, just use it simply
SELECT target_id, edge_type
FROM edges
WHERE source_id = ? AND edge_type IN ('ASSOCIATION', 'REFERENCE')
```

### Removed: Vector Embeddings

**Round 1 proposed**: 384-dim embeddings with similarity search
**Round 2 removes**: Adds complexity, RAM, and drift issues

**Alternative**: FTS5 BM25 handles 80% of cases

**Migration path**: Add embeddings later if FTS5 proves insufficient

### Removed: ColBERT/BGE Reranking

**Round 1 proposed**: Neural reranking for top results
**Round 2 removes**: Requires GPU or significant CPU

**Alternative**: Trust BM25 ranking, limit to top-5

### Removed: MCP Protocol

**Round 1 proposed**: Full inter-agent context protocol
**Round 2 removes**: Premature optimization, standard still maturing

**Alternative**: Use existing filesystem-based coordination

### Removed: 5-Tier Memory

**Round 1 proposed**: Core, Working, Episodic, Semantic, Archival
**Round 2 simplifies to 3**: Core, Working, Archival

**Rationale**: Episodic and Semantic distinction is academic. Both go to archival.

---

## Implementation Path

### Phase 1: Wire Up Existing Code (Day 1)

Already have:
- `memory.py` - MemoryManager with load/save methods
- `memory_store.py` - MemoryStore with facts/preferences
- `memory_db.py` - SQLite with FTS5

Need to add:
1. `search_memory()` function using FTS5
2. `retrieve_context()` that combines all three tiers
3. Hook into chat handler

### Phase 2: Add Automatic Archival (Day 2)

Add:
1. Auto-extract facts from conversations
2. Store to semantic_nodes with FTS trigger
3. Session summarization on close

### Phase 3: Tune Retrieval (Day 3-5)

Add:
1. Recall query detection
2. Context budget enforcement
3. Confidence scoring on facts

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Memory layer RAM | <50 MB | `htop` during operation |
| FTS5 query latency | <50 ms | Time search_memory() |
| Context retrieval | <100 ms | Time retrieve_context() |
| Relevant results | 3/5 top results | Manual review |
| Session recovery | 100% | Restart and verify context |

---

## Anti-Patterns Avoided

1. **No external vector DB** - SQLite FTS5 is enough for now
2. **No graph database** - edges table handles simple relationships
3. **No embedding models** - BM25 first, add later if needed
4. **No MCP complexity** - filesystem coordination works
5. **No over-engineering** - 3 tiers not 5
6. **No future-proofing** - Build for today's 5 agents, not 100

---

## Decision Record

| Decision | Rationale |
|----------|-----------|
| Use FTS5 not vectors | 10MB vs 500MB+, BM25 is good enough |
| 3 tiers not 5 | Simpler to implement and maintain |
| No reranking | Trust BM25 ranking, limit top-K |
| Filesystem for tiers 1-2 | Already works, zero new dependencies |
| SQLite for tier 3 | Already have memory_db.py with FTS5 |
| Skip MCP | Standard is immature, defer 6+ months |

---

## Files to Modify

```
backend/
├── memory.py              # Add search_memory(), retrieve_context()
├── services/
│   ├── memory_store.py    # No changes needed
│   └── memory_db.py       # Add search wrapper methods
└── websocket/
    └── chat_handler.py    # Hook in retrieve_context()
```

**Lines of code**: ~100-150 new lines total

---

*Round 2 Complete - Architecture refined for 8GB reality*
