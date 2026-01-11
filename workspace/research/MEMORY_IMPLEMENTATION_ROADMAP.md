# Memory System Implementation Roadmap

**Date**: 2026-01-06
**Status**: FINAL DELIVERABLE
**Constraint**: 8GB RAM Mac Mini
**Total Estimated Lines of Code**: ~400-500 new lines

---

## Executive Summary

This roadmap synthesizes Round 1 (Quick Wins, Architecture, Risks) and Round 2 (Synthesis, MVP Architecture) into an actionable implementation plan. The guiding principle: **90% of value at 10% of complexity** using SQLite FTS5 + filesystem + minimal Python.

### Core Decision: What We're Building

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       3-TIER MEMORY ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────────────┤
│  TIER 1: CORE         │  TIER 2: WORKING      │  TIER 3: ARCHIVAL         │
│  (Always Loaded)      │  (Session Scoped)     │  (On-Demand Search)       │
│                       │                       │                           │
│  memory/core/         │  memory/sessions/     │  SQLite FTS5              │
│  ~2K tokens           │  ~8K tokens           │  Unbounded                │
│  Updates: Rare        │  Updates: Per turn    │  Access: BM25 search      │
└────────────────────────────────────────────────────────────────────────────┘
```

### What We're NOT Building

| Eliminated | Reason |
|------------|--------|
| Neo4j/Graph DB | 4GB+ RAM minimum |
| ChromaDB/Weaviate | 500MB-2GB RAM per index |
| Large embedding models | Can't fit alongside Ollama |
| MCP Protocol | Standard still maturing |
| 5-tier memory | Simplified to 3 tiers |

---

## Phase 1: This Week Quick Wins

**Focus**: Prevent context bloat, enable cross-session memory
**Total RAM Cost**: ~30MB
**Total LOC**: ~150-200 lines

---

### 1.1 STATE.md Hierarchical Compression

**Priority**: 1 (Highest)
**Effort**: 2/5 | **Impact**: 5/5 | **RAM**: ~0MB

#### What to Build
Create a compression tool that maintains a "sliding window" of detailed entries (last 10) while summarizing older entries into weekly digests.

#### Files to Modify/Create

| File | Action | Lines |
|------|--------|-------|
| `tools/compress_state.py` | **CREATE** | ~80 |
| `swarms/swarm_dev/workspace/STATE.md` | MODIFY template | ~10 |
| `memory/archives/` | CREATE directory | - |

#### Implementation Details

```python
# tools/compress_state.py - NEW FILE
"""
STATE.md Hierarchical Compression Tool

Functions:
- parse_state_md(path) -> List[Entry]
- group_by_week(entries) -> Dict[str, List[Entry]]
- generate_weekly_summary(entries) -> str
- compress_state(state_path, archive_dir)
- main() - CLI interface
"""
```

#### Algorithm
1. Parse STATE.md for `## Progress Log` entries
2. Keep entries from last 7 days verbatim
3. Group older entries by week
4. Generate 2-3 sentence summaries per week
5. Move detailed entries to `memory/archives/state_YYYY_WW.md`
6. Update STATE.md with compressed version

#### Assignable Agent
- **Primary**: `implementer` agent
- **Review**: `critic` agent (validate no data loss)

#### Dependencies
- None (uses Python stdlib only)

#### Success Metric
- STATE.md token reduction: **≥50%**
- Archive files created with full history

---

### 1.2 Bounded Memory Collections

**Priority**: 2
**Effort**: 1/5 | **Impact**: 3/5 | **RAM**: ~0MB

#### What to Build
Add size limits to memory collections with automatic rotation of oldest entries.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `backend/services/memory_store.py` | MODIFY | ~30 |
| `memory/archives/facts/` | CREATE directory | - |

#### Implementation Details

```python
# backend/services/memory_store.py - ADD CONSTANTS
MAX_FACTS = 200
MAX_SESSIONS = 100

# ADD METHOD
def _enforce_bounds(self):
    """Evict oldest facts when limit exceeded."""
    if len(self.facts) > MAX_FACTS:
        sorted_facts = sorted(self.facts.items(), key=lambda x: x[1].get('updated_at', 0))
        evicted = sorted_facts[:len(self.facts) - MAX_FACTS]
        self._archive_facts(evicted)
        for key, _ in evicted:
            del self.facts[key]

def _archive_facts(self, facts: list):
    """Archive evicted facts to memory/archives/facts/"""
    pass
```

#### Assignable Agent
- **Primary**: `implementer` agent

#### Dependencies
- None

#### Success Metric
- Memory files stable at **<50KB** after 100 sessions
- Zero data loss (archived before eviction)

---

### 1.3 Session Memory Auto-Load

**Priority**: 3
**Effort**: 2/5 | **Impact**: 4/5 | **RAM**: ~10-20MB (cache)

#### What to Build
Automatically load relevant session summaries on COO startup based on recency and keyword relevance.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `backend/services/memory_store.py` | MODIFY | ~40 |
| `backend/memory.py` | MODIFY | ~20 |
| `backend/websocket/chat_handler.py` | MODIFY | ~10 |

#### Implementation Details

```python
# backend/services/memory_store.py - ADD METHOD
def load_recent_sessions(self, limit: int = 5, days: int = 7) -> List[dict]:
    """
    Load most recent session summaries.

    Returns sessions from last N days, limited to most recent K.
    """
    sessions_dir = Path("memory/sessions")
    summaries = []
    cutoff = datetime.now() - timedelta(days=days)

    for session_file in sessions_dir.glob("*.md"):
        mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
        if mtime > cutoff:
            summaries.append({
                "path": session_file,
                "mtime": mtime,
                "content": self._extract_summary(session_file)
            })

    return sorted(summaries, key=lambda x: x["mtime"], reverse=True)[:limit]
```

```python
# backend/memory.py - MODIFY build_coo_system_prompt()
def build_coo_system_prompt(self, session_id: str = None) -> str:
    """Include recent session context in system prompt."""
    base_prompt = self._get_base_prompt()

    # NEW: Load recent sessions
    recent = self.memory_store.load_recent_sessions(limit=5)
    if recent:
        session_context = self._format_session_context(recent)
        base_prompt += f"\n\n## Previous Sessions\n{session_context}"

    return base_prompt
```

#### Assignable Agent
- **Primary**: `implementer` agent
- **Research**: `researcher` agent (if unclear on existing session format)

#### Dependencies
- 1.2 Bounded Memory (recommended to complete first)

#### Success Metric
- **3+ relevant facts** loaded per session
- Startup time increase: **<500ms**

---

## Phase 2: Near-Term Architecture

**Focus**: Quality gates, unified access, search infrastructure
**Total RAM Cost**: ~20MB additional
**Total LOC**: ~150-200 lines

---

### 2.1 Confidence Scoring for Escalations

**Priority**: 4
**Effort**: 2/5 | **Impact**: 4/5 | **RAM**: ~0MB

#### What to Build
Add 0-1 confidence score to all escalation decisions with configurable auto-escalation threshold.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `shared/escalation_protocol.py` | MODIFY | ~30 |
| `backend/routes/escalations.py` | MODIFY | ~15 |

#### Implementation Details

```python
# shared/escalation_protocol.py - MODIFY Escalation dataclass
@dataclass
class Escalation:
    id: str
    level: EscalationLevel
    reason: EscalationReason
    source_agent: str
    target_agent: str
    message: str
    context: Dict[str, Any]
    confidence: float = 1.0  # NEW: 0.0-1.0
    created_at: datetime = field(default_factory=datetime.utcnow)

# ADD CONSTANTS
CONFIDENCE_THRESHOLD = 0.8
AUTO_ESCALATE_THRESHOLD = 0.6

# MODIFY escalate_to_coo()
def escalate_to_coo(
    self,
    reason: EscalationReason,
    message: str,
    context: Dict[str, Any],
    confidence: float = 1.0  # NEW PARAMETER
) -> Escalation:
    """Create escalation with confidence scoring."""
    if confidence < AUTO_ESCALATE_THRESHOLD:
        reason = EscalationReason.UNCERTAINTY
        message = f"[LOW CONFIDENCE: {confidence:.2f}] {message}"

    # ... rest of implementation
```

#### Assignable Agent
- **Primary**: `implementer` agent

#### Dependencies
- None

#### Success Metric
- **100%** of escalations have confidence scores
- Auto-escalation triggers at <0.6 confidence

---

### 2.2 Unified Context Endpoint

**Priority**: 5
**Effort**: 3/5 | **Impact**: 4/5 | **RAM**: ~0MB

#### What to Build
Single `/api/context` endpoint that aggregates all context sources for agent consumption.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `backend/routes/context.py` | MODIFY (or CREATE if minimal) | ~60 |
| `backend/main.py` | MODIFY (add route) | ~5 |

#### Implementation Details

```python
# backend/routes/context.py - ADD/MODIFY
from fastapi import APIRouter
from backend.services.memory_store import MemoryStore
from shared.escalation_protocol import EscalationManager
from shared.work_ledger import WorkLedger
from shared.agent_mailbox import AgentMailbox

router = APIRouter(prefix="/api/context", tags=["context"])

@router.get("")
async def get_unified_context(
    agent_id: str = "coo",
    include_escalations: bool = True,
    include_work: bool = True,
    include_messages: bool = True
) -> dict:
    """
    Unified context endpoint - single API call for all context.

    Returns:
    {
        "facts": [...],           # User facts from MemoryStore
        "escalations": [...],     # Pending escalations
        "active_work": [...],     # Current work items
        "messages": [...],        # Pending mailbox messages
        "sessions": [...]         # Recent session summaries
    }
    """
    context = {
        "facts": memory_store.get_context_for_prompt(),
        "sessions": memory_store.load_recent_sessions(limit=3)
    }

    if include_escalations:
        context["escalations"] = escalation_manager.get_pending()

    if include_work:
        context["active_work"] = work_ledger.get_active_work()

    if include_messages:
        context["messages"] = agent_mailbox.get_pending_for(agent_id)

    return context

@router.get("/formatted")
async def get_formatted_context(agent_id: str = "coo") -> str:
    """Return context as formatted string for prompt injection."""
    ctx = await get_unified_context(agent_id)
    return format_context_for_prompt(ctx)
```

#### Assignable Agent
- **Primary**: `implementer` agent
- **Review**: `architect` agent (validate API design)

#### Dependencies
- 1.3 Session Memory Auto-Load (for sessions field)

#### Success Metric
- Single API call for all context
- Response time: **<500ms**

---

### 2.3 SQLite FTS5 Search Integration

**Priority**: 6
**Effort**: 3/5 | **Impact**: 3/5 | **RAM**: ~10MB

#### What to Build
Wire up existing FTS5 infrastructure for BM25-based memory search.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `backend/services/memory_db.py` | MODIFY | ~40 |
| `backend/memory.py` | MODIFY | ~30 |

#### Implementation Details

```python
# backend/services/memory_db.py - ADD METHODS
def search_memory(self, query: str, limit: int = 5) -> list[dict]:
    """
    BM25 search across all semantic nodes.

    Uses existing memory_fts FTS5 virtual table.
    """
    cursor = self.conn.execute("""
        SELECT node_id, node_type, label, description,
               bm25(memory_fts) as rank
        FROM memory_fts
        WHERE memory_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query, limit))

    return [
        {
            "node_id": row[0],
            "node_type": row[1],
            "label": row[2],
            "description": row[3],
            "rank": row[4]
        }
        for row in cursor.fetchall()
    ]

def is_recall_query(self, query: str) -> bool:
    """Detect if query needs memory retrieval."""
    recall_patterns = [
        "what was", "what did", "remember when",
        "previous", "earlier", "last time",
        "you said", "we discussed", "decided"
    ]
    return any(p in query.lower() for p in recall_patterns)
```

```python
# backend/memory.py - ADD METHOD
def retrieve_context(self, query: str, session_id: str) -> str:
    """
    Retrieve relevant context for a query.

    Strategy:
    1. Always include core context (Tier 1)
    2. Include session context (Tier 2)
    3. BM25 search for relevant facts (Tier 3) - only if recall query
    """
    sections = []

    # Tier 1: Core (always)
    core = self.memory_store.get_context_for_prompt()
    if core:
        sections.append(core)

    # Tier 2: Session context
    summary = self.load_session_summary(session_id)
    if summary:
        sections.append(f"## Previous Context\n{summary}")

    # Tier 3: Search if recall query
    if self.memory_db.is_recall_query(query):
        results = self.memory_db.search_memory(query, limit=3)
        if results:
            sections.append("## Relevant Knowledge")
            for r in results:
                sections.append(f"- {r['label']}: {r['description']}")

    return "\n\n".join(sections)
```

#### Assignable Agent
- **Primary**: `implementer` agent
- **Testing**: `tester` agent (search quality validation)

#### Dependencies
- Existing `memory_db.py` with FTS5 table (already exists)

#### Success Metric
- Search latency: **<100ms**
- Relevant results in **top-5**

---

## Phase 3: Future Enhancements

**Focus**: Only implement if Phase 1-2 prove insufficient
**Trigger**: Retrieval accuracy measured <70%

---

### 3.1 Small Embedding Model (Conditional)

**Priority**: 7 (Deferred)
**Effort**: 3/5 | **Impact**: 3/5 | **RAM**: ~80MB

#### Trigger Condition
- FTS5/BM25 retrieval accuracy <70%
- Semantic similarity needed (synonyms, paraphrasing)

#### What to Build
Add sentence-transformers MiniLM for semantic search fallback.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `backend/services/embedding_service.py` | MODIFY | ~50 |
| `backend/services/memory_db.py` | MODIFY (add vector column) | ~30 |
| `requirements.txt` | MODIFY (add sentence-transformers) | ~2 |

#### Implementation Details

```python
# backend/services/embedding_service.py
from sentence_transformers import SentenceTransformer

# ONLY load if needed
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB
    return _model

def embed_text(text: str) -> list[float]:
    """Generate 384-dim embedding for text."""
    model = get_embedding_model()
    return model.encode(text).tolist()
```

#### Assignable Agent
- **Primary**: `implementer` agent
- **Review**: `architect` agent (memory budget validation)

#### Dependencies
- 2.3 SQLite FTS5 (must prove insufficient first)

#### Risk Mitigation
- Lazy load model (don't load at startup)
- Monitor RAM with `htop` after loading
- Version pin sentence-transformers to prevent drift

---

### 3.2 Meta-Learning Source Tracking (Deferred)

**Priority**: 8 (Deferred)
**Effort**: 3/5 | **Impact**: 3/5 | **RAM**: ~0MB

#### Trigger Condition
- Core memory system working well
- Need to optimize context inclusion

#### What to Build
Track which knowledge sources lead to successful outcomes.

#### Files to Modify

| File | Action | Lines |
|------|--------|-------|
| `backend/services/memory_store.py` | MODIFY | ~40 |
| `backend/routes/context.py` | MODIFY (add stats endpoint) | ~20 |

#### Implementation Details

```python
# backend/services/memory_store.py - ADD
source_effectiveness: Dict[str, Dict[str, int]] = {}
# Format: {"source_id": {"successes": 5, "uses": 10}}

def record_source_use(self, source_id: str, success: bool):
    """Track source effectiveness."""
    if source_id not in self.source_effectiveness:
        self.source_effectiveness[source_id] = {"successes": 0, "uses": 0}

    self.source_effectiveness[source_id]["uses"] += 1
    if success:
        self.source_effectiveness[source_id]["successes"] += 1

def get_effectiveness_score(self, source_id: str) -> float:
    """Return success rate for source."""
    stats = self.source_effectiveness.get(source_id, {})
    uses = stats.get("uses", 0)
    if uses == 0:
        return 0.5  # Default neutral score
    return stats.get("successes", 0) / uses
```

#### Assignable Agent
- **Primary**: `implementer` agent

#### Dependencies
- Phase 1-2 complete

---

### 3.3 Agent Self-Awareness Document (Deferred)

**Priority**: 9 (Low)
**Effort**: 2/5 | **Impact**: 3/5 | **RAM**: ~0MB

#### Trigger Condition
- Agents hallucinating capabilities
- Need clearer self-delegation

#### What to Build
Auto-generate capability manifest from agent definitions.

#### Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `tools/generate_self_awareness.py` | **CREATE** | ~60 |
| `memory/agents/{agent}_self.md` | **CREATE** (generated) | - |

#### Assignable Agent
- **Primary**: `implementer` agent

#### Dependencies
- None (but low priority)

---

## Implementation Matrix

### Phase 1 Summary

| Item | Files | Agent | Dependencies | LOC |
|------|-------|-------|--------------|-----|
| 1.1 STATE.md Compression | `tools/compress_state.py` (new) | implementer | None | ~80 |
| 1.2 Bounded Memory | `backend/services/memory_store.py` | implementer | None | ~30 |
| 1.3 Session Auto-Load | `memory_store.py`, `memory.py`, `chat_handler.py` | implementer | 1.2 (soft) | ~70 |

### Phase 2 Summary

| Item | Files | Agent | Dependencies | LOC |
|------|-------|-------|--------------|-----|
| 2.1 Confidence Scoring | `shared/escalation_protocol.py`, `routes/escalations.py` | implementer | None | ~45 |
| 2.2 Unified Context | `backend/routes/context.py`, `main.py` | implementer | 1.3 | ~65 |
| 2.3 FTS5 Search | `memory_db.py`, `memory.py` | implementer | None | ~70 |

### Phase 3 Summary (Conditional)

| Item | Files | Agent | Trigger | LOC |
|------|-------|-------|---------|-----|
| 3.1 Embeddings | `embedding_service.py`, `memory_db.py` | implementer | FTS5 <70% accuracy | ~80 |
| 3.2 Meta-Learning | `memory_store.py`, `routes/context.py` | implementer | Phase 1-2 complete | ~60 |
| 3.3 Self-Awareness | `tools/generate_self_awareness.py` (new) | implementer | Capability hallucinations | ~60 |

---

## Resource Budget

| Component | RAM Allocation | Phase |
|-----------|---------------|-------|
| macOS System | 2.5 GB | Fixed |
| Ollama (loaded) | 3-4 GB | Fixed |
| Python Backend | 300 MB | Fixed |
| **Memory Layer** | **<200 MB** | Target |
| ├── SQLite + FTS5 | 10 MB | Phase 2 |
| ├── Session Cache | 20 MB | Phase 1 |
| ├── Context Cache | 20 MB | Phase 2 |
| └── MiniLM (if needed) | 80 MB | Phase 3 |
| Safety Buffer | 0.5-1 GB | Reserve |

---

## Success Criteria

### Phase 1 Exit Criteria

| Metric | Target |
|--------|--------|
| STATE.md token reduction | ≥50% |
| Memory file growth | Bounded <50KB |
| Cross-session recall | 3+ relevant facts |

### Phase 2 Exit Criteria

| Metric | Target |
|--------|--------|
| Escalations with confidence | 100% |
| Context API latency | <500ms |
| FTS5 search latency | <100ms |

### Phase 3 Entry Criteria

| Trigger | Measurement |
|---------|-------------|
| FTS5 insufficient | Retrieval accuracy <70% |
| Semantic needed | Manual review of missed queries |

---

## Agent Assignment Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ASSIGNMENT MATRIX                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1 (Quick Wins)                                           │
│  ├── 1.1 STATE.md Compression     → implementer (+ critic)      │
│  ├── 1.2 Bounded Memory           → implementer                 │
│  └── 1.3 Session Auto-Load        → implementer (+ researcher)  │
│                                                                  │
│  PHASE 2 (Architecture)                                         │
│  ├── 2.1 Confidence Scoring       → implementer                 │
│  ├── 2.2 Unified Context          → implementer (+ architect)   │
│  └── 2.3 FTS5 Search              → implementer (+ tester)      │
│                                                                  │
│  PHASE 3 (Future - Conditional)                                 │
│  ├── 3.1 Embeddings               → implementer (+ architect)   │
│  ├── 3.2 Meta-Learning            → implementer                 │
│  └── 3.3 Self-Awareness           → implementer                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

```
Phase 1:
   1.1 ────────────────────────────┐
                                   │
   1.2 ─────┬──────────────────────┤
            │                      │
            └──▶ 1.3 ──────────────┤
                                   │
Phase 2:                           │
   2.1 ────────────────────────────┤
                                   │
   1.3 ───────▶ 2.2 ───────────────┤
                                   │
   2.3 ────────────────────────────┘
                 │
Phase 3:         │
   2.3 ───────▶ 3.1 (conditional)

   All Phase 1-2 ───▶ 3.2, 3.3 (low priority)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Compression loses important detail | Archive full entries before compression |
| Session loading slows startup | Limit to 5 sessions, async if needed |
| Confidence scoring is subjective | Start at 0.8 threshold, make configurable |
| FTS5 misses semantic similarity | Add embeddings in Phase 3 if <70% accuracy |
| RAM exceeds 8GB | Monitor with htop, lazy-load optional components |

---

## Decision Log

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| SQLite FTS5 over ChromaDB | 10MB vs 500MB+, proven stability | No native vector search |
| 3 tiers not 5 | Simpler mental model, less code | Episodic/Semantic merged |
| Defer embeddings | 80MB cost, BM25 may be sufficient | May miss semantic matches |
| File-based archives | Zero infrastructure, portable | Manual rotation |
| Confidence threshold 0.8 | Matches MYND v3 pattern | May need tuning |

---

## Next Steps

1. **Assign Phase 1.1** to implementer agent: Create `tools/compress_state.py`
2. **Parallel**: Assign Phase 1.2 to implementer agent: Update `memory_store.py`
3. **After 1.2**: Assign Phase 1.3 to implementer agent: Session auto-load
4. **Validation Gate**: Test cross-session recall before Phase 2
5. **Phase 2**: Proceed with confidence scoring, unified context, FTS5

---

*FINAL DELIVERABLE - Memory Implementation Roadmap Complete*
*Ready for implementation starting with Phase 1.1: STATE.md Compression*
