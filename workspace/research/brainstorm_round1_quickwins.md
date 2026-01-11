# Brainstorm Round 1: Quick Wins for Context Storage/Retrieval

**Date**: 2026-01-06
**Based on**: `workspace/research/context_advancements.md`
**Focus**: Leveraging EXISTING infrastructure (Memory API, STATE.md, session files)
**Constraint**: Zero external dependencies, can be done by our agents

---

## Existing Infrastructure Summary

Before identifying quick wins, here's what we already have:

| Component | Location | Current State |
|-----------|----------|---------------|
| **MemoryManager** | `backend/memory.py` | Hierarchical context loading (COO → VP → Agent), session summarization, progress updates |
| **MemoryStore** | `backend/services/memory_store.py` | JSON key-value facts + preferences, thread-safe, persistence |
| **SemanticMemory** | `backend/services/semantic_memory.py` | SQLite-backed nodes with confidence, activation (ACT-R), FTS5 search, edges/relationships |
| **EpisodicMemory** | `backend/services/episodic_memory.py` | Episodes with Ebbinghaus decay, consolidation tracking, gzip compression |
| **STATE.md files** | `swarms/*/workspace/STATE.md` | Per-swarm progress logs, objectives, decisions (flat markdown) |
| **Session files** | `memory/sessions/*.md` | Session transcripts and summaries |
| **Core memory** | `memory/core/{vision,priorities,decisions}.md` | Organizational context files |
| **Work Ledger** | `workspace/ledger/index.json` | Task tracking (currently empty) |

---

## Quick Win Ideas

### 1. STATE.md Auto-Compression (Hierarchical Summarization)

**Problem**: STATE.md files grow unbounded, consuming context window space. The swarm_dev STATE.md is already 200+ lines.

**Solution**: Implement automatic compression of older STATE.md entries using the pattern from research:
- Keep recent 10 entries verbatim (per `_extract_recent_entries` in memory.py)
- Summarize entries older than 7 days into weekly summaries
- Archive monthly summaries to `memory/swarms/{swarm}/history.md`

**Implementation**:
```python
# Extend MemoryManager with:
def compress_state_md(self, swarm_name: str, keep_recent: int = 10):
    # 1. Read STATE.md
    # 2. Parse entries by date
    # 3. Keep recent N verbatim
    # 4. Generate weekly summary for older entries (can use Claude or simple extraction)
    # 5. Write compressed STATE.md + archive old content
```

**Why This Works**:
- Research shows 3-4x compression possible while maintaining accuracy
- MemoryManager already has `_extract_recent_entries()` and `_extract_summary()` methods
- Matches RAPTOR's hierarchical summarization pattern

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 2/5 | Build on existing MemoryManager methods |
| **Impact** | 4/5 | Direct context window savings, cleaner STATE.md |
| **Dependencies** | None | Pure Python, existing file structure |

---

### 2. Session-to-Episodic Memory Bridge

**Problem**: Session summaries exist (`memory/sessions/*.md`) but aren't connected to EpisodicMemory system. EpisodicMemory has sophisticated decay/consolidation but no data.

**Solution**: Auto-create Episode records when sessions end, bridging session files to the cognitive memory system.

**Implementation**:
```python
# On session end:
def save_session_as_episode(session_id: str, summary: str, duration: int):
    from backend.services.episodic_memory import get_episodic_memory

    ep = get_episodic_memory().create_episode(
        summary=summary,
        transcript=full_conversation,  # Will be gzip compressed
        duration_seconds=duration,
        encoding_strength=0.7,  # Sessions are important
        social_context=["CEO", "COO"],
    )

    # Also save to markdown for human readability
    save_session_summary(session_id, summary)
```

**Why This Works**:
- EpisodicMemory already has Ebbinghaus decay, consolidation tracking, emotional tagging
- Session files already exist - just need to dual-write
- Enables "what happened last week?" queries via `get_episodes_in_range()`

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 2/5 | Hook into existing session save flow |
| **Impact** | 4/5 | Unlocks episodic queries, decay-based forgetting |
| **Dependencies** | None | Existing EpisodicMemory + session flow |

---

### 3. Confidence-Scored Auto-Escalation

**Problem**: Agents escalate manually without quality gates. Research shows 0.8 confidence threshold prevents noise accumulation.

**Solution**: Add confidence scoring to agent outputs, auto-escalate when confidence < 0.8.

**Implementation**:
```python
# In agent response handling:
def process_agent_response(response: str, task: str) -> dict:
    # Extract confidence from response (or infer from hedging language)
    confidence = extract_confidence(response)

    if confidence < 0.8:
        return {
            "response": response,
            "confidence": confidence,
            "escalate": True,
            "reason": "Low confidence - needs human review"
        }

    return {"response": response, "confidence": confidence, "escalate": False}

def extract_confidence(text: str) -> float:
    # Simple heuristic: look for uncertainty markers
    uncertainty_markers = ["might", "maybe", "not sure", "could be", "possibly", "unclear"]
    count = sum(1 for marker in uncertainty_markers if marker in text.lower())
    return max(0.3, 1.0 - (count * 0.15))
```

**Why This Works**:
- MYND v3 uses 0.8 threshold for knowledge distillation
- Research confirms confidence scoring prevents noise accumulation
- SemanticMemory already has `update_confidence()` with Bayesian updates

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 2/5 | Simple text analysis + threshold check |
| **Impact** | 4/5 | Quality gate for all agent outputs |
| **Dependencies** | None | Text parsing only |

---

### 4. Cross-Session Fact Extraction

**Problem**: MemoryStore has facts/preferences but they're manually set. Valuable facts are buried in conversations.

**Solution**: Auto-extract facts from conversations using pattern matching, store in MemoryStore.

**Implementation**:
```python
# Fact extraction patterns
FACT_PATTERNS = [
    (r"my name is (\w+)", "user_name"),
    (r"I (?:work|am) (?:at|with) (.+?)(?:\.|,|$)", "user_workplace"),
    (r"I prefer (.+?)(?:\.|,|$)", "user_preference"),
    (r"the (\w+) (?:project|swarm|team) (?:is|should) (.+?)(?:\.|,|$)", "project_decision"),
]

def extract_facts_from_message(message: str) -> list[tuple[str, str, str]]:
    facts = []
    for pattern, fact_type in FACT_PATTERNS:
        matches = re.findall(pattern, message, re.IGNORECASE)
        for match in matches:
            facts.append((fact_type, match, "extracted"))
    return facts

# On each user message:
def process_user_message(message: str):
    facts = extract_facts_from_message(message)
    store = get_memory_store()
    for fact_type, value, source in facts:
        store.set_fact(fact_type, value, source)
```

**Why This Works**:
- MemoryStore already has `set_fact()`, `get_context_for_prompt()` for injection
- Matches Mem0's ADD/UPDATE operations from conversations
- Simple regex patterns, no ML required

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 2/5 | Regex patterns + existing MemoryStore API |
| **Impact** | 3/5 | Gradual personalization, reduces repetition |
| **Dependencies** | None | Pure Python regex |

---

### 5. Semantic Memory Auto-Population from STATE.md

**Problem**: SemanticMemory has sophisticated node/edge system with confidence, activation tracking - but needs manual population.

**Solution**: Parse STATE.md entries and auto-create SemanticNodes for decisions, goals, and key entities.

**Implementation**:
```python
def populate_semantic_from_state(swarm_name: str):
    state_content = read_state_md(swarm_name)
    sem = get_semantic_memory()

    # Extract decisions (## headers with decision keywords)
    for decision in extract_decisions(state_content):
        sem.add_node(
            label=decision.title,
            node_type=SemanticNodeType.DECISION,
            description=decision.content,
            confidence=0.8,
            source="STATE.md",
            provenance={"swarm": swarm_name, "date": decision.date}
        )

    # Extract goals/objectives
    for goal in extract_goals(state_content):
        sem.add_node(
            label=goal,
            node_type=SemanticNodeType.GOAL,
            confidence=0.9,
            source="STATE.md"
        )

    # Link related nodes
    sem.add_edge(goal_node.id, decision_node.id, "INFORMS")
```

**Why This Works**:
- SemanticMemory already has node types: DECISION, GOAL, CONCEPT, FACT
- Has FTS5 search via `search_fts()` and activation tracking
- STATE.md already structured with ## headers for easy parsing

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 3/5 | Need robust markdown parsing |
| **Impact** | 4/5 | Unlocks semantic search, relationship queries |
| **Dependencies** | None | Existing SemanticMemory + markdown parsing |

---

### 6. Context Loading Optimization (Smart Pruning)

**Problem**: `load_coo_context()` loads ALL swarm contexts, potentially exceeding context limits.

**Solution**: Implement smart context pruning based on recency, relevance, and activation.

**Implementation**:
```python
def load_coo_context_optimized(self, max_tokens: int = 30000) -> str:
    sections = []

    # Always include: vision (summary only), priorities (full)
    sections.append(self._extract_summary(self._read_file("core/vision.md")))
    sections.append(self._read_file("core/priorities.md"))

    # Load swarm contexts ordered by recency
    swarm_contexts = []
    for swarm_dir in self.memory_path.glob("swarms/*/context.md"):
        content = self._read_file(swarm_dir)
        mtime = swarm_dir.stat().st_mtime
        swarm_contexts.append((mtime, swarm_dir.parent.name, content))

    # Sort by modification time (most recent first)
    swarm_contexts.sort(reverse=True)

    # Add swarms until token budget exhausted
    remaining_tokens = max_tokens - self.estimate_tokens("\n".join(sections))
    for _, swarm_name, content in swarm_contexts:
        if self.estimate_tokens(content) < remaining_tokens:
            sections.append(f"## {swarm_name}\n{content}")
            remaining_tokens -= self.estimate_tokens(content)
        else:
            # Add summary only
            sections.append(f"## {swarm_name} (summary)\n{self._extract_summary(content)}")

    return "\n".join(sections)
```

**Why This Works**:
- MemoryManager already has `estimate_tokens()` method
- Already has `_extract_summary()` for fallback
- Research shows prioritizing recent context is optimal

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 2/5 | Extend existing MemoryManager methods |
| **Impact** | 4/5 | Prevents context overflow, maintains quality |
| **Dependencies** | None | Existing infrastructure |

---

### 7. Work Ledger Activation

**Problem**: Work Ledger (`workspace/ledger/index.json`) exists but is empty. Missing task tracking.

**Solution**: Integrate Work Ledger with agent task assignment to enable "what's everyone working on?" queries.

**Implementation**:
```python
# In task assignment flow:
def assign_task(agent: str, task: str, swarm: str) -> str:
    ledger = load_ledger()
    task_id = f"task-{ledger['id_counter']}"

    ledger['items'][task_id] = {
        "id": task_id,
        "title": task,
        "owner": agent,
        "swarm": swarm,
        "status": "in_progress",
        "created_at": datetime.now().isoformat(),
    }

    ledger['id_counter'] += 1
    ledger['by_status']['in_progress'].append(task_id)
    ledger['by_owner'].setdefault(agent, []).append(task_id)
    ledger['by_swarm'].setdefault(swarm, []).append(task_id)

    save_ledger(ledger)
    return task_id

# Add to COO context:
def get_active_work_summary() -> str:
    ledger = load_ledger()
    in_progress = ledger['by_status']['in_progress']

    summary = ["## Active Work"]
    for task_id in in_progress:
        task = ledger['items'][task_id]
        summary.append(f"- [{task['owner']}] {task['title']}")

    return "\n".join(summary)
```

**Why This Works**:
- Ledger structure already exists with proper indexes
- Enables "what's blocked?", "what did X complete?" queries
- Matches industry pattern of centralized task visibility

| Metric | Rating | Notes |
|--------|--------|-------|
| **Effort** | 2/5 | Simple JSON updates |
| **Impact** | 3/5 | Task visibility, coordination improvement |
| **Dependencies** | None | Existing ledger structure |

---

## Implementation Priority Matrix

| Rank | Quick Win | Effort | Impact | Score (I/E) | Rationale |
|------|-----------|--------|--------|-------------|-----------|
| 1 | Confidence-Scored Auto-Escalation | 2 | 4 | 2.0 | Quality gate, prevents noise |
| 2 | Context Loading Optimization | 2 | 4 | 2.0 | Immediate context savings |
| 3 | STATE.md Auto-Compression | 2 | 4 | 2.0 | Long-term context health |
| 4 | Session-to-Episodic Bridge | 2 | 4 | 2.0 | Unlocks memory queries |
| 5 | Semantic Memory Auto-Population | 3 | 4 | 1.3 | Search capability |
| 6 | Cross-Session Fact Extraction | 2 | 3 | 1.5 | Gradual personalization |
| 7 | Work Ledger Activation | 2 | 3 | 1.5 | Task coordination |

---

## Recommended Week 1 Sprint

**Day 1-2**: Implement #1 (Confidence Scoring) + #2 (Context Optimization)
- Both are low effort, high impact
- Context optimization provides immediate relief
- Confidence scoring adds quality gates

**Day 3-4**: Implement #3 (STATE.md Compression) + #4 (Session-to-Episodic Bridge)
- STATE.md compression prevents future bloat
- Episodic bridge activates existing memory infrastructure

**Day 5**: Implement #7 (Work Ledger Activation)
- Simple integration
- Enables "what's the team doing?" queries

**Stretch Goals (if time permits)**:
- #5 Semantic Memory Auto-Population
- #6 Cross-Session Fact Extraction

---

## Key Insights from Research

1. **Don't over-engineer**: Simple heuristics (confidence thresholds, recency sorting) work well
2. **Dual-write pattern**: Save to both human-readable (markdown) and machine-queryable (SQLite)
3. **Bounded storage**: MYND uses max 200 insights, 100 corrections - prevent unbounded growth
4. **Compression is free lunch**: 3-4x compression with minimal quality loss is proven
5. **Activation > Recency**: ACT-R style activation (SemanticMemory already has this) better than pure recency

---

*Brainstorm completed by COO on 2026-01-06*
