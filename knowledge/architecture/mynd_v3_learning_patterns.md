# MYND v3 Learning System Patterns Research

**Research Date**: 2026-01-06
**Repository**: https://github.com/JKILLR/mynd-v3

## Executive Summary

MYND v3 is a distributed intelligence system that demonstrates how to effectively combine local ML models with Claude for continuous learning. The key insight: **MYND's power comes NOT from fine-tuning a local model for generation, but from the bidirectional learning loop between Claude and local specialized models.**

---

## 1. How MYND Uses Local Models

### Multi-Model Architecture

MYND uses **multiple specialized small models** instead of one general-purpose model:

| Model | Purpose | Parameters | Memory |
|-------|---------|------------|--------|
| **all-MiniLM-L6-v2** | Text embeddings | 22M | ~100MB |
| **Graph Transformer** | Connection prediction | 11.5M (custom) | ~50MB |
| **Whisper Base** | Audio transcription | 74M | ~150MB |
| **CLIP ViT-B-32** | Image understanding | 151M | ~300MB |

### Key Design Decision: No Local Generation

Local models are used for **classification and prediction only**, NOT for text generation:

- **Embeddings**: Semantic similarity search (not chat)
- **Graph Transformer**: Predicts if two concepts should connect (not explains why)
- **Whisper/CLIP**: Converts multimodal input to text (preprocessing)
- **Claude**: Handles ALL natural language generation and reasoning

### Graph Transformer Architecture

```python
# Custom architecture details:
- 512 hidden dimensions
- 8 attention heads
- 3 transformer layers
- Graph positional encoding (depth, connectivity, centrality)
- Edge-aware attention with adjacency biasing
- Lazy initialization (activates only when needed)
```

The GT learns from user feedback:
```python
# When user creates a connection between nodes:
gt.train_connection_step(
    source_embedding=node_a_emb,
    target_embedding=node_b_emb,
    should_connect=True,  # User confirmed this connection
    weight=0.5  # Confidence weight
)
```

---

## 2. The Feedback Learning Loop with Claude

### Bidirectional Knowledge Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    LEARNING LOOP                             │
│                                                              │
│  ┌──────────┐     Context      ┌─────────┐                  │
│  │  Brain   │ ───────────────► │ Claude  │                  │
│  │          │                  │         │                  │
│  │ - Memory │ ◄─────────────── │ - Chat  │                  │
│  │ - GT     │   Structured     │ - Tools │                  │
│  │ - Meta   │   Insights       │         │                  │
│  └──────────┘                  └─────────┘                  │
│       │                              │                       │
│       │                              │                       │
│       ▼                              ▼                       │
│  ┌─────────────────────────────────────────────┐            │
│  │             USER INTERACTION                 │            │
│  │  - Creates connections (trains GT)          │            │
│  │  - Accepts/rejects suggestions (feedback)   │            │
│  │  - Corrects Claude (stored for context)     │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Four Learning Loops

| Loop | Trigger | What Learns | How |
|------|---------|-------------|-----|
| **Feedback** | User accepts/rejects | Graph Transformer | `train_connection_step()` |
| **Correction** | User corrects Claude | Correction Memory | Stored for future context |
| **Pattern Extraction** | Session end | Long-term Memory | Behavioral patterns saved |
| **Code Analysis** | Code changes | Self-Awareness Doc | Auto-regenerates |

### Knowledge Distillation from Claude

When Claude responds, the brain extracts learnable data:

```python
def receive_from_claude(self, data):
    """Extract high-confidence insights from Claude response."""

    # 1. Parse structured data (if Claude returned JSON)
    insights = data.get('insights', [])
    patterns = data.get('patterns', [])
    corrections = data.get('corrections', [])

    # 2. Filter by confidence threshold
    high_conf_insights = [i for i in insights if i['confidence'] >= 0.8]

    # 3. Store in bounded collections
    self.knowledge_distiller.add_insights(high_conf_insights)  # Max 200
    self.knowledge_distiller.add_corrections(corrections)       # Max 100

    # 4. Update meta-learner
    self.meta_learner.record_source_effectiveness(
        source='claude_response',
        was_useful=len(insights) > 0
    )
```

### Context Synthesis for Claude

Before each Claude call, the brain compiles context:

```python
def get_context(self):
    """Build comprehensive context document."""
    return {
        # Identity
        'self_awareness': self.get_identity_document(),

        # Knowledge
        'distilled_insights': self.get_relevant_insights(limit=20),
        'recent_corrections': self.get_corrections(limit=10),

        # State
        'memory': {
            'short_term': self.memory.get_recent(5),
            'working': self.memory.get_active_goals(),
            'relevant_long_term': self.memory.search(query)
        },

        # Neural Predictions
        'gt_predictions': self.gt.get_likely_connections(current_focus),

        # Meta-learning
        'effective_sources': self.meta_learner.get_top_sources()
    }
```

---

## 3. Patterns Applicable to Agent-Swarm

### Pattern 1: Unified Context Endpoint

**MYND**: Single `/brain/context` endpoint replaces 19+ fragmented providers.

**Agent-Swarm Opportunity**:
```python
# New endpoint: /api/context
def get_unified_context(agent_type: str, task: str):
    return {
        'memory': session_memory.get_relevant(task),
        'escalations': escalation_manager.get_pending(),
        'work_ledger': work_ledger.get_in_progress(),
        'knowledge': knowledge_store.search(task),
        'agent_manifest': get_agent_capabilities(agent_type)
    }
```

### Pattern 2: Self-Awareness Document

**MYND**: Auto-generated document describing system capabilities and limitations.

**Agent-Swarm Opportunity**:
```markdown
# Agent-Swarm Self-Awareness

## Current Capabilities
- 6 agent types in swarm_dev
- REST API delegation (real agents)
- Task tool delegation (quick research only)
- WebSocket chat for COO

## Known Limitations
- Task tool spawns internal subagents, not real agent processes
- 5 concurrent agent limit via executor pool
- 8GB RAM constraint

## Recent Learnings
- REST API more reliable than Task for multi-step work
- Critic reviews catch ~40% of issues
```

### Pattern 3: Knowledge Distillation from Agent Outputs

**MYND**: Extracts insights from every Claude response.

**Agent-Swarm Opportunity**:
```python
# After agent completes task:
def distill_from_agent_output(agent_output: str, task_type: str):
    """Extract learnable patterns from agent work."""

    # Parse for patterns
    patterns = extract_patterns(agent_output)

    # Store successful approaches
    if task_succeeded:
        knowledge_store.add_pattern(
            task_type=task_type,
            pattern=patterns,
            confidence=0.8
        )

    # Track what worked
    meta_learner.record(
        agent_type=agent_type,
        task_type=task_type,
        success=task_succeeded
    )
```

### Pattern 4: Meta-Learning for Source Effectiveness

**MYND**: Tracks which knowledge sources improve outcomes.

**Agent-Swarm Opportunity**:
```python
class MetaLearner:
    """Track what context helps agents succeed."""

    def __init__(self):
        self.source_stats = {}  # source -> {used: int, helpful: int}

    def record(self, source: str, was_helpful: bool):
        self.source_stats[source]['used'] += 1
        if was_helpful:
            self.source_stats[source]['helpful'] += 1

    def get_attention_weights(self) -> dict:
        """Return weights for context sources."""
        return {
            source: stats['helpful'] / stats['used']
            for source, stats in self.source_stats.items()
        }
```

### Pattern 5: Evolution Daemon (Background Learning)

**MYND**: Runs every 30 minutes during idle, generates insights autonomously.

**Agent-Swarm Opportunity**:
```python
# Add to backend/jobs.py
async def evolution_daemon():
    """Background learning process."""
    while True:
        await asyncio.sleep(30 * 60)  # 30 minutes

        if is_user_idle():
            # Analyze recent work
            recent_tasks = work_ledger.get_completed(hours=24)

            # Extract patterns with Claude
            insights = await analyze_with_claude(recent_tasks)

            # Store high-confidence insights
            for insight in insights:
                if insight['confidence'] >= 0.7:
                    knowledge_store.add_insight(insight)
```

### Pattern 6: Three-Layer Memory

**MYND**: Short-term -> Working -> Long-term memory structure.

**Agent-Swarm Mapping**:
| MYND Layer | Agent-Swarm Equivalent |
|------------|----------------------|
| Short-term | Session messages |
| Working | STATE.md + Work Ledger |
| Long-term | `/knowledge/` + `/memory/` |

### Pattern 7: Bounded Storage

**MYND**: Caps at 200 insights, 100 corrections to prevent unbounded growth.

**Agent-Swarm Opportunity**:
```python
# Add to knowledge store
MAX_INSIGHTS = 200
MAX_CORRECTIONS = 100

def add_insight(insight: dict):
    if len(self.insights) >= MAX_INSIGHTS:
        # Remove oldest
        self.insights.pop(0)
    self.insights.append(insight)
```

---

## 4. Recommended Implementation Priority

| Priority | Pattern | Effort | Impact |
|----------|---------|--------|--------|
| **1** | Unified context endpoint `/api/context` | Low | High |
| **2** | Knowledge distillation from agent outputs | Medium | High |
| **3** | Meta-learning for source effectiveness | Medium | Medium |
| **4** | Self-awareness document generator | Low | Medium |
| **5** | Evolution daemon (background learning) | High | High |
| **6** | Embeddings for semantic search | Medium | Medium |
| **7** | Graph-based relationship learning | High | Optional |

---

## 5. Key Architectural Lessons

### What MYND Gets Right

1. **Separation of Concerns**: Local models do what they're good at (classification, prediction, preprocessing). Claude does what it's good at (reasoning, generation).

2. **Bidirectional Learning**: Claude teaches the brain (knowledge distillation). Brain teaches Claude (context injection). This creates continuous improvement.

3. **Confidence Thresholds**: Only high-confidence (>=0.8) insights are stored. Prevents noise accumulation.

4. **Bounded Collections**: Caps on storage prevent memory bloat. FIFO rotation keeps content fresh.

5. **Lazy Initialization**: Models only load when needed, preserving RAM.

### What to Avoid

1. **Don't try to replace Claude with local model**: MYND explicitly avoids using local models for chat/generation.

2. **Don't fine-tune for general tasks**: Training a small model to be a "mini Claude" doesn't work well. Use specialized models for specific tasks.

3. **Don't store everything**: Filter by confidence, cap storage, rotate old data.

---

## 6. Comparison: MYND vs Our Current Design

| Aspect | MYND v3 | Agent-Swarm Current | Gap |
|--------|---------|---------------------|-----|
| **Context Endpoint** | Single `/brain/context` | Multiple sources (manual) | Need unified endpoint |
| **Model Purpose** | Classification only | Considering generation | Align with MYND approach |
| **Knowledge Distillation** | From every Claude response | None | Critical gap |
| **Meta-Learning** | Tracks source effectiveness | None | Missing |
| **Confidence Scoring** | 0.8 threshold | None | Critical gap |
| **Background Learning** | Evolution daemon | recover_orphaned_work only | Opportunity |
| **Bounded Storage** | 200 insights, 100 corrections | Unbounded | Risk of bloat |

---

## 7. Next Steps for Agent-Swarm

### Immediate (This Week)
1. Create `/api/context` unified endpoint
2. Add distillation parsing to agent output handling
3. Implement confidence thresholds for learnings

### Short-term (This Month)
4. Build meta-learning tracker
5. Create self-awareness document generator
6. Add bounded storage to knowledge files

### Medium-term (Next Quarter)
7. Implement evolution daemon
8. Add embeddings for semantic search
9. Consider graph transformer for agent relationship learning

---

## Appendix: MYND Code References

**Repository Structure**:
```
mynd-brain/
├── brain/
│   ├── unified_brain.py      # 120KB - Main orchestrator
│   └── context_synthesizer.py # 68KB - Context assembly
├── models/
│   ├── graph_transformer.py   # 37KB - Connection prediction
│   ├── knowledge_extractor.py # 12KB - Insight extraction
│   ├── living_asa.py         # 109KB - Adaptive Semantic Agent
│   └── embeddings.py         # 4KB - MiniLM wrapper
├── evolution_daemon.py        # 16KB - Background learning
└── server.py                  # 221KB - API server
```

**Key Files to Study**:
- `brain/unified_brain.py`: Context synthesis, bidirectional learning
- `models/graph_transformer.py`: Local model training pattern
- `evolution_daemon.py`: Background learning implementation
- `UNIFIED_BRAIN_DESIGN.md`: Architecture documentation
