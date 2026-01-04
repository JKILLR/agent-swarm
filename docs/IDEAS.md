---
created: 2026-01-03
updated: 2026-01-03
---

# Ideas Backlog

> **Purpose:** Parking lot for interesting ideas and feature requests. Not prioritized, not committed — just captured.

---

## 1. Push Notifications

**Date Added:** 2026-01-03
**Priority:** Nice to have
**Effort:** Low-Medium

Send notifications when agents complete work or need attention.

**Options:**
1. **macOS Native** - Easiest, `osascript` notifications (Mac only)
2. **ntfy.sh** - Free push service, works on iOS/Android
3. **Pushover** - $5 one-time, polished app

**Use Cases:**
- Agent completes long-running task
- Error/escalation needs attention
- Daily summary of swarm activity

**Status:** Backlogged

---

## 2. Training-Free GRPO (Agent RL Without Fine-Tuning)

**Date Added:** 2026-01-03
**Source:** Recent research on training-free Group Relative Policy Optimization

**Summary:** Approach that achieves agent improvement through experience distillation into prompts rather than gradient updates. Claims $8 vs $1000s cost for traditional RL fine-tuning.

**Key Concepts:**
- Experience distillation into meta-prompts
- Contrastive learning from successful vs failed trajectories
- No weight updates required — just better context

**What's Interesting:**
1. **Economic Pragmatism** — Democratizes agent improvement for teams that can't afford traditional RL
2. **Architectural Elegance** — "Experience as prompt" approximates what weight updates achieve
3. **Contrastive Insight** — Comparing success/failure trajectories captures *why* things work, not just *that* they work

**Concerns/Limitations:**
1. **Prompt Bloat** — Meta-prompts accumulate, burning context window on "lessons learned"
2. **Context Limits** — As experiences grow, may hit token limits
3. **Quality Ceiling** — May not achieve same performance as true RL

**Potential Swarm Relevance:**
- Could apply to agent systems in this swarm
- Meta-learning patterns for constraint satisfaction?
- Low-cost alternative to fine-tuning swarm-aware models

**Status:** Backlogged

---

## 3. Swarm Brain Server (Agent Learning System)

**Date Added:** 2026-01-03
**Source:** MYND app brain architecture + Training-Free GRPO synthesis
**Priority:** High
**Effort:** High

**Summary:** A local ML server that gives agents persistent memory and learning capabilities, enabling the swarm to improve over time by distilling experience into better context.

**Key Concepts:**
- **Experience Memory**: Vector-indexed storage of task outcomes (success/failure trajectories)
- **Learning Engine**: Training-Free GRPO with contrastive analysis (compare what worked vs what didn't)
- **Knowledge Distillation**: Extract patterns from successful Claude outputs into reusable context
- **Swarm Awareness**: Track which agents excel at what tasks, recommend optimal delegation
- **Three-Tier Memory**: Global (org-wide) → Swarm (team) → Agent (individual) learning

**Inspired By:**
- MYND app's Axel brain server (localhost:8420)
- UnifiedBrain architecture: embeddings, Graph Transformer, context synthesis
- "Axel Continuity": memory writes become training signal, not just storage

**Architecture:**
```
Swarm Brain Server (localhost:8421)
├── Context Synthesizer    # Unified context for any agent
├── Experience Memory      # Task trajectories + outcomes
├── Learning Engine        # Training-Free GRPO patterns
├── Knowledge Distiller    # Extract insights from outputs
├── Swarm Awareness        # Agent capability tracking
└── Embedding Engine       # Semantic search (384-dim)
```

**How It Ties to Training-Free GRPO:**
1. Store task trajectories with success/failure signals
2. Find similar past tasks via semantic search
3. Extract contrastive patterns: "when X worked vs when X failed"
4. Inject distilled patterns into agent context (not weight updates)
5. Apply pattern compression to prevent prompt bloat

**What's Interesting:**
1. **Collective Learning** — Entire swarm improves from each agent's experience
2. **No Fine-Tuning Required** — $8 vs $1000s cost
3. **Proven Pattern** — MYND brain already works with local ML
4. **Incremental Value** — Each component useful independently

**Concerns/Limitations:**
1. Additional service to run (complexity)
2. Latency overhead for brain calls
3. Pattern quality depends on outcome signal accuracy
4. Meta-prompt bloat if not actively compressed

**Design Document:** `/docs/designs/swarm-brain-architecture.md`

**Incremental Path:**
1. MVP: Experience memory only (store task outcomes)
2. Add semantic search (embeddings + ChromaDB)
3. Build context synthesis endpoint
4. Implement contrastive learning engine
5. Add swarm awareness + recommendations

**Status:** Design Complete — Ready for Implementation

---

## Template for New Ideas

```markdown
## [Idea Title]

**Date Added:** YYYY-MM-DD
**Source:** [Paper, conversation, observation]
**Priority:** High / Medium / Nice to have
**Effort:** Low / Medium / High

**Summary:** [1-2 sentence description]

**Key Concepts:**
- ...

**What's Interesting:**
- ...

**Concerns/Limitations:**
- ...

**Potential Swarm Relevance:**
- ...

**Status:** Backlogged / Under Consideration / Exploring / Integrated
```
