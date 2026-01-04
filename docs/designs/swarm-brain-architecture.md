# Swarm Brain Architecture Design

**Author**: System Architect
**Date**: 2026-01-03
**Status**: DESIGN COMPLETE
**ADR**: ADR-006

---

## Executive Summary

This document presents a comprehensive architecture for a "Swarm Brain Server" that provides persistent memory, learning capabilities, and unified context to the agent swarm system. The design is inspired by the MYND app's brain server while being tailored to the unique requirements of multi-agent orchestration.

The Swarm Brain enables agents to learn from task success/failure patterns, share knowledge across the swarm, and improve over time through experience distillation into better context -- without requiring expensive fine-tuning.

---

## 1. Architecture Decision

### Context
The agent swarm currently has:
- File-based memory (`/memory/` directory with markdown files)
- Session persistence via Work Ledger and Mailbox systems
- No learning from past successes/failures
- No semantic search or retrieval
- No shared knowledge distillation across agents

The MYND brain demonstrates a powerful pattern:
- Unified context endpoint for all queries
- Multi-modal learning (embeddings, graph transformer, feedback loops)
- Knowledge distillation from Claude responses
- Self-awareness and capability tracking

### Decision
Build a **Swarm Brain Server** as a FastAPI service on `localhost:8421` that provides:
1. **Unified Context Endpoint** - Single source of truth for agent context
2. **Experience Memory** - Vector-indexed storage of task outcomes
3. **Learning Engine** - Training-free GRPO with contrastive trajectory analysis
4. **Knowledge Distillation** - Extract patterns from successful agent outputs
5. **Swarm Awareness** - Track which agents excel at what tasks

### Rationale
1. **Separation of Concerns** - Brain as a service allows independent evolution
2. **Training-Free GRPO** - Achieves learning without fine-tuning cost ($8 vs $1000s)
3. **MYND Patterns** - Proven architecture for local ML integration
4. **Incremental Value** - Each component provides immediate utility

### Consequences
- **Positive**: Agents improve over time, shared learning, semantic retrieval
- **Negative**: Additional service to run, complexity, latency overhead
- **Trade-off**: Meta-prompt bloat vs learning quality (addressed in design)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
+------------------+     +------------------+     +------------------+
|   COO Agent      |     |  Swarm Agents    |     |  Task Tool       |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------------------+
|                         Agent Swarm Backend                            |
|                     (FastAPI @ localhost:8000)                         |
+--------+-----------------------+----------------------+----------------+
         |                       |                      |
         v                       v                      v
+--------+---------+    +--------+---------+   +-------+--------+
| /api/brain/*     |    | WebSocket        |   | Work Ledger    |
| (proxy endpoints)|    | Events           |   | + Mailbox      |
+--------+---------+    +--------+---------+   +----------------+
         |                       |
         v                       v
+------------------------------------------------------------------------+
|                         Swarm Brain Server                             |
|                     (FastAPI @ localhost:8421)                         |
|                                                                        |
|  +----------------+  +----------------+  +----------------+            |
|  | Context        |  | Experience     |  | Learning       |            |
|  | Synthesizer    |  | Memory         |  | Engine         |            |
|  +----------------+  +----------------+  +----------------+            |
|                                                                        |
|  +----------------+  +----------------+  +----------------+            |
|  | Knowledge      |  | Swarm          |  | Embedding      |            |
|  | Distiller      |  | Awareness      |  | Engine         |            |
|  +----------------+  +----------------+  +----------------+            |
+------------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------------+
|                         Storage Layer                                  |
|  +----------------+  +----------------+  +----------------+            |
|  | Vector DB      |  | Experience     |  | Pattern        |            |
|  | (ChromaDB)     |  | Store (JSON)   |  | Library (JSON) |            |
|  +----------------+  +----------------+  +----------------+            |
+------------------------------------------------------------------------+
```

### 2.2 Component Overview

| Component | Purpose | MYND Equivalent |
|-----------|---------|-----------------|
| **Context Synthesizer** | Build unified context for any agent | ContextSynthesizer |
| **Experience Memory** | Store task trajectories with outcomes | MemorySystem |
| **Learning Engine** | Apply Training-Free GRPO patterns | MetaLearner |
| **Knowledge Distiller** | Extract insights from Claude outputs | KnowledgeDistiller |
| **Swarm Awareness** | Track agent capabilities and preferences | SelfAwareness |
| **Embedding Engine** | Semantic search and similarity | EmbeddingEngine |

---

## 3. Core Components

### 3.1 Swarm Brain Server (`brain/server.py`)

```python
"""
Swarm Brain Server
==================
Local ML server for agent swarm learning and context.
Port: 8421 (next to main backend at 8000)
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize brain components on startup."""
    # Load embedding model
    brain.embedder = EmbeddingEngine()

    # Load experience memory from disk
    brain.experience = ExperienceMemory("./brain_data/experiences")

    # Initialize learning engine
    brain.learner = LearningEngine(brain.experience, brain.embedder)

    # Load distilled patterns
    brain.patterns = PatternLibrary("./brain_data/patterns")

    yield

    # Cleanup: save state
    brain.save_all()

app = FastAPI(title="Swarm Brain", lifespan=lifespan)
```

### 3.2 Experience Memory (`brain/experience.py`)

Stores task trajectories with outcomes for contrastive learning.

```python
@dataclass
class TaskExperience:
    """A recorded task execution with outcome."""

    # Identity
    id: str
    timestamp: datetime

    # Task context
    task_type: str           # "code_change", "research", "review", etc.
    task_prompt: str         # What was asked
    agent_type: str          # "implementer", "researcher", "architect", etc.
    swarm_name: str | None

    # Trajectory (what happened)
    trajectory: list[dict]   # Sequence of (action, observation) pairs
    tool_calls: list[str]    # Tools used
    files_touched: list[str] # Files read/written

    # Outcome
    success: bool            # Did it succeed?
    outcome_signal: str      # How we know (user_approved, tests_passed, error, etc.)
    duration_seconds: float
    token_count: int

    # Learning features
    embedding: list[float]   # Task prompt embedding for similarity search
    extracted_patterns: list[str]  # What worked/didn't work

    def to_contrastive_pair(self) -> dict:
        """Format for contrastive learning."""
        return {
            "prompt": self.task_prompt,
            "trajectory_summary": self._summarize_trajectory(),
            "outcome": "success" if self.success else "failure",
            "key_patterns": self.extracted_patterns,
            "agent": self.agent_type
        }
```

### 3.3 Learning Engine (`brain/learner.py`)

Implements Training-Free GRPO through experience distillation.

```python
class LearningEngine:
    """
    Training-Free GRPO Implementation
    ==================================

    Instead of gradient updates, we:
    1. Collect successful vs failed trajectories
    2. Extract contrastive patterns (what worked vs what didn't)
    3. Distill into meta-prompts that guide future behavior
    4. Manage prompt budget to prevent bloat
    """

    def __init__(
        self,
        experience: ExperienceMemory,
        embedder: EmbeddingEngine,
        max_pattern_tokens: int = 2000  # Budget for learned patterns in context
    ):
        self.experience = experience
        self.embedder = embedder
        self.max_pattern_tokens = max_pattern_tokens

        # Pattern storage by task type
        self.patterns: dict[str, list[LearnedPattern]] = {}

        # Pattern compression threshold
        self.pattern_similarity_threshold = 0.85

    async def learn_from_outcome(
        self,
        task_prompt: str,
        trajectory: list[dict],
        success: bool,
        agent_type: str,
        outcome_signal: str
    ) -> LearnedPattern | None:
        """
        Learn from a completed task.

        This is the core learning loop:
        1. Store the experience
        2. Find similar past experiences
        3. Identify contrastive patterns (success vs failure)
        4. Distill into a learnable pattern
        5. Compress pattern library to prevent bloat
        """

        # Store experience
        exp = TaskExperience(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            task_type=self._classify_task(task_prompt),
            task_prompt=task_prompt,
            agent_type=agent_type,
            trajectory=trajectory,
            success=success,
            outcome_signal=outcome_signal,
            embedding=await self.embedder.embed(task_prompt)
        )
        self.experience.store(exp)

        # Find similar experiences (both success and failure)
        similar = await self.experience.find_similar(
            embedding=exp.embedding,
            limit=20
        )

        # Need both successes and failures for contrastive learning
        successes = [e for e in similar if e.success]
        failures = [e for e in similar if not e.success]

        if len(successes) >= 2 and len(failures) >= 1:
            # Extract contrastive pattern
            pattern = await self._extract_contrastive_pattern(
                successes=successes,
                failures=failures,
                current=exp
            )

            if pattern:
                # Add to pattern library with compression
                self._add_pattern_with_compression(pattern)
                return pattern

        return None

    async def _extract_contrastive_pattern(
        self,
        successes: list[TaskExperience],
        failures: list[TaskExperience],
        current: TaskExperience
    ) -> LearnedPattern | None:
        """
        Use Claude to extract what differentiates success from failure.

        This is the "knowledge distillation" step - we're using Claude's
        reasoning to understand WHY things work, not just THAT they work.
        """

        prompt = f"""Analyze these task trajectories to identify what makes the difference between success and failure.

## Task Type: {current.task_type}

## Successful Trajectories:
{self._format_trajectories(successes[:3])}

## Failed Trajectories:
{self._format_trajectories(failures[:2])}

## Extract a Pattern

Identify ONE specific, actionable pattern that explains the difference.

Format:
- **Pattern Name**: (short descriptive name)
- **When to Apply**: (task conditions)
- **What to Do**: (specific guidance, max 2 sentences)
- **What to Avoid**: (common failure mode, max 1 sentence)
- **Confidence**: (high/medium/low based on evidence)

Be specific and actionable. This will be used to guide future agents."""

        # Call Claude for pattern extraction
        response = await self._call_claude_for_pattern(prompt)

        return self._parse_pattern_response(response, current.task_type)

    def _add_pattern_with_compression(self, pattern: LearnedPattern):
        """
        Add pattern while managing token budget.

        Compression strategies:
        1. Merge similar patterns (>85% embedding similarity)
        2. Decay old patterns (reduce weight over time)
        3. Prune low-confidence patterns when at capacity
        """
        task_type = pattern.task_type
        if task_type not in self.patterns:
            self.patterns[task_type] = []

        existing = self.patterns[task_type]

        # Check for similar pattern to merge
        for i, existing_pattern in enumerate(existing):
            similarity = self._pattern_similarity(pattern, existing_pattern)
            if similarity > self.pattern_similarity_threshold:
                # Merge: boost confidence, combine guidance
                existing[i] = self._merge_patterns(existing_pattern, pattern)
                return

        # Add new pattern
        existing.append(pattern)

        # Compress if over budget
        total_tokens = sum(p.estimated_tokens for p in existing)
        while total_tokens > self.max_pattern_tokens and len(existing) > 3:
            # Remove lowest confidence pattern
            existing.sort(key=lambda p: p.confidence, reverse=True)
            removed = existing.pop()
            total_tokens -= removed.estimated_tokens

    def get_relevant_patterns(
        self,
        task_prompt: str,
        agent_type: str,
        max_tokens: int = 500
    ) -> str:
        """
        Get learned patterns relevant to a task, formatted for context.

        This is what gets injected into agent prompts to apply learning.
        """
        task_type = self._classify_task(task_prompt)
        patterns = self.patterns.get(task_type, [])

        if not patterns:
            return ""

        # Sort by relevance and confidence
        patterns = sorted(
            patterns,
            key=lambda p: (p.confidence, p.success_rate),
            reverse=True
        )

        # Format for context
        lines = ["## Learned Patterns (from past experience)\n"]
        tokens_used = 10

        for pattern in patterns:
            if tokens_used + pattern.estimated_tokens > max_tokens:
                break
            lines.append(pattern.format_for_context())
            tokens_used += pattern.estimated_tokens

        return "\n".join(lines)
```

### 3.4 Knowledge Distiller (`brain/distiller.py`)

Extracts reusable insights from Claude's responses.

```python
class KnowledgeDistiller:
    """
    Extracts structured knowledge from Claude interactions.

    Unlike MYND which distills for a single user, this distills
    for the entire swarm - capturing organizational patterns.
    """

    def __init__(self, pattern_library: PatternLibrary):
        self.patterns = pattern_library
        self.pending_insights: list[dict] = []

    async def process_agent_response(
        self,
        agent_type: str,
        task_prompt: str,
        response: str,
        tool_calls: list[dict],
        outcome: str  # "success", "failure", "pending"
    ):
        """
        Extract knowledge from an agent's completed work.

        Types of knowledge extracted:
        1. Tool usage patterns (which tools for which tasks)
        2. File organization patterns (where things go)
        3. Code patterns (naming, structure)
        4. Communication patterns (how to phrase things)
        """

        # Only learn from successes for now
        if outcome != "success":
            return

        # Extract tool usage pattern
        if tool_calls:
            tool_pattern = {
                "type": "tool_usage",
                "agent": agent_type,
                "task_keywords": self._extract_keywords(task_prompt),
                "tools_used": [t["name"] for t in tool_calls],
                "tool_sequence": self._extract_tool_sequence(tool_calls),
                "timestamp": datetime.now().isoformat()
            }
            self.pending_insights.append(tool_pattern)

        # Extract file patterns
        files_read = [t for t in tool_calls if t["name"] == "Read"]
        files_written = [t for t in tool_calls if t["name"] in ("Write", "Edit")]

        if files_written:
            file_pattern = {
                "type": "file_organization",
                "agent": agent_type,
                "task_keywords": self._extract_keywords(task_prompt),
                "files_referenced": [t["params"].get("file_path") for t in files_read],
                "files_modified": [t["params"].get("file_path") for t in files_written],
                "timestamp": datetime.now().isoformat()
            }
            self.pending_insights.append(file_pattern)

        # Batch process insights periodically
        if len(self.pending_insights) >= 10:
            await self._consolidate_insights()

    async def _consolidate_insights(self):
        """
        Consolidate pending insights into patterns.

        This runs periodically to:
        1. Find recurring patterns across insights
        2. Promote high-frequency patterns to the library
        3. Age out one-off observations
        """
        insights = self.pending_insights
        self.pending_insights = []

        # Group by type and agent
        grouped: dict[str, list[dict]] = {}
        for insight in insights:
            key = f"{insight['type']}:{insight['agent']}"
            grouped.setdefault(key, []).append(insight)

        # Find patterns with 3+ occurrences
        for key, group in grouped.items():
            if len(group) >= 3:
                pattern = self._extract_recurring_pattern(group)
                if pattern:
                    self.patterns.add(pattern)
```

### 3.5 Context Synthesizer (`brain/context.py`)

Builds unified context for any agent request.

```python
class ContextSynthesizer:
    """
    Unified context builder for all agent requests.

    Combines:
    1. Static memory (from /memory/ files)
    2. Learned patterns (from LearningEngine)
    3. Relevant experiences (from ExperienceMemory)
    4. Swarm awareness (what other agents are doing)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        learner: LearningEngine,
        experience: ExperienceMemory,
        swarm_tracker: SwarmAwareness
    ):
        self.memory = memory_manager
        self.learner = learner
        self.experience = experience
        self.swarm = swarm_tracker

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        """
        Build complete context for an agent.

        Token budget allocation:
        - 40% for static memory (role-appropriate)
        - 25% for learned patterns
        - 20% for relevant experiences
        - 15% for swarm awareness
        """

        parts = []
        breakdown = {}

        # 1. Static memory (role-appropriate)
        static_context = self._get_static_context(request)
        parts.append(("memory", static_context))
        breakdown["memory"] = self._count_tokens(static_context)

        # 2. Learned patterns (relevant to task)
        if request.include_patterns:
            patterns = self.learner.get_relevant_patterns(
                task_prompt=request.task_prompt,
                agent_type=request.agent_type,
                max_tokens=int(request.max_tokens * 0.25)
            )
            if patterns:
                parts.append(("patterns", patterns))
                breakdown["patterns"] = self._count_tokens(patterns)

        # 3. Relevant experiences
        if request.include_experiences:
            experiences = await self._get_relevant_experiences(request)
            if experiences:
                parts.append(("experiences", experiences))
                breakdown["experiences"] = self._count_tokens(experiences)

        # 4. Swarm awareness
        if request.include_swarm_state:
            swarm_state = self.swarm.get_context(request.swarm_name)
            if swarm_state:
                parts.append(("swarm", swarm_state))
                breakdown["swarm"] = self._count_tokens(swarm_state)

        # Combine with priority ordering
        context = self._combine_context(parts, request.max_tokens)

        return ContextResponse(
            context=context,
            token_count=sum(breakdown.values()),
            breakdown=breakdown
        )

    async def _get_relevant_experiences(
        self,
        request: ContextRequest,
        max_experiences: int = 3
    ) -> str:
        """
        Find past experiences similar to current task.

        Only include successful experiences as examples.
        """
        embedding = await self.learner.embedder.embed(request.task_prompt)

        similar = await self.experience.find_similar(
            embedding=embedding,
            agent_type=request.agent_type,
            success_only=True,
            limit=max_experiences
        )

        if not similar:
            return ""

        lines = ["## Similar Past Tasks (successful)\n"]
        for exp in similar:
            lines.append(f"**Task**: {exp.task_prompt[:100]}...")
            lines.append(f"**Approach**: {exp.extracted_patterns[0] if exp.extracted_patterns else 'N/A'}")
            lines.append(f"**Tools Used**: {', '.join(exp.tool_calls[:5])}")
            lines.append("")

        return "\n".join(lines)
```

### 3.6 Swarm Awareness (`brain/awareness.py`)

Tracks agent capabilities and current state.

```python
class SwarmAwareness:
    """
    The brain's understanding of the swarm.

    Tracks:
    - Which agents exist and their capabilities
    - Current active work across agents
    - Agent performance statistics
    - Cross-agent coordination state
    """

    def __init__(self):
        self.agents: dict[str, AgentProfile] = {}
        self.active_work: dict[str, list[str]] = {}  # agent -> work_ids
        self.performance: dict[str, AgentPerformance] = {}

    def update_agent_profile(
        self,
        agent_type: str,
        swarm_name: str | None,
        capabilities: list[str],
        recent_tasks: list[str]
    ):
        """Update knowledge about an agent's capabilities."""
        key = f"{swarm_name or 'global'}:{agent_type}"
        self.agents[key] = AgentProfile(
            agent_type=agent_type,
            swarm_name=swarm_name,
            capabilities=capabilities,
            recent_tasks=recent_tasks[-10:],
            last_seen=datetime.now()
        )

    def record_performance(
        self,
        agent_type: str,
        task_type: str,
        success: bool,
        duration: float
    ):
        """Record an agent's performance on a task."""
        key = f"{agent_type}:{task_type}"
        if key not in self.performance:
            self.performance[key] = AgentPerformance(
                agent_type=agent_type,
                task_type=task_type,
                successes=0,
                failures=0,
                avg_duration=0.0
            )

        perf = self.performance[key]
        if success:
            perf.successes += 1
        else:
            perf.failures += 1

        # Running average
        total = perf.successes + perf.failures
        perf.avg_duration = (perf.avg_duration * (total - 1) + duration) / total

    def get_best_agent_for_task(self, task_type: str) -> str | None:
        """Suggest the best agent type for a task based on past performance."""
        candidates = [
            (k, v) for k, v in self.performance.items()
            if k.endswith(f":{task_type}")
        ]

        if not candidates:
            return None

        # Score by success rate with confidence weighting
        def score(perf: AgentPerformance) -> float:
            total = perf.successes + perf.failures
            if total < 3:
                return 0.5  # Not enough data
            success_rate = perf.successes / total
            confidence = min(total / 20, 1.0)  # Max confidence at 20 samples
            return success_rate * confidence

        candidates.sort(key=lambda x: score(x[1]), reverse=True)
        return candidates[0][1].agent_type

    def get_context(self, swarm_name: str | None) -> str:
        """Format swarm awareness for context injection."""
        lines = ["## Swarm State\n"]

        # Active agents
        active = [
            a for a in self.agents.values()
            if (swarm_name is None or a.swarm_name == swarm_name)
            and (datetime.now() - a.last_seen).seconds < 300
        ]

        if active:
            lines.append("### Active Agents")
            for agent in active:
                lines.append(f"- **{agent.agent_type}**: {', '.join(agent.recent_tasks[-3:])}")

        # Performance insights
        if self.performance:
            lines.append("\n### Agent Strengths")
            for task_type in set(k.split(":")[1] for k in self.performance.keys()):
                best = self.get_best_agent_for_task(task_type)
                if best:
                    lines.append(f"- {task_type}: Best handled by **{best}**")

        return "\n".join(lines)
```

---

## 4. API Endpoints

### 4.1 Brain Server Endpoints

```python
# ═══════════════════════════════════════════════════════════════
# CONTEXT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/brain/context")
async def get_brain_context(request: ContextRequest) -> ContextResponse:
    """
    Get unified context for an agent.

    This is the primary endpoint agents call to get their context.
    Replaces loading memory files directly.
    """
    return await brain.synthesizer.get_context(request)


@app.get("/brain/patterns/{task_type}")
async def get_patterns(
    task_type: str,
    agent_type: str = None,
    max_tokens: int = 500
) -> PatternResponse:
    """Get learned patterns for a task type."""
    patterns = brain.learner.get_relevant_patterns(
        task_prompt=f"[{task_type}]",
        agent_type=agent_type or "any",
        max_tokens=max_tokens
    )
    return PatternResponse(patterns=patterns)


# ═══════════════════════════════════════════════════════════════
# LEARNING ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/brain/learn")
async def learn_from_task(request: LearnRequest) -> LearnResponse:
    """
    Record a task outcome for learning.

    Called when:
    - Task tool completes (success or failure)
    - User provides explicit feedback
    - Tests pass/fail after code change
    """
    pattern = await brain.learner.learn_from_outcome(
        task_prompt=request.task_prompt,
        trajectory=request.trajectory,
        success=request.success,
        agent_type=request.agent_type,
        outcome_signal=request.outcome_signal
    )

    return LearnResponse(
        learned=pattern is not None,
        pattern_name=pattern.name if pattern else None,
        patterns_count=len(brain.learner.patterns.get(request.task_type, []))
    )


@app.post("/brain/distill")
async def distill_knowledge(request: DistillRequest) -> DistillResponse:
    """
    Extract knowledge from an agent's response.

    Called after successful agent completions to extract:
    - Tool usage patterns
    - File organization patterns
    - Code patterns
    """
    await brain.distiller.process_agent_response(
        agent_type=request.agent_type,
        task_prompt=request.task_prompt,
        response=request.response,
        tool_calls=request.tool_calls,
        outcome=request.outcome
    )

    return DistillResponse(processed=True)


# ═══════════════════════════════════════════════════════════════
# EXPERIENCE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/brain/experience/search")
async def search_experiences(request: SearchRequest) -> SearchResponse:
    """
    Search past experiences by semantic similarity.

    Used for:
    - Finding similar past tasks as examples
    - Debugging (what worked before for similar problems)
    - Agent selection (who handled this well)
    """
    embedding = await brain.embedder.embed(request.query)

    results = await brain.experience.find_similar(
        embedding=embedding,
        agent_type=request.agent_type,
        success_only=request.success_only,
        limit=request.limit
    )

    return SearchResponse(
        experiences=[e.to_summary() for e in results],
        count=len(results)
    )


@app.get("/brain/experience/stats")
async def experience_stats() -> StatsResponse:
    """Get statistics about stored experiences."""
    return StatsResponse(
        total_experiences=brain.experience.count(),
        by_agent=brain.experience.count_by_agent(),
        by_task_type=brain.experience.count_by_task_type(),
        success_rate=brain.experience.overall_success_rate(),
        patterns_learned=sum(len(p) for p in brain.learner.patterns.values())
    )


# ═══════════════════════════════════════════════════════════════
# SWARM AWARENESS ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/brain/swarm/agent-active")
async def record_agent_active(request: AgentActiveRequest):
    """Record that an agent is active."""
    brain.swarm.update_agent_profile(
        agent_type=request.agent_type,
        swarm_name=request.swarm_name,
        capabilities=request.capabilities,
        recent_tasks=request.recent_tasks
    )
    return {"status": "recorded"}


@app.post("/brain/swarm/task-complete")
async def record_task_complete(request: TaskCompleteRequest):
    """Record task completion for performance tracking."""
    brain.swarm.record_performance(
        agent_type=request.agent_type,
        task_type=request.task_type,
        success=request.success,
        duration=request.duration
    )
    return {"status": "recorded"}


@app.get("/brain/swarm/recommend-agent")
async def recommend_agent(task_type: str) -> AgentRecommendation:
    """Get recommended agent for a task type."""
    best = brain.swarm.get_best_agent_for_task(task_type)
    return AgentRecommendation(
        recommended_agent=best,
        confidence="high" if best else "none",
        based_on_samples=sum(
            p.successes + p.failures
            for p in brain.swarm.performance.values()
            if p.task_type == task_type
        )
    )


# ═══════════════════════════════════════════════════════════════
# HEALTH & DEBUG ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/brain/health")
async def health() -> HealthResponse:
    """Health check with component status."""
    return HealthResponse(
        status="healthy",
        components={
            "embedder": brain.embedder is not None,
            "experience": brain.experience.count() > 0,
            "patterns": len(brain.learner.patterns) > 0,
            "swarm_aware": len(brain.swarm.agents) > 0
        },
        uptime_seconds=time.time() - brain.started_at
    )


@app.get("/brain/patterns/export")
async def export_patterns() -> dict:
    """Export all learned patterns (for debugging/backup)."""
    return {
        task_type: [p.to_dict() for p in patterns]
        for task_type, patterns in brain.learner.patterns.items()
    }
```

### 4.2 Backend Integration Points

Add proxy endpoints in the main backend to forward brain requests:

```python
# In backend/routes/brain.py

from fastapi import APIRouter
import httpx

router = APIRouter(prefix="/api/brain", tags=["brain"])

BRAIN_URL = "http://localhost:8421"

@router.post("/context")
async def get_context(request: ContextRequest):
    """Proxy to brain server for context."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BRAIN_URL}/brain/context",
            json=request.dict()
        )
        return response.json()

@router.post("/learn")
async def learn(request: LearnRequest):
    """Proxy to brain server for learning."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BRAIN_URL}/brain/learn",
            json=request.dict()
        )
        return response.json()

# ... additional proxies as needed
```

---

## 5. Learning Mechanism Deep Dive

### 5.1 Training-Free GRPO Implementation

```
+------------------+     +------------------+     +------------------+
| Task Execution   | --> | Outcome Signal   | --> | Experience Store |
| (agent + tools)  |     | (success/fail)   |     | (with embedding) |
+------------------+     +------------------+     +--------+---------+
                                                          |
                                                          v
+------------------+     +------------------+     +--------+---------+
| Pattern          | <-- | Contrastive      | <-- | Find Similar     |
| Library          |     | Extraction       |     | (success+failure)|
+------------------+     +------------------+     +------------------+
         |
         v
+------------------+     +------------------+
| Context Injection| --> | Improved Agent   |
| (via /context)   |     | Behavior         |
+------------------+     +------------------+
```

### 5.2 Success/Failure Signal Sources

| Signal | Source | Reliability |
|--------|--------|-------------|
| `user_approved` | User explicitly says "looks good" | High |
| `tests_passed` | Test run after code change | High |
| `no_errors` | Task completed without errors | Medium |
| `user_rejected` | User says "that's wrong" | High (failure) |
| `tests_failed` | Test run failed | High (failure) |
| `timeout` | Agent timed out | Medium (failure) |
| `escalation` | Task was escalated | Low (could be either) |

### 5.3 Pattern Compression Strategy

To prevent prompt bloat, patterns are actively compressed:

```python
def compress_patterns(patterns: list[LearnedPattern], max_tokens: int):
    """
    Compression strategies:

    1. Merge Similar Patterns
       - If two patterns have >85% embedding similarity
       - Combine their guidance into a single pattern
       - Boost the confidence of the merged pattern

    2. Generalization
       - If many task-specific patterns emerge
       - Extract the common principle
       - Replace specifics with general rule

    3. Confidence Decay
       - Old patterns decay in confidence over time
       - Low-confidence patterns get pruned first

    4. Token Budgeting
       - High-confidence, high-impact patterns get more tokens
       - Low-value patterns get summarized or removed
    """
    pass
```

### 5.4 Example Learned Pattern

```markdown
## Learned Patterns (from past experience)

### Pattern: Read-Before-Edit
**Applies to**: code_change, refactor
**Confidence**: high (based on 15 successes, 3 failures)

When editing a file, ALWAYS read it first using the Read tool. Edits that
fail often skip this step and use stale or incorrect assumptions about
file contents.

**Do**: `Read(file) -> Edit(file, old_string, new_string)`
**Avoid**: `Edit(file, assumed_content, new_content)` without reading

---

### Pattern: Test-After-Change
**Applies to**: code_change, bug_fix
**Confidence**: medium (based on 8 successes, 2 failures)

After modifying code, run the relevant test suite. Changes that skip
testing often introduce regressions caught later.

**Do**: Make change -> Run tests -> Report results
**Avoid**: Declaring success without verification
```

---

## 6. Memory Layers

### 6.1 Three-Tier Memory Architecture

```
+------------------------------------------------------------------+
|                    SWARM-WIDE MEMORY (Shared)                    |
|                                                                  |
|  - Organizational patterns (how we do things here)              |
|  - Cross-swarm insights (what works across teams)               |
|  - Global performance stats (best agents for task types)        |
|                                                                  |
|  Storage: brain_data/global/                                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                   PER-SWARM MEMORY (Team-Level)                  |
|                                                                  |
|  - Swarm-specific patterns (coding style, tools, preferences)   |
|  - Team performance stats                                       |
|  - Recent swarm experiences                                     |
|                                                                  |
|  Storage: brain_data/swarms/{swarm_name}/                       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                  PER-AGENT MEMORY (Individual)                   |
|                                                                  |
|  - Agent's own experiences (what it learned)                    |
|  - Personal performance stats                                   |
|  - Recent task context                                          |
|                                                                  |
|  Storage: brain_data/agents/{agent_type}/                       |
+------------------------------------------------------------------+
```

### 6.2 Memory Flow

```python
# When context is requested, memories flow DOWN (inheritance):
#
# 1. Start with global patterns
# 2. Add swarm-specific patterns (override/extend global)
# 3. Add agent-specific patterns (override/extend swarm)

async def build_agent_context(agent_type: str, swarm_name: str) -> str:
    context_parts = []

    # Global patterns (all agents should know)
    global_patterns = await load_patterns("global")
    context_parts.append(format_patterns(global_patterns, "Organization-Wide"))

    # Swarm patterns (team-specific)
    if swarm_name:
        swarm_patterns = await load_patterns(f"swarms/{swarm_name}")
        context_parts.append(format_patterns(swarm_patterns, f"{swarm_name} Team"))

    # Agent patterns (individual learning)
    agent_patterns = await load_patterns(f"agents/{agent_type}")
    context_parts.append(format_patterns(agent_patterns, f"{agent_type} Expertise"))

    return "\n\n".join(context_parts)
```

### 6.3 Knowledge Promotion

Patterns can be "promoted" from individual to team to global:

```python
class KnowledgePromoter:
    """
    Promotes successful patterns up the hierarchy.

    Promotion criteria:
    - Individual -> Swarm: Pattern used successfully by 2+ agents in swarm
    - Swarm -> Global: Pattern used successfully across 3+ swarms
    """

    async def check_promotions(self):
        # Check for patterns ready to promote
        for swarm in self.swarms:
            swarm_patterns = await self.get_swarm_patterns(swarm)

            for pattern in swarm_patterns:
                if pattern.used_by_agents >= 2 and pattern.success_rate > 0.8:
                    # Check if similar pattern exists in other swarms
                    other_swarms_with_pattern = await self.find_similar_across_swarms(
                        pattern, exclude=swarm
                    )

                    if len(other_swarms_with_pattern) >= 2:
                        await self.promote_to_global(pattern)
```

---

## 7. Integration with Existing Systems

### 7.1 Agent Execution Flow (Modified)

```python
# In shared/agent_executor_pool.py

async def execute_with_brain(
    self,
    context: AgentExecutionContext,
    prompt: str,
    system_prompt: str | None = None
):
    """Execute agent with brain-enhanced context."""

    # 1. Get enhanced context from brain
    brain_context = await self._get_brain_context(
        agent_type=context.agent_type,
        swarm_name=context.swarm_name,
        task_prompt=prompt
    )

    # 2. Inject into system prompt
    enhanced_system = self._inject_brain_context(
        base_system=system_prompt or context.system_prompt,
        brain_context=brain_context
    )

    # 3. Execute agent
    trajectory = []
    async for event in self.execute(context, prompt, enhanced_system):
        # Track trajectory for learning
        if event["type"] == "tool_use":
            trajectory.append({
                "action": event["tool"],
                "params": event["params"],
                "timestamp": time.time()
            })
        yield event

    # 4. Learn from outcome (async, non-blocking)
    asyncio.create_task(self._learn_from_execution(
        agent_type=context.agent_type,
        task_prompt=prompt,
        trajectory=trajectory,
        outcome=self._determine_outcome(events)
    ))
```

### 7.2 COO Integration

The COO (Supreme Orchestrator) uses brain for delegation decisions:

```python
# In COO system prompt addition

## Brain-Enhanced Delegation

When delegating tasks, you can query the brain for recommendations:

1. **Best Agent Selection**
   The brain tracks which agent types perform best on different task types.
   Consider its recommendations but use judgment for novel tasks.

2. **Pattern Awareness**
   Learned patterns from past successes are automatically included in
   agent context. You don't need to repeat standard guidance.

3. **Experience References**
   For complex tasks, ask the brain for similar past experiences to
   provide as examples to agents.
```

### 7.3 Work Ledger Integration

```python
# In shared/work_ledger.py

class WorkLedger:
    def complete_work(self, work_id: str, result: dict):
        """Mark work complete and notify brain."""
        # ... existing completion logic ...

        # Notify brain of outcome
        asyncio.create_task(self._notify_brain_completion(
            work_item=work_item,
            result=result
        ))

    async def _notify_brain_completion(self, work_item: WorkItem, result: dict):
        """Send completion to brain for learning."""
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:8421/brain/learn",
                json={
                    "task_prompt": work_item.description,
                    "agent_type": work_item.assigned_to,
                    "success": result.get("status") == "success",
                    "outcome_signal": result.get("signal", "completion"),
                    "trajectory": result.get("trajectory", [])
                }
            )
```

---

## 8. Incremental Implementation Path

### Phase 1: MVP (Week 1) - Experience Memory Only

**Goal**: Get basic experience tracking working.

1. **Create brain server skeleton**
   - `brain/server.py` - FastAPI app
   - `brain/experience.py` - Experience storage (JSON files)
   - Health endpoint

2. **Add experience recording**
   - POST `/brain/experience` - Store task outcome
   - GET `/brain/experience/search` - Find similar (keyword match for now)

3. **Integrate with Work Ledger**
   - Call brain on work completion
   - Store task prompt + outcome

**Files to Create**:
```
brain/
  __init__.py
  server.py
  experience.py
  models.py
```

**Value Delivered**: Historical record of all task outcomes.

---

### Phase 2: Semantic Search (Week 2)

**Goal**: Enable similarity-based experience retrieval.

1. **Add Embedding Engine**
   - `brain/embeddings.py` - Sentence transformers
   - Embed all experiences on store
   - Vector similarity search

2. **Add ChromaDB for vectors**
   - Replace JSON search with vector search
   - Persistent collection

3. **Experience retrieval endpoint**
   - Find similar past tasks
   - Return as examples

**Files to Modify/Create**:
```
brain/
  embeddings.py
  vector_store.py (ChromaDB wrapper)
```

**Value Delivered**: "Find me tasks like this" capability.

---

### Phase 3: Context Synthesis (Week 3)

**Goal**: Unified context endpoint for all agents.

1. **Context Synthesizer**
   - `brain/context.py`
   - Combine memory files + experiences
   - Token budget management

2. **Brain Context endpoint**
   - POST `/brain/context`
   - Returns formatted context

3. **Integrate with agents**
   - Modify agent executor to call brain
   - Inject enhanced context

**Files to Modify/Create**:
```
brain/
  context.py
backend/routes/brain.py (proxy)
shared/agent_executor_pool.py (integration)
```

**Value Delivered**: Agents get relevant context automatically.

---

### Phase 4: Learning Engine (Week 4)

**Goal**: Extract patterns from success/failure.

1. **Learning Engine**
   - `brain/learner.py`
   - Contrastive pattern extraction
   - Pattern storage with compression

2. **Distill from Claude**
   - `brain/distiller.py`
   - Extract patterns from successful responses

3. **Pattern injection**
   - Include learned patterns in context
   - Test on real tasks

**Files to Modify/Create**:
```
brain/
  learner.py
  distiller.py
  patterns.py
```

**Value Delivered**: Agents learn from past successes.

---

### Phase 5: Swarm Awareness (Week 5)

**Goal**: Track and leverage agent performance.

1. **Swarm Awareness**
   - `brain/awareness.py`
   - Agent profiles and performance

2. **Recommendation endpoint**
   - Best agent for task type
   - Performance stats

3. **COO integration**
   - Use recommendations for delegation
   - Display in UI

**Files to Modify/Create**:
```
brain/
  awareness.py
frontend/ (stats display)
```

**Value Delivered**: Data-driven agent selection.

---

### Phase 6: Polish & Optimization (Week 6+)

1. **Pattern promotion** (individual -> swarm -> global)
2. **Pattern compression** optimization
3. **Performance tuning** (caching, batch embedding)
4. **UI for pattern inspection**
5. **A/B testing framework** for pattern effectiveness

---

## 9. Technical Requirements

### 9.1 Dependencies

```python
# brain/requirements.txt

fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
httpx>=0.24.0
torch>=2.0.0  # For sentence-transformers
numpy>=1.24.0
```

### 9.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| Storage | 5GB for models | 20GB+ |
| GPU | Not required | Apple MPS for faster embedding |

### 9.3 Storage Layout

```
brain_data/
  experiences/
    index.json           # Experience index
    {year}/{month}/      # Experiences by date
      {id}.json

  patterns/
    global.json          # Global patterns
    swarms/
      {swarm_name}.json  # Per-swarm patterns
    agents/
      {agent_type}.json  # Per-agent patterns

  vectors/
    chroma/              # ChromaDB data

  checkpoints/
    {timestamp}/         # Periodic backups
```

---

## 10. Comparison with MYND Brain

| Aspect | MYND Brain | Swarm Brain |
|--------|------------|-------------|
| **Purpose** | Personal mind mapping assistant | Multi-agent orchestration |
| **Users** | Single human user | Multiple AI agents |
| **Learning Source** | User feedback, conversations | Task outcomes, trajectories |
| **Context Type** | Map nodes, AI memories | Task prompts, patterns |
| **Training** | Graph Transformer (weights) | Training-Free GRPO (prompts) |
| **Neural Network** | 6.7M param GT | Embeddings only (no training) |
| **Knowledge** | Personal knowledge graph | Organizational patterns |
| **Memory** | Session-based with consolidation | Hierarchical (global/swarm/agent) |

### Key Differences

1. **No Neural Network Training**
   - MYND trains GT weights from feedback
   - Swarm Brain uses prompt engineering (Training-Free GRPO)
   - Rationale: Multiple agents, no consistent "self" to train

2. **Collective vs Individual**
   - MYND optimizes for one person's preferences
   - Swarm Brain optimizes for organizational patterns
   - Patterns must generalize across agents

3. **Task-Centric vs Concept-Centric**
   - MYND connects concepts in a knowledge graph
   - Swarm Brain connects task patterns to outcomes
   - Different substrate for learning

### Transferred Patterns

1. **Unified Context Endpoint** - Single source of truth
2. **Knowledge Distillation** - Extract patterns from Claude
3. **Self-Awareness** - Track capabilities and state
4. **Context Lens** - Relevance-weighted retrieval
5. **Feedback Loops** - Learn from outcomes

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pattern bloat | Medium | Medium | Active compression, token budgets |
| Wrong patterns learned | Low | High | Confidence thresholds, decay |
| Brain unavailable | Low | Medium | Fallback to static memory |
| Performance overhead | Medium | Low | Caching, async operations |
| Pattern conflicts | Low | Medium | Hierarchy (specific overrides general) |

---

## 12. Success Metrics

### Quantitative

| Metric | Target | Measurement |
|--------|--------|-------------|
| Task success rate | +10% improvement | Compare before/after brain |
| Time to completion | -15% reduction | Average task duration |
| Token efficiency | -20% waste | Context tokens vs useful |
| Pattern coverage | 80% of task types | Tasks with relevant patterns |

### Qualitative

- Agents show consistent behavior across sessions
- Common mistakes are not repeated
- Best practices propagate across swarm
- Less explicit guidance needed in prompts

---

## 13. Open Questions

1. **Pattern Conflict Resolution**
   - What if global and swarm patterns conflict?
   - Current proposal: More specific wins

2. **Human Feedback Collection**
   - How to get explicit success/failure signals?
   - Consider: Inline rating, periodic review

3. **Pattern Explainability**
   - How to show users what the brain learned?
   - UI for pattern inspection

4. **Cold Start**
   - How to bootstrap with no history?
   - Consider: Seed with best practices document

---

## 14. Files to Create

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `brain/__init__.py` | Package init | 20 |
| `brain/server.py` | FastAPI server | 300 |
| `brain/experience.py` | Experience storage | 200 |
| `brain/learner.py` | Learning engine | 400 |
| `brain/distiller.py` | Knowledge distillation | 200 |
| `brain/context.py` | Context synthesis | 250 |
| `brain/awareness.py` | Swarm awareness | 150 |
| `brain/embeddings.py` | Embedding engine | 100 |
| `brain/models.py` | Pydantic models | 150 |
| `backend/routes/brain.py` | Proxy endpoints | 100 |

**Total**: ~1,870 lines of new code

---

## 15. Related Documents

- `/docs/MEMORY_ARCHITECTURE.md` - Current memory system
- `/workspace/LOCAL_NEURAL_BRAIN_DESIGN.md` - Local brain research
- `/docs/IDEAS.md` - Training-Free GRPO concept
- `/swarms/mynd_app/workspace/mynd-server/mynd-brain/` - MYND reference

---

## Appendix A: Request/Response Models

```python
# brain/models.py

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ContextRequest(BaseModel):
    """Request for agent context."""
    agent_type: str
    swarm_name: Optional[str] = None
    task_prompt: str
    max_tokens: int = 4000
    include_patterns: bool = True
    include_experiences: bool = True
    include_swarm_state: bool = True

class ContextResponse(BaseModel):
    """Response with synthesized context."""
    context: str
    token_count: int
    breakdown: Dict[str, int]

class LearnRequest(BaseModel):
    """Request to learn from task outcome."""
    task_prompt: str
    agent_type: str
    trajectory: List[Dict[str, Any]]
    success: bool
    outcome_signal: str  # "user_approved", "tests_passed", etc.

class LearnResponse(BaseModel):
    """Response from learning."""
    learned: bool
    pattern_name: Optional[str] = None
    patterns_count: int

class SearchRequest(BaseModel):
    """Request to search experiences."""
    query: str
    agent_type: Optional[str] = None
    success_only: bool = True
    limit: int = 5

class SearchResponse(BaseModel):
    """Response with matching experiences."""
    experiences: List[Dict[str, Any]]
    count: int

class LearnedPattern(BaseModel):
    """A pattern learned from experiences."""
    name: str
    task_type: str
    applies_to: List[str]
    guidance: str
    avoid: str
    confidence: str  # "high", "medium", "low"
    success_rate: float
    sample_count: int
    estimated_tokens: int
    created_at: datetime
    last_used: datetime
```

---

## Appendix B: Example Context Output

```markdown
# Agent Context

## Role: Implementer in swarm_dev

You are an Implementation Specialist focused on writing clean, tested code.

---

## Learned Patterns (from past experience)

### Pattern: Read-Before-Edit
**Applies to**: code_change, refactor
**Confidence**: high (15 successes, 3 failures)

When editing a file, ALWAYS read it first using the Read tool. Edits that
fail often skip this step and use stale assumptions about file contents.

**Do**: `Read(file) -> Edit(file, old_string, new_string)`
**Avoid**: `Edit(file, assumed_content, new_content)` without reading

### Pattern: Test-After-Change
**Applies to**: code_change, bug_fix
**Confidence**: medium (8 successes, 2 failures)

After modifying code, run the relevant test suite.

---

## Similar Past Tasks (successful)

**Task**: "Add validation to the user input handler"
**Approach**: Read existing handler, understand pattern, add validation, run tests
**Tools Used**: Read, Edit, Bash

---

## Swarm State

### Active Agents
- **researcher**: investigating caching options
- **architect**: reviewing PR #67

### Agent Strengths
- code_change: Best handled by **implementer**
- research: Best handled by **researcher**

---

## Current Context

### swarm_dev Mission
Build and maintain the agent swarm infrastructure.

### Active Work
- [ ] Main.py refactoring - Phase 3 in progress
- [x] COO tool restriction - Complete
```
