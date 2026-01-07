# Optimal Hierarchical Delegation Pattern

**Author:** System Architect
**Date:** 2026-01-03
**Status:** PROPOSED

---

## Executive Summary

This document defines the optimal delegation pattern for the agent-swarm hierarchical system, addressing inconsistencies where the COO (Supreme Orchestrator) claims to delegate but does not actually spawn agents, does work itself that should be delegated, or spawns agents without proper follow-through.

---

## 1. Current State Analysis

### 1.1 How Delegation Currently Works

**Execution Flow:**
1. User sends message via WebSocket (`backend/main.py:2496-2676`)
2. COO receives message and processes via Claude CLI with streaming JSON output
3. COO uses the `Task` tool to spawn sub-agents
4. Task tool input is streamed via `input_json_delta` events
5. When Task tool detected, `agent_spawn` event sent to frontend (`main.py:2260-2270`)
6. Sub-agent runs as nested Claude CLI process
7. Results stream back to COO for synthesis

**Key Code Locations:**
- COO system prompt: `backend/main.py:2550-2606`
- Task tool detection: `backend/main.py:2239-2271`
- Agent executor pool: `shared/agent_executor_pool.py`
- Swarm agent definitions: `swarms/*/swarm.yaml`

### 1.2 Identified Problems

**Problem 1: COO Claims Delegation But Doesn't Act**
- Root cause: COO's system prompt describes delegation patterns but doesn't enforce them
- The COO can describe "I'll delegate to researcher" without using the Task tool
- No verification that Task tool was actually invoked

**Problem 2: COO Does Work Itself When It Should Delegate**
- Root cause: No clear decision criteria in the system prompt
- COO optimizes for speed/simplicity rather than organizational structure
- Small tasks that could benefit from specialist review are done directly

**Problem 3: Spawned Agents Without Follow-Through**
- Root cause: Fire-and-forget pattern - no result tracking
- Agent spawns but COO doesn't wait for or synthesize results
- No visibility into whether delegated work completed

**Problem 4: Race Condition in Agent Stack (Existing Issue)**
- At `content_block_start`, tool input is empty (streams via `input_json_delta`)
- Agent pushed to stack with potentially incomplete data
- Known issue documented in `STATE.md:55-60`

### 1.3 What's Working Well

- Task tool streaming detection (`main.py:2239-2271`) correctly accumulates JSON
- Agent stack tracking concept is sound
- WebSocket event system provides visibility
- Work Ledger and Mailbox systems exist for persistence (ADR-003, ADR-004)

---

## 2. Optimal Hierarchy Model

### 2.1 The Three-Tier Hierarchy

```
CEO (Human)
    |
    v
COO (Supreme Orchestrator)
    |
    +-- Executive Team (direct reports)
    |   |-- Chief of Staff
    |   |-- Project Manager
    |   |-- Context Keeper
    |   +-- VP Operations
    |
    +-- Swarm Orchestrators
        |
        +-- Swarm Agents (implementer, critic, etc.)
```

### 2.2 Role Definitions

**COO (Supreme Orchestrator)**
- Scope: Strategic decisions, cross-swarm coordination, CEO interaction
- Tools: Task (primary), Read (for context), Bash (git status, system health)
- Should NOT: Write code, edit files, run tests, deep research

**Swarm Orchestrator**
- Scope: Intra-swarm coordination, task breakdown, agent selection
- Tools: Task (primary), Read, Glob (discovery)
- Should NOT: Implement, test, or review directly

**Swarm Agents (implementer, critic, etc.)**
- Scope: Actual work - implementation, testing, review
- Tools: Read, Write, Edit, Bash, Grep, Glob
- Should NOT: Use Task tool (leaf nodes)

### 2.3 Delegation Rules

**Mandatory Delegation (COO MUST delegate):**
1. Any task > 5 lines of code
2. Any multi-file change
3. Any task requiring tests
4. Any architecture decision
5. Deep research (> 2 sources)
6. Code review

**Optional Delegation (COO MAY do directly):**
1. Single file read for context
2. Simple grep/glob for discovery
3. Git status check
4. Reading STATE.md
5. Synthesizing agent results
6. Answering meta-questions about the system

**Never Delegate:**
1. CEO-decision escalations
2. Cross-swarm priority conflicts
3. Strategic planning synthesis

---

## 3. Decision Tree: When to Do vs Delegate

```
START: User Request Received
    |
    v
[Is this a CEO question/decision?]
    |
    +--YES--> Handle directly, format as CEO DECISION REQUIRED
    |
    +--NO--> [Does it require file changes?]
                |
                +--YES--> [How many files?]
                |            |
                |            +--1 file, <5 lines--> Consider doing (see criteria)
                |            |
                |            +--Otherwise--> DELEGATE to implementer
                |
                +--NO--> [Is it research/information gathering?]
                            |
                            +--YES--> [How deep?]
                            |            |
                            |            +--Single file read--> Do directly
                            |            |
                            |            +--Multi-source--> DELEGATE to researcher
                            |
                            +--NO--> [Is it code review/quality?]
                                        |
                                        +--YES--> DELEGATE to critic
                                        |
                                        +--NO--> [Is it status/briefing?]
                                                    |
                                                    +--YES--> Spawn executive team in parallel
                                                    |
                                                    +--NO--> Route to appropriate swarm
```

### 3.1 "Consider Doing" Criteria

If task appears simple (1 file, <5 lines), still delegate if:
- It touches critical code (auth, payments, security)
- It requires domain expertise
- It could benefit from review
- User explicitly asked for quality over speed

---

## 4. Anti-Patterns to Avoid

### Anti-Pattern 1: Delegation Theater
**Description:** COO describes the delegation in prose but doesn't invoke Task tool
**Example:**
```
"I'll have the implementer handle this..."
[No Task tool call follows]
```
**Solution:** System must verify Task tool invocation when delegation described

### Anti-Pattern 2: Over-Delegation
**Description:** COO delegates trivial work, adding overhead
**Example:** Spawning researcher to read a single README
**Solution:** Clear criteria in decision tree; simple reads done directly

### Anti-Pattern 3: Fire-and-Forget
**Description:** Agent spawned but results not tracked or synthesized
**Example:**
```
Task(subagent_type="implementer", prompt="...")
[COO immediately responds without waiting]
```
**Solution:** Track pending tasks, require synthesis step

### Anti-Pattern 4: Sequential Over-Caution
**Description:** COO runs agents sequentially when parallel is possible
**Example:** Wait for researcher before spawning implementer
**Solution:** Explicit parallel patterns in prompt; spawn together when possible

### Anti-Pattern 5: Context Loss
**Description:** Delegated agent lacks context from prior conversation
**Example:** Agent doesn't know about previous decisions/constraints
**Solution:** Include context in Task prompt; reference STATE.md

### Anti-Pattern 6: Recursive Delegation
**Description:** Agent A delegates to B who delegates to C who delegates...
**Example:** Orchestrator -> Implementer -> (creates subtask) -> another implementer
**Solution:** Limit delegation depth; only orchestrators should use Task

---

## 5. Concrete Implementation Recommendations

### 5.1 System Prompt Improvements

**Current COO prompt location:** `backend/main.py:2550-2606`

**Recommended additions:**

```python
# Add after "## When to Do vs Delegate" section

## ENFORCEMENT RULES

1. **Never describe delegation without acting.** If you say "I'll delegate to X",
   immediately follow with a Task() call. Do not just describe the plan.

2. **Always wait for results.** After spawning agents, explicitly wait and
   synthesize their responses before completing your response.

3. **Track your delegations.** Before responding, verify each Task() you planned
   actually completed. Report: "Delegated to: [list]. Received results from: [list]."

4. **Use parallel spawning.** When spawning multiple agents, do it in one turn:
   - Task(subagent_type="researcher", ...)
   - Task(subagent_type="implementer", ...)
   - Task(subagent_type="critic", ...)
   Then synthesize all results together.

5. **Include context in every Task.** Every Task prompt should include:
   - Reference to STATE.md for swarm context
   - Specific success criteria
   - How results will be used
```

### 5.2 Delegation Tracking

**Integration with existing Work Ledger (`shared/work_ledger.py`):**

When COO spawns a Task:
1. Create WorkItem with `type=TASK`, parent_id=session_work_id
2. Set status to `IN_PROGRESS`
3. Track in context as pending_delegations
4. On agent completion, update WorkItem status
5. COO must synthesize before WorkItem marked `COMPLETED`

```python
# Pseudocode for delegation tracking
class DelegationTracker:
    def on_task_spawn(self, agent_name: str, prompt: str, parent_work_id: str) -> str:
        """Called when Task tool invoked. Returns delegation_id."""
        work_id = work_ledger.create_subtask(
            parent_id=parent_work_id,
            title=f"Delegation to {agent_name}",
            work_type=WorkType.TASK,
            owner=agent_name,
            context={"prompt": prompt},
        )
        return work_id

    def on_task_complete(self, delegation_id: str, result: str):
        """Called when spawned agent completes."""
        work_ledger.complete_work(delegation_id, result=result)

    def get_pending(self, parent_work_id: str) -> list[WorkItem]:
        """Get all pending delegations for a work item."""
        return work_ledger.get_children(parent_work_id, status=WorkStatus.IN_PROGRESS)
```

### 5.3 Result Synthesis Enforcement

**Add to WebSocket chat handler (`backend/main.py`):**

Before sending `chat_complete`:
1. Check context["pending_tasks"]
2. If any pending, require synthesis in response
3. Log warning if COO response doesn't reference delegated work

```python
# Add before line 2668 (await manager.send_event chat_complete)

# Check for unresolved delegations
pending_tasks = context.get("pending_tasks", {})
completed_tasks = context.get("completed_tasks", set())
unresolved = set(pending_tasks.keys()) - completed_tasks

if unresolved:
    logger.warning(
        f"COO completed with {len(unresolved)} unresolved delegations: "
        f"{[pending_tasks[t] for t in unresolved]}"
    )
    # Could inject a synthesis prompt here in future
```

### 5.4 Parallel Spawning Pattern

**Recommended prompt pattern for COO:**

```
For multi-step work, spawn agents in parallel using multiple Task calls in the same response:

Task(subagent_type="researcher", prompt="Research X. Write findings to workspace/research_X.md")
Task(subagent_type="implementer", prompt="Implement Y. Read architect's design first.")
Task(subagent_type="critic", prompt="Review the implementation in workspace/. Report issues.")

All three will execute concurrently. Wait for all results before synthesizing.
```

### 5.5 Agent Prompt Updates

**Orchestrator agents should include:**
- Explicit delegation authority
- Clear list of available sub-agents
- Pattern examples for parallel spawning
- STATE.md read/update requirements

**Worker agents (implementer, critic) should include:**
- No Task tool access
- Direct work only
- Report results to STATE.md
- Explicit completion signals

---

## 6. How Agents Should Report Back Results

### 6.1 Synchronous Reporting (Default)

Claude CLI streams results back via stdout JSON events. The parent agent receives:
- Tool result with agent output
- This happens automatically for Task tool

### 6.2 Persistent Reporting (For Long Tasks)

For background/long-running tasks:

1. **STATE.md Updates**
   - Agent writes progress to `workspace/STATE.md`
   - Include timestamp, agent name, action taken
   - Example: `### 2026-01-03 - implementer - Fixed auth bug in login.py`

2. **Mailbox Messages (for async handoffs)**
   - Use `agent_mailbox.py` for structured handoffs
   - Agent sends message on completion
   - Orchestrator checks mailbox for background results

3. **Work Ledger Updates**
   - Agent marks WorkItem as COMPLETED with result summary
   - Orchestrator queries ledger for work status

### 6.3 Recommended Result Format

Agents should end their response with a structured summary:

```
## Work Complete

**Task:** [What was assigned]
**Status:** COMPLETE | PARTIAL | BLOCKED
**Actions Taken:**
- [List of actions]
**Files Modified:**
- [List of files]
**Result:** [Summary of outcome]
**STATE.md Updated:** Yes/No
**Next Steps:** [If any]
```

---

## 7. Ensuring Delegated Work Completes

### 7.1 Timeout and Recovery

**Process-level timeouts:**
- `AgentExecutorPool` has `context.timeout` (default 600s)
- On timeout, process killed, error event emitted

**Session-level tracking:**
- Work Ledger tracks all work items
- `recover_orphaned_work()` reclaims stale items
- COO can query work status at session start

### 7.2 Completion Verification Checklist

Before COO marks a delegated task as done:

1. [ ] Task tool completed (tool_result received)
2. [ ] Agent output parsed (no error events)
3. [ ] Result referenced in synthesis
4. [ ] STATE.md updated (if required)
5. [ ] WorkItem status updated (if tracking)

### 7.3 Failure Handling

On delegation failure:
1. Log error with agent name and prompt
2. Update WorkItem to FAILED
3. Inform COO in response stream
4. COO decides: retry, escalate, or abort

---

## 8. Implementation Plan

### Phase 1: Prompt Hardening (1 day)
1. Update COO system prompt with enforcement rules
2. Add delegation tracking reminders
3. Include parallel spawning examples
4. Test with sample requests

**Files to modify:**
- `backend/main.py:2550-2606` (system_prompt)

### Phase 2: Result Tracking (2-3 days)
1. Add delegation tracking to work ledger
2. Log unresolved delegations
3. Add synthesis verification warning
4. Surface pending work in UI

**Files to modify:**
- `backend/main.py` (chat handler)
- `shared/work_ledger.py` (add parent/child queries)

### Phase 3: Agent Prompt Updates (1 day)
1. Update orchestrator prompts with delegation patterns
2. Update worker prompts to remove Task tool
3. Add result format requirements

**Files to modify:**
- `swarms/*/agents/*.md`
- `supreme/agents/*.md`

### Phase 4: Observability (1-2 days)
1. Add delegation metrics logging
2. Surface pending delegations in ActivityPanel
3. Add completion rate tracking

**Files to modify:**
- `frontend/components/ActivityPanel.tsx`
- `backend/main.py` (metrics)

---

## 9. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Delegation claims without Task call | Unknown | 0% |
| Fire-and-forget delegations | Unknown | <5% |
| Parallel spawning usage | Low | >60% for multi-agent work |
| Result synthesis inclusion | Variable | 100% |
| Work completion rate | Unknown | >95% |

---

## 10. Key Takeaways

1. **Explicit is better than implicit.** Don't describe delegation - do it.
2. **Track everything.** Use Work Ledger for persistent tracking.
3. **Parallel by default.** Spawn agents together, not sequentially.
4. **Synthesize always.** Every delegation requires result synthesis.
5. **Update STATE.md.** It's the shared memory across agent lifecycles.

---

## Appendix A: Task Tool Reference

**Tool signature:**
```
Task(subagent_type="agent_name", prompt="What to do", description="Short summary")
```

**Valid subagent_types:**
- `researcher` - Deep research, documentation analysis
- `architect` - System design, planning
- `implementer` - Code implementation
- `critic` - Code review, quality assessment
- `tester` - Test creation and verification
- Swarm-qualified: `swarm_dev/implementer`, `asa/researcher`

**Best practices:**
- Include success criteria in prompt
- Reference STATE.md for context
- Keep prompts focused and specific
- Use description for ActivityPanel visibility

---

## Appendix B: Related Documents

- `/workspace/MAILBOX_DESIGN.md` - Agent Mailbox system (ADR-003)
- `/workspace/WORK_LEDGER_DESIGN.md` - Work Ledger system (ADR-004)
- `/workspace/STATE.md` - Current state and progress log
- `/docs/ROADMAP.md` - Organizational priorities
