# Review Workflow Audit

**Date**: 2026-01-03
**Auditor**: Researcher Agent
**Scope**: Swarm Dev review process, gates, and enforcement mechanisms

---

## Executive Summary

The agent-swarm system has a **well-documented but poorly enforced** review workflow. The review process is defined in configuration files and agent prompts, but **lacks technical enforcement at the code level**. Reviews currently rely entirely on agent cooperation and orchestrator discipline.

**Overall Assessment**: PARTIALLY IMPLEMENTED - Advisory reviews exist, but no blocking gates

---

## 1. Current Review Process

### 1.1 Documented Workflow (swarm.yaml lines 64-75)

```yaml
review_workflow:
  - step: implement
    agent: implementer
  - step: review
    agent: reviewer
    required: true
  - step: critique
    agent: critic
    required: true
  - step: refactor_check
    agent: refactorer
    required: false
```

### 1.2 What Triggers a Review?

| Trigger | Source | Enforcement |
|---------|--------|-------------|
| Orchestrator instruction | orchestrator.md lines 44-52 | Advisory - orchestrator told to "Spawn reviewer and critic IN PARALLEL after implementation" |
| STATE.md policy | STATE.md line 89 | "All implementations must be reviewed by critic before merging" |
| Swarm config | swarm.yaml `required: true` | **NOT ENFORCED** - config is read but not acted upon |

**Finding**: There is **no automated trigger**. Reviews happen only if:
1. The orchestrator agent remembers to spawn reviewer/critic
2. Or a human explicitly requests review

### 1.3 Who Performs Reviews?

| Agent | Role | Review Focus | Blocking Power |
|-------|------|--------------|----------------|
| **reviewer** | L1 Review | Correctness, bugs, logic errors | Advisory only |
| **critic** | L2 Review | Design patterns, security, edge cases | "Security issues are BLOCKING" (per prompt, not code) |
| **refactorer** | Optional | Technical debt, cleanup | Non-blocking |
| **architect** | Design review | Architecture fit, consistency | Advisory |

### 1.4 What Happens After Review?

**Current State**:
1. Reviewer/Critic produce markdown reports with "Approved: Yes/No" fields
2. Reports are written to workspace or STATE.md
3. If issues found, STATE.md is updated (e.g., "NEEDS_CHANGES")
4. **No automatic blocking** - implementer can continue even if not approved

**Evidence from STATE.md (lines 61-64)**:
```
### 2026-01-02: Phase 0.1.2 - Critic Review: NEEDS_CHANGES
- **Critic** reviewed backend integration of AgentExecutorPool
- **Result**: NEEDS_CHANGES before proceeding
- Issues identified and now fixed (see entry above)
```

This worked because the orchestrator (or human) respected the NEEDS_CHANGES status.

---

## 2. Review Gates Analysis

### 2.1 Gate: Before Code is Implemented

| Gate | Status | Location |
|------|--------|----------|
| Design review by architect | **MISSING** | No gate exists |
| "Ready for implementation" approval | **MISSING** | Implementation can start without explicit design approval |

**Gap**: Architect produces design documents (e.g., `DESIGN_AGENT_EXECUTOR.md`), but there is no formal approval step before implementer begins.

### 2.2 Gate: Before Code is Merged

| Gate | Status | Location |
|------|--------|----------|
| Reviewer L1 approval | **ADVISORY** | orchestrator.md states workflow, not enforced |
| Critic L2 approval | **ADVISORY** | swarm.yaml marks as `required: true` but no code checks |
| Test passing | **MISSING** | No automated test gate |

**Gap**: The `required: true` in swarm.yaml is purely documentation - no backend code reads or enforces this field.

### 2.3 Gate: Before Deployment

| Gate | Status | Location |
|------|--------|----------|
| Branch protection | **EXTERNAL** | GitHub branch protection (if configured) |
| PR approval | **EXTERNAL** | Depends on GitHub settings |
| Test CI | **PARTIAL** | CI exists but not integrated with agent workflow |

**Gap**: Agents can push to feature branches without any internal gates. Deployment gates rely entirely on external GitHub configuration.

---

## 3. Critic Integration Analysis

### 3.1 Critic Agent Definition (agents/critic.md)

The critic is well-defined with:
- OWASP security checklist (lines 35-39)
- Pattern violation examples (lines 42-71)
- Review format template (lines 80-97)
- Explicit statement: "Security issues are BLOCKING" (line 101)

### 3.2 Where is Critic Supposed to be Used?

| Location | Reference | Status |
|----------|-----------|--------|
| orchestrator.md line 49 | "critic validates design/security" | Documented |
| swarm.yaml line 72 | `required: true` | Configured, not enforced |
| STATE.md line 89 | "All implementations must be reviewed by critic before merging" | Policy, not code |
| ROADMAP.md line 42 | "Critic can check for security issues" | Success criteria |
| ROADMAP.md line 112 | "Critic must participate in every consensus round" | Future goal |

### 3.3 Is Critic Review Enforced?

**NO** - There is no code in the backend that:
1. Checks if critic has reviewed before allowing merge
2. Blocks implementation completion without critic approval
3. Prevents git push without review status

**Evidence**: Searched `backend/main.py` and `backend/jobs.py` - neither file references:
- `critic`
- `review_workflow`
- The `required` field from swarm.yaml

### 3.4 Critic Effectiveness (Historical)

From STATE.md, the critic has been effective when used:

| Date | Review | Outcome |
|------|--------|---------|
| 2026-01-02 | Phase 0.1.2 review | NEEDS_CHANGES -> Fixed 4 critical issues |
| 2026-01-02 | Security review | MEDIUM-HIGH risk identified |

This shows **human-in-the-loop enforcement** - reviews worked because someone respected the output.

---

## 4. Gap Analysis

### 4.1 Missing Review Gates

| Gap | Priority | Impact |
|-----|----------|--------|
| **No "ready for implementation" gate** | HIGH | Design documents exist but no approval step prevents premature implementation |
| **No "implementation approved" gate** | CRITICAL | Nothing blocks merging unapproved code except external GitHub protections |
| **No automated test gate** | HIGH | Tests exist but aren't required to pass before code proceeds |
| **Review workflow not enforced** | CRITICAL | `required: true` is documentation only |

### 4.2 Are Reviews Actually Blocking?

**NO** - Reviews are entirely advisory:

1. **Orchestrator can ignore review results** - Nothing stops it from proceeding
2. **No state machine** - There's no formal state tracking (e.g., PENDING_REVIEW -> APPROVED -> MERGEABLE)
3. **No enforcement code** - The `review_workflow` config in swarm.yaml is never read by any backend code
4. **Git operations unblocked** - Implementer can push even without reviews

### 4.3 What Would Make Reviews Blocking?

To enforce reviews, the system would need:

1. **State Machine**: Track work items through states:
   ```
   DESIGNED -> READY_FOR_IMPL -> IMPLEMENTED -> REVIEWED -> APPROVED -> MERGED
   ```

2. **Gate Checks**: Before state transitions, verify:
   - Has reviewer approved? (L1)
   - Has critic approved? (L2)
   - Do tests pass?

3. **Backend Enforcement**: Modify `backend/jobs.py` or a new module to:
   - Read `review_workflow` from swarm.yaml
   - Track review status per work item
   - Block git push/merge without approvals

4. **Integration with Escalation**: When reviews reject, auto-escalate

---

## 5. Recommendations

### 5.1 Immediate (0-1 weeks)

1. **Add Review Status to STATE.md Format**
   - Add a "Review Status" section to STATE.md template
   - Track: `PENDING_REVIEW | NEEDS_CHANGES | APPROVED`
   - Make orchestrator read this before allowing merge

2. **Update Orchestrator Prompt**
   - Add explicit gate checks: "DO NOT allow merge until critic has approved"
   - Add checklist the orchestrator must verify

3. **Document Review Requirements**
   - Create `workspace/REVIEW_GATES.md` defining mandatory gates
   - List what requires review vs what can skip

### 5.2 Short-term (1-4 weeks)

4. **Create WorkItem Tracking**
   - Build on existing `shared/work_ledger.py` (found in git status)
   - Add review_status field
   - Add gate checks before state transitions

5. **Enforce Review Workflow in Jobs**
   - Modify `backend/jobs.py` to read `review_workflow` from swarm.yaml
   - Add review check before job completion

6. **Add Review-Required Flag to API**
   - Add endpoint: `GET /api/work/{id}/review_status`
   - Block merge API calls without approved status

### 5.3 Medium-term (1-3 months)

7. **Implement Consensus Protocol Integration**
   - Per ROADMAP.md line 112: "Critic must participate in every consensus round"
   - Build `shared/consensus.py` with mandatory critic voting

8. **Add Pre-commit Hooks**
   - Create `.claude/hooks/pre-commit` that checks review status
   - Block git commits without approved state

9. **Build Review Dashboard**
   - Show pending reviews in frontend
   - Alert when reviews are overdue

---

## 6. Key Files Identified

### Configuration
| File | Purpose |
|------|---------|
| `/Users/jellingson/agent-swarm/swarms/swarm_dev/swarm.yaml` | review_workflow config (not enforced) |
| `/Users/jellingson/agent-swarm/docs/ROADMAP.md` | Strategic review goals |

### Agent Definitions
| File | Purpose |
|------|---------|
| `/Users/jellingson/agent-swarm/swarms/swarm_dev/agents/critic.md` | L2 security/design review |
| `/Users/jellingson/agent-swarm/swarms/swarm_dev/agents/reviewer.md` | L1 correctness review |
| `/Users/jellingson/agent-swarm/swarms/swarm_dev/agents/orchestrator.md` | Workflow coordination (advisory) |

### Backend (Enforcement Points)
| File | Purpose |
|------|---------|
| `/Users/jellingson/agent-swarm/backend/main.py` | API endpoints (no review checks) |
| `/Users/jellingson/agent-swarm/backend/jobs.py` | Job execution (no review gates) |

### Potential New Files
| File | Purpose |
|------|---------|
| `shared/review_gates.py` | Enforce review status transitions |
| `shared/work_item_state.py` | State machine for work items |

---

## 7. Summary

### What Works
- Review agents (reviewer, critic) are well-defined with clear checklists
- Review workflow is documented in swarm.yaml
- STATE.md captures review outcomes when reviews happen
- Historical evidence shows reviews catch real issues

### What's Missing
- **No technical enforcement** - reviews are advisory only
- **No state machine** - work items have no formal lifecycle
- **No gates** - implementation and merge proceed without approval
- **Backend ignores config** - `review_workflow` and `required: true` are never read

### Risk Assessment
**MEDIUM-HIGH** - The system relies on:
1. Orchestrator agent discipline (can be inconsistent)
2. Human oversight (not scalable)
3. External GitHub protections (not integrated)

Without blocking gates, low-quality code can enter the codebase despite defined review processes.

---

## 8. Next Steps

1. **Architect**: Design `shared/review_gates.py` with state machine
2. **Implementer**: Build review status tracking into work_ledger.py
3. **Critic**: Review this audit and proposed design
4. **Orchestrator**: Update workflow to include explicit gate checks

---

*Audit complete. Findings should be discussed in STATE.md Progress Log.*
