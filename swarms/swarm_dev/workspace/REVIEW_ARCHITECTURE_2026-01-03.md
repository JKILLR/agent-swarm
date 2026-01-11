# Architecture Review: Agent-Swarm System
**Date**: 2026-01-03
**Reviewer**: Architect Agent
**Scope**: System connections, integration points, orphaned code

---

## Executive Summary

The agent-swarm system has a well-designed modular architecture with several execution pathways. However, there are **significant integration gaps** where new components have been built but are **not fully wired into the main execution flow**. The system has redundant execution mechanisms that could cause confusion about which pathway is actually used.

### Overall Status: NEEDS ATTENTION

| Category | Status | Notes |
|----------|--------|-------|
| Backend API to Frontend | CONNECTED | WebSocket streaming works |
| Workspace Manager | CONNECTED | Properly initialized in startup |
| Executor Pool | PARTIALLY CONNECTED | Integrated but bypassed in main flow |
| Work Ledger | NOT CONNECTED | Built but not wired |
| Agent Mailbox | NOT CONNECTED | Built but not wired |
| Escalation Protocol | NOT CONNECTED | Built but not wired |
| Old AgentExecutor | ORPHANED | Superseded by pool |

---

## 1. System Connection Analysis

### 1.1 Backend API (backend/main.py)

**Status: PROPERLY CONNECTED**

The FastAPI backend correctly:
- Initializes `WorkspaceManager` and `AgentExecutorPool` in `startup_event()` (lines 131-141)
- Sets up event broadcasting callback for pool events (lines 136-140)
- Exposes new endpoints: `/api/agents/execute` and `/api/agents/pool/status`

**WebSocket Flow (MAIN EXECUTION PATH):**
```
/ws/chat (line 1887)
  -> stream_claude_response() (line 1299)
    -> asyncio.create_subprocess_exec() (line 1353)
    -> parse_claude_stream() (line 1365)
```

**CRITICAL FINDING**: The main WebSocket chat flow **DOES NOT USE** the AgentExecutorPool. It directly spawns Claude CLI processes via `stream_claude_response()`.

The executor pool is only used by:
1. The `/api/agents/execute` REST endpoint (line 454)
2. The job system via `_execute_with_pool()` when `use_pool=True`

### 1.2 Agent Execution Flow

**Three Parallel Execution Mechanisms Exist:**

| Mechanism | File | Used By | Status |
|-----------|------|---------|--------|
| `stream_claude_response()` | backend/main.py:1299 | WebSocket chat (COO) | PRIMARY |
| `AgentExecutorPool.execute()` | shared/agent_executor_pool.py:80 | REST API, Jobs (optional) | SECONDARY |
| `AgentExecutor.execute()` | shared/agent_executor.py:70 | Not used by main flows | ORPHANED |

**Recommendation**: Consolidate on AgentExecutorPool for all agent execution to ensure consistent workspace isolation, event streaming, and resource management.

### 1.3 Workspace Isolation

**Status: PROPERLY INITIALIZED BUT INCONSISTENTLY USED**

`WorkspaceManager` is correctly:
- Initialized in `startup_event()` with `PROJECT_ROOT` (line 132)
- Used in `/api/agents/execute` endpoint (line 476)
- Used in jobs.py `_execute_with_pool()` (line 541)

**Gap**: The main WebSocket chat flow (`stream_claude_response()`) does NOT use workspace isolation - it runs in PROJECT_ROOT regardless of swarm context.

```python
# line 1344-1345 in main.py
cwd = str(workspace) if workspace else None
# ... but workspace is always PROJECT_ROOT for COO
```

### 1.4 Jobs System

**Status: CONNECTED WITH DUAL PATHS**

`backend/jobs.py` properly:
- Imports workspace manager and executor pool (lines 27-30)
- Has `use_pool` flag (line 265, default True)
- Implements `_execute_with_pool()` method (line 539)
- Falls back to `_execute_chat_job()` when `use_pool=False`

**Issue**: The fallback path still uses the old import:
```python
# line 429
from main import stream_claude_response  # Uses relative import
```

This was flagged as fixed in STATE.md but the fallback path still exists.

---

## 2. Orphaned/Disconnected Components

### 2.1 shared/agent_executor.py - ORPHANED

**File**: `/Users/jellingson/agent-swarm/shared/agent_executor.py`

This file provides `AgentExecutor` class and helper functions:
- `get_executor()`
- `execute_agent()`
- `stream_agent()`

**Analysis**: These are **NOT imported or used** anywhere in the codebase. They have been superseded by:
- `AgentExecutorPool` (for concurrent execution with isolation)
- Direct CLI invocation in `stream_claude_response()`

**Recommendation**: Either:
1. Remove this file as dead code
2. Or refactor to use it as the base for both execution paths

### 2.2 shared/work_ledger.py - NOT INTEGRATED

**File**: `/Users/jellingson/agent-swarm/shared/work_ledger.py`
**Exported in**: `shared/__init__.py` (lines 16-26)

The WorkLedger provides:
- Persistent work item tracking
- Crash-resilient task state
- Dependency management
- Progress tracking

**Integration Status**:
- Exported from shared module
- **NOT imported** in backend/main.py
- **NOT imported** in backend/jobs.py
- **NOT used** by any execution flow

**Recommendation**: Integrate with:
1. Job system (create WorkItem for each job)
2. Task tool delegations (track spawned agent work)
3. COO orchestration (manage work breakdown)

### 2.3 shared/agent_mailbox.py - NOT INTEGRATED

**File**: `/Users/jellingson/agent-swarm/shared/agent_mailbox.py`
**Exported in**: `shared/__init__.py` (lines 28-40)

The MailboxManager provides:
- Agent-to-agent messaging
- Structured handoffs
- Priority queuing
- Thread-based conversations

**Integration Status**:
- Exported from shared module
- **NOT imported** anywhere else
- **NOT used** by any agent communication

**Recommendation**: Integrate with:
1. Task tool completions (handoff results)
2. Agent wake-up notifications
3. COO to agent communication

### 2.4 shared/escalation_protocol.py - NOT INTEGRATED

**File**: `/Users/jellingson/agent-swarm/shared/escalation_protocol.py`

The EscalationManager provides:
- Agent-to-COO escalations
- COO-to-CEO escalations
- Blocked work tracking

**Integration Status**:
- Implemented per DESIGN_ESCALATION_PROTOCOL.md
- **NOT imported** in backend/main.py
- **NOT exposed** via any API endpoint
- **NOT used** in WebSocket events

**Recommendation (per STATE.md Next Steps):**
1. Add escalation WebSocket events to backend/main.py
2. Update agent system prompts with escalation guidance
3. Add REST endpoints: `/api/escalations`, `/api/escalations/{id}`

---

## 3. Integration Gaps Summary

### 3.1 Critical Gaps

| Gap | Impact | Fix Complexity |
|-----|--------|----------------|
| Main chat bypasses executor pool | No isolation for COO | Medium |
| Work ledger not connected | Work not persisted | Medium |
| Escalations not exposed | No escalation flow | Low |

### 3.2 Medium Gaps

| Gap | Impact | Fix Complexity |
|-----|--------|----------------|
| Mailbox not connected | No agent messaging | Medium |
| Orphaned agent_executor.py | Code confusion | Low |
| Dual execution paths | Inconsistent behavior | Medium |

### 3.3 Low Priority Gaps

| Gap | Impact | Fix Complexity |
|-----|--------|----------------|
| `allowed_tools` not enforced | Informational only | N/A (design choice) |
| Unused `_running` dict in pool | Dead code | Low |

---

## 4. Connection Diagram

```
Frontend (Next.js)
    |
    v
[WebSocket /ws/chat]           [REST /api/agents/execute]
    |                                   |
    v                                   v
stream_claude_response()       AgentExecutorPool.execute()
    |                                   |
    |    NOT CONNECTED                  |
    |    +----------------+             |
    |    | WorkLedger     |             |
    |    | Mailbox        |             |
    |    | Escalations    |             |
    |    +----------------+             |
    |                                   |
    +--------->  WorkspaceManager  <----+
                      |
                      v
               Claude CLI
```

---

## 5. Recommendations

### 5.1 Immediate Actions (Phase 0.1.3)

1. **Wire escalation protocol to WebSocket**
   - Import `get_escalation_manager()` in main.py
   - Add WebSocket event types for escalation create/resolve
   - Add REST endpoints for escalation CRUD

2. **Unify execution paths**
   - Modify `stream_claude_response()` to use `AgentExecutorPool`
   - Or create wrapper that maintains WebSocket streaming while using pool

3. **Remove or archive orphaned code**
   - Move `shared/agent_executor.py` to `shared/_deprecated/`
   - Update any remaining imports

### 5.2 Short-term Actions (Phase 0.2)

1. **Integrate Work Ledger**
   - Create WorkItem when Job is created
   - Update WorkItem status as Job progresses
   - Use for crash recovery on restart

2. **Integrate Agent Mailbox**
   - Use for Task tool result handoffs
   - Use for structured agent-to-agent communication
   - Add to agent system prompts

### 5.3 Architecture Improvements

1. **Single execution entry point**
   - All agent execution should go through AgentExecutorPool
   - Add WebSocket-aware wrapper for streaming

2. **Event-driven architecture**
   - All components emit events to central bus
   - WebSocket broadcasts from event bus
   - Work ledger listens for completion events

---

## 6. Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| backend/main.py | 2176 | FastAPI backend, WebSocket handling |
| backend/jobs.py | 628 | Background job system |
| shared/workspace_manager.py | 318 | Workspace isolation |
| shared/agent_executor_pool.py | 634 | Concurrent agent execution |
| shared/execution_context.py | 124 | Execution context dataclass |
| shared/escalation_protocol.py | 518 | Escalation management |
| shared/agent_executor.py | 381 | Legacy executor (orphaned) |
| shared/work_ledger.py | 1065 | Work item persistence |
| shared/agent_mailbox.py | 1005 | Agent messaging |
| shared/__init__.py | 86 | Module exports |
| supreme/orchestrator.py | 460 | Swarm orchestration |

---

## 7. Conclusion

The agent-swarm architecture is well-designed with proper separation of concerns. However, significant work remains to connect newly built components (Work Ledger, Mailbox, Escalation Protocol) into the main execution flow. The most critical issue is that the main WebSocket chat flow bypasses the carefully designed executor pool and workspace isolation.

**Recommended Priority:**
1. Connect escalation protocol (already planned)
2. Unify execution paths through pool
3. Integrate work ledger for crash resilience
4. Add mailbox for structured agent communication

---

*Review completed by Architect Agent - 2026-01-03*
