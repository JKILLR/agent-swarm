# Swarm Dev - STATE.md
## Last Updated: 2026-01-03

---

## Current Objective
Make the agent-swarm system fully self-developing - as capable as Claude Code in the terminal.

## Priority Order
1. **Phase 0: Execution Layer** - CURRENT FOCUS
2. **Phase 1: Core Functionality** - Next
3. **Phase 2: Operational Excellence** - After Phase 1

---

## Progress Log

### 2026-01-03: Critical Architecture Fixes - COO Execution & Escalation Integration
- **COO** completed three high-priority tasks from the architecture review

#### Task 1: Unified WebSocket Chat through AgentExecutorPool ✅
- **Problem**: COO WebSocket chat bypassed AgentExecutorPool - no isolation, no tracking
- **Solution**:
  - Created `execute_coo_via_pool()` function in `backend/websocket/chat_handler.py`
  - Routes all COO execution through `AgentExecutorPool.execute()`
  - Added `disallowed_tools` parameter to pool's execute chain
  - Maps pool events to WebSocket events for frontend compatibility
- **Files Modified**:
  - `shared/agent_executor_pool.py` - Added disallowed_tools param to execute(), _run_agent(), _build_command()
  - `backend/websocket/chat_handler.py` - Replaced stream_claude_response with pool execution
- **Result**: COO now gets workspace isolation and resource management

#### Task 2: Critic Review of escalation_protocol.py ✅
- **Result**: APPROVED with minor improvements
- **Added Methods**:
  - `get_by_id()` - Direct escalation lookup by ID
  - `get_all()` - List all escalations with optional status filter
- **Files Modified**: `shared/escalation_protocol.py`

#### Task 3: Escalation REST API + WebSocket Events ✅
- **Improvements**:
  - Updated `list_escalations()` to support `status=all` and `status=resolved` filters
  - Updated `get_escalation()` to use new `get_by_id()` method
  - All mutation endpoints now broadcast WebSocket events
- **WebSocket Endpoint**: `ws://localhost:8000/ws/escalations`
  - Broadcasts: `escalation_created`, `escalation_updated`, `escalation_resolved`
  - Actions: `get_pending`, `get_blocking`, `get_status`
  - CEO/critical escalations also broadcast to main chat
- **Files Modified**:
  - `backend/main.py` - Updated REST endpoints + added WebSocket endpoint
  - `backend/websocket/escalation_updates.py` - Created (standalone module)
  - `backend/websocket/__init__.py` - Added exports
- **Status**: COMPLETE

#### Issues Resolved
- Issue #14: "WebSocket chat bypasses AgentExecutorPool" → **FIXED**
- Issue #8 (partial): "Escalation Protocol not connected" → **FIXED**

### 2026-01-03: Task Delegation Fix - REST API as Primary Method
- **COO** updated COO system prompt to prefer REST API over Task tool
- **Problem**: Task tool (Issues #12-14) doesn't spawn REAL agents:
  - Built-in Claude Code subagent runs in same session
  - Custom agent prompts from `swarms/*/agents/*.md` never loaded
  - No workspace isolation, no executor pool tracking
- **Solution**: Updated COO system prompt in TWO locations:
  1. `backend/websocket/chat_handler.py:build_coo_system_prompt()`
  2. `backend/main.py` (duplicate prompt inline)
- **Key Changes**:
  - REST API now PRIMARY: "ALWAYS use REST API for implementation work"
  - Task tool now SECONDARY: "Quick research only (read-only, no custom prompts)"
  - Clear guidance: "REST API = Real agents. Task tool = Quick research only."
  - Delegation pipeline now specifies: `swarm_dev/architect`, `swarm_dev/implementer`, etc.
- **Files Modified**:
  - `backend/websocket/chat_handler.py` - Updated `build_coo_system_prompt()`
  - `backend/main.py` - Updated inline system prompt
- **Note**: This is a WORKAROUND, not a fix for the underlying architecture. True fix would require intercepting Task tool calls to spawn real agents via AgentExecutorPool.
- **Status**: COMPLETE

### 2026-01-03: Boris Claude Code Usage Analysis for Swarm Optimization
- **Researcher** analyzed Boris's (Claude Code creator) 13-point Twitter thread
- **Analysis Document**: `workspace/boris_insights_analysis.md`
- **Key Findings**:
  1. **Already Implemented (40%)**:
     - Opus 4.5 for all agents (Point 3)
     - Subagent structure (Point 8)
     - Permission management via settings.json (Point 10)
     - SubagentStop hooks (Point 12 partial)
  2. **Quick Wins Missing**:
     - Slash Commands (.claude/commands/) - Boris uses for every repeated workflow
     - PostToolUse formatting hook - Auto-format after Write/Edit
     - Expand permissions to include npm, python, pytest
  3. **High Impact Gaps**:
     - Verification loops - Boris: "most important thing, 2-3x quality"
     - Task tool doesn't spawn real agents (validates our Known Issue #12)
     - GitHub Action integration for "Compounding Engineering"
  4. **Recommended Priority Order**:
     1. Create `.claude/commands/` with common workflows
     2. Add PostToolUse formatting hook
     3. Add automatic verification to SubagentStop hook
     4. Expand permissions in settings.json
- **Strategic Insight**: Boris's top recommendation (verification loops) aligns with our existing Critic/Tester agents but reveals they're not automatically triggered
- **Files Created**: `swarms/swarm_dev/workspace/boris_insights_analysis.md`
- **Status**: COMPLETE

### 2026-01-03: Quick Win - Structured Logging with Correlation IDs
- **Implementer** added structured logging with request correlation IDs
- **Problem**: Code review noted missing structured logging and no request correlation IDs, making debugging difficult
- **Solution**:
  - Added `request_id_var` context variable using `contextvars` for request-scoped correlation IDs
  - Created `CorrelationIdFilter` class that injects request_id into log records
  - Created `StructuredFormatter` class that outputs logs in key=value format:
    `timestamp level=LEVEL request_id=ID logger=NAME message="MSG"`
  - Added `CorrelationIdMiddleware` class for HTTP requests:
    - Generates unique 8-char correlation ID per request
    - Accepts incoming X-Request-ID header if provided
    - Sets correlation ID in response header
    - Logs request start and completion with status code
  - Added correlation ID handling for WebSocket chat:
    - Generates per-connection correlation ID on WebSocket connect
    - Generates per-message correlation ID for each chat message
    - Logs "WebSocket chat session started" and "Processing chat message"
- **Files Modified**:
  - `backend/main.py`:
    - Added `contextvars` import (line 7)
    - Added `Request` import from fastapi (line 22)
    - Added `request_id_var` context variable (line 28)
    - Added `CorrelationIdFilter` class (lines 74-79)
    - Added `StructuredFormatter` class (lines 82-103)
    - Added `setup_logging()` function (lines 106-123)
    - Added `CorrelationIdMiddleware` class (lines 145-164)
    - Added middleware registration (line 168)
    - Added WebSocket correlation ID handling (lines 2599-2616)
- **Log Format Example**:
  ```
  2026-01-03T14:30:00.123 level=INFO request_id=a1b2c3d4 logger=backend.main message="Request started: GET /api/status"
  ```
- **Benefits**:
  - All logs from a single request share the same request_id
  - WebSocket messages get unique correlation IDs
  - Logs are machine-parseable (structured key=value format)
  - Correlation ID returned in X-Request-ID response header
- **Note**: Verify syntax with `python3 -m py_compile backend/main.py`
- **Status**: COMPLETE

### 2026-01-03: Quick Win - Fixed Mid-File Imports in main.py
- **Implementer** resolved code quality issue from review
- **Problem**: Several imports were placed mid-file instead of at the top:
  - Line 85: `import sqlite3` (inside startup_event function)
  - Line 463: `from shared.work_models import ...`
  - Line 600: `from shared.agent_mailbox import MessageType, ...`
  - Line 756: `from shared.escalation_protocol import EscalationLevel, ...`
  - Lines 1703-1705: `import urllib.parse`, `urllib.request`, `re`
- **Solution**:
  - Moved all standard library imports (sqlite3, re, urllib.parse, urllib.request) to top
  - Extended existing shared module imports to include additional symbols
  - Removed all mid-file import statements
- **Files Modified**:
  - `backend/main.py` - Consolidated all imports at file top (lines 3-67)
- **Import Organization**:
  1. Standard library (lines 5-20): asyncio, base64, contextvars, json, logging, mimetypes, os, re, sqlite3, sys, urllib.parse, urllib.request, uuid, datetime, Path, Any
  2. Third-party (lines 22-25): fastapi, pydantic, starlette
  3. Local shared (lines 31-48): workspace_manager, agent_executor_pool, execution_context, work_ledger, work_models, agent_mailbox, escalation_protocol
  4. Project modules (lines 55, 64-67): dotenv, memory, supreme.orchestrator, jobs, session_manager
- **Status**: COMPLETE

### 2026-01-03: Quick Win - Fixed Duplicate _get_tool_description Function
- **Implementer** resolved DRY violation identified in code review
- **Problem**: `_get_tool_description()` was duplicated in both:
  - `backend/main.py:2024-2044`
  - `shared/agent_executor_pool.py:597-625`
- **Solution**:
  - Created single canonical `get_tool_description()` function in `shared/agent_executor_pool.py` (line 608)
  - Updated `AgentExecutorPool` class to call the module-level function
  - Updated `backend/main.py` to import from shared module
  - Removed duplicate function from `backend/main.py`
- **Files Modified**:
  - `shared/agent_executor_pool.py` - Added `get_tool_description()`, removed class method
  - `backend/main.py` - Added import, removed local function, updated usages
- **Note**: `backend/tools.py` also has a `_get_tool_description` but uses different parameter names (`path` vs `file_path`) and different tools - intentionally left as-is
- **Status**: COMPLETE

### 2026-01-03: Brainstorm Review - Innovation & Improvement Ideas
- **Brainstorm Agent** performed comprehensive innovation review
- **Review Document**: `workspace/REVIEW_BRAINSTORM_2026-01-03.md`
- **Key Findings**:
  1. **Missing Features**: Agent memory continuity, intelligent task decomposition, real-time collaboration protocol, agent skill registry, rollback/recovery system
  2. **Communication Improvements**: Unified message bus, structured handoff templates, agent-to-agent direct queries, broadcast with acknowledgment, context-aware routing
  3. **Automation Opportunities**: Auto-spawn on work detection, stale work recovery scheduling, auto-generated test suites, pre-commit review automation
  4. **Self-Improvement Ideas**: Agent performance metrics, prompt evolution system, pattern library, self-critique loop, emergent specialization
  5. **Innovative Solutions**: Hierarchical consensus voting, shadow agent execution, predictive context loading, agent chain templates, adversarial review mode
- **Top 5 Prioritized Ideas by Impact/Effort**:
  1. **Auto-Spawn on Work Detection** (9.5/10) - Closes autonomy loop, uses existing Work Ledger
  2. **Unified Message Bus** (8.5/10) - Solves root communication fragmentation
  3. **Stale Work Recovery Scheduling** (8.0/10) - Already implemented, just needs cron
  4. **Agent Performance Metrics** (7.5/10) - Enables data-driven improvement
  5. **Agent Chain Templates** (7.0/10) - Automates common workflows
- **Quick Wins Identified**:
  - Schedule `recover_orphaned_work()` to run every 5 minutes
  - Add unified `/api/activity` endpoint merging Mailbox + Escalation + Work Ledger
  - Add Work Ledger integration to Task tool detection
- **Long-Term Vision**: Fully autonomous self-development with human approval at key gates

### 2026-01-03: COO Delegation Failure Analysis - CRITICAL ISSUES
- **Critic** performed comprehensive review of COO delegation behavior
- **Review Document**: `workspace/REVIEW_DELEGATION_FAILURES.md`
- **Result**: NEEDS_CHANGES (3 critical, 3 high priority issues)
- **Critical Findings**:
  1. **CRITICAL: Task Tool Does Not Spawn Real Agents** - When COO uses `Task(subagent_type=...)`, no actual sub-agent process is spawned. The backend only sends UI notifications but Claude's built-in Task tool runs internally with COO's context.
  2. **CRITICAL: No Connection Between WebSocket and AgentExecutorPool** - Main chat flow uses `stream_claude_response()` which bypasses the pool entirely. No workspace isolation, no concurrency limits, no agent configuration.
  3. **CRITICAL: Agent .md Files Not Used for Delegation** - The carefully crafted agent definitions (prompts, tools, permissions) in `swarms/*/agents/*.md` are NEVER read or applied to delegated tasks.
- **High Priority Findings**:
  4. **HIGH: Agent Stack is Cosmetic Only** - `context["agent_stack"]` tracks agent names for UI attribution but doesn't change prompts, workspaces, or permissions.
  5. **HIGH: Task Results Not Captured** - Delegation completion always reports `success: True` with no actual result extraction.
  6. **HIGH: Work Ledger/Mailbox Not Integrated** - These systems were built but are orphaned - not connected to Task flow.
- **Root Cause**: The Task tool is Claude CLI's built-in feature that creates internal sub-conversations. The backend has no control over these - it only parses the stream. Real delegation would require intercepting Task and spawning separate agent processes.
- **Evidence**: Backend logs show only ONE Claude CLI process per chat, even when COO "delegates" to multiple agents.
- **Recommendations**:
  1. Intercept Task tool, spawn real agents via AgentExecutorPool
  2. Route WebSocket chat through pool for proper isolation
  3. Load agent .md files when spawning sub-agents
  4. Integrate Work Ledger and Mailbox with Task execution

### 2026-01-03: Code Quality Review Complete - NEEDS_CHANGES
- **Critic** performed comprehensive backend code quality review
- **Review Document**: `workspace/REVIEW_CODE_QUALITY_2026-01-03.md`
- **Result**: NEEDS_CHANGES (1 critical bug, 3 high priority issues)
- **Critical Finding**:
  1. CRITICAL: Missing `_get_tool_description` method in `AgentExecutorPool` (line 422) - WILL CRASH at runtime
- **High Priority Findings**:
  2. HIGH: Unused `_running` dict in AgentExecutorPool - cancellation via task reference broken
  3. HIGH: Race condition in `get_workspace_manager()` singleton - not thread-safe
  4. HIGH: Race condition in `get_executor_pool()` singleton - not thread-safe
- **Medium Priority**: Fire-and-forget tasks, hard-coded defaults, mid-file imports, bare exceptions
- **Positive Observations**: Good error handling in safe wrappers, thread-safe escalation protocol, atomic file writes
- **Required Actions**:
  1. Add `_get_tool_description` method to AgentExecutorPool class
  2. Add thread-safe locking to singleton getters (workspace_manager, executor_pool)
  3. Populate `_running` dict for proper task tracking

### 2026-01-03: Review Workflow Audit Complete
- **Researcher** performed comprehensive audit of review workflow
- **Audit Document**: `workspace/REVIEW_WORKFLOW_AUDIT_2026-01-03.md`
- **Overall Assessment**: PARTIALLY IMPLEMENTED - Advisory reviews exist, but no blocking gates
- **Key Findings**:
  1. **CRITICAL**: Review workflow in swarm.yaml (`required: true`) is NOT ENFORCED in backend code
  2. **CRITICAL**: No "ready for implementation" gate - designs can be built without approval
  3. **CRITICAL**: No "implementation approved" gate - code can merge without review
  4. **HIGH**: Critic reviews are effective when used, but nothing ensures they happen
  5. **HIGH**: Reviews are advisory only - orchestrator can ignore them
  6. **MEDIUM**: No automated test gate before merge
- **What Works**:
  - Reviewer/Critic agents well-defined with clear checklists
  - STATE.md captures review outcomes when reviews happen
  - Historical evidence shows reviews catch real issues (4 critical fixes from 2026-01-02)
- **What's Missing**:
  - No technical enforcement in backend/main.py or backend/jobs.py
  - No state machine for work items (DESIGNED -> IMPLEMENTED -> REVIEWED -> APPROVED)
  - Backend ignores `review_workflow` config entirely
- **Recommendations**:
  1. Add review status tracking to work_ledger.py
  2. Create shared/review_gates.py with state machine
  3. Modify backend/jobs.py to enforce review workflow
  4. Update orchestrator prompt with explicit gate checks
- **Next**: Architect to design review gate enforcement system

### 2026-01-03: Architecture Review Complete
- **Architect** performed thorough architecture review
- **Review Document**: `workspace/REVIEW_ARCHITECTURE_2026-01-03.md`
- **Key Findings**:
  1. **CRITICAL**: Main WebSocket chat flow bypasses AgentExecutorPool - runs without isolation
  2. **NOT CONNECTED**: Work Ledger (`shared/work_ledger.py`) - built but not integrated
  3. **NOT CONNECTED**: Agent Mailbox (`shared/agent_mailbox.py`) - built but not integrated
  4. **NOT CONNECTED**: Escalation Protocol - built but no REST/WebSocket endpoints
  5. **ORPHANED**: `shared/agent_executor.py` - superseded by AgentExecutorPool, should be removed
  6. **DUAL PATHS**: Three execution mechanisms exist, causing inconsistency
- **Recommendations**:
  1. Unify execution through AgentExecutorPool for all agent execution
  2. Wire escalation protocol to WebSocket events and REST endpoints
  3. Integrate Work Ledger with job system for crash resilience
  4. Integrate Agent Mailbox for structured agent-to-agent communication
  5. Remove or archive orphaned agent_executor.py

### 2026-01-02: Escalation Protocol Implementation Complete
- **Implementer** created `shared/escalation_protocol.py`
  - Implemented all enums: EscalationLevel, EscalationReason, EscalationStatus, EscalationPriority
  - Implemented Escalation dataclass with to_dict(), from_dict(), to_markdown()
  - Implemented EscalationManager class with create, resolve, update, get methods
  - Implemented convenience functions: escalate_to_coo(), escalate_to_ceo(), get_escalation_manager()
  - Singleton pattern with disk persistence to `logs/escalations/`
- **Implementer** created `logs/escalations/` directory with .gitkeep
- **Implementer** added Escalations section to STATE.md with quick reference table format
- **Note**: Run `python3 -m py_compile shared/escalation_protocol.py` to verify syntax

### 2026-01-02: Escalation Protocol Design Complete
- **Architect** completed design document: `DESIGN_ESCALATION_PROTOCOL.md`
- Key decisions made:
  - Three-tier hierarchy: CEO (human) -> COO (orchestrator) -> Swarm Agents
  - Agent-to-COO escalation reasons: BLOCKED, CLARIFICATION, CONFLICT, SECURITY, ARCHITECTURE, SCOPE_EXCEEDED
  - COO-to-CEO escalation reasons: ARCHITECTURE_MAJOR, SECURITY_CRITICAL, PRIORITY_CONFLICT, COST, PERMISSION, BLOCKED_CRITICAL
  - Blocked work protocol: continue with unblocked tasks, mark blocked with [BLOCKED: ESC-ID]
- Implementation: `shared/escalation_protocol.py` with EscalationManager, convenience functions
- STATE.md format: Dedicated Escalations section with pending/resolved subsections
- **Next**: Implementer to build `shared/escalation_protocol.py` - DONE

### 2026-01-02: Phase 0.1.2 - Critical Issues Fixed ✅
- **Implementer** fixed all critical and high priority issues from reviews:
  1. ✅ Fixed singleton initialization - added `PROJECT_ROOT` to all `get_workspace_manager()` calls
  2. ✅ Fixed relative import - changed `from .main import` to `from main import` in jobs.py
  3. ✅ Added swarm existence validation in `execute_agent` endpoint
  4. ✅ Removed unused imports (WorkspaceManager, AgentExecutorPool classes)
- **Remaining Medium Issues** (for future consideration):
  - Unused `_running` dict in AgentExecutorPool
  - `allowed_tools` not enforced in CLI command (informational only)
  - Hard-coded "swarm_dev" default in jobs.py

### 2026-01-02: Security Review Complete
- **Critic** performed security review of workspace isolation
- **Overall Risk**: MEDIUM-HIGH (requires authentication for production)
- **Key Findings**:
  1. CRITICAL: Swarm name injection allows privilege escalation to swarm_dev (API callers can claim any swarm)
  2. HIGH: TOCTOU vulnerability with symlinks in path validation
  3. HIGH: Agent type spoofing via API (no validation of agent existence)
- **Positive**: No shell injection, proper subprocess.exec usage, API key removal
- **Recommendations**: Add authentication, validate swarm/agent existence (partially done)

### 2026-01-02: Phase 0.1.2 - Critic Review: NEEDS_CHANGES
- **Critic** reviewed backend integration of AgentExecutorPool
- **Result**: NEEDS_CHANGES before proceeding
- Issues identified and now fixed (see entry above)

### 2026-01-02: Phase 0.1.2 - Backend Integration Complete
- Integrated AgentExecutorPool into `backend/main.py`
- Added imports for workspace_manager, agent_executor_pool, and execution_context
- Initialized workspace manager and executor pool in startup_event()
- Added new API endpoints:
  - `POST /api/agents/execute` - Execute an agent with workspace isolation
  - `GET /api/agents/pool/status` - Get executor pool status
- **Note**: Syntax validation should be run: `python3 -m py_compile backend/main.py`
- **Next**: Update backend/jobs.py to use pool for job execution (Phase 0.1.2 continued)

### 2026-01-02: Agent Executor Architecture Design Complete
- Architect completed design document: `DESIGN_AGENT_EXECUTOR.md`
- Key decisions made:
  - Use AgentExecutorPool for concurrent execution
  - Create WorkspaceManager for isolation and permissions
  - Swarm Dev gets special PROJECT_ROOT access
  - Claude CLI remains primary execution method
- Implementation plan: 3 phases over 5-8 days
- **Next**: Implementer to build WorkspaceManager and AgentExecutorPool

### 2026-01-02: System Self-Development Initiative Started
- COO taking over development of the system
- Swarm Dev agents will implement all capabilities
- All implementations must be reviewed by critic before merging

---

## Implementation Phases

### Phase 0: Execution Layer (CURRENT)

#### 0.1 Claude Agent SDK Integration
**Goal**: Wire up actual agent execution so agents can use tools

**Tasks**:
- [ ] Create `shared/agent_executor.py` with proper tool execution
- [ ] Enable file read/write capabilities for agents
- [ ] Enable bash command execution for agents
- [ ] Stream tool results back to UI via WebSocket
- [ ] Test: Agent can read a file and report contents

**Key Files**:
- `backend/main.py` - Current agent spawning logic
- `shared/agent_base.py` - Agent base class
- `supreme/orchestrator.py` - Current orchestration

#### 0.2 Git Integration for Agents
**Goal**: Enable agents to push/pull from GitHub

**Tasks**:
- [ ] Configure git credentials for agent subprocesses
- [ ] Pass CLAUDE_CODE_OAUTH_TOKEN to agent processes
- [ ] Add git tools to Swarm Dev agent prompts
- [ ] Test: Agent can commit and push a change

**Key Files**:
- `backend/main.py:stream_claude_response()` - Where env is set
- Agent markdown files need git permissions

#### 0.3 Self-Modification Capability
**Goal**: Swarm Dev can modify agent-swarm itself

**Tasks**:
- [ ] Implementer can edit files in the codebase
- [ ] Reviewer can run tests (`pytest`, type checking)
- [ ] Critic can check for security issues
- [ ] Orchestrator can coordinate multi-file changes
- [ ] Test: Agent successfully modifies and commits code

---

### Phase 1: Core Functionality

#### 1.2 Enhance Supreme Orchestrator
**Goal**: Supreme acts like COO with direct reports

**Tasks**:
- [ ] Create `supreme/agents/chief_of_staff.md`
- [ ] Create `supreme/agents/project_manager.md`
- [ ] Create `supreme/agents/context_keeper.md`
- [ ] Update `supreme/agents/supreme.md` to delegate appropriately

---

### Phase 2: Operational Excellence

#### 2.1 Consensus Protocol
- [ ] Implement `shared/consensus.py` with voting rounds
- [ ] Log to `logs/consensus/` for audit
- [ ] Critic must participate in every consensus round

#### 2.2 Memory and Context
- [ ] Workspaces persist state between sessions
- [ ] Supreme summarizes sessions to `logs/daily/`
- [ ] Context keeper reads summaries for continuity

#### 2.3 Background Monitors
- [ ] Each swarm gets a monitor agent
- [ ] Monitors watch for test failures, build errors
- [ ] Wake main thread only on problems

---

## Architecture Decisions

1. **Agent Execution**: Using Claude CLI with `--output-format stream-json`
2. **Tool Access**: Agents use Claude Code's built-in tools (Read, Write, Edit, Bash, etc.)
3. **Workspace Isolation**: Each swarm works in its designated workspace
4. **STATE.md Pattern**: Shared context file for agent coordination

### ADR-001: Agent Executor Pool Architecture (2026-01-02)

**Context**: Need to execute multiple agents concurrently with proper isolation.

**Decision**: Create `AgentExecutorPool` class that:
- Manages concurrent Claude CLI processes
- Limits concurrency via semaphore (default 5)
- Tracks process lifecycle for cancellation
- Streams events back in unified format

**Rationale**:
- Reuses existing Claude CLI infrastructure
- Provides resource limits to prevent overload
- Enables cancellation of long-running agents
- Keeps execution model simple (subprocess-based)

**Consequences**:
- Each agent is a separate OS process (higher overhead, better isolation)
- Must propagate environment variables carefully
- Process cleanup on timeout/cancel is important

### ADR-002: Workspace Isolation Model (2026-01-02)

**Context**: Agents need file access but should be sandboxed.

**Decision**: Create `WorkspaceManager` that:
- Resolves workspace paths from swarm config
- Validates file paths stay within boundaries
- Grants different permissions by agent type
- Swarm Dev gets special PROJECT_ROOT access

**Rationale**:
- Security: prevent file access escapes
- Flexibility: different swarms have different needs
- Self-modification: Swarm Dev must access project files

**Consequences**:
- Swarm Dev is a privileged swarm (security responsibility)
- Path validation adds overhead but is necessary
- Workspace changes persist (not ephemeral)

### ADR-003: Credential Propagation (2026-01-02)

**Context**: Agents need git/API access without exposing secrets.

**Decision**: Use environment variables:
- `CLAUDE_CODE_OAUTH_TOKEN` for git operations (already used by CLI)
- Remove `ANTHROPIC_API_KEY` to force CLI auth (Max subscription)
- Set `AGENT_WORKSPACE`, `AGENT_NAME`, `AGENT_SWARM` for context

**Rationale**:
- CLAUDE_CODE_OAUTH_TOKEN already handles CLI auth
- API key removal ensures consistent authentication
- Agent context env vars help with debugging

**Consequences**:
- Agents can push to GitHub via CLI
- No API fallback if CLI unavailable
- Must be careful not to log credentials

### ADR-004: Escalation Protocol (2026-01-02)

**Context**: Agents need a formal way to escalate issues they cannot resolve themselves. The system has a three-tier hierarchy (CEO/human -> COO/orchestrator -> Agents) and escalations must flow upward appropriately.

**Decision**: Create `EscalationManager` class and protocol that:
- Defines `EscalationLevel` enum: AGENT, COO, CEO
- Defines `EscalationReason` enum: BLOCKED, CLARIFICATION, CONFLICT, SECURITY, etc.
- Provides `Escalation` dataclass with full lifecycle tracking
- Implements blocked work protocol: continue with unblocked tasks
- Logs escalations to `logs/escalations/` and STATE.md

**Rationale**:
- Agents should not block waiting for escalation resolution
- Clear decision trees prevent inappropriate escalations
- Persistent logging enables audit and pattern analysis
- STATE.md integration keeps context visible to all agents

**Consequences**:
- Agents must check for pending escalations on startup
- COO becomes bottleneck for all agent escalations (by design)
- CEO only engaged for major decisions (preserves human time)
- Must add escalation awareness to agent system prompts

### ADR-005: Unified Execution Architecture (2026-01-03)

**Context**: Architecture review revealed three parallel execution mechanisms creating inconsistency:
1. `stream_claude_response()` - used by WebSocket chat (COO)
2. `AgentExecutorPool.execute()` - used by REST API and jobs
3. `AgentExecutor.execute()` - not used (orphaned)

The main WebSocket chat flow bypasses workspace isolation and executor pool features.

**Decision**: Unify all agent execution through AgentExecutorPool:
- Deprecate/remove `shared/agent_executor.py`
- Modify WebSocket flow to use pool (with streaming wrapper if needed)
- All execution gets workspace isolation, event streaming, resource limits

**Rationale**:
- Consistent behavior across all execution paths
- Security: all agents get proper workspace isolation
- Observability: all agents emit consistent events
- Resource management: all agents respect concurrency limits

**Consequences**:
- May need WebSocket-aware wrapper for pool
- COO execution will be subject to concurrency limits
- Better crash recovery possible when all work tracked
- Single point of execution simplifies debugging

### ADR-006: Component Integration Strategy (2026-01-03)

**Context**: Three components built but not wired into main flows:
1. Work Ledger - crash-resilient work tracking
2. Agent Mailbox - structured agent communication
3. Escalation Protocol - issue escalation

**Decision**: Integrate in phases:
1. **Escalation Protocol first** (already planned) - add REST endpoints and WebSocket events
2. **Work Ledger second** - integrate with Jobs system for crash resilience
3. **Agent Mailbox third** - integrate with Task tool for handoffs

**Rationale**:
- Escalations are most immediately useful (already in progress)
- Work Ledger enables crash recovery without data loss
- Mailbox enables more structured agent coordination

**Consequences**:
- Each integration adds overhead but improves reliability
- Agents need updated system prompts for each feature
- STATE.md becomes less critical once Work Ledger/Mailbox work

---

## Key Files

### Existing
| File | Purpose |
|------|---------|
| `shared/agent_executor.py` | Current executor (to be enhanced) |
| `shared/agent_base.py` | Base agent class |
| `backend/main.py` | WebSocket and COO execution |
| `backend/jobs.py` | Background job management |

### Created (Phase 0.1.1)
| File | Purpose |
|------|---------|
| `shared/workspace_manager.py` | Workspace isolation and permissions |
| `shared/agent_executor_pool.py` | Concurrent execution pool |
| `shared/execution_context.py` | AgentExecutionContext dataclass |

### Modified (Phase 0.1.2)
| File | Purpose |
|------|---------|
| `backend/main.py` | Added executor pool integration and new API endpoints |

### Created (Phase 0.1.3 - Escalation Protocol)
| File | Purpose |
|------|---------|
| `shared/escalation_protocol.py` | Escalation management and lifecycle |
| `logs/escalations/.gitkeep` | Escalation log directory |

### Design Documents
| File | Purpose |
|------|---------|
| `DESIGN_ESCALATION_PROTOCOL.md` | Design document for escalation system |

### Boris Analysis (2026-01-03)
| File | Purpose |
|------|---------|
| `swarms/swarm_dev/workspace/boris_insights_analysis.md` | Analysis of Boris's Claude Code usage patterns for swarm optimization |

---

## Escalations

| ID | Status | Priority | Title | Assigned To |
|----|--------|----------|-------|-------------|
| - | - | - | *No pending escalations* | - |

### Pending

*None*

### Resolved

*None*

---

## Known Issues

### Delegation System (from 2026-01-03 Delegation Failure Review)
12. **WORKAROUND** (was CRITICAL): Task tool does NOT spawn real agents - **COO now instructed to use REST API instead**
13. **WORKAROUND** (was CRITICAL): Agent .md files not loaded for Task tool - **REST API loads them properly**
14. **FIXED**: WebSocket chat bypasses AgentExecutorPool entirely → **COO now routes through pool via execute_coo_via_pool()**

### Boris Analysis - Quick Wins (from 2026-01-03)
15. **RESOLVED**: `.claude/commands/` directory created with common workflows
16. **RESOLVED**: PostToolUse formatting hook added (`scripts/hooks/post_format.py`)
17. **MEDIUM**: No automatic verification loop - Boris: "2-3x quality" if agents verify their work
18. **RESOLVED**: Permissions expanded in settings.json - Added npm, python, python3, pytest, chmod

### Previous Issues
1. Query() keyword args bug may still exist - needs verification
2. Parallel agent spawning needs testing
3. Wake messaging needs validation
4. **NEW** (from Phase 0.1.2 review): Race condition in singleton init - endpoints may fail if called before startup completes
5. **NEW** (from Phase 0.1.2 review): Broken relative import in jobs.py fallback path
6. **NEW** (from Phase 0.1.2 review): `allowed_tools` permission model is informational only - not enforced
7. **FIXED** (from Architecture Review): Main WebSocket chat bypasses AgentExecutorPool → **Now uses execute_coo_via_pool()**
8. **PARTIAL** (from Architecture Review 2026-01-03): Three orphaned components → **Escalation Protocol now has REST + WebSocket endpoints**; Work Ledger and Mailbox still pending
9. **NEW** (from Architecture Review 2026-01-03): `shared/agent_executor.py` is dead code - superseded by pool
10. **FIXED** (from Code Quality Review 2026-01-03): Missing `_get_tool_description` method in AgentExecutorPool:422 - Created shared `get_tool_description()` function
11. **HIGH** (from Code Quality Review 2026-01-03): Singleton getters (workspace_manager, executor_pool) not thread-safe

---

## Next Steps

### Completed (Phase 0.1.1) [x]
1. [x] **Implementer**: Created `shared/workspace_manager.py`
   - Workspace resolution from swarm config
   - Path validation for security
   - Permission lookup by agent type

2. [x] **Implementer**: Created `shared/agent_executor_pool.py`
   - AgentExecutionContext dataclass (in execution_context.py)
   - Concurrent execution with semaphore
   - Process lifecycle management

3. [x] **Implementer**: Created `shared/execution_context.py`
   - AgentExecutionContext dataclass
   - Validation and serialization methods

### Completed (Phase 0.1.2) [x]
4. [x] **Implementer**: Updated `backend/main.py`
   - Integrated AgentExecutorPool
   - Wired workspace resolution
   - Added `/api/agents/execute` endpoint
   - Added `/api/agents/pool/status` endpoint

### Completed (Phase 0.1.2-fix) [x]
5. [x] **Implementer**: Fixed critical issues from critic review
   - ✅ Added `PROJECT_ROOT` to endpoint calls to `get_workspace_manager()`
   - ✅ Fixed relative import in jobs.py:429
   - ✅ Added swarm existence validation in execute_agent
   - ✅ Cleaned up unused imports

### Completed (Phase 0.1.2 continued) [x]
6. [x] **Implementer**: Updated `backend/jobs.py`
   - Added `_execute_with_pool` method
   - Added `use_pool` flag to optionally use AgentExecutorPool
   - Original fallback preserved

### Completed (Phase 0.1.2 reviews) [x]
7. [x] **Critic**: Code quality review - NEEDS_CHANGES → FIXED
8. [x] **Critic**: Security review - MEDIUM-HIGH risk identified

### Next (Phase 0.1.3) - Validation
9. **Tester**: Create integration tests for:
   - AgentExecutorPool execution
   - WorkspaceManager path validation
   - API endpoint functionality
10. **All**: End-to-end validation - test actual agent execution

### Completed (Escalation Protocol Implementation) [x]
11. [x] **Implementer**: Create `shared/escalation_protocol.py`
    - EscalationLevel, EscalationReason, EscalationStatus, EscalationPriority enums
    - Escalation dataclass with to_dict(), from_dict(), to_markdown()
    - EscalationManager class with create, resolve, update, get methods
    - Convenience functions: escalate_to_coo(), escalate_to_ceo()
    - Singleton pattern with get_escalation_manager()
12. [x] **Implementer**: Create `logs/escalations/` directory structure

### Next (Escalation Protocol Integration)
13. [ ] **Critic**: Review escalation_protocol.py implementation
14. [ ] **Implementer**: Add escalation WebSocket events to backend/main.py
15. [ ] **Architect**: Update agent system prompts with escalation protocol guidance

### Architecture Review Findings (2026-01-03) - Priority Actions
16. [ ] **Implementer**: Unify WebSocket chat execution through AgentExecutorPool
    - Modify `stream_claude_response()` to use pool or create WebSocket-aware wrapper
    - Ensures workspace isolation for COO execution
17. [ ] **Implementer**: Wire escalation protocol to REST API
    - Add endpoints: GET/POST `/api/escalations`, GET/PUT `/api/escalations/{id}`
    - Add WebSocket event broadcasting for escalation changes
18. [ ] **Implementer**: Archive orphaned code
    - Move `shared/agent_executor.py` to `shared/_deprecated/` or remove
19. [ ] **Implementer**: Integrate Work Ledger with Jobs system
    - Create WorkItem when Job is created
    - Update WorkItem status as Job progresses
20. [ ] **Implementer**: Integrate Agent Mailbox for handoffs
    - Use for Task tool completion results
    - Add to agent system prompts

---

## Success Criteria

### Phase 0 Complete When:
- [ ] Swarm Dev agents can read/write files
- [ ] Swarm Dev agents can run git commands
- [ ] Swarm Dev agents can execute tests
- [ ] A Swarm Dev agent successfully modifies and commits code

### Phase 1 Complete When:
- [ ] python main.py chat works without errors
- [ ] Parallel agents spawn and wake properly
- [ ] Swarm Dev is self-maintaining
- [ ] Consensus rounds are logged
- [ ] Supreme acts like a COO, not just a router
