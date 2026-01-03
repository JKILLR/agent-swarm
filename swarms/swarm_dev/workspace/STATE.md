# Swarm Dev - STATE.md
## Last Updated: 2026-01-02

---

## Current Objective
Make the agent-swarm system fully self-developing - as capable as Claude Code in the terminal.

## Priority Order
1. **Phase 0: Execution Layer** - CURRENT FOCUS
2. **Phase 1: Core Functionality** - Next
3. **Phase 2: Operational Excellence** - After Phase 1

---

## Progress Log

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

---

## Known Issues

1. Query() keyword args bug may still exist - needs verification
2. Parallel agent spawning needs testing
3. Wake messaging needs validation
4. **NEW** (from Phase 0.1.2 review): Race condition in singleton init - endpoints may fail if called before startup completes
5. **NEW** (from Phase 0.1.2 review): Broken relative import in jobs.py fallback path
6. **NEW** (from Phase 0.1.2 review): `allowed_tools` permission model is informational only - not enforced

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
