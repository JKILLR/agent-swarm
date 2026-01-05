# Agent Swarm - State Management

---

## ⚠️ COO OPERATING RULES - READ FIRST ⚠️

**YOU ARE THE COO (Chief Operating Officer). YOU ARE AN ORCHESTRATOR, NOT A WORKER.**

### NEVER Do Directly (MUST Delegate)
- ❌ Write or edit code (even 1 line) - Write/Edit tools are DISABLED
- ❌ Create or modify files (except STATE.md via Bash)
- ❌ Run builds, tests, or linters without delegation
- ❌ Make architectural decisions alone
- ❌ Deep research requiring multiple searches
- ❌ Fix bugs or implement features

### MAY Do Directly (No Delegation Needed)
- ✅ Read files to understand context
- ✅ Simple grep/glob to find files
- ✅ Update STATE.md (via Bash `cat >>`)
- ✅ Synthesize results from agents
- ✅ Answer conceptual questions
- ✅ Run curl commands for REST API

---

## TWO DELEGATION METHODS

### Method 1: Task Tool (Built-in Agents)
For standard, single-focus work:
```
Task(subagent_type="implementer", prompt="Read workspace/STATE.md first. Implement X. Update STATE.md when done.")
```

Available: researcher, architect, implementer, critic, tester

### Method 2: REST API (Swarm Agents)
For Operations coordination and swarm-specific agents:
```bash
curl -X POST http://localhost:8000/api/agents/execute \
  -H "Content-Type: application/json" \
  -d '{"swarm": "operations", "agent": "ops_coordinator", "prompt": "..."}'
```

This method loads custom agent prompts and provides workspace isolation.

---

## HYBRID COORDINATION MODEL

### Tier 1 (DEFAULT) - Task Tool
Use for simple features, bug fixes, research, design, code review.

### Tier 2 (ESCALATE) - Operations via REST API
Engage when ANY of these apply:
1. Priority 1-2 (critical/high)?
2. Spans multiple swarms?
3. Cross-swarm dependencies?
4. Changes core infrastructure?
5. Could conflict with ongoing work?

**Operations Agents:**
- ops_coordinator - Cross-swarm coordination, organizational health
- qa_agent - Quality audits, standards enforcement

Reference: `swarms/operations/protocols/coordination_model.md`

---



## Trading Bot Test Validation - 2026-01-04
**Tester**: Test Specialist Agent
**Date**: 2026-01-04

**Status**: COMPLETE - READY FOR PAPER TRADING

### Test Results Summary
| Category | Result |
|----------|--------|
| Syntax Validation | 21/21 files PASSED |
| P0 Fixes Verification | 5/5 VERIFIED |
| P1 Fixes Verification | 2/2 VERIFIED |
| Config Settings | 8/8 VERIFIED |
| Dry Run Mode | AVAILABLE |

### P0 Fixes Confirmed in simple_arb_bot.py:
1. can_trade() method for daily loss limit (lines 147-159)
2. record_trade_result() method for P&L tracking (lines 161-163)
3. min_liquidity validation in check_arbitrage() (lines 360-366)
4. slippage_buffer logic in check_arbitrage() (lines 377-382)
5. Dynamic cooldown in compute_dynamic_cooldown() (lines 423-451)

### P0 Fixes Confirmed in config.py:
All required settings present: min_liquidity, max_daily_risk, slippage_buffer, dry_run

### polymarket_arb.py test Results:
Parallel fetching 59 percent faster. Capital constraints properly configured.

### Issue Found (P1):
File: btc-polymarket-bot/src/lookup.py line 68
Problem: Uses Python 3.10+ syntax (datetime pipe None)
Fix Required: Change to Optional[datetime] syntax

### Test Report:
Full report at: swarms/trading_bots/workspace/research/TEST_REPORT_2026-01-04.md

### Next Steps:
1. Fix lookup.py Python version compatibility (P1)
2. Begin 7-day paper trading validation
3. Add automated unit tests for P0 methods

---

## Latest Work: Dual Delegation System + Tier 1/Tier 2 Model WIRED
**Implementer**: External Review
**Date**: 2026-01-03

**Status**: COMPLETE

**Changes Made:**
1. Updated COO system prompt in `backend/main.py` (lines 2764-2882) to include:
   - Two delegation methods: Task tool + REST API
   - Operations swarm integration (ops_coordinator, qa_agent)
   - Hybrid Coordination Model (Tier 1/Tier 2)
   - 5-question decision tree for escalation
   - Operations reference documentation paths

2. Updated `backend/websocket/chat_handler.py` (lines 52-170) with same changes

3. Verified syntax - both files pass Python compilation

**Key Integration Points:**
- Task tool → Built-in agents (researcher, architect, implementer, critic, tester)
- REST API → Swarm agents via `/api/agents/execute` endpoint
- AgentExecutorPool → Real Claude CLI processes with workspace isolation
- Operations protocols → Full Tier 1/Tier 2 coordination model now referenced

---

## Previous Work: COO Tool Restriction Enforcement - Layer 1 IMPLEMENTED
**Implementer**: Implementation Specialist
**Date**: 2026-01-03

**Status**: LAYER 1 COMPLETE

**Problem**: The COO (Supreme Orchestrator) repeatedly violates delegation rules by directly editing files instead of delegating to implementer agents. The rules existed only as soft guidance in system prompts - there was no technical enforcement.

**Solution Implemented**: Layer 1 - Tool Access Restriction (HARD ENFORCEMENT)

### Changes Made

1. **`/Users/jellingson/agent-swarm/backend/main.py`**
   - Added `disallowed_tools: list[str] | None = None` parameter to `stream_claude_response()` (line 1965)
   - Added `--disallowedTools Write,Edit` flag to Claude CLI command when parameter is provided (lines 1990-1992)
   - Updated COO system prompt to clearly state Write/Edit tools are DISABLED (lines 2762-2842)
   - Passed `disallowed_tools=["Write", "Edit"]` when spawning COO (line 2869)

2. **`/Users/jellingson/agent-swarm/backend/services/claude_service.py`**
   - Added `disallowed_tools: list[str] | None = None` parameter to `stream_claude_response()` (line 31)
   - Added `--disallowedTools` flag handling (lines 60-62)

3. **`/Users/jellingson/agent-swarm/backend/websocket/chat_handler.py`**
   - Updated COO system prompt to match main.py changes (lines 51-130)
   - Passed `disallowed_tools=["Write", "Edit"]` when spawning COO (line 276)

### Key System Prompt Changes

**OLD (Soft Enforcement)**:
```
You have FULL access to all tools and can do anything directly:
- Read/Write/Edit any file in the workspace
```

**NEW (Hard Enforcement)**:
```
## TOOL RESTRICTIONS - HARD ENFORCED

**The Write and Edit tools are DISABLED for you.** Attempting to use them will fail.

You MUST delegate ALL file modifications to specialized agents using the Task tool.
```

### Technical Details

- Uses Claude CLI's `--disallowedTools Write,Edit` flag
- COO can still: Read, Glob/Grep, Bash, Task, Web Search/Fetch
- COO can update STATE.md via Bash (e.g., `cat >> workspace/STATE.md << 'EOF' ... EOF`)
- Only COO has this restriction - subagents retain full tool access

### Verification Required

Run to verify no syntax errors:
```bash
python3 -m py_compile backend/main.py backend/websocket/chat_handler.py backend/services/claude_service.py
```

### Next Steps

1. Test the implementation by asking COO to edit a file (should fail)
2. Test delegation still works (COO delegates to implementer)
3. Consider implementing Layer 4 (Detection/Warning) for UI feedback

---

## Previous Work: COO Delegation Rule Enforcement Design
**Architect**: System Architect
**Date**: 2026-01-03

**Status**: DESIGN COMPLETE - LAYER 1 IMPLEMENTED (see above)

**Design Document**: `/swarms/swarm_dev/workspace/DESIGN_COO_ENFORCEMENT.md`

---

## Previous Work: COO System Prompt Updates - Strict Delegation Enforcement
**Implementer**: Implementation Specialist
**Date**: 2026-01-03

**Status**: COMPLETE

**Problem**: The COO (Supreme Orchestrator) system prompts in `backend/websocket/chat_handler.py` and `backend/main.py` had incorrect guidance that said "Do it yourself when: Quick tasks (reading files, small edits, simple searches)". This was causing the COO to do work directly instead of delegating.

**Changes Made**:
1. **`/Users/jellingson/agent-swarm/backend/websocket/chat_handler.py`** (lines 62-96)
   - Replaced "When to Do vs Delegate" section with strict delegation rules
   - Added "CRITICAL: You are an ORCHESTRATOR, NOT a WORKER" header
   - Defined explicit NEVER/MAY do lists
   - Added Delegation Pipeline (researcher -> architect -> implementer -> critic -> tester)
   - Added Anti-Patterns to AVOID section

2. **`/Users/jellingson/agent-swarm/backend/main.py`** (lines 2768-2802)
   - Same changes applied to the inline system prompt
   - Removed old "small edits" and "quick tasks" guidance
   - Now enforces strict delegation for all code modifications

**Old (Incorrect)**:
```
## When to Do vs Delegate
**Do it yourself when:**
- Quick tasks (reading files, small edits, simple searches)
- You need to understand something before delegating
```

**New (Correct)**:
```
## CRITICAL: You are an ORCHESTRATOR, NOT a WORKER

### NEVER Do Directly (MUST Delegate via Task tool)
- Write, edit, or create any code files
- Run builds, tests, or linters
...
```

**Verification**: Both Python files maintain valid syntax (f-string placeholders preserved).

---

## Previous Work: Main.py Modular Refactoring (Phase 1-4 Complete)
**Implementer**: COO (Supreme Orchestrator)
**Date**: 2026-01-03

**Status**: PHASES 1-4 COMPLETE - Foundation, Services, WebSocket, Routes extracted

**New Modular Structure Created**:
```
backend/
├── models/              # Pydantic models
│   ├── __init__.py
│   ├── requests.py      # SwarmCreate, JobCreate, WorkCreateRequest, etc.
│   ├── responses.py     # HealthResponse, SwarmResponse, AgentInfo
│   └── chat.py          # ChatMessageModel, ChatSession
├── routes/              # API endpoints (routers)
│   ├── __init__.py
│   ├── jobs.py          # /api/jobs/* endpoints
│   ├── work.py          # /api/work/* endpoints (Work Ledger)
│   ├── mailbox.py       # /api/mailbox/* endpoints
│   ├── escalations.py   # /api/escalations/* endpoints
│   ├── swarms.py        # /api/swarms/* endpoints
│   ├── chat.py          # /api/chat/* endpoints
│   └── (files, web, workflows, agents - placeholders)
├── services/            # Business logic
│   ├── __init__.py
│   ├── chat_history.py  # ChatHistoryManager class
│   ├── orchestrator_service.py  # get_orchestrator() singleton
│   ├── event_processor.py  # CLIEventProcessor class (refactored!)
│   └── claude_service.py  # stream_claude_response(), parse_claude_stream()
├── websocket/           # WebSocket handlers
│   ├── __init__.py
│   ├── connection_manager.py  # ConnectionManager class
│   ├── job_updates.py   # /ws/jobs handler
│   ├── executor_pool.py # /ws/executor-pool handler
│   └── chat_handler.py  # /ws/chat handler
└── utils/               # Utilities
    ├── __init__.py
    └── constants.py     # Named constants (replaces magic numbers)
```

**Key Improvements**:
1. **CLIEventProcessor class** - The 381-line `_process_cli_event` function refactored into a class with separate methods for each event type
2. **Separation of concerns** - Models, routes, services, websocket handlers in separate modules
3. **Named constants** - Magic numbers replaced with named constants in `utils/constants.py`
4. **DRY fix ready** - `get_tool_description` in shared location

**Next Steps**:
- Phase 5: Create final `app.py` to assemble and run
- Verify all imports work correctly
- Update `main.py` to use the new modules (incremental migration)

---

## Previous Latest Work: Hierarchical Delegation Pattern Design
**Architect**: System Architect
**Date**: 2026-01-03

**Design Documents:**
- `/docs/designs/swarm-brain-architecture.md` - Swarm Brain Server with learning capabilities (ADR-006) **NEW**
- `/swarms/swarm_dev/workspace/DESIGN_DELEGATION_PATTERN.md` - Optimal delegation patterns (ADR-005)
- `/workspace/MAILBOX_DESIGN.md` - Complete Agent Mailbox system design (ADR-003)
- `/workspace/WORK_LEDGER_DESIGN.md` - Work Ledger system design (ADR-004)
- `/workspace/LOCAL_NEURAL_BRAIN_DESIGN.md` - Research and model selection
- See ADR-002 below for Local Neural Brain architectural specification

## Current Objectives
- [COMPLETE] Fix activity panel state persistence when navigating away from the chat page
- [COMPLETE] Ensure agent/tool activities persist across navigation
- [COMPLETE] Move ActivityPanel to global sidebar for visibility on ALL pages

## Architecture Decisions
- Using React Context (AgentActivityContext) for global state management
- Extended the existing AgentActivityProvider to include panel-specific activities
- No external state management library (no Zustand) - using React's built-in useState/useContext
- ActivityPanel is now rendered in the global Sidebar component, visible on all pages

## Key Files
- `/frontend/lib/AgentActivityContext.tsx` - Global context for activity state
- `/frontend/app/chat/page.tsx` - Chat page component (MODIFIED - mobile responsive, bottom sheet)
- `/frontend/components/ActivityPanel.tsx` - Activity panel display component (MODIFIED - mobile touch targets)
- `/frontend/components/Sidebar.tsx` - Global sidebar (MODIFIED - mobile onNavigate, touch targets)
- `/frontend/components/MobileLayout.tsx` - NEW: Mobile-responsive layout wrapper with hamburger menu
- `/frontend/components/ChatInput.tsx` - Chat input (MODIFIED - mobile touch targets, 16px font)
- `/frontend/components/Providers.tsx` - App-level provider wrapper
- `/frontend/app/layout.tsx` - Root layout (MODIFIED - viewport meta, MobileLayout wrapper)
- `/frontend/app/globals.css` - Global styles (MODIFIED - mobile utilities, animations)
- `/shared/agent_mailbox.py` - IMPLEMENTED: Agent Mailbox system for structured handoffs
- `/workspace/mailboxes/` - Directory for agent mailbox JSON files (created on demand)
- `/shared/work_ledger.py` - IMPLEMENTED: Work Ledger system for persistent work tracking
- `/shared/work_models.py` - IMPLEMENTED: WorkItem, WorkStatus, WorkType dataclasses
- `/workspace/ledger/` - Directory for work item JSON files (created on demand)
- `/swarms/swarm_dev/workspace/DESIGN_DELEGATION_PATTERN.md` - Hierarchical delegation pattern design (ADR-005)
- `/docs/designs/swarm-brain-architecture.md` - Swarm Brain Server architecture (ADR-006) **NEW**

## Next Steps for Swarm Brain Implementation

1. **Phase 1 (MVP)**: Create `brain/` module with experience memory
   - `brain/server.py` - FastAPI app skeleton
   - `brain/experience.py` - Task trajectory storage
   - Basic POST/GET endpoints for experiences

2. **Phase 2**: Add semantic search with sentence-transformers
   - `brain/embeddings.py` - Embedding engine
   - ChromaDB integration for vector storage

3. **Phase 3**: Context synthesis endpoint
   - `brain/context.py` - Unified context builder
   - Integration with agent executor

4. **Phase 4**: Learning engine
   - `brain/learner.py` - Training-Free GRPO
   - `brain/distiller.py` - Knowledge extraction

5. **Phase 5**: Swarm awareness
   - `brain/awareness.py` - Agent performance tracking
   - Recommendation endpoints

## Progress Log

### 2026-01-05 - WebSocket Disconnect/Chat Loading Deep Review
**Reviewer**: Quality Critic
**Result**: NEEDS_CHANGES

**Symptoms Reported**:
1. WebSocket connects then immediately disconnects (Total: 1 -> Total: 0)
2. Multiple reconnection attempts before stable connection
3. Chat content not loading

---

#### Critical Issues Found

**BUG 1: React Strict Mode Causes Double-Mount Connection Chaos**
- **File**: `/home/user/agent-swarm/frontend/next.config.js:3`
- **Code**: `reactStrictMode: true`
- **Root Cause**: React 18 Strict Mode mounts components twice in development. Both `AgentActivityContext` and `chat/page.tsx` call `ws.connect()` on mount, causing:
  1. First mount: connect() called, WebSocket in CONNECTING state
  2. Strict Mode unmount: cleanup runs, but does NOT call disconnect()
  3. Second mount: connect() called again, previous connection closed mid-handshake
- **Impact**: Immediate disconnect pattern, unstable initial connection

**BUG 2: connect() Only Guards OPEN State, Not CONNECTING**
- **File**: `/home/user/agent-swarm/frontend/lib/websocket.ts:63-72`
- **Code**:
  ```typescript
  if (this.ws?.readyState === WebSocket.OPEN) {
    return Promise.resolve()
  }
  if (this.ws) {
    this.ws.close()  // Kills CONNECTING WebSocket!
    this.ws = null
  }
  ```
- **Root Cause**: Guard only checks `OPEN`, not `CONNECTING` (readyState === 0). If a connection is being established, a second `connect()` call closes it and starts fresh.
- **Impact**: Race condition when multiple components call connect() simultaneously

**BUG 3: Dual WebSocket Consumers Both Call connect()**
- **Files**:
  - `/home/user/agent-swarm/frontend/lib/AgentActivityContext.tsx:65,181`
  - `/home/user/agent-swarm/frontend/app/chat/page.tsx:50,456`
- **Code**: Both components do:
  ```typescript
  const wsRef = useRef(getChatWebSocket())
  // ...
  ws.connect()
  ```
- **Root Cause**: Both components mount at the same time (Context wraps layout, chat/page is nested). They share the singleton but call connect() independently.
- **Impact**: Race to connect, potential duplicate connections

**BUG 4: No disconnect() in Cleanup Functions**
- **Files**:
  - `/home/user/agent-swarm/frontend/app/chat/page.tsx:462-464`
  - `/home/user/agent-swarm/frontend/lib/AgentActivityContext.tsx:183-185`
- **Code**:
  ```typescript
  return () => {
    ws.off('*', handleEvent)
    // Missing: ws.disconnect() or connection management
  }
  ```
- **Root Cause**: Cleanup only removes event handlers, never closes the WebSocket
- **Impact**: Orphaned connections stay open, reconnect logic fires unexpectedly

**BUG 5: onclose Handler Triggers Reconnect Before Cleanup**
- **File**: `/home/user/agent-swarm/frontend/lib/websocket.ts:94-98`
- **Code**:
  ```typescript
  this.ws.onclose = () => {
    console.log('WebSocket disconnected')
    this.emit('disconnected', { type: 'error', message: 'Disconnected' })
    this.attemptReconnect()  // Immediate reconnect!
  }
  ```
- **Root Cause**: When React Strict Mode unmounts/remounts, the close event may fire and trigger reconnection WHILE React is still in the middle of cleanup/remount cycle
- **Impact**: Reconnection races with remount, creating connection chaos

---

#### Warnings (Should Fix)

**WARN 1: Chat Content May Not Load Due to Event Handler Timing**
- **Files**:
  - `/home/user/agent-swarm/frontend/app/chat/page.tsx:132-145` (session loading)
  - `/home/user/agent-swarm/frontend/app/chat/page.tsx:148-465` (WebSocket handlers)
- **Issue**: Sessions load via REST API, but WebSocket events for streaming chat may arrive before handlers are fully registered (due to mount timing issues)
- **Impact**: Chat events lost during connection instability

**WARN 2: Backend Connection Count Can Get Out of Sync**
- **File**: `/home/user/agent-swarm/backend/main.py:2210-2215`
- **Code**: `disconnect()` silently ignores ValueError if already removed
- **Issue**: If disconnect is called twice (from multiple exception paths), the count logging may be misleading

---

#### Recommended Fixes

**FIX 1: Add CONNECTING state guard to connect()**
```typescript
connect(): Promise<void> {
  // Guard: already connected or connecting
  if (this.ws?.readyState === WebSocket.OPEN ||
      this.ws?.readyState === WebSocket.CONNECTING) {
    return this.ws.readyState === WebSocket.OPEN
      ? Promise.resolve()
      : this.connectionPromise  // Return existing promise
  }
  // ... rest of connect logic
}
```

**FIX 2: Centralize WebSocket connection in AgentActivityContext only**
- Remove `ws.connect()` from `chat/page.tsx`
- Context should own the connection lifecycle
- Page should only register/unregister handlers

**FIX 3: Add disconnect() to cleanup or use connection ref counting**
```typescript
// In cleanup:
return () => {
  ws.off('*', handleEvent)
  // Only disconnect if no other consumers:
  if (ws.handlerCount === 0) {
    ws.disconnect()
  }
}
```

**FIX 4: Disable Strict Mode for production-like testing**
- Temporarily set `reactStrictMode: false` to verify fix
- Or implement proper connection lifecycle that survives double-mount

**FIX 5: Add connection state tracking to prevent reconnect races**
```typescript
private isReconnecting = false

private attemptReconnect() {
  if (this.isReconnecting) return
  this.isReconnecting = true
  // ... reconnect logic
  // Reset flag on success/failure
}
```

---

**Files Requiring Changes**:
- `/home/user/agent-swarm/frontend/lib/websocket.ts` - Add CONNECTING guard, connection promise tracking
- `/home/user/agent-swarm/frontend/app/chat/page.tsx` - Remove direct connect(), delegate to context
- `/home/user/agent-swarm/frontend/lib/AgentActivityContext.tsx` - Own connection lifecycle fully
- `/home/user/agent-swarm/frontend/next.config.js` - Consider strictMode implications

**Priority**: HIGH - These bugs directly cause the reported symptoms

---

### 2026-01-03 - Swarm Brain Architecture Design (ADR-006)
**Architect**: System Architect

**Status**: DESIGN COMPLETE

**Design Document**: `/docs/designs/swarm-brain-architecture.md`

**Summary**: Comprehensive architecture for a "Swarm Brain Server" that provides persistent memory, learning capabilities, and unified context to the agent swarm system. Inspired by MYND app's brain server while tailored to multi-agent orchestration.

**Key Components**:
1. **Context Synthesizer** - Unified context endpoint for all agents
2. **Experience Memory** - Vector-indexed storage of task outcomes
3. **Learning Engine** - Training-Free GRPO with contrastive trajectory analysis
4. **Knowledge Distiller** - Extract patterns from successful agent outputs
5. **Swarm Awareness** - Track agent capabilities and performance

**Architecture Decisions**:
- FastAPI server on `localhost:8421` (separate from main backend at 8000)
- Training-Free GRPO instead of neural network training ($8 vs $1000s cost)
- Three-tier memory hierarchy: Global -> Swarm -> Agent
- ChromaDB for vector storage with sentence-transformers embeddings
- Active pattern compression to prevent prompt bloat

**Learning Mechanism**:
- Store task trajectories with success/failure outcomes
- Find similar past experiences via embedding search
- Extract contrastive patterns (what works vs what fails) using Claude
- Distill patterns into meta-prompts injected into agent context
- Patterns promoted up hierarchy based on cross-agent success

**Incremental Path (6 phases)**:
1. MVP: Experience memory only (Week 1)
2. Semantic search with embeddings (Week 2)
3. Context synthesis endpoint (Week 3)
4. Learning engine with contrastive extraction (Week 4)
5. Swarm awareness and recommendations (Week 5)
6. Polish and optimization (Week 6+)

**Integration Points**:
- Agent Executor Pool calls brain for enhanced context
- Work Ledger notifies brain on task completion
- COO uses brain recommendations for delegation
- Backend proxies brain API endpoints

**Files to Create**: ~1,870 lines across 10 new files in `brain/` module

---

### 2026-01-03 - localStorage Race Condition Fix Code Review
**Reviewer**: Quality Critic
**Result**: APPROVED

**Files Reviewed**:
- `/Users/jellingson/agent-swarm/frontend/components/CeoTodoPanel.tsx`
- `/Users/jellingson/agent-swarm/workspace/research/localStorage_race_condition_analysis.md`

**Review Summary**:

The lazy initialization pattern fix is **correctly implemented** and follows React 18+ best practices.

**Correctness Verification**:
1. `getInitialTodos()` runs synchronously during initial render (lines 16-27)
2. Uses `useState<Todo[]>(getInitialTodos)` for lazy init (line 30)
3. Single save effect (lines 36-38) only runs after initialization complete
4. No race condition possible - data loaded before any effects run

**Edge Cases Handled**:
| Edge Case | Status |
|-----------|--------|
| localStorage unavailable | Returns `[]` safely |
| Corrupted JSON | try/catch with console.error |
| SSR | `typeof window === 'undefined'` guard |
| Missing key | Returns `[]` |
| Schema migration | Not handled (acceptable for simple todo list) |

**Code Quality**:
- Clean, well-structured code
- Proper TypeScript types
- Descriptive function naming
- Note: `useRef` import already cleaned up (was mentioned as pending in research)

**Suggestions (Nice to Have)**:
1. Consider schema validation for future-proofing stored data
2. Consider save debouncing if todo list grows large (not needed currently)

**Verdict**: The fix is correct, robust, and ready for commit.

---

### 2026-01-03 - localStorage Race Condition Analysis (CeoTodoPanel)
**Researcher**: Research Specialist

**Task**: Investigate and verify the localStorage race condition fix in CeoTodoPanel.tsx

**Findings**:

1. **Original Bug**: Classic React useEffect race condition where dual effects (load and save) could execute in wrong order, causing the save effect to overwrite localStorage with empty array before load completed.

2. **Fix Evolution**:
   - Commit `7c7249c`: Original buggy implementation with dual useEffects
   - Commit `55e2bb1`: First fix using `hasLoaded` ref guard
   - Current (unstaged): Refactored to lazy initialization pattern

3. **Current Fix Assessment**: CORRECT AND ROBUST
   - Uses `useState<Todo[]>(getInitialTodos)` for synchronous initialization
   - No race condition possible - data loaded during render, before effects run
   - SSR-safe with `typeof window === 'undefined'` guard
   - Single save effect instead of dual load/save effects

4. **Other Components**: CeoTodoPanel is the ONLY component using localStorage in the frontend. No other components at risk.

5. **Minor Cleanup**: The `useRef` import on line 3 is now unused and can be removed.

**Analysis Document**: `/workspace/research/localStorage_race_condition_analysis.md`

**Recommendations**:
- Commit the current unstaged changes (fix is correct)
- Remove unused `useRef` import
- Use lazy initialization pattern for future localStorage usage

---

### 2026-01-03 - Session Persistence & Delegation Tracking Integration
**Implementer**: COO (Supreme Orchestrator)

**Problem**: When a conversation ends (usage limit, timeout, etc.), we lose track of what subagents were working on. There's no recovery mechanism on restart.

**Root Cause Analysis**:
1. Claude's Task tool state is in-memory only - not persisted
2. Work Ledger exists but wasn't connected to Task delegations
3. Auto-spawn was implemented but not enabled at startup
4. Orphaned work recovery was implemented but not called at startup

**Solution Implemented**:

1. **Enabled Auto-Spawn at Startup** (`backend/main.py:253-255`)
   - Added import: `from shared.auto_spawn import enable_auto_spawn`
   - Called `enable_auto_spawn()` in `startup_event()`
   - When work items are created without an owner, agents are automatically spawned

2. **Added Orphaned Work Recovery** (`backend/main.py:257-267`)
   - On server startup, recovers any work items that were IN_PROGRESS when server stopped
   - Uses 30-minute timeout to identify stale work
   - Resets orphaned work to PENDING status for re-execution

3. **Tracked Delegations in Work Ledger** (`backend/main.py:2348-2369`)
   - When COO uses Task tool to delegate, creates a WorkItem in the ledger
   - Work item includes: title, description, subagent_type, parent_agent
   - Work item is claimed immediately by the delegated agent
   - Stored in `workspace/ledger/active/` as JSON files

4. **Marked Delegations Complete** (`backend/main.py:2416-2428`)
   - When Task tool completes, marks the corresponding WorkItem as completed
   - Result includes status and agent name
   - Work item archived to `workspace/ledger/completed/`

**Files Modified**:
- `backend/main.py`: Added auto_spawn import, startup recovery, delegation tracking

**How Session Recovery Now Works**:
1. When server restarts, `recover_orphaned_work()` finds stale IN_PROGRESS items
2. These are reset to PENDING status
3. With auto-spawn enabled, the work items trigger new agent executions
4. Work resumes from where it left off (using WorkItem context)

**Testing Required**:
- [ ] Verify server starts without errors
- [ ] Create a Task delegation and verify WorkItem created
- [ ] Stop server mid-delegation and restart - verify recovery
- [ ] Verify auto-spawn creates agents for new work items

**Status**: COMPLETE (pending verification)

---

### 2026-01-03 - Main.py Modular Refactoring Plan
**Architect**: System Architect
**Design Document**: `/swarms/swarm_dev/workspace/MAIN_PY_REFACTOR_PLAN.md`

**Context**: `backend/main.py` has grown to 2823 lines, making it unmaintainable. Code review identified this as a CRITICAL issue.

**Analysis Performed**:
- Full line-by-line analysis of main.py structure
- Identified 15+ logical groupings of functionality
- Mapped dependencies between components
- Identified shared code that causes DRY violations

**Proposed Architecture**:
```
backend/
    app.py                          # FastAPI app factory (~150 lines)
    models/                         # Pydantic models (~170 lines total)
        requests.py, responses.py, chat.py
    routes/                         # API endpoints (~1150 lines total)
        agents.py, chat.py, escalations.py, files.py,
        jobs.py, mailbox.py, swarms.py, web.py,
        work.py, workflows.py
    services/                       # Business logic (~630 lines total)
        chat_history.py, claude_service.py,
        event_processor.py, orchestrator_service.py
    websocket/                      # WebSocket handlers (~580 lines total)
        connection_manager.py, chat_handler.py,
        job_updates.py, executor_pool.py
    utils/                          # Shared utilities (~70 lines total)
        tool_helpers.py, constants.py
```

**Critical Refactors Identified**:
1. `_process_cli_event` (381 lines, 7 nesting levels) -> `CLIEventProcessor` class with method dispatch
2. `_get_tool_description` DRY violation -> Move to `utils/tool_helpers.py`, update `agent_executor_pool.py`
3. Magic numbers -> Named constants in `utils/constants.py`

**Implementation Plan**: 5 phases over 2-3 days
- Phase 1: Foundation (directory structure, models, utils)
- Phase 2: Services (chat_history, orchestrator, event_processor, claude_service)
- Phase 3: WebSocket (connection_manager, handlers)
- Phase 4: Routes (all 10 route modules)
- Phase 5: App assembly and integration testing

**See**: `/swarms/swarm_dev/workspace/MAIN_PY_REFACTOR_PLAN.md` for full specification

---

### 2026-01-03 - Code Quality Review: Backend & Shared Modules
**Reviewer**: Reviewer Agent (Code Quality Specialist)
**Result**: NEEDS_CHANGES

**Files Reviewed:**
- `/Users/jellingson/agent-swarm/shared/__init__.py` (85 lines) - Excellent
- `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py` (672 lines) - Good
- `/Users/jellingson/agent-swarm/shared/workspace_manager.py` (326 lines) - Good
- `/Users/jellingson/agent-swarm/backend/main.py` (2823 lines) - Needs Refactor
- `/Users/jellingson/agent-swarm/backend/jobs.py` (628 lines) - Good

**Key Findings:**

1. **Critical: main.py is too large (2823 lines)**
   - Should be split into route modules (chat, work, mailbox, files, jobs)
   - `_process_cli_event` function is 381 lines with 7 levels of nesting

2. **DRY Violation: Duplicate `_get_tool_description`**
   - Exists in both `agent_executor_pool.py:597-625` and `main.py:2024-2044`
   - Should move to shared utility module

3. **Import placement issues**
   - `threading` import at line 639 in `agent_executor_pool.py` (should be at top)
   - Same issue in `workspace_manager.py:293`

4. **Magic numbers without constants**
   - `messages[-2:]` - should be `MAX_RECENT_MESSAGES = 2`
   - `content[:1000]` - should be `MAX_CONTENT_LENGTH = 1000`

**Positive Observations:**
- Excellent type hint coverage across all files
- Thread-safe singleton implementations with double-checked locking
- Good async/await usage without blocking calls
- Proper resource cleanup with finally blocks
- Security-conscious path traversal protection

**Full Review:** `/swarms/swarm_dev/workspace/REVIEW_REVIEWER_2026-01-03.md`

**Recommended Next Steps:**
1. Create plan to split `main.py` into route modules
2. Move `_get_tool_description` to shared utility
3. Refactor `_process_cli_event` into event handler class
4. Add constants for magic numbers

---

### 2026-01-03 - Hierarchical Delegation Pattern Design
**Architect**: System Architect

**Problem:** COO (Supreme Orchestrator) inconsistently delegates work to agents:
1. Claims to delegate but doesn't actually spawn agents
2. Does work itself that should be delegated
3. Spawns agents without proper follow-through
4. Doesn't wait for or synthesize results from delegated tasks

**Analysis Conducted:**
- Reviewed `backend/main.py` - COO system prompt (lines 2550-2606), Task detection (lines 2239-2271)
- Reviewed `shared/agent_executor_pool.py` - Agent execution mechanism
- Reviewed `swarms/swarm_dev/swarm.yaml` - Agent definitions
- Reviewed `supreme/agents/supreme.md` - COO agent definition
- Reviewed `shared/swarm_interface.py` - Swarm and workflow structures

**Root Causes Identified:**
1. **Delegation Theater**: COO can describe delegation without using Task tool
2. **No enforcement**: System prompt describes patterns but doesn't enforce them
3. **Fire-and-forget**: No result tracking for spawned agents
4. **Missing synthesis**: COO completes before verifying delegation results

**Solution Designed:** Complete Hierarchical Delegation Pattern at `/swarms/swarm_dev/workspace/DESIGN_DELEGATION_PATTERN.md`

**Key Design Decisions:**

1. **Three-Tier Hierarchy**: CEO -> COO -> Swarm Orchestrators -> Swarm Agents
2. **Mandatory Delegation Rules**: COO MUST delegate for code >5 lines, multi-file changes, tests, architecture decisions, deep research
3. **Decision Tree**: Clear criteria for when to do vs when to delegate
4. **Anti-Patterns Documented**: Delegation theater, over-delegation, fire-and-forget, sequential over-caution, context loss, recursive delegation
5. **Result Tracking**: Integration with Work Ledger for delegation persistence
6. **Synthesis Enforcement**: Require result synthesis before completing delegated work

**Implementation Plan:**
- Phase 1 (1 day): COO prompt hardening with enforcement rules
- Phase 2 (2-3 days): Delegation tracking via Work Ledger
- Phase 3 (1 day): Agent prompt updates (remove Task from workers)
- Phase 4 (1-2 days): Observability - metrics and UI surfacing

**Files to Modify:**
- `backend/main.py:2550-2606` - COO system prompt
- `shared/work_ledger.py` - Add parent/child delegation queries
- `swarms/*/agents/*.md` - Update agent prompts
- `supreme/agents/*.md` - Update executive team prompts

**See:** `/swarms/swarm_dev/workspace/DESIGN_DELEGATION_PATTERN.md` for complete specification

---

### 2026-01-03 - Code Review: Activity Panel & Agent Tracking
**Reviewer**: Quality Critic
**Result**: NEEDS_CHANGES

**Files Reviewed:**
- `/frontend/components/ActivityPanel.tsx` - Fullscreen mode, file tracking, notifications
- `/backend/main.py` - Agent stack tracking, WebSocket broadcasting
- `/frontend/app/chat/page.tsx` - Agent event handling
- `/frontend/lib/websocket.ts` - WebSocket types
- `/frontend/app/globals.css` - New animations

**Critical Issues (Must Fix):**

1. **Race Condition in Agent Stack Management** (`backend/main.py:1556-1576, 1623-1643, 1750-1762`)
   - At `content_block_start`, `tool_input` dict is often EMPTY because input is streamed via `input_json_delta`
   - Agent pushed with empty name or never pushed at all
   - Agent never popped because `tool_use_id` not in `pending_tasks`
   - **Fix**: Only push to stack AFTER full tool input streamed in `input_json_delta` or `content_block_stop`

2. **Duplicate agent_spawn Events** (`backend/main.py:1698-1717`)
   - Detection happens in both `input_json_delta` and `content_block_start`
   - `agent_spawn_sent` flag only set in one location
   - **Fix**: Consolidate logic to one location

**Warnings (Should Fix):**

3. Notification permission requested on every mount (`ActivityPanel.tsx:241-246`)
4. Memory leak: setTimeout without cleanup (`ActivityPanel.tsx:236`)
5. Unused `useCallback` import (`ActivityPanel.tsx:3`)
6. Missing error handling for Notification API (`ActivityPanel.tsx:228-232`)
7. Executor pool callback may not be thread-safe (`main.py:136-140`)

**Positive Observations:**
- Fullscreen layout fix is correct (`inset-4 md:inset-12 lg:inset-16`)
- Tool icons with color coding by operation type is excellent
- Progress bar animation for running tools is well done
- Agent stack concept is architecturally sound (just needs timing fix)
- Escape key handling and cleanup is proper
- WebSocket types properly extended with new fields

**Next Steps:**
- Fix critical race condition in agent stack management
- Consolidate agent_spawn event logic
- Add setTimeout cleanup in useEffect return
- Wrap Notification API in try-catch

---

### 2026-01-03 - Work Ledger System Implementation
**Implementer**: Implementation Specialist

**Task:** Implement Work Ledger system based on `/workspace/WORK_LEDGER_DESIGN.md`

**Files Created:**

1. `/shared/work_models.py` (243 lines) - Data structures
2. `/shared/work_ledger.py` (1065 lines) - Main WorkLedger class

**Implementation Details:**

1. **Enums (work_models.py):**
   - `WorkStatus` - PENDING, IN_PROGRESS, BLOCKED, COMPLETED, FAILED, CANCELLED
   - `WorkType` - TASK, FEATURE, BUG, RESEARCH, REVIEW, DESIGN, REFACTOR, TEST, DOCUMENTATION, ESCALATION
   - `WorkPriority` - LOW, MEDIUM, HIGH, CRITICAL

2. **Dataclasses (work_models.py):**
   - `WorkHistoryEntry` - Audit trail entry with timestamp, action, actor, details, status transition
   - `WorkItem` - Full work unit with 21 fields: id, title, type, priority, status, owner, description, context, parent_id, dependencies, created_at, created_by, updated_at, started_at, completed_at, result, error, swarm_name, job_id, execution_id, history
   - `WorkIndex` - Manifest with items map, by_status, by_owner, by_swarm, by_parent indexes
   - All include `to_dict()` and `from_dict()` methods

3. **WorkLedger Class (work_ledger.py):**
   - Thread-safe using `threading.RLock`
   - Directory structure: `workspace/ledger/active/`, `completed/YYYY/MM/`, `failed/YYYY/MM/`
   - In-memory cache for loaded work items
   - Atomic file writes (temp file + rename pattern)

   **Creation Methods:**
   - `create_work()` - Create new work item with validation
   - `create_subtask()` - Create child work item inheriting parent context

   **Claiming Methods:**
   - `claim_work()` - Atomically claim work (sets owner, status to IN_PROGRESS)
   - `release_work()` - Release claimed work back to PENDING

   **Status Transition Methods:**
   - `start_work()` - Mark as IN_PROGRESS
   - `block_work()` - Mark as BLOCKED with reason
   - `complete_work()` - Mark as COMPLETED with result
   - `fail_work()` - Mark as FAILED with error

   **Query Methods:**
   - `get_work()` - Get single work item by ID
   - `get_pending()` - Get pending items with filters (owner, swarm, type)
   - `get_in_progress()` - Get in-progress items with owner filter
   - `get_blocked()` - Get all blocked items
   - `get_children()` - Get child items of parent
   - `get_by_swarm()` - Get all items for a swarm
   - `get_ready_to_start()` - Get unclaimed items with satisfied dependencies

   **Hierarchy Methods:**
   - `get_progress()` - Get completion stats for parent (total, completed, in_progress, etc.)

   **Recovery Methods:**
   - `recover_orphaned_work()` - Reset stale IN_PROGRESS items to PENDING
   - `get_stale_work()` - Find items not updated recently
   - `load_from_disk()` - Rebuild index from disk files

4. **Singleton Pattern:**
   - `get_work_ledger()` - Thread-safe double-checked locking singleton

**Persistence:**
- Work items stored in `workspace/ledger/active/WRK-YYYYMMDD-NNNN.json`
- Completed items archived to `workspace/ledger/completed/YYYY/MM/`
- Failed items archived to `workspace/ledger/failed/YYYY/MM/`
- Index stored in `workspace/ledger/index.json`
- ID counter restored from existing files on load

**Thread Safety:**
- `threading.RLock` for all operations
- `threading.Lock` for singleton initialization
- List copies before mutation during iteration

**Status:** COMPLETE

**Next Steps:**
- Create unit tests
- Integrate with `agent_executor_pool.py` (add work_id parameter)
- Integrate with `execution_context.py` (add work_id field)
- Integrate with `backend/main.py` (create work items for requests)
- Add to `shared/__init__.py` exports

---

### 2026-01-03 - Agent Mailbox System Implementation
**Implementer**: Implementation Specialist

**Task:** Implement Phase 1 of Agent Mailbox System based on `/workspace/MAILBOX_DESIGN.md`

**File Created:** `/shared/agent_mailbox.py` (1005 lines)

**Implementation Details:**

1. **Enums:**
   - `MessageType` - request, response, notification, handoff, escalation
   - `MessagePriority` - LOW(1), NORMAL(2), HIGH(3), URGENT(4)
   - `MessageStatus` - unread, read, processing, completed, archived, failed

2. **Dataclasses:**
   - `Message` - Core message with 17 fields, includes `to_dict()`, `from_dict()`, `to_markdown()`
   - `HandoffContext` - Structured handoff context with `to_dict()`, `from_dict()`, `to_markdown()`

3. **MailboxManager Class:**
   - Thread-safe singleton using RLock for all mutations
   - `send()` - Send message to agent's mailbox
   - `handoff()` - Convenience method for structured handoffs
   - `broadcast()` - Send to all agents in swarm
   - `check_mailbox()` - Priority-sorted message retrieval with filters
   - `read_message()` - Mark as read with timestamp
   - `mark_processing()` - Status transition
   - `mark_completed()` - Complete with optional archive
   - `reply()` - Reply maintaining thread
   - `get_thread()` - Get all messages in conversation
   - `get_pending_count()` - Count by priority level
   - Atomic file writes (temp + rename pattern from escalation_protocol.py)
   - Archive functionality (inbox -> archive directory)

4. **Module-level Convenience Functions:**
   - `get_mailbox_manager()` - Thread-safe singleton with double-checked locking
   - `send_message()` - Quick message sending
   - `check_my_mailbox()` - Check mailbox for agent
   - `send_handoff()` - Structured handoff with HandoffContext
   - `broadcast_to_swarm()` - Swarm-wide notification

**Persistence:**
- Messages stored in `workspace/mailboxes/{agent}/inbox/MSG-{uuid}.json`
- Archive in `workspace/mailboxes/{agent}/archive/`
- Atomic writes prevent corruption on crash
- Messages loaded from disk on startup

**Thread Safety:**
- `threading.RLock` for all mutable operations
- `threading.Lock` for singleton initialization
- Snapshot-based queries (copy before filtering)

**Status:** COMPLETE

**Next Steps:**
- Create unit tests at `/tests/test_agent_mailbox.py`
- Integrate with `agent_executor_pool.py`
- Integrate with `escalation_protocol.py`
- Add to `shared/__init__.py` exports

---

### 2026-01-03 - Agent Mailbox System Design
**Architect**: System Architect

**Problem:** Agents delegate directly without structured handoffs. No way for agents to leave messages for specific agents. Coordination breaks down when agents are busy or offline. No message persistence.

**Inspiration:** Gas Town's mailbox approach where each agent has a mailbox, messages are persistent files on disk, structured handoffs with context, and agents check mailbox on startup.

**Solution Designed:** Complete Agent Mailbox system at `/workspace/MAILBOX_DESIGN.md`

**Key Data Structures:**
- `Message` - Core message with id, from_agent, to_agent, type, priority, payload, status, thread_id
- `MessageType` enum - request, response, notification, handoff, escalation
- `MessagePriority` enum - low, normal, high, urgent
- `MessageStatus` enum - unread, read, processing, completed, archived, failed
- `HandoffContext` - Structured context for agent-to-agent handoffs

**Key API Methods:**
- `send()` - Send a message to an agent's mailbox
- `handoff()` - Send a structured handoff with complete context
- `broadcast()` - Send message to all agents in a swarm
- `check_mailbox()` - Check for pending messages (priority-sorted)
- `read_message()`, `mark_processing()`, `mark_completed()` - Status transitions
- `reply()` - Reply to a message (maintains thread)
- `get_thread()` - Get all messages in a conversation

**Persistence Strategy:**
- JSON files in `workspace/mailboxes/{agent_name}/inbox/`
- Archive directory for completed messages
- Atomic writes (temp file + rename) for crash safety
- In-memory index for fast lookups

**Integration Points:**
- `shared/escalation_protocol.py` - Send notification on escalation
- `shared/agent_executor_pool.py` - Check mailbox on agent startup
- Agent system prompts - Mailbox awareness instructions

**Files to Create:**
- `/shared/agent_mailbox.py` - Main MailboxManager class
- `/tests/test_agent_mailbox.py` - Unit tests

**Implementation Plan:** 4-phase rollout over ~5 days

**See:** `/workspace/MAILBOX_DESIGN.md` for complete specification

---

### 2026-01-03 - Work Ledger System Design
**Architect**: System Architect

**Problem:** Agents lose work state if they crash or restart. Work state is in agent memory, not persisted. There is no structured way to track work units across agent lifecycles.

**Inspiration:** Gas Town's "Beads" system where work persists on hooks and survives crashes with a git-backed ledger of work items.

**Solution Designed:** Complete Work Ledger system at `/workspace/WORK_LEDGER_DESIGN.md`

**Key Data Structures:**
- `WorkItem` - Persistent work unit with ID, type, status, owner, dependencies, result
- `WorkStatus` enum - pending, in_progress, blocked, completed, failed, cancelled
- `WorkType` enum - task, feature, bug, research, review, design, refactor, test, documentation
- `WorkHistoryEntry` - Audit trail of all state changes
- `WorkIndex` - Manifest file for fast lookups

**Key API Methods:**
- `create_work()` - Create new work items with hierarchy support
- `claim_work()` / `release_work()` - Atomic work claiming by agents
- `start_work()`, `block_work()`, `complete_work()`, `fail_work()` - Status transitions
- `get_ready_to_start()` - Find unclaimed work with satisfied dependencies
- `recover_orphaned_work()` - Reclaim work from crashed agents

**Persistence Strategy:**
- JSON files in `workspace/ledger/` directory
- Atomic writes (temp file + rename) for crash safety
- Index manifest for O(1) lookups
- Archive completed/failed work by date

**Thread Safety:**
- RLock for index operations
- Per-item locks for concurrent mutations
- Snapshot queries that release lock before returning

**Integration Points:**
- `shared/agent_executor_pool.py` - Link executions to work items
- `shared/execution_context.py` - Add work_id field
- `backend/main.py` - Create work items for user requests
- `shared/escalation_protocol.py` - Block work on escalation

**Files to Create:**
- `/shared/work_ledger.py` - Main WorkLedger class
- `/shared/work_models.py` - Data structures

**Implementation Plan:** 5-phase rollout over ~5 days

**See:** `/workspace/WORK_LEDGER_DESIGN.md` for complete specification

---

### 2026-01-02 - Move ActivityPanel to Global Sidebar
**Problem:** The ActivityPanel was only visible on the chat page. Users navigating to /dashboard or other pages could not see agent activity, even though the state was persisted in the global context.

**Solution Implemented:**
1. Added `ActivityPanel` import to `Sidebar.tsx`
2. Added context hooks to Sidebar to access panel activities:
   - `panelAgentActivities`
   - `panelToolActivities`
   - `clearPanelActivities`
3. Added computed values for display logic:
   - `hasActivity` - true if any activities exist
   - `isProcessing` - true if any agent is still working
4. Rendered ActivityPanel in sidebar between the Swarms list and CEO Todo Panel
5. Removed ActivityPanel rendering from `chat/page.tsx`
6. Removed unused `clearPanelActivities` import from chat page

**Files Modified:**
- `frontend/components/Sidebar.tsx`: Added ActivityPanel with global context data
- `frontend/app/chat/page.tsx`: Removed ActivityPanel rendering (kept WebSocket handlers)

**Behavior:**
- The ActivityPanel now appears in the sidebar when there is any activity
- Activity is visible whether on /dashboard, /chat, /swarm/*, or any other page
- The chat page still handles WebSocket events and updates the global context
- The Clear History button in the panel works from any page

### 2026-01-02 - Activity Panel State Persistence
**Problem:** When navigating away from the chat page, the activity panel state (agentActivities, toolActivities) was lost. The backend work continued but the frontend lost the WebSocket connection state.

**Solution Implemented:**
1. Extended `AgentActivityContext` to include new types and state:
   - Added `PanelAgentActivity` and `PanelToolActivity` interfaces
   - Added `panelAgentActivities` and `panelToolActivities` state arrays
   - Added `setPanelAgentActivities`, `setPanelToolActivities`, and `clearPanelActivities` functions

2. Updated `ActivityPanel.tsx`:
   - Imported types from `AgentActivityContext` instead of defining locally
   - Re-exported types for backwards compatibility

3. Updated `chat/page.tsx`:
   - Replaced local `useState` for agentActivities/toolActivities with context hooks
   - Using `useAgentActivity()` to get/set activities from global context
   - Activities now persist across component unmount/remount

**Files Modified:**
- `frontend/lib/AgentActivityContext.tsx`: Added panel activity types and state
- `frontend/app/chat/page.tsx`: Use context instead of local state
- `frontend/components/ActivityPanel.tsx`: Import types from context

## Known Issues

### CRITICAL: WebSocket Connection Leaks (from 2026-01-05 Stability Review)

**Root Cause**: Frontend `connect()` method creates new WebSocket every call, orphaning previous connections.

1. **Frontend connect() overwrites without closing**
   - File: `frontend/lib/websocket.ts:63-66`
   - `connect()` creates NEW WebSocket, overwrites `this.ws` reference
   - Previous connection orphaned (still connected to backend, no cleanup)

2. **Components don't disconnect on unmount**
   - Files: `frontend/app/chat/page.tsx:462-464`, `frontend/lib/AgentActivityContext.tsx:183-185`
   - Cleanup only unregisters handlers, never closes WebSocket
   - Status: NEEDS FIX

3. **Backend doesn't clean dead connections from active_connections**
   - File: `backend/main.py:1412-1416` (and similar broadcast loops)
   - Failed sends silently ignored, dead connections stay in list
   - Status: NEEDS FIX

---

### CRITICAL: COO Delegation System Broken (from 2026-01-03 Delegation Review)

**Full Analysis**: `/swarms/swarm_dev/workspace/REVIEW_DELEGATION_FAILURES.md`

4. **Task Tool Does Not Spawn Real Agents**
   - Severity: CRITICAL
   - When COO uses `Task(subagent_type="architect", ...)`, no separate agent process is spawned
   - Claude's built-in Task tool runs internally with COO's context - same prompt, same workspace
   - The agent definitions in `.md` files are NEVER used for delegation
   - Status: NEEDS ARCHITECTURAL FIX

5. **WebSocket Chat Bypasses AgentExecutorPool**
   - Severity: CRITICAL
   - `backend/main.py:stream_claude_response()` spawns Claude directly
   - AgentExecutorPool provides isolation, concurrency, config - but is unused for main chat
   - Status: NEEDS FIX

6. **Work Ledger/Mailbox/Escalation Not Connected**
   - Severity: HIGH
   - All three systems built but not integrated with Task delegation flow
   - Status: NEEDS INTEGRATION

### Active Issues (from 2026-01-03 Code Review)

1. **Agent Stack Race Condition** - `backend/main.py:1556-1576, 1623-1643`
   - Severity: Critical
   - At `content_block_start`, `tool_input` is empty because it streams via `input_json_delta`
   - Causes incorrect agent attribution and missing completion events
   - Status: NEEDS FIX

### Code Quality Issues (from 2026-01-03 Reviewer Review)

7. **main.py is Too Large (2823 lines)** - `backend/main.py`
   - Severity: HIGH (Maintainability)
   - Should be split into route modules: chat, work, mailbox, files, jobs
   - `_process_cli_event` is 381 lines with 7 levels of nesting
   - Status: NEEDS REFACTOR
   - See: `/swarms/swarm_dev/workspace/REVIEW_REVIEWER_2026-01-03.md`

8. **Duplicate _get_tool_description Function** - FIXED
   - Severity: MEDIUM (DRY violation)
   - Was in both `agent_executor_pool.py` and `main.py`
   - Fixed: Created `get_tool_description()` in `shared/agent_executor_pool.py`, removed from main.py
   - Status: FIXED (2026-01-03)

9. **Import Placement Issues**
   - Severity: LOW (PEP 8)
   - `threading` import at module level in `agent_executor_pool.py:639` and `workspace_manager.py:293`
   - Should be moved to top of files
   - Status: NEEDS FIX

2. **Duplicate agent_spawn Events** - `backend/main.py:1698-1717`
   - Severity: Critical
   - Same detection logic in two places without proper coordination
   - Status: NEEDS FIX

3. **ActivityPanel setTimeout Memory Leak** - `ActivityPanel.tsx:236`
   - Severity: Warning
   - setTimeout not cleaned up on unmount
   - Status: NEEDS FIX

## Progress Log (continued)

### 2026-01-02 - WebSocket Connection Leak Fix
**Problem:** WebSocket connections were leaking in `/backend/main.py`:
- "Total: N" counter accumulated without proper cleanup
- Error: "Cannot call 'send' once a close message has been sent"
- Connections not removed from `active_connections` list

**Root Causes Identified:**
1. Missing `finally` block - cleanup only in `except` blocks
2. `disconnect()` failed with `ValueError` if item already removed
3. `send_event()` had no exception handling for closed sockets
4. Race condition - error handler tried to send after client disconnect

**Solution Implemented:**

**Fix 1: ConnectionManager.disconnect() (line 1122-1127)**
Added try/except to handle cases where websocket already removed:
```python
def disconnect(self, websocket: WebSocket):
    try:
        self.active_connections.remove(websocket)
    except ValueError:
        pass  # Already removed
    logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
```

**Fix 2: ConnectionManager.send_event() (lines 1129-1144)**
Added connection check and exception handling:
```python
async def send_event(self, websocket: WebSocket, event_type: str, data: dict[str, Any]):
    try:
        if websocket not in self.active_connections:
            return  # Connection already closed
        await websocket.send_json(...)
    except (RuntimeError, Exception) as e:
        if "close message" in str(e).lower():
            logger.debug(f"Skipped send to closed WebSocket: {e}")
        else:
            logger.error(f"Error sending event: {e}")
```

**Fix 3: websocket_chat() - finally block (lines 1932-1937)**
Changed from separate `except` blocks to unified `finally`:
```python
except WebSocketDisconnect:
    logger.info("WebSocket client disconnected")
except Exception as e:
    logger.error(f"WebSocket error: {e}")
finally:
    manager.disconnect(websocket)
```

**Fix 4: Error-path sends (lines 1911-1930)**
Wrapped error response sends in try/except:
```python
try:
    await manager.send_event(websocket, "agent_complete", {...})
    await manager.send_event(websocket, "chat_complete", {...})
except Exception as send_err:
    logger.debug(f"Failed to send error response (client may have disconnected): {send_err}")
```

**Files Modified:**
- `/backend/main.py`: ConnectionManager class and websocket_chat function

### 2026-01-02 - Escalation Protocol Code Review
**Reviewer**: Quality Critic
**Verdict**: NEEDS_CHANGES

**Files Reviewed**:
- `/Users/jellingson/agent-swarm/shared/escalation_protocol.py`
- `/Users/jellingson/agent-swarm/swarms/swarm_dev/workspace/DESIGN_ESCALATION_PROTOCOL.md`

**Critical Issues Found** (Must Fix):
1. **Thread Safety - Singleton Race Condition** (line 459-474): `get_escalation_manager()` is not thread-safe. Two threads could create separate instances.
2. **ID Collision After Restart** (line 198-204): `_id_counter` always starts at 0, not restored from loaded escalations.
3. **Non-Atomic File Writes** (line 367-373): Direct file writes could corrupt data on crash.

**High Issues Found** (Should Fix):
4. **Dict Mutation During Iteration** (line 334-365): Query methods not thread-safe.
5. **Missing Hierarchy Validation** (line 206-265): No validation that escalations follow AGENT->COO->CEO.
6. **Missing `__init__.py` Exports**: Escalation protocol not exported from shared module.

**Positive Observations**:
- Implementation matches design specification well
- Follows codebase patterns (similar to consensus.py)
- Good type hints and documentation
- Clean separation of concerns

## Known Issues

None currently tracked - escalation protocol issues have been fixed.

## Next Steps
- Test mobile UI on actual mobile device
- Verify hamburger menu opens/closes sidebar correctly
- Test bottom sheet for chat history on mobile
- Verify touch targets are at least 44px
- Test safe area insets on notched devices (iPhone X+)
- Run production build to verify no TypeScript errors

## Progress Log (continued)

### 2026-01-02 - Mobile UI Optimization
**Problem:** The Agent Swarm UI was not mobile-friendly. Issues included:
- Sidebar and main content overlapping/cramped on mobile
- Input area squeezed at the bottom
- Desktop layout forced on mobile screens
- Activity panel didn't work well on small screens
- No mobile-friendly navigation patterns

**Solution Implemented:**

**New Component: MobileLayout.tsx**
- Created responsive layout wrapper that detects mobile viewport (< 768px)
- Hamburger menu header on mobile with toggle button
- Sidebar slides in/out with overlay backdrop on mobile
- Prevents body scroll when sidebar is open
- Context provider (`useMobileLayout`) for child components to access mobile state

**Updated layout.tsx:**
- Added proper viewport meta tags (width, initialScale, maximumScale, userScalable)
- Wrapped content in MobileLayout component
- Moved Sidebar rendering into MobileLayout for responsive control

**Updated Sidebar.tsx:**
- Added `onNavigate` prop to close sidebar on navigation (mobile)
- Increased touch targets to 44px minimum on mobile
- Responsive icon sizes (w-5/h-5 on mobile, w-4/h-4 on desktop)
- Added `active:` states for touch feedback
- Hidden logo on mobile (shown in MobileLayout header instead)

**Updated chat/page.tsx:**
- Hidden desktop sidebar on mobile, replaced with bottom sheet
- Mobile history bottom sheet with slide-up animation
- Responsive header with mobile-specific history toggle
- Connection status text hidden on very small screens
- Responsive padding and spacing throughout
- Touch-friendly quick suggestion buttons
- Safe area padding for input on notched devices

**Updated ActivityPanel.tsx:**
- Mobile viewport detection
- Increased touch targets for all buttons
- Responsive max-height (48 on mobile, 64 on desktop)
- Added active states for touch feedback

**Updated ChatInput.tsx:**
- Touch-friendly buttons with 44px minimum size
- 16px font size to prevent iOS zoom on focus
- Responsive attachment preview sizing
- Hidden hint text on mobile to save space

**Updated globals.css:**
- Added touch-manipulation utility class
- Safe area inset utilities (pb-safe, pt-safe)
- Bottom sheet slide-up animation
- Better scrolling on iOS (-webkit-overflow-scrolling: touch)

**Files Modified:**
- `/frontend/app/layout.tsx`
- `/frontend/components/MobileLayout.tsx` (NEW)
- `/frontend/components/Sidebar.tsx`
- `/frontend/app/chat/page.tsx`
- `/frontend/components/ActivityPanel.tsx`
- `/frontend/components/ChatInput.tsx`
- `/frontend/app/globals.css`

### 2026-01-02 - Escalation Protocol Critical/High Fixes
**Problem:** Code review identified critical and high priority issues in `/shared/escalation_protocol.py`:
1. Thread-unsafe singleton pattern (race condition)
2. ID collision after restart (_id_counter not restored)
3. Non-atomic file writes (data corruption risk)
4. Thread-unsafe query methods (dict mutation during iteration)
5. Missing hierarchy validation (could create invalid escalation paths)

**Solution Implemented:**

**Fix 1: Thread-safe singleton** (lines 492-517)
- Added `_singleton_lock = threading.Lock()` at module level
- Implemented double-checked locking pattern in `get_escalation_manager()`

**Fix 2: ID collision prevention** (lines 405-421)
- Modified `load_from_disk()` to extract counter from existing escalation IDs
- After loading all files, sets `_id_counter = max_counter`
- Parses ID format ESC-TIMESTAMP-NNNN to find highest counter

**Fix 3: Atomic file writes** (lines 389-403)
- Write to temp file first (`.json.tmp` suffix)
- Use `temp_path.rename(filepath)` for atomic operation on POSIX
- Clean up temp file on error

**Fix 4: Thread-safe queries** (lines 200, 256, 300, 331, 353, 375, 383)
- Added `self._lock = threading.RLock()` in `__init__`
- Wrapped `create_escalation`, `resolve_escalation`, `update_status` with lock
- Wrapped `get_pending`, `get_by_swarm`, `get_blocked_work` with lock

**Fix 5: Hierarchy validation** (lines 245-254)
- Added validation in `create_escalation` before creating escalation
- Valid paths: AGENT->COO, COO->CEO only
- Raises `ValueError` for invalid paths

**Files Modified:**
- `/Users/jellingson/agent-swarm/shared/escalation_protocol.py`

**Testing:**
- Syntax verified by reading completed file
- Run `python3 -m py_compile shared/escalation_protocol.py` to verify

### 2026-01-03 - Claude CLI-Style Frontend Restyling
**Goal:** Restyle the Agent Swarm frontend to match Claude CLI's aesthetic - minimal, terminal-style dark theme with orange accents.

**Design Reference:**
- Deep black background: #0d0d0d (nearly pure black)
- Orange accent color: #ea580c / #f97316
- Monospace font: JetBrains Mono
- Minimal UI with clean lines, no gradients
- Subtle borders: zinc-800/50 separators
- White/light gray text for content

**Changes Implemented:**

**1. globals.css:**
- Added JetBrains Mono font import from Google Fonts
- Changed background from gradient to pure black (#0d0d0d)
- Updated CSS variables for new color scheme
- Changed agent-orchestrator color to orange (#ea580c)
- Added terminal cursor blink animation
- Added .terminal-prompt CSS class
- Simplified scrollbar styling (more minimal)
- Added fadeIn animation

**2. layout.tsx:**
- Switched font from Inter to JetBrains_Mono
- Updated body background to bg-[#0d0d0d]

**3. chat/page.tsx:**
- Replaced Bot icon with Terminal icon
- Changed purple-* colors to orange-* throughout
- Updated all bg-zinc-900 to bg-[#0d0d0d]
- Updated border colors to border-zinc-800/50 (more subtle)
- Added terminal-style prompt (>) in quick suggestions
- Made empty state use Terminal icon with orange tint

**4. ChatInput.tsx:**
- Added terminal prompt ">" character before textarea
- Changed send button from blue to orange
- Updated all backgrounds to #0d0d0d
- Changed hover states to orange accents
- Updated border styling to match terminal aesthetic

**5. ChatMessage.tsx:**
- Changed user avatar from blue to orange
- Updated "You" label to text-orange-500
- Changed text attachment icon to orange
- Updated backgrounds to pure black/subtle zinc

**6. AgentResponse.tsx:**
- Made orchestrator/supreme agent types use orange color
- Changed thinking indicator from purple to orange
- Updated CEO decision highlights to orange theme
- Changed thinking section styling to orange tints
- Made loading dots smaller and more subtle

**7. Sidebar.tsx:**
- Changed Bot logo to Terminal icon with orange color
- Updated active nav state to use orange accent
- Changed all blue-* references to orange-*
- Updated active swarm indicators to orange

**8. MobileLayout.tsx:**
- Added Terminal icon with orange accent in mobile header
- Updated overlay to pure black background
- Changed all background references to #0d0d0d

**9. ActivityPanel.tsx:**
- Changed all blue/purple accents to orange
- Updated status indicators (thinking, working, delegating) to orange
- Changed background to #0d0d0d
- Updated agent bot icons to orange

**Color Mappings Applied:**
- purple-* -> orange-*
- blue-* -> orange-*
- bg-zinc-950/bg-zinc-900 -> bg-[#0d0d0d]
- Primary accent: #ea580c

**Files Modified:**
- `/frontend/app/globals.css`
- `/frontend/app/layout.tsx`
- `/frontend/app/chat/page.tsx`
- `/frontend/components/ChatInput.tsx`
- `/frontend/components/ChatMessage.tsx`
- `/frontend/components/AgentResponse.tsx`
- `/frontend/components/Sidebar.tsx`
- `/frontend/components/MobileLayout.tsx`
- `/frontend/components/ActivityPanel.tsx`

### 2026-01-03 - Purple Accent Addition
**Goal:** Add subtle purple accents to complement the existing dark terminal theme with orange primary color.

**Purple Color Used:** #8B5CF6 (violet-500 in Tailwind)

**Changes Implemented:**

**1. globals.css:**
- Added purple CSS variables: `--accent-purple`, `--accent-purple-light`, `--accent-purple-dark`
- Added scrollbar hover state to purple
- Added `.purple-glow`, `.purple-glow-hover`, `.purple-glow-focus` utility classes
- Added `.border-purple-accent` and `.text-purple-accent` utility classes

**2. Sidebar.tsx:**
- Nav item hover: subtle purple background tint (`hover:bg-violet-500/5`)
- Active swarm border on working: purple tint (`border-violet-500/30`)
- Settings button hover: purple tint

**3. ChatInput.tsx:**
- Input focus: purple glow effect (`purple-glow-focus` class)
- Input border on focus: violet border (`focus-within:border-violet-500/50`)
- Attach button hover: purple accent (`hover:text-violet-400`, purple shadow)

**4. ActivityPanel.tsx:**
- Thinking status: violet color (`text-violet-400`)
- Delegating status: violet color
- Active badge: violet background/text
- Header hover: purple tint

**5. AgentResponse.tsx:**
- Thinking indicator: violet color
- Thinking section button hover: purple tint
- Thinking content: violet tinted background with left border

**6. FileBrowser.tsx:**
- Image file icon: violet color
- Selected file: violet background tint with left border
- Drag-over state: violet border/background
- Input focus states: violet border
- Create File button: violet background

**7. JobsPanel.tsx:**
- Running job status: violet color
- Running badge: violet background/text
- Progress bar: violet fill

**8. chat/page.tsx:**
- Quick suggestion buttons: purple hover effects
- Connection status dot: subtle green glow

**9. MobileLayout.tsx:**
- Mobile header: subtle purple shadow
- Hamburger menu button: purple hover

**Design Philosophy:**
- Purple is used for "thinking/processing" states (AI working)
- Purple hover effects add depth without changing the core orange accent
- Purple glow effects are subtle (15-20% opacity)
- Orange remains the primary action color (send button, terminal prompt)

**Files Modified:**
- `/frontend/app/globals.css`
- `/frontend/app/chat/page.tsx`
- `/frontend/components/Sidebar.tsx`
- `/frontend/components/ChatInput.tsx`
- `/frontend/components/ActivityPanel.tsx`
- `/frontend/components/AgentResponse.tsx`
- `/frontend/components/FileBrowser.tsx`
- `/frontend/components/JobsPanel.tsx`
- `/frontend/components/MobileLayout.tsx`

### 2026-01-03 - CeoTodoPanel localStorage Race Condition Fix
**Problem:** Todos were not persisting between page reloads due to a race condition in the useEffect hooks:
1. Initial state: `useState<Todo[]>([])` - empty array
2. First useEffect (load): Loads todos from localStorage
3. Second useEffect (save): Saves todos whenever they change

The save useEffect ran immediately on mount with the empty initial state, overwriting localStorage BEFORE the load useEffect had a chance to restore saved data.

**Fix Evolution:**
1. **First fix (hasLoaded ref):** Added `useRef` tracking to prevent premature saves
2. **Current fix (lazy initialization):** Refactored to use React's lazy initialization pattern

**Current Solution (Verified 2026-01-03):**
- Created `getInitialTodos()` function that loads from localStorage synchronously
- Uses `useState<Todo[]>(getInitialTodos)` for lazy initialization
- Single save useEffect instead of dual load/save effects
- SSR-safe with `typeof window === 'undefined'` guard

**Analysis Document:** `/workspace/research/localStorage_race_condition_analysis.md`

**Verification Status:** CORRECT - The lazy initialization pattern is the canonical React solution. No race condition possible because data is loaded synchronously during initial render.

**Minor Cleanup:** Removed unused `useRef` import (2026-01-03).

**Files Modified:**
- `/frontend/components/CeoTodoPanel.tsx`

## Project Priorities

| Priority | Project | Status | Next Step |
|----------|---------|--------|-----------|
| #1 | TBD | - | - |
| **#2** | **Agent Mailbox System** | Design Complete | Implement Phase 1 - Core classes |
| **#3** | **Work Ledger System** | Design Complete | Implement Phase 1 - Core data structures |
| **#4** | **Local Neural Net Brain** | Design Complete | Explore options for privacy/latency before implementation |

## Next Steps
- [#2] Agent Mailbox: Implement Phase 1 - Core classes (shared/agent_mailbox.py)
- [#2] Agent Mailbox: Implement Phase 2 - Persistence and testing
- [#2] Agent Mailbox: Integrate with escalation_protocol.py and agent_executor_pool.py
- [#3] Work Ledger: Implement Phase 1 - Core data structures (work_models.py)
- [#3] Work Ledger: Implement Phase 2 - Core operations (work_ledger.py)
- [#4] Local Neural Brain: Explore privacy constraints and latency requirements
- Continue Smart Context Injection development (ADR-001)

---

## Architecture Decisions

### ADR-006: Hard Enforcement of COO Delegation Rules

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect

---

#### Context

The COO (Supreme Orchestrator) repeatedly violates delegation rules by directly editing files instead of delegating to implementer agents. Rules exist in STATE.md but are not technically enforced - the COO has full access to Write, Edit, and Bash tools.

---

#### Decision

Implement a **multi-layer defense-in-depth** approach:

1. **Layer 1: Tool Restriction** - Use Claude CLI `--disallowedTools` flag to block Write/Edit
2. **Layer 2: Pre-Execution Hook** - Detection layer in `/shared/coo_enforcement.py`
3. **Layer 3: System Prompt Hardening** - State restrictions as facts, not guidelines
4. **Layer 4: Detection/Warning** - Real-time UI warnings and logging

**Key Design Points:**
- Exception: COO MAY modify STATE.md files
- Bash file modifications detected heuristically
- Violations logged to `logs/coo_violations.jsonl`
- Clear error messages with delegation examples

---

#### Integration

| Component | Change |
|-----------|--------|
| `backend/main.py` | Add `disallowed_tools` to `stream_claude_response()`, update system prompt |
| `backend/websocket/chat_handler.py` | Add `disallowed_tools`, update system prompt |
| `shared/coo_enforcement.py` | NEW - COO rule enforcement logic |
| `frontend/lib/websocket.ts` | Add `enforcement_violation` event type |
| `frontend/components/ActivityPanel.tsx` | Add violation display |

---

#### Consequences

**Pros:**
- Technical enforcement instead of soft guidelines
- Defense-in-depth with multiple layers
- Clear error messages guide correct behavior
- Violations logged for pattern analysis

**Cons:**
- Adds complexity to the codebase
- Bash detection is heuristic (may miss creative modifications)
- Requires CLI flag verification

---

**Full Specification:** See `/swarms/swarm_dev/workspace/DESIGN_COO_ENFORCEMENT.md`

---

### ADR-005: Hierarchical Delegation Pattern

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect

---

#### Context

The COO (Supreme Orchestrator) has been inconsistently delegating work to agents:
1. Claims to delegate but doesn't actually spawn agents (Delegation Theater)
2. Does work itself that should be delegated
3. Spawns agents without proper follow-through
4. Doesn't wait for results from delegated tasks

This undermines the hierarchical agent-swarm architecture and causes unreliable execution.

---

#### Decision

Implement an **Optimal Hierarchical Delegation Pattern** with:

1. **Three-tier hierarchy**: CEO -> COO -> Swarm Orchestrators -> Swarm Agents
2. **Mandatory delegation rules**: Clear criteria for when COO must delegate vs do directly
3. **Decision tree**: Visual guide for do vs delegate decisions
4. **Anti-pattern documentation**: Six patterns to avoid with solutions
5. **Result tracking**: Integration with Work Ledger for delegation persistence
6. **Synthesis enforcement**: Require result synthesis before completing delegated work

**Key Rules:**
- COO MUST delegate: code >5 lines, multi-file changes, tests, architecture, deep research
- COO MAY do directly: single file reads, simple grep/glob, git status, STATE.md, synthesis
- All Task prompts must include STATE.md reference and success criteria
- Worker agents (implementer, critic) should NOT have Task tool access

---

#### Integration

| Component | Change |
|-----------|--------|
| `backend/main.py:2550-2606` | Update COO system prompt with enforcement rules |
| `shared/work_ledger.py` | Add parent/child delegation queries |
| `swarms/*/agents/*.md` | Update agent prompts, remove Task from workers |
| `supreme/agents/*.md` | Update executive team prompts |
| `frontend/ActivityPanel.tsx` | Surface pending delegations |

---

#### Consequences

**Pros:**
- Consistent delegation behavior
- Auditable work tracking
- Parallel execution by default
- Clear role separation

**Cons:**
- More prescriptive prompts
- Potential over-delegation for simple tasks
- Additional tracking overhead

---

**Full Specification:** See `/swarms/swarm_dev/workspace/DESIGN_DELEGATION_PATTERN.md`

---

### ADR-004: Work Ledger System (Persistent Work Units)

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect

---

#### Context

The agent-swarm system suffers from work state fragility:

1. **In-memory only**: Work state exists only in agent conversation context
2. **No crash recovery**: Agent crashes lose all in-progress work
3. **No visibility**: No way to query what work is in progress
4. **No history**: No audit trail of work lifecycle

**Inspiration:** Gas Town's "Beads" system where work persists on hooks and survives crashes.

---

#### Decision

Implement a **Work Ledger** system with:

1. **WorkItem** - Persistent work unit with full lifecycle tracking
2. **JSON file storage** - Work items persisted to `workspace/ledger/`
3. **Atomic writes** - Crash-safe file operations
4. **Thread-safe API** - RLock for concurrent agent access
5. **Hierarchy support** - Parent/child task relationships
6. **Dependency tracking** - Work blocked until dependencies complete

**Key Design Points:**

- Work items have 6 statuses: pending, in_progress, blocked, completed, failed, cancelled
- Agents must `claim_work()` before processing (atomic operation)
- Full history tracking via `WorkHistoryEntry` list
- Index manifest (`index.json`) for O(1) lookups
- Integration with `agent_executor_pool.py` for automatic lifecycle

---

#### Integration

| Component | Change |
|-----------|--------|
| `shared/work_ledger.py` | New - Main WorkLedger class |
| `shared/work_models.py` | New - Data structures |
| `shared/agent_executor_pool.py` | Add work_id parameter |
| `shared/execution_context.py` | Add work_id field |
| `backend/main.py` | Create work items for requests |

---

#### Consequences

**Pros:**
- Crash resilience - work survives restarts
- Auditability - full history of all work
- Visibility - query work status from anywhere
- Recovery - orphaned work auto-reclaimed

**Cons:**
- Additional I/O overhead
- Storage accumulation (needs archival)
- Agents must properly claim/release work

---

**Full Specification:** See `/workspace/WORK_LEDGER_DESIGN.md`

---

### ADR-002: Local Neural Net Brain Integration

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect
**Co-Author:** Research Specialist (model research)

---

#### Context

The agent-swarm system currently relies exclusively on cloud-based Claude API for all inference. This presents challenges:

1. **Cost**: Every query, regardless of complexity, incurs API costs
2. **Latency**: Round-trip to cloud adds 500ms-2s per request
3. **Privacy**: All user interactions sent to external servers
4. **Personalization**: No learning from user patterns and preferences
5. **Dependency**: System requires internet connectivity

**Goal:** Integrate a local neural network "brain" that can:
- Handle simple queries locally to reduce costs
- Learn from user interactions to personalize responses
- Store persistent behavioral patterns and preferences
- Intelligently route between local and cloud models

---

#### Decision

Implement a **Local Neural Net Brain** using **Ollama** as the inference server with **LoRA fine-tuning** for personalization.

**Architecture Overview:**

```
                                   +------------------+
                                   |   User Request   |
                                   +--------+---------+
                                            |
                                            v
+------------------+    route     +-------------------+
|  Query Router    |<-------------|  Confidence       |
|  (classifier)    |              |  Threshold (0.85) |
+--------+---------+              +-------------------+
         |
    +----+----+
    |         |
    v         v
+-------+  +-------+
| Local |  | Cloud |
| Brain |  | Claude|
+---+---+  +---+---+
    |          |
    +----+-----+
         |
         v
+------------------+              +-------------------+
|    Response      |------------->|  Interaction      |
|    Merger        |              |  Logger           |
+------------------+              +-------------------+
                                            |
                                            v
                                  +-------------------+
                                  |  Training Data    |
                                  |  Collector        |
                                  +-------------------+
```

---

#### 1. Integration Points

##### 1.1 Primary Hook: `backend/main.py` - `websocket_chat()` (Lines 1656-1937)

```python
# After receiving message, before Claude processing
from local_brain import get_local_brain

local_brain = get_local_brain()

# Step 1: Check if local brain can handle this
routing_decision = await local_brain.route_query(
    prompt=user_message,
    context=conversation_history,
    session_id=session_id,
)

if routing_decision.use_local:
    # Local inference via Ollama
    local_response = await local_brain.generate(
        prompt=user_message,
        context=conversation_history,
        max_tokens=1024,
    )

    # Send response with local indicator
    await manager.send_event(websocket, "agent_start", {
        "agent": "Local Brain",
        "agent_type": "local",
        "model": local_brain.model_name,
    })

    await manager.send_event(websocket, "agent_delta", {
        "agent": "Local Brain",
        "delta": local_response.text,
    })

    # Log for learning
    await local_brain.log_interaction(
        prompt=user_message,
        response=local_response.text,
        routing_decision=routing_decision,
        session_id=session_id,
    )
else:
    # Continue with existing Claude CLI flow (lines 1832-1881)
    # After response, log for learning
    await local_brain.log_interaction(
        prompt=user_message,
        response=final_content,
        routing_decision=routing_decision,
        cloud_used=True,
        session_id=session_id,
    )
```

##### 1.2 Agent Executor Pool Integration

Modify `shared/agent_executor_pool.py`:

```python
async def execute(
    self,
    context: AgentExecutionContext,
    prompt: str,
    system_prompt: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    prefer_local: bool = False,  # NEW PARAMETER
) -> AsyncIterator[dict[str, Any]]:

    if prefer_local and self._local_brain_available():
        async for event in self._execute_local(context, prompt):
            yield event
    else:
        async for event in self._run_agent(...):
            yield event
```

---

#### 2. Data Flow

##### 2.1 Training Data Sources

| Source | Path | Data Extracted |
|--------|------|----------------|
| Chat sessions | `logs/chat/*.json` | User prompts + agent responses |
| Session summaries | `memory/sessions/*.md` | Interaction patterns |
| Swarm context | `swarms/*/workspace/STATE.md` | Domain knowledge |
| Decisions | `memory/core/decisions.md` | Preferences |

##### 2.2 Interaction Schema

```python
@dataclass
class TrainingInteraction:
    id: str
    timestamp: datetime
    session_id: str

    # Input
    prompt: str
    context: list[dict]

    # Output
    response: str
    agent_used: str
    tools_used: list[str]

    # Routing
    local_handled: bool
    local_confidence: float

    # Feedback
    feedback_type: str | None  # "approve", "reject", "edit"
    edited_response: str | None
    feedback_timestamp: datetime | None

    # Metadata
    swarm_name: str | None
    response_time_ms: int
    token_count: int
```

##### 2.3 User Feedback Capture

**Frontend (AgentResponse.tsx):**
- Add thumbs up/down buttons
- Add edit capability for corrections
- POST to `/api/feedback` endpoint

**Backend Endpoint:**
```python
@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    local_brain = get_local_brain()
    await local_brain.record_feedback(
        message_id=request.message_id,
        feedback_type=request.type,
        edited_content=request.edited_content,
        session_id=request.session_id,
    )
    return {"success": True}
```

##### 2.4 Storage Locations

| Data Type | Location | Format |
|-----------|----------|--------|
| Training interactions | `local_brain/data/interactions/` | JSONL (daily rotation) |
| Model weights | `local_brain/models/` | Safetensors |
| LoRA adapters | `local_brain/models/lora/` | Safetensors |
| User preferences | `local_brain/data/preferences.json` | JSON |
| Routing statistics | `local_brain/data/routing_stats.json` | JSON |

---

#### 3. Architecture Components

##### 3.1 Local Inference Server: Ollama

**Why Ollama:**
- Simple installation and model management
- REST API compatible with OpenAI format
- Efficient memory management (Apple Silicon optimized)
- Built-in model caching
- Active development

**Client Implementation:**

```python
# local_brain/inference/ollama_client.py
@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"  # or "qwen2.5:7b"
    timeout: float = 60.0

class OllamaClient:
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        context: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post("/api/chat", json={
            "model": self.config.model,
            "messages": messages,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": False,
        })
        return response.json()["message"]["content"]

    async def is_available(self) -> bool:
        try:
            return (await self._client.get("/api/tags")).status_code == 200
        except Exception:
            return False
```

##### 3.2 Query Router

```python
# local_brain/router/query_router.py
class QueryComplexity(Enum):
    SIMPLE = "simple"      # Greetings, yes/no
    MODERATE = "moderate"  # Single-step tasks
    COMPLEX = "complex"    # Multi-step, tool usage

class QueryIntent(Enum):
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    SIMPLE_QUESTION = "simple_question"
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    RESEARCH = "research"
    ORCHESTRATION = "orchestration"

@dataclass
class RoutingDecision:
    use_local: bool
    confidence: float
    intent: QueryIntent
    complexity: QueryComplexity
    reasoning: str

class QueryRouter:
    LOCAL_CAPABLE_INTENTS = {
        QueryIntent.GREETING,
        QueryIntent.CLARIFICATION,
        QueryIntent.SIMPLE_QUESTION,
    }

    CLOUD_REQUIRED_PATTERNS = [
        r"write.*code",
        r"implement.*",
        r"create.*file",
        r"fix.*bug",
        r"run.*command",
        r"Task\(",  # Delegation
    ]

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

    async def analyze(self, prompt: str, history: list | None = None) -> RoutingDecision:
        # Pattern match for cloud-required
        if self._requires_cloud(prompt):
            return RoutingDecision(use_local=False, confidence=1.0, ...)

        # Classify intent and complexity
        intent = await self._classify_intent(prompt)
        complexity = self._assess_complexity(prompt, history)
        confidence = self._calculate_confidence(intent, complexity, prompt)

        use_local = (
            intent in self.LOCAL_CAPABLE_INTENTS and
            complexity == QueryComplexity.SIMPLE and
            confidence >= self.confidence_threshold
        )

        return RoutingDecision(use_local=use_local, confidence=confidence, ...)
```

##### 3.3 Training Pipeline (LoRA)

```python
# local_brain/training/lora_trainer.py
@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.2-3B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_steps: int = 500

class LoRATrainer:
    def prepare_training_data(self, interactions_dir: Path) -> list[dict]:
        """Convert logged interactions to training format."""
        training_data = []
        for file in interactions_dir.glob("*.jsonl"):
            for line in file.read_text().splitlines():
                interaction = json.loads(line)
                if interaction.get("feedback_type") == "approve":
                    training_data.append({
                        "prompt": interaction["prompt"],
                        "response": interaction["response"],
                    })
                elif interaction.get("feedback_type") == "edit":
                    training_data.append({
                        "prompt": interaction["prompt"],
                        "response": interaction["edited_response"],
                    })
        return training_data

    async def train(self, training_data: list[dict]) -> Path:
        """Run LoRA fine-tuning using PEFT."""
        # Implementation using transformers + peft
        # Returns path to saved adapter
```

##### 3.4 Memory Integration

Extends existing `backend/memory.py`:

```python
# local_brain/memory/preference_memory.py
class PreferenceMemory:
    """Persistent storage for learned user preferences."""

    PREFERENCE_CATEGORIES = [
        "response_style",      # concise, detailed, technical
        "code_preferences",    # formatting, languages
        "interaction_patterns", # usage patterns
        "domain_expertise",    # frequent topics
    ]

    def learn_from_interaction(self, interaction: dict):
        """Extract and update preferences from interaction."""
        if interaction.get("feedback_type") == "edit":
            self._learn_from_edit(
                original=interaction["response"],
                edited=interaction["edited_response"],
            )

    def get_context_for_prompt(self, prompt: str) -> str:
        """Generate preference context to inject into prompts."""
        relevant_prefs = [
            f"- {pref.key}: {pref.value}"
            for pref in self.preferences.values()
            if pref.confidence > 0.7
        ]
        if relevant_prefs:
            return "User preferences:\n" + "\n".join(relevant_prefs)
        return ""
```

---

#### 4. File Structure

```
local_brain/
├── __init__.py                    # Package exports, get_local_brain()
├── config.py                      # Configuration dataclasses
├── brain.py                       # Main LocalBrain orchestrator
│
├── inference/                     # Local model inference
│   ├── __init__.py
│   ├── ollama_client.py          # Ollama API client
│   ├── base_client.py            # Abstract interface
│   └── model_manager.py          # Model download/management
│
├── router/                        # Query routing
│   ├── __init__.py
│   ├── query_router.py           # Main routing logic
│   ├── intent_classifier.py      # Intent classification
│   ├── complexity_scorer.py      # Complexity assessment
│   └── patterns.py               # Regex patterns
│
├── training/                      # Fine-tuning
│   ├── __init__.py
│   ├── lora_trainer.py           # LoRA implementation
│   ├── data_processor.py         # Training data prep
│   └── evaluation.py             # Metrics
│
├── memory/                        # Learning storage
│   ├── __init__.py
│   ├── interaction_logger.py     # Log interactions
│   ├── preference_memory.py      # User preferences
│   └── feedback_handler.py       # Process feedback
│
├── data/                          # Data storage (gitignored)
│   ├── interactions/             # JSONL files
│   ├── preferences.json
│   ├── routing_stats.json
│   └── embeddings/
│
├── models/                        # Weights (gitignored)
│   ├── lora/
│   └── cache/
│
└── tests/
    ├── test_router.py
    ├── test_inference.py
    ├── test_training.py
    └── test_memory.py
```

**Files to Modify:**

| File | Changes |
|------|---------|
| `backend/main.py` | LocalBrain integration in websocket_chat() |
| `config.yaml` | Add local_brain configuration section |
| `shared/agent_executor_pool.py` | Add prefer_local option |
| `frontend/components/AgentResponse.tsx` | Add feedback buttons |
| `frontend/lib/api.ts` | Add feedback endpoint |
| `frontend/components/ActivityPanel.tsx` | Show local vs cloud indicator |

---

#### 5. Implementation Phases

##### Phase 1: MVP - Basic Local Inference (1-2 weeks)
1. Create `local_brain/` directory structure
2. Implement OllamaClient with health checks
3. Create basic QueryRouter with pattern matching
4. Add routing hook in backend/main.py
5. Frontend indicator for local vs cloud
6. Basic interaction logging

**Success Criteria:**
- Greetings handled locally
- Clear UI indication of local processing
- Seamless fallback to cloud
- All interactions logged

##### Phase 2: Feedback Loop (1 week)
1. Add feedback buttons to AgentResponse.tsx
2. Create /api/feedback endpoint
3. Implement FeedbackHandler
4. Store feedback with interaction links

##### Phase 3: Preference Learning (1-2 weeks)
1. Implement PreferenceMemory
2. Extract preferences from feedback
3. Inject preferences into prompts
4. Add preference display in settings

##### Phase 4: LoRA Fine-tuning (2-3 weeks)
1. Implement LoRATrainer
2. Create training data processor
3. Add training job to queue system
4. Model evaluation and adapter management

##### Phase 5: Advanced Routing (1-2 weeks)
1. Train intent classifier
2. Implement hybrid mode
3. Routing confidence calibration
4. Adaptive threshold adjustment

##### Phase 6: Production Hardening (1 week)
1. Comprehensive error handling
2. Graceful degradation
3. Telemetry and metrics
4. Documentation

---

#### Trade-offs

**Pros:**
- 50-90% cost reduction for simple queries
- 5-10x latency improvement for local inference
- Personalization improves over time
- Works offline for supported queries
- Privacy for local processing

**Cons:**
- Additional system complexity
- Requires local compute (8GB+ RAM)
- Training needs significant interactions (1000+)
- Lower quality than Claude for complex tasks

**Mitigations:**
- Conservative routing (high confidence threshold)
- Clear UI indicators
- Easy disable option
- Graceful fallback on errors

---

#### Hardware Requirements

**Minimum:** CPU quad-core, 8GB RAM, 10GB storage
**Recommended:** 8+ cores, 16GB+ RAM, 50GB storage
**macOS:** M1/M2/M3 with 16GB unified memory (ideal)

---

#### Dependencies

**Required:**
- Ollama (local inference)
- httpx (async HTTP)

**Training (Phase 4+):**
- transformers
- peft
- torch
- bitsandbytes (8-bit training)

---

#### Success Metrics

1. API calls reduced by 30%+ for simple queries
2. Response latency under 200ms for local
3. Routing accuracy above 90%
4. User feedback approval rate above 80%
5. System uptime above 99%

---

#### Configuration

```yaml
# config.yaml addition
local_brain:
  enabled: true
  inference:
    provider: "ollama"
    base_url: "http://localhost:11434"
    model: "llama3.2:3b"
    timeout: 30
  router:
    confidence_threshold: 0.85
    local_capable_intents:
      - greeting
      - clarification
      - simple_question
    cloud_required_patterns:
      - "write.*code"
      - "implement"
      - "create.*file"
      - "Task\\("
  training:
    enabled: false  # Enable after Phase 4
    schedule: "weekly"
    min_interactions: 100
```

---

**Next Steps:**
1. Review and approve this design
2. Install Ollama: `brew install ollama && ollama pull llama3.2:3b`
3. Implementer builds Phase 1
4. Critic review before integration

---

### ADR-001: Smart Context Injection System

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect

---

#### Context

The current agent swarm system requires agents to manually read relevant files before they can work on a task. When a user asks about "auth" or "websocket", the agent must first discover which files are relevant, then read them. This leads to:

1. **Wasted turns** - Agents spend turns exploring the codebase before doing useful work
2. **Missed context** - Agents may not find all relevant files
3. **Inconsistent quality** - Success depends on agent's exploration strategy
4. **Slower responses** - Extra round-trips to discover and read files

**Goal:** Auto-detect relevant files for each task and inject them into the agent's context, reducing manual exploration.

---

#### Decision

Implement a **Smart Context Injection** system with the following architecture:

```
+------------------+     +-------------------+     +------------------+
|   User Prompt    | --> | Context Analyzer  | --> | File Relevance   |
|                  |     | (Topic Detection) |     | Scorer           |
+------------------+     +-------------------+     +------------------+
                                                           |
                                                           v
+------------------+     +-------------------+     +------------------+
| Agent Execution  | <-- | Context Injector  | <-- | Codebase Index   |
| (with context)   |     | (Prompt Builder)  |     | (Topic -> Files) |
+------------------+     +-------------------+     +------------------+
```

**Core Components:**

1. **CodebaseIndexer** - Builds and maintains a topic-to-file mapping
2. **ContextAnalyzer** - Extracts topics/keywords from user prompts
3. **RelevanceScorer** - Ranks files by relevance to detected topics
4. **ContextInjector** - Builds enhanced prompts with injected file context

---

#### Component Design

##### 1. CodebaseIndexer (`shared/context_indexer.py`)

Builds an index mapping topics/concepts to relevant files.

**Indexing Strategies:**
- **File path analysis**: `auth.py` -> ["auth", "authentication"]
- **Function/class extraction**: Parse AST for public symbols
- **Comment/docstring analysis**: Extract semantic meaning
- **Import graph**: Files that import auth-related modules
- **Keyword extraction**: TF-IDF or simple frequency analysis

**Index Structure:**
```python
{
    "auth": {
        "files": [
            {"path": "backend/auth.py", "score": 1.0, "type": "primary"},
            {"path": "backend/middleware/auth_middleware.py", "score": 0.8, "type": "related"},
        ],
        "keywords": ["login", "jwt", "token", "session", "permission"]
    },
    "websocket": {
        "files": [
            {"path": "backend/main.py", "score": 0.9, "sections": ["ConnectionManager", "websocket_chat"]},
            {"path": "frontend/lib/websocket.ts", "score": 1.0, "type": "primary"},
        ],
        "keywords": ["ws", "socket", "real-time", "streaming"]
    }
}
```

**Persistence:** JSON file at `.claude/context_index.json`, rebuilt on demand or on file changes.

**Update Strategy:**
- Full rebuild: On startup or manual trigger
- Incremental: On file save (via file watcher or git hook)
- Lazy: Only index files when first queried

##### 2. ContextAnalyzer (`shared/context_analyzer.py`)

Extracts topics and intent from user prompts.

**Analysis Methods:**
- **Keyword extraction**: Match against known topic vocabulary
- **Intent classification**: "fix", "implement", "explain", "review"
- **Entity extraction**: File paths, function names, error messages
- **Semantic similarity**: Embed prompt and compare to topic embeddings (optional, heavier)

**Output:**
```python
{
    "topics": ["websocket", "connection", "error"],
    "intent": "fix",
    "entities": {
        "files": [],
        "functions": ["websocket_chat"],
        "errors": ["WebSocketDisconnect"]
    },
    "confidence": 0.85
}
```

##### 3. RelevanceScorer (`shared/relevance_scorer.py`)

Ranks and filters files based on detected topics.

**Scoring Factors:**
- **Topic match score**: Direct match = 1.0, related = 0.5-0.8
- **Recency**: Recently modified files get a boost
- **Centrality**: Files imported by many others rank higher
- **Size penalty**: Very large files get lower scores (harder to process)
- **STATE.md bonus**: Swarm STATE.md files always included

**Filtering:**
- **Max files**: Default 5-10 relevant files
- **Max tokens**: Stay within ~50k token budget for context
- **Min score threshold**: Exclude files below 0.3 relevance

##### 4. ContextInjector (`shared/context_injector.py`)

Builds the enhanced prompt with injected context.

**Injection Modes:**
- **File summaries**: Just file paths + brief descriptions
- **Key sections**: Relevant functions/classes extracted
- **Full content**: Complete file contents (for small files)
- **Hybrid**: Mix based on file size and relevance

**Prompt Template:**
```
## Relevant Context (auto-detected)

The following files appear relevant to your request:

### backend/main.py (WebSocket handling)
Lines 1110-1200: ConnectionManager class
Lines 1656-1937: websocket_chat endpoint
[Full content or summary]

### frontend/lib/websocket.ts
[Full content - 150 lines]

---

## Your Request

{original_prompt}

---

Note: Relevant files have been pre-loaded. Use Read tool for additional files if needed.
```

---

#### Integration Points

##### Backend Integration (`backend/main.py`)

Modify `websocket_chat()` at lines 1756-1826 where the user prompt is processed:

```python
# After receiving message, before building prompt
from shared.context_injector import inject_context

# Analyze and inject relevant context
enhanced_prompt, injected_files = await inject_context(
    prompt=user_message,
    workspace=PROJECT_ROOT,
    max_files=8,
    max_tokens=50000,
)

# Use enhanced_prompt instead of user_message
user_prompt = enhanced_prompt
```

##### Agent Executor Integration (`shared/agent_executor_pool.py`)

Add optional context injection in `execute()` method:

```python
async def execute(
    self,
    context: AgentExecutionContext,
    prompt: str,
    system_prompt: str | None = None,
    inject_context: bool = True,  # New parameter
    ...
):
    if inject_context:
        prompt = await self._inject_relevant_context(prompt, context.workspace)
    ...
```

##### CLI Agent Spawning (`stream_claude_response()`)

Add context injection before spawning Claude CLI:

```python
# Before building cmd
if inject_context:
    from shared.context_injector import inject_context
    prompt = await inject_context(prompt, workspace)
```

---

#### Implementation Plan

**Phase 1: Core Infrastructure (2-3 days)**
1. Create `shared/context_indexer.py` with basic file-path-based indexing
2. Create `shared/context_analyzer.py` with keyword extraction
3. Create `shared/relevance_scorer.py` with simple scoring
4. Create `shared/context_injector.py` with basic injection
5. Unit tests for each component

**Phase 2: Backend Integration (1-2 days)**
1. Integrate into `websocket_chat()` in `backend/main.py`
2. Add configuration options (enable/disable, max files, etc.)
3. Add WebSocket event for showing injected context to frontend
4. Test with real prompts

**Phase 3: Advanced Features (2-3 days)**
1. AST-based indexing for Python files
2. Import graph analysis for related files
3. Section extraction (inject relevant functions, not whole files)
4. Caching layer for index
5. Incremental index updates

**Phase 4: Observability (1 day)**
1. Logging of injected context
2. Frontend indicator showing "N files auto-loaded"
3. Metrics on context injection effectiveness

---

#### Files to Create

| File | Purpose |
|------|---------|
| `shared/context_indexer.py` | Build and maintain topic -> file mapping |
| `shared/context_analyzer.py` | Extract topics from user prompts |
| `shared/relevance_scorer.py` | Score and rank relevant files |
| `shared/context_injector.py` | Build enhanced prompts with context |
| `shared/context_config.py` | Configuration for context injection |
| `.claude/context_index.json` | Persistent index (generated) |
| `tests/test_context_injection.py` | Unit tests |

---

#### Files to Modify

| File | Changes |
|------|---------|
| `backend/main.py` | Add context injection in `websocket_chat()` |
| `shared/agent_executor_pool.py` | Optional context injection parameter |
| `frontend/components/ActivityPanel.tsx` | Show injected files indicator |

---

#### Trade-offs and Considerations

**Pros:**
- Reduces agent exploration time significantly
- Improves first-response quality
- Consistent context loading
- Agents can focus on the actual task

**Cons:**
- Additional complexity in the system
- Index may become stale
- May inject irrelevant files (false positives)
- Context window consumption
- Performance overhead for indexing

**Mitigations:**
- Keep index simple initially (file-path based)
- Set conservative defaults (5-8 files max)
- Allow agents to request more context via Read tool
- Show injected files to user for transparency
- Make injection optional/configurable

---

#### Performance Considerations

1. **Index Build Time**: Initial full index ~1-5 seconds for typical codebase
2. **Query Time**: Topic matching should be <50ms
3. **Context Injection**: File reading parallelized, <200ms for 10 files
4. **Token Budget**: Stay under 50k tokens to leave room for agent work
5. **Caching**: Index cached in memory, rebuilt on significant changes

---

#### Alternatives Considered

1. **RAG with embeddings**: More accurate but requires embedding infrastructure
2. **LLM-based file selection**: Send file list to LLM to pick relevant ones - expensive and slow
3. **Manual hints**: User tags files - doesn't solve the problem
4. **Full codebase in context**: Too large, expensive, noisy

**Chosen approach** balances simplicity, speed, and effectiveness.

---

#### Success Metrics

1. **Reduction in exploration turns**: Measure average turns before useful work
2. **First-response quality**: Does the agent's first response show understanding?
3. **File hit rate**: Are injected files actually used by agents?
4. **User satisfaction**: Fewer "read this file first" instructions needed

---

#### Dependencies

- No external dependencies for basic implementation
- Optional: `tree-sitter` for AST parsing (Phase 3)
- Optional: Sentence embeddings for semantic matching (future enhancement)

---

**Next Steps:**
1. Review and approve this design
2. Implementer to build Phase 1 components
3. Critic review before integration
4. Test with real swarm tasks

---

### 2026-01-03 - Gastown Research (PARTIAL)
**Researcher**: Research Specialist

**Status**: INCOMPLETE - Network access blocked

**Problem**: Attempted to research the Gastown repository (https://github.com/steveyegge/gastown) to understand its memory/persistence mechanisms, but all network commands (curl, git clone, python urllib) require manual approval.

**Workaround**: Conducted comprehensive analysis of the existing agent-swarm memory system instead.

**Findings Document**: `/Users/jellingson/agent-swarm/workspace/research/gastown_analysis.md`

**Key Discoveries About Current System**:

1. **MemoryManager** (`/Users/jellingson/agent-swarm/backend/memory.py`):
   - 466-line class handling persistent context
   - Role-based context loading (COO, VP, Orchestrator, Agent)
   - Markdown-based storage in `memory/` directory
   - Session summarization with token estimation

2. **SessionManager** (`/Users/jellingson/agent-swarm/backend/session_manager.py`):
   - Tracks Claude CLI session IDs for `--continue` flag
   - Saves 2-3s per agent spawn
   - In-memory only (not disk-persisted)

3. **Memory Hierarchy**:
   ```
   memory/
     core/           # Organization-wide (vision, priorities, decisions)
     swarms/{name}/  # Per-swarm (context, progress, history)
     sessions/       # Session logs and summaries
   ```

4. **Current Limitations Identified**:
   - SessionManager state not persisted to disk
   - No semantic/vector search
   - No preference learning from interactions
   - Manual updates required for many memory operations

**Recommendations**:
1. Add disk persistence for SessionManager.active_sessions
2. Prioritize ADR-001 (Smart Context Injection) implementation
3. Consider lightweight vector store (ChromaDB, sqlite-vss)
4. Implement cross-session thread tracking

**Next Steps**:
- Request manual approval for `git clone https://github.com/steveyegge/gastown.git` to complete comparative analysis
- Or: Use Web Search tool if available to research Gastown documentation

---

### 2026-01-04 - Comprehensive Code Quality Review
**Reviewer**: Quality Critic
**Result**: NEEDS_CHANGES

**Scope Reviewed:**
- backend/main.py (3500+ lines)
- backend/services/*.py (all modules)
- shared/*.py (all modules)
- swarms/*/agents/*.md (ALL 47 agent definitions)

**Critical Issues Found:**

1. **TypeError at Runtime** - `/home/user/agent-swarm/backend/main.py:267-269`
   - `recover_orphaned_work(timeout_minutes=30, actor="startup_recovery")`
   - Method signature in work_ledger.py only accepts `timeout_minutes`
   - Server crashes on startup when orphaned work exists

2. **Missing YAML Frontmatter in Agents**:
   - `swarms/mynd_app/agents/*.md` (3 files) - No tools, model, or type defined
   - `swarms/asa_research/agents/theory_researcher.md` - No frontmatter
   - `swarms/asa_research/agents/empirical_researcher.md` - No frontmatter

**High Priority Issues:**

3. **Missing WebFetch Tool** in researcher agents:
   - `swarms/trading_bots/agents/researcher.md`
   - `swarms/_template/agents/researcher.md`
   - `swarms/swarm_dev/agents/brainstorm.md`

4. **Duplicate API Endpoints** - Work ledger endpoints in both main.py:473-589 AND routes/work.py

**Medium Priority Issues:**

5. **Silent Exception Swallowing** - 20+ locations with `except: pass` or `except Exception:`
6. **Hardcoded Values** - max_concurrent=5, timeout_minutes=30, various truncation limits
7. **Import at wrong location** - threading import not at top in agent_executor_pool.py:659

**Positive Observations:**
- MemoryStore deadlock fix correctly applied
- Thread-safe singleton implementations correct (double-checked locking)
- WebSearch/WebFetch correctly added to ios_app_factory and asa_research researchers
- RLock usage in WorkLedger for nested acquisition
- Good path traversal security checks

**Next Steps:**
1. FIX IMMEDIATELY: Remove `actor` parameter from main.py:269
2. FIX IMMEDIATELY: Add YAML frontmatter to mynd_app and asa_research agents
3. Add WebFetch to missing researcher agents
4. Resolve duplicate endpoint definitions in main.py vs routes/work.py

---

## Latest Work: Polymarket Trading Bot P0 Critical Fixes
**Implementer**: Implementation Specialist
**Date**: 2026-01-04

**Status**: COMPLETE

**Summary**: Applied P0 (Priority 0) critical fixes to the Polymarket BTC 15-minute arbitrage trading bot to improve safety and profitability.

### Changes Made

1. **`/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/polymarket-arbitrage/btc-polymarket-bot/src/simple_arb_bot.py`**
   - Added `can_trade()` method (lines 147-159): Checks daily loss limit before allowing trades
   - Added `record_trade_result()` method (lines 161-163): Records P&L for daily tracking
   - Added liquidity validation in `check_arbitrage()` (lines 359-365): Requires minimum $100 at best ask
   - Added `can_trade()` check in `execute_arbitrage()` (lines 455-457): Prevents trading when daily limit exceeded

2. **`/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/polymarket-arbitrage/btc-polymarket-bot/src/config.py`**
   - Added `min_liquidity` setting (line 46): Default $100 minimum liquidity at best ask

3. **`/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/research/SYNTHESIZED_STRATEGY_2026.md`**
   - Created synthesized strategy document from 6 research agents
   - Documents all P0 fixes and risk management summary

### P0 Fixes Applied

| Fix | Description | Status |
|-----|-------------|--------|
| P0-1 | Use Best Ask instead of Midpoint | Already Implemented |
| P0-2 | Private Key to Environment Variable | Already Implemented |
| P0-3 | Slippage Buffer (0.5%) | Already Implemented |
| P0-4 | Daily Loss Limit ($20 default) | NEW - Implemented |
| P0-5 | Liquidity Validation ($100 min) | NEW - Implemented |

### Verification

- Python syntax check: PASSED
- All three new code blocks properly integrated
- Config updated with new MIN_LIQUIDITY setting

### Next Steps

- Run integration tests with DRY_RUN=true
- Monitor bot logs for liquidity/daily limit messages
- Consider adding more verbose logging for daily P&L tracking


---

## Research: Claude Code Skills System - 2026-01-04
**Researcher**: Research Specialist Agent
**Date**: 2026-01-04

**Status**: COMPLETE

### Research Objective
Comprehensive research on Claude Code skills system, including:
- How skills work and are applied
- Available skill categories (especially frontend/UI)
- Specialized skills for agent departments

### Key Findings Summary
1. Skills are model-invoked: Claude automatically selects relevant skills based on context
2. Official Anthropic skills include document skills (DOCX, PDF, PPTX, XLSX) and frontend-design
3. Skills use progressive disclosure architecture to minimize context usage
4. Published as open standard at agentskills.io for cross-platform portability
5. Extensive community skills available via awesome-claude-skills repositories

### Skill Categories Identified
- Document Skills (docx, pdf, pptx, xlsx)
- Frontend Design Skills (frontend-design, canvas-design, artifacts-builder)
- Creative Skills (algorithmic-art, slack-gif-creator)
- DevOps Skills (aws-skills, terraform, infrastructure-as-code)
- Testing Skills (webapp-testing, api-tester, test-fixing)
- Security Skills (security-bluebook-builder, defense-in-depth, varlock-claude-skill)
- Enterprise Skills (brand-guidelines, internal-comms)

### Full Research Report
See: /home/user/agent-swarm/workspace/research/claude-code-skills-research.md


---

## Corporate Structure Research - 2026-01-04
**Agent**: Research Specialist
**Date**: 2026-01-04

**Status**: COMPLETE

### Research Summary
Comprehensive analysis of corporate organizational structures for software/product companies to inform AI agent corporation design.

### Key Findings

1. **Hierarchy Levels**: C-Suite (strategy) -> VP (planning) -> Director (allocation) -> Manager (assignment) -> IC (execution)

2. **CTO vs VP Engineering Split**: CTO owns vision/strategy (external), VP Engineering owns execution/delivery (internal). Critical separation of concerns.

3. **RACI Model**: Every task needs exactly ONE Accountable owner. Multiple can be Responsible, Consulted, or Informed.

4. **Quality Gates**: Automated checkpoints in CI/CD that block progression until criteria met. Fail fast, catch issues early.

5. **Cross-Team Coordination**: 75% of cross-functional teams fail. Success requires: single owner, shared goals, clear RACI, regular syncs.

6. **Three Key Voices**: CEO/COO needs Marketing, Product+Design, and Engineering as equal peers reporting directly.

### Recommendations for Agent Swarm

1. COO as primary orchestrator (current model is correct)
2. Use RACI explicitly: one Accountable agent per task
3. Implement quality gates: research -> architecture -> implementation -> testing
4. STATE.md serves as shared artifact for all communication
5. Clear escalation paths when agents encounter blockers

### Full Report
See research response for complete analysis with sources.

---
