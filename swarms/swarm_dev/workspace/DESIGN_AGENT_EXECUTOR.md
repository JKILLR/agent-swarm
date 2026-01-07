# Agent Executor Design Document

## Phase 0.1: Claude Agent SDK Integration

**Author**: Swarm Dev Architect
**Date**: 2026-01-02
**Status**: PROPOSED

---

## 1. Problem Statement

The agent-swarm system currently has partial agent execution capability. The COO (Supreme Orchestrator) can execute via Claude CLI, but individual swarm agents spawned via the Task tool do not have isolated execution environments. This document designs a complete agent execution layer that enables:

1. Full tool access (Read, Write, Edit, Bash, Glob, Grep, etc.)
2. Workspace isolation per agent/swarm
3. Streaming results back to the UI via WebSocket
4. Security sandboxing

---

## 2. Architecture Overview

### 2.1 Current State

```
+-------------+     +-----------------+     +------------+
|  Frontend   |<--->|  WebSocket      |<--->| Claude CLI |
|  (Next.js)  |     |  /ws/chat       |     | (COO only) |
+-------------+     +-----------------+     +------------+
                           ^
                           | JSON stream
                           v
                    +---------------+
                    | main.py       |
                    | (stream_      |
                    | claude_       |
                    | response)     |
                    +---------------+
```

**Limitations**:
- Only COO uses Claude CLI directly
- Task tool spawns subagents but they share context
- No true isolation between swarm workspaces
- agent_executor.py exists but is not fully wired into Task tool execution

### 2.2 Target State

```
+-------------+     +------------------+     +------------------+
|  Frontend   |<--->| WebSocket        |<--->| AgentExecutor    |
|  (Next.js)  |     | /ws/chat         |     | Pool             |
+-------------+     | /ws/jobs         |     +--------+---------+
                    +--------+---------+              |
                             |                        v
                    +--------v---------+     +--------+---------+
                    | Connection       |     | WorkspaceManager |
                    | Manager          |     +--------+---------+
                    +------------------+              |
                                                      v
                    +--------------------------------------------------+
                    |                  Execution Layer                  |
                    |  +-------------+  +-------------+  +-------------+|
                    |  | Agent 1     |  | Agent 2     |  | Agent 3     ||
                    |  | (workspace  |  | (workspace  |  | (workspace  ||
                    |  |  /swarm_a/) |  |  /swarm_a/) |  |  /swarm_b/) ||
                    |  | Claude CLI  |  | Claude CLI  |  | Claude CLI  ||
                    |  +-------------+  +-------------+  +-------------+|
                    +--------------------------------------------------+
```

---

## 3. Key Components

### 3.1 AgentExecutor (Enhanced)

**Location**: `/Users/jellingson/agent-swarm/shared/agent_executor.py`

**Current State**: Basic execution via Claude CLI or Anthropic API with streaming.

**Enhancements Required**:

1. **Workspace Isolation**: Each agent process runs in an isolated working directory
2. **Environment Propagation**: Pass git credentials, API tokens to subprocesses
3. **Permission Modes**: Support different tool permission levels per agent type
4. **Process Management**: Track and manage multiple concurrent agent processes
5. **Event Normalization**: Unified event format across CLI and API backends

```python
@dataclass
class AgentExecutionContext:
    """Context for agent execution."""

    agent_name: str
    agent_type: str  # orchestrator, implementer, critic, etc.
    swarm_name: str
    workspace: Path

    # Permissions
    allowed_tools: list[str]
    permission_mode: str  # "acceptEdits", "default", "readonly"

    # Credentials (passed as env vars)
    git_credentials: bool = False
    web_access: bool = False

    # Limits
    max_turns: int = 25
    timeout: float = 600.0

    # Tracking
    job_id: str | None = None
    parent_agent: str | None = None  # For tracing task delegation
```

### 3.2 WorkspaceManager (New)

**Location**: `/Users/jellingson/agent-swarm/shared/workspace_manager.py`

**Purpose**: Manages workspace isolation and security boundaries.

```python
class WorkspaceManager:
    """Manages workspace isolation for agents."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.swarms_dir = project_root / "swarms"

    def get_workspace(self, swarm_name: str) -> Path:
        """Get the workspace path for a swarm."""
        swarm_path = self.swarms_dir / swarm_name
        workspace = swarm_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def validate_path_access(
        self,
        path: Path,
        workspace: Path,
        allow_project_root: bool = False,
    ) -> bool:
        """Check if a path is within allowed boundaries."""
        resolved = path.resolve()

        # Always allow workspace access
        if resolved.is_relative_to(workspace.resolve()):
            return True

        # Some agents (swarm_dev) can access project root
        if allow_project_root:
            return resolved.is_relative_to(self.project_root.resolve())

        return False

    def get_agent_permissions(
        self,
        agent_type: str,
        swarm_name: str,
    ) -> dict:
        """Get permissions based on agent type and swarm."""
        # Swarm Dev gets special permissions
        if swarm_name == "swarm_dev":
            return {
                "allow_project_root": True,
                "git_access": True,
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
            }

        # Permission levels by agent type
        permissions_by_type = {
            "orchestrator": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
            },
            "implementer": {
                "allow_project_root": False,
                "git_access": True,  # Can commit in their workspace
                "bash_allowed": True,
                "permission_mode": "acceptEdits",
            },
            "critic": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": False,  # Read-only
                "permission_mode": "default",
            },
            "researcher": {
                "allow_project_root": False,
                "git_access": False,
                "bash_allowed": True,
                "permission_mode": "default",
            },
        }

        return permissions_by_type.get(agent_type, {
            "allow_project_root": False,
            "git_access": False,
            "bash_allowed": True,
            "permission_mode": "default",
        })
```

### 3.3 AgentExecutorPool (New)

**Location**: `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py`

**Purpose**: Manages concurrent agent execution with resource limits.

```python
class AgentExecutorPool:
    """Pool for managing concurrent agent executions."""

    def __init__(
        self,
        max_concurrent: int = 5,
        workspace_manager: WorkspaceManager = None,
    ):
        self.max_concurrent = max_concurrent
        self.workspace_manager = workspace_manager
        self._running: dict[str, asyncio.Task] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        context: AgentExecutionContext,
        prompt: str,
        system_prompt: str | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute an agent with the given context."""
        async with self._semaphore:
            execution_id = str(uuid.uuid4())

            try:
                async for event in self._run_agent(
                    execution_id, context, prompt, system_prompt
                ):
                    if on_event:
                        on_event(event)
                    yield event
            finally:
                self._cleanup(execution_id)

    async def _run_agent(
        self,
        execution_id: str,
        context: AgentExecutionContext,
        prompt: str,
        system_prompt: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a single agent execution."""
        # Build command
        cmd = self._build_command(context, prompt, system_prompt)

        # Build environment
        env = self._build_environment(context)

        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(context.workspace),
            env=env,
        )

        self._processes[execution_id] = process

        # Stream output
        async for event in self._parse_stream(process, context):
            yield event

        await process.wait()

    def _build_command(
        self,
        context: AgentExecutionContext,
        prompt: str,
        system_prompt: str | None,
    ) -> list[str]:
        """Build the Claude CLI command."""
        cmd = [
            "claude",
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", context.permission_mode,
        ]

        # Add system prompt
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Add prompt
        cmd.append(prompt)

        return cmd

    def _build_environment(
        self,
        context: AgentExecutionContext,
    ) -> dict[str, str]:
        """Build environment variables for the agent."""
        env = os.environ.copy()

        # Remove API key to force CLI auth
        env.pop("ANTHROPIC_API_KEY", None)

        # Add git credentials if allowed
        if context.git_credentials:
            # CLAUDE_CODE_OAUTH_TOKEN enables git operations
            oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            if oauth_token:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token

        # Add workspace info for agent context
        env["AGENT_WORKSPACE"] = str(context.workspace)
        env["AGENT_NAME"] = context.agent_name
        env["AGENT_SWARM"] = context.swarm_name

        return env

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running agent execution."""
        if execution_id in self._processes:
            process = self._processes[execution_id]
            process.kill()
            await process.wait()
            return True
        return False

    def _cleanup(self, execution_id: str):
        """Clean up after execution."""
        self._processes.pop(execution_id, None)
        self._running.pop(execution_id, None)
```

### 3.4 Task Tool Integration

**Location**: `/Users/jellingson/agent-swarm/backend/main.py` (modifications)

The Task tool is built into Claude CLI. When the COO uses Task, we need to:

1. Intercept the tool call
2. Resolve the agent context (swarm, workspace, permissions)
3. Execute via AgentExecutorPool
4. Stream results back

**Key Insight**: Claude CLI's Task tool spawns subprocesses internally. We need to ensure:
- The subprocesses inherit correct working directories
- Credentials are propagated
- Results are streamed back to the parent

---

## 4. Data Flow

### 4.1 Direct Agent Execution

```
User Message
    |
    v
WebSocket /ws/chat
    |
    v
COO System Prompt + Context
    |
    v
stream_claude_response()
    |
    v
Claude CLI Process (PROJECT_ROOT)
    |
    +---> stdout (stream-json)
    |         |
    |         v
    |     parse_claude_stream()
    |         |
    |         v
    |     WebSocket Events (agent_delta, tool_start, etc.)
    |
    +---> Task tool call detected
              |
              v
          [Current: handled internally by CLI]
          [Target: intercept and route via pool]
```

### 4.2 Task Delegation Flow

```
COO executes Task tool
    |
    v
AgentExecutorPool.execute()
    |
    v
Resolve AgentExecutionContext
    - Get workspace from swarm config
    - Get permissions from agent type
    - Get system prompt from agent markdown
    |
    v
Start Claude CLI subprocess
    - CWD = swarm workspace
    - ENV = credentials + agent info
    - Permission mode per agent type
    |
    v
Stream events back to parent
    - tool_start: "Agent X starting..."
    - agent_delta: streamed content
    - tool_complete: "Agent X completed"
    |
    v
Results bubble up to COO
    |
    v
COO synthesizes and responds
    |
    v
WebSocket to Frontend
```

---

## 5. WebSocket Event Protocol

### 5.1 Existing Events (Unchanged)

| Event Type | Purpose | Payload |
|------------|---------|---------|
| `chat_start` | User message received | `{ message }` |
| `agent_start` | Agent begins processing | `{ agent, agent_type }` |
| `agent_delta` | Streaming text chunk | `{ agent, delta }` |
| `agent_complete` | Agent finished | `{ agent, content, thinking }` |
| `chat_complete` | Full conversation done | `{ success }` |
| `thinking_start` | Extended thinking begins | `{ agent }` |
| `thinking_delta` | Thinking chunk | `{ agent, delta }` |
| `thinking_complete` | Thinking done | `{ agent, thinking }` |
| `tool_start` | Tool execution begins | `{ tool, description, input }` |
| `tool_complete` | Tool execution done | `{ tool, success, summary }` |
| `agent_spawn` | Subagent spawned | `{ agent, description }` |
| `error` | Error occurred | `{ message }` |

### 5.2 New Events (Proposed)

| Event Type | Purpose | Payload |
|------------|---------|---------|
| `agent_execution_start` | Pool begins agent | `{ execution_id, agent, swarm, workspace }` |
| `agent_execution_progress` | Progress update | `{ execution_id, agent, progress, activity }` |
| `agent_execution_complete` | Pool finishes agent | `{ execution_id, agent, success, result_summary }` |
| `workspace_write` | File written in workspace | `{ agent, path, action }` |

---

## 6. Security Model

### 6.1 Workspace Boundaries

```
/Users/jellingson/agent-swarm/              <- PROJECT_ROOT
    |
    +-- swarms/
    |       |
    |       +-- swarm_dev/
    |       |       +-- workspace -> ../..   <- Points to PROJECT_ROOT (special)
    |       |       +-- agents/
    |       |
    |       +-- asa_research/
    |       |       +-- workspace/           <- Isolated
    |       |       +-- agents/
    |       |
    |       +-- mynd_app/
    |               +-- workspace/           <- Isolated
    |               +-- agents/
    |
    +-- backend/                             <- Protected from normal swarms
    +-- frontend/                            <- Protected from normal swarms
    +-- shared/                              <- Protected from normal swarms
```

### 6.2 Permission Modes by Agent Type

| Agent Type | Read Files | Write Files | Execute Bash | Git Access | Web Access |
|------------|------------|-------------|--------------|------------|------------|
| orchestrator | workspace | workspace | yes | no | yes |
| implementer | workspace | workspace | yes | workspace | yes |
| researcher | workspace | workspace | yes | no | yes |
| critic | workspace | no | no | no | yes |
| reviewer | workspace | no | limited | no | yes |
| monitor | logs only | no | limited | no | no |

### 6.3 Swarm Dev Special Permissions

The `swarm_dev` swarm is special - its workspace points to PROJECT_ROOT:
- Can read/write any file in the project
- Can run git commands on the main repo
- Can execute tests and linters
- Required for self-modification capability

**Risk Mitigation**:
- All changes must go through critic review
- Commits go to feature branches, not main
- Audit logging of all modifications

---

## 7. Implementation Plan

### Phase 0.1.1: Core Infrastructure (2-3 days)

1. **Create WorkspaceManager** (`/Users/jellingson/agent-swarm/shared/workspace_manager.py`)
   - Workspace resolution
   - Path validation
   - Permission lookup

2. **Create AgentExecutorPool** (`/Users/jellingson/agent-swarm/shared/agent_executor_pool.py`)
   - Concurrent execution management
   - Process lifecycle
   - Event streaming

3. **Enhance AgentExecutor** (`/Users/jellingson/agent-swarm/shared/agent_executor.py`)
   - Add AgentExecutionContext support
   - Environment propagation
   - Improved error handling

### Phase 0.1.2: Backend Integration (2-3 days)

4. **Update main.py** (`/Users/jellingson/agent-swarm/backend/main.py`)
   - Integrate AgentExecutorPool
   - Wire workspace resolution for swarm agents
   - Add new WebSocket events

5. **Update jobs.py** (`/Users/jellingson/agent-swarm/backend/jobs.py`)
   - Use AgentExecutorPool for job execution
   - Track agent hierarchies (parent/child)

### Phase 0.1.3: Testing & Validation (1-2 days)

6. **Create Integration Tests**
   - Test workspace isolation
   - Test agent tool access
   - Test streaming to WebSocket
   - Test parallel execution

7. **End-to-End Validation**
   - Swarm Dev agent reads a file
   - Swarm Dev agent writes a file
   - Swarm Dev agent runs git status
   - Result streams to UI correctly

---

## 8. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `shared/workspace_manager.py` | Workspace isolation and permissions |
| `shared/agent_executor_pool.py` | Concurrent execution pool |
| `shared/execution_context.py` | AgentExecutionContext dataclass |

### Modified Files

| File | Changes |
|------|---------|
| `shared/agent_executor.py` | Add context-aware execution |
| `shared/agent_base.py` | Wire to new executor |
| `backend/main.py` | Integrate pool, new events |
| `backend/jobs.py` | Use pool for job execution |
| `frontend/lib/websocket.ts` | Add new event types |

---

## 9. Dependencies

### External
- Claude CLI installed and authenticated (Max subscription)
- Git configured with SSH keys or credential helper

### Internal
- Existing WebSocket infrastructure (already working)
- Existing agent_executor.py (will be enhanced)
- Existing swarm configuration system

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Claude CLI subprocess doesn't inherit auth | High | Ensure CLAUDE_CODE_OAUTH_TOKEN is passed in env |
| Task tool spawns outside pool control | Medium | May need to use --no-task-tool and implement our own |
| File operations escape workspace | High | Path validation in WorkspaceManager |
| Too many concurrent agents | Medium | Semaphore in AgentExecutorPool |
| Long-running agents block pool | Medium | Per-agent timeouts, cancellation support |

---

## 11. Success Criteria

Phase 0.1 is complete when:

- [ ] Agent can read a file in its workspace
- [ ] Agent can write a file in its workspace
- [ ] Agent can run bash commands
- [ ] Results stream to UI in real-time
- [ ] Swarm Dev agent can read/modify project files
- [ ] Multiple agents can run concurrently
- [ ] Agent errors are reported clearly

---

## 12. Open Questions

1. **Task Tool Interception**: Claude CLI's Task tool spawns internal subprocesses. Should we:
   - Let CLI handle it (simpler, less control)
   - Disable Task and implement our own delegation (more control, more work)

   **Recommendation**: Start with CLI handling, add interception later if needed.

2. **Git Authentication**: How should we handle git push/pull?
   - Option A: Use existing CLAUDE_CODE_OAUTH_TOKEN
   - Option B: Configure SSH keys in agent environment

   **Recommendation**: Use CLAUDE_CODE_OAUTH_TOKEN for consistency with CLI auth.

3. **Workspace Persistence**: Should agent workspace changes persist across sessions?
   - Yes (current approach) - easier for continuity
   - No (ephemeral) - cleaner, but loses context

   **Recommendation**: Persist, with STATE.md as the main context carrier.

---

## 13. Appendix: ASCII Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              Agent Swarm System                                    |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  +-------------+                                                                   |
|  |   Frontend  |<---.                                                              |
|  |  (Next.js)  |    |                                                              |
|  +-------------+    |                                                              |
|                     |                                                              |
|                     v                                                              |
|  +------------------+----------------+                                             |
|  |              Backend              |                                             |
|  |            (FastAPI)              |                                             |
|  |                                   |                                             |
|  |  +-------------+  +------------+  |                                             |
|  |  | /ws/chat    |  | /ws/jobs   |  |                                             |
|  |  +------+------+  +------+-----+  |                                             |
|  |         |                |        |                                             |
|  |         v                v        |                                             |
|  |  +-------------+  +------------+  |                                             |
|  |  | Connection  |  | Job        |  |                                             |
|  |  | Manager     |  | Manager    |  |                                             |
|  |  +------+------+  +------+-----+  |                                             |
|  |         |                |        |                                             |
|  +---------+----------------+--------+                                             |
|            |                |                                                      |
|            v                v                                                      |
|  +-----------------------------------+                                             |
|  |        AgentExecutorPool          |                                             |
|  |                                   |                                             |
|  |  - Max concurrent: 5              |                                             |
|  |  - Process tracking               |                                             |
|  |  - Event streaming                |                                             |
|  +----------------+------------------+                                             |
|                   |                                                                |
|                   v                                                                |
|  +-----------------------------------+                                             |
|  |        WorkspaceManager           |                                             |
|  |                                   |                                             |
|  |  - Path validation                |                                             |
|  |  - Permission lookup              |                                             |
|  |  - Workspace resolution           |                                             |
|  +----------------+------------------+                                             |
|                   |                                                                |
|                   v                                                                |
|  +-----------------------------------------------------------------------+         |
|  |                        Execution Layer                                 |         |
|  |  +-----------------+  +-----------------+  +-----------------+         |         |
|  |  | Claude CLI #1   |  | Claude CLI #2   |  | Claude CLI #3   |         |         |
|  |  | Agent: COO      |  | Agent: impl1    |  | Agent: critic   |         |         |
|  |  | CWD: /project   |  | CWD: /swarm/ws  |  | CWD: /swarm/ws  |         |         |
|  |  +-----------------+  +-----------------+  +-----------------+         |         |
|  +-----------------------------------------------------------------------+         |
|                                                                                    |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                              File System                                           |
|                                                                                    |
|  /Users/jellingson/agent-swarm/                                                   |
|  +-- swarms/                                                                      |
|  |   +-- swarm_dev/workspace -> ../.. (PROJECT_ROOT)                             |
|  |   +-- asa_research/workspace/                                                  |
|  |   +-- mynd_app/workspace/                                                      |
|  +-- backend/                                                                     |
|  +-- frontend/                                                                    |
|  +-- shared/                                                                      |
|  +-- logs/                                                                        |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-02
**Next Review**: After Phase 0.1.1 implementation
