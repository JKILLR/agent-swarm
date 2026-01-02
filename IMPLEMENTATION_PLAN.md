# Agent Swarm Architecture Improvements - Implementation Plan

**Created:** January 2, 2026
**Updated:** January 2, 2026 (incorporated reviewer feedback)
**Based on:** ARCHITECTURE_REVIEW.md findings
**Goal:** Transform the system into a "well-oiled machine" with 3-5x speed improvement

---

## Executive Summary

This plan addresses the critical bottlenecks identified in the architecture review:

| Issue | Solution | Expected Impact |
|-------|----------|-----------------|
| Sequential tool execution | `asyncio.gather()` parallel execution | 3-5x speedup |
| CLI subprocess overhead | Session continuity (`--continue` flag) | 2-3s saved per agent |
| Context pollution | Compact COO prompt + simple truncation | Cleaner context |
| No coordination | Hooks system + shared memory | Better cohesion |
| Missing verification | PreToolUse/PostToolUse hooks | Self-correcting agents |

---

## Phase 1: Parallel Execution (Day 1) - CRITICAL

### 1.1 Parallel Tool Execution in Agentic Loop

**File:** `backend/main.py`

**Current Problem:** Tools execute sequentially in `run_agentic_chat()`.

**Change:** Modify the tool execution loop to use `asyncio.gather()`:

```python
# Location: run_agentic_chat() tool execution section

# BEFORE: Sequential
for tool_use in tool_uses:
    result = await tool_executor.execute(tool_use.name, tool_use.input)
    tool_results.append(...)

# AFTER: Parallel
if tool_uses:
    results = await asyncio.gather(*[
        tool_executor.execute(tu.name, tu.input, workspace)
        for tu in tool_uses
    ], return_exceptions=True)

    tool_results = []
    for tu, result in zip(tool_uses, results):
        if isinstance(result, Exception):
            result = f"Error: {result}"
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": str(result)
        })
```

**Testing:**
- Send a message that triggers multiple Task tool calls
- Verify agents spawn concurrently (check timestamps in logs)

### 1.2 Parallel Subprocess Spawning in Task Tool (CRITICAL FIX)

**File:** `backend/tools.py`

**Current Problem:** Each `Task` tool spawns Claude subprocess sequentially, even with `asyncio.gather()` at the tool level.

**Change:** Ensure the internal subprocess creation is truly parallel:

```python
async def _execute_parallel_tasks(self, input: dict[str, Any]) -> str:
    """Execute multiple tasks with TRUE parallel subprocess spawning."""
    tasks = input.get("tasks", [])

    if not tasks:
        return "No tasks provided"

    # Create subprocess coroutines (don't await yet!)
    coroutines = [
        self._spawn_agent_subprocess(t.get("agent"), t.get("prompt"))
        for t in tasks
    ]

    # NOW they run truly in parallel
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Format results
    output_lines = ["## Parallel Execution Results\n"]
    for task, result in zip(tasks, results):
        agent = task.get("agent", "unknown")
        if isinstance(result, Exception):
            output_lines.append(f"### {agent}\n**Error:** {result}\n")
        else:
            # Truncate long results
            result_str = str(result)
            if len(result_str) > 2000:
                result_str = self._truncate_result(result_str, 2000)
            output_lines.append(f"### {agent}\n{result_str}\n")

    return "\n".join(output_lines)

async def _spawn_agent_subprocess(self, agent: str, prompt: str) -> str:
    """Spawn a Claude subprocess for the agent. This is the unit of parallelism."""
    # Build command
    cmd = [
        "claude",
        "-p",
        "--output-format", "stream-json",
        "--permission-mode", "default",
        prompt,
    ]

    # ... subprocess creation and handling ...
    # Returns the result string
```

### 1.3 Update _execute_task to Support Concurrent Spawning

**File:** `backend/tools.py`

Ensure `_execute_task` creates the subprocess in a way that allows concurrent execution:

```python
async def _execute_task(self, input: dict[str, Any]) -> str:
    """Execute a single Task - must be async-safe for concurrent use."""
    agent = input.get("agent", "")
    prompt = input.get("prompt", "")

    # Don't block - create subprocess asynchronously
    result = await self._spawn_agent_subprocess(agent, prompt)

    # Truncate if needed (no LLM call - too slow)
    if len(result) > 2000:
        result = self._truncate_result(result, 2000)

    return result
```

---

## Phase 2: Session Continuity (Day 1-2) - HIGH PRIORITY

### 2.1 Session Manager

**File:** `backend/session_manager.py` (NEW)

```python
"""Session manager for persistent Claude sessions."""

import asyncio
import os
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ClaudeSession:
    session_id: str
    chat_id: str
    created_at: datetime
    last_used: datetime

# Singleton instance
_session_manager: Optional["SessionManager"] = None

class SessionManager:
    """Maintain persistent Claude sessions per chat."""

    def __init__(self):
        self.active_sessions: Dict[str, ClaudeSession] = {}
        self._lock = asyncio.Lock()

    async def get_session(self, chat_id: str) -> Optional[str]:
        """Get existing session ID for a chat."""
        async with self._lock:
            session = self.active_sessions.get(chat_id)
            if session:
                session.last_used = datetime.now()
                return session.session_id
            return None

    async def register_session(self, chat_id: str, session_id: str):
        """Register a new session from Claude output."""
        async with self._lock:
            self.active_sessions[chat_id] = ClaudeSession(
                session_id=session_id,
                chat_id=chat_id,
                created_at=datetime.now(),
                last_used=datetime.now(),
            )

    async def end_session(self, chat_id: str):
        """End a session."""
        async with self._lock:
            self.active_sessions.pop(chat_id, None)

    def get_continue_flags(self, chat_id: str) -> list[str]:
        """Get --continue flags if session exists (sync for command building)."""
        session = self.active_sessions.get(chat_id)
        if session:
            return ["--continue", session.session_id]
        return []

def get_session_manager() -> SessionManager:
    """Get the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
```

### 2.2 Integrate Session Manager with stream_claude_response()

**File:** `backend/main.py`

**CRITICAL:** This is the missing integration code:

```python
from session_manager import get_session_manager

async def stream_claude_response(
    prompt: str,
    chat_id: str,  # ADD THIS PARAMETER
    workspace: Path | None = None,
) -> asyncio.subprocess.Process:
    """Stream response from Claude CLI with session continuity."""

    session_mgr = get_session_manager()

    cmd = [
        "claude",
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "acceptEdits",
    ]

    # ADD SESSION CONTINUITY
    continue_flags = session_mgr.get_continue_flags(chat_id)
    if continue_flags:
        cmd.extend(continue_flags)  # ["--continue", "<session_id>"]

    cmd.append(prompt)

    # Build environment
    env = os.environ.copy()

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace) if workspace else None,
        env=env,
    )

    return process
```

### 2.3 Update WebSocket Handler to Pass chat_id

**File:** `backend/main.py`

```python
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = None):
    # ... existing code ...

    # When calling stream_claude_response, pass the session_id
    process = await stream_claude_response(
        prompt=full_prompt,
        chat_id=session_id,  # PASS THIS
        workspace=PROJECT_ROOT,
    )

    # After first message, capture and register the Claude session ID
    # (Claude outputs session ID in its stream - capture it)
```

### 2.4 Capture Session ID from Claude Output

**File:** `backend/main.py`

When parsing the Claude stream, look for session information:

```python
async def parse_claude_stream(process, websocket, session_id):
    """Parse Claude CLI stream and capture session ID."""
    session_mgr = get_session_manager()

    async for line in process.stdout:
        event = json.loads(line.decode())

        # Look for session info in the stream
        if event.get("type") == "session_start":
            claude_session_id = event.get("session_id")
            if claude_session_id:
                await session_mgr.register_session(session_id, claude_session_id)

        # ... rest of event handling ...
```

---

## Phase 3: Hooks & Coordination (Day 2)

### 3.1 Fix Hook Paths and Permissions

**CRITICAL:** Hooks must use proper paths and be executable.

**File:** `.claude/settings.json`

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Task",
        "hooks": [
          {"type": "command", "command": "./scripts/hooks/pre_task.py"}
        ]
      },
      {
        "matcher": "Write|Edit",
        "hooks": [
          {"type": "command", "command": "./scripts/hooks/pre_write.py"}
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          {"type": "command", "command": "./scripts/hooks/agent_complete.py"}
        ]
      }
    ]
  }
}
```

### 3.2 Make Hook Scripts Executable

**Run these commands:**

```bash
chmod +x scripts/hooks/pre_task.py
chmod +x scripts/hooks/pre_write.py
chmod +x scripts/hooks/agent_complete.py
```

### 3.3 Fix DB_PATH in Hook Scripts

**File:** `scripts/hooks/pre_task.py`

Use environment variable for robust path resolution:

```python
#!/usr/bin/env python3
"""Pre-task hook for agent coordination."""

import json
import sys
import sqlite3
import os
from datetime import datetime
from pathlib import Path

# Use environment variable or fallback to relative path
SWARM_ROOT = Path(os.environ.get("SWARM_ROOT", Path(__file__).parent.parent.parent))
DB_PATH = SWARM_ROOT / ".claude" / "coordination.db"

def init_db():
    """Initialize coordination database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_log (
            id INTEGER PRIMARY KEY,
            agent TEXT,
            prompt TEXT,
            started_at TEXT,
            completed_at TEXT,
            status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            agent TEXT,
            timestamp TEXT,
            UNIQUE(namespace, key)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_agent ON task_log(agent, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ns ON decisions(namespace)")
    conn.commit()
    return conn

def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        sys.exit(0)  # Allow on parse error

    tool_input = hook_input.get("tool_input", {})
    agent_name = tool_input.get("agent", "unknown")
    prompt = tool_input.get("prompt", "")[:500]

    conn = init_db()

    # Log the task as starting
    conn.execute(
        "INSERT INTO task_log (agent, prompt, started_at, status) VALUES (?, ?, ?, ?)",
        (agent_name, prompt, datetime.now().isoformat(), "starting")
    )
    conn.commit()

    # Check for conflicts
    running = conn.execute(
        "SELECT id, agent FROM task_log WHERE status = 'running' AND agent = ?",
        (agent_name,)
    ).fetchall()

    if running:
        print(json.dumps({
            "message": f"Note: {agent_name} already has {len(running)} running task(s)",
            "continue": True
        }))

    # Update to running
    conn.execute(
        "UPDATE task_log SET status = 'running' WHERE agent = ? AND status = 'starting'",
        (agent_name,)
    )
    conn.commit()
    conn.close()

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### 3.4 Initialize Coordination DB on Startup

**File:** `backend/main.py`

Add initialization on app startup:

```python
@app.on_event("startup")
async def init_coordination_db():
    """Initialize the coordination database on startup."""
    db_path = PROJECT_ROOT / ".claude" / "coordination.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_log (
            id INTEGER PRIMARY KEY,
            agent TEXT,
            prompt TEXT,
            started_at TEXT,
            completed_at TEXT,
            status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            agent TEXT,
            timestamp TEXT,
            UNIQUE(namespace, key)
        )
    """)
    conn.commit()
    conn.close()
```

---

## Phase 4: Context Optimization (Day 2-3)

### 4.1 Simple Result Truncation (NOT LLM Summarization)

**File:** `backend/tools.py`

**IMPORTANT:** LLM summarization adds 0.5-1s latency. Use simple truncation instead:

```python
def _truncate_result(self, result: str, max_length: int = 2000) -> str:
    """Truncate result while preserving structure."""
    if len(result) <= max_length:
        return result

    # Find the last complete section/paragraph
    lines = result.split('\n')
    truncated = []
    char_count = 0

    for line in lines:
        if char_count + len(line) + 1 > max_length - 100:  # Leave room for notice
            break
        truncated.append(line)
        char_count += len(line) + 1

    truncated.append("")
    truncated.append(f"... [Truncated {len(result) - char_count} chars]")
    truncated.append("Use `Read` tool to see full output if needed.")

    return '\n'.join(truncated)
```

### 4.2 Add get_compact_state() to Memory Manager

**File:** `backend/memory.py`

```python
def get_compact_state(self) -> str:
    """Get compact state for COO - NOT detailed implementation."""
    db_path = PROJECT_ROOT / ".claude" / "coordination.db"

    if not db_path.exists():
        return "## Current State\nNo active tasks."

    conn = sqlite3.connect(str(db_path))

    # Get active tasks
    active = conn.execute(
        "SELECT agent, prompt, started_at FROM task_log WHERE status = 'running'"
    ).fetchall()

    # Get recent decisions
    recent = conn.execute(
        "SELECT namespace, key, agent FROM decisions ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()

    conn.close()

    lines = ["## Current State (Compact)"]

    if active:
        lines.append("\n### Active Tasks")
        for agent, prompt, started in active:
            lines.append(f"- **{agent}**: {prompt[:50]}...")

    if recent:
        lines.append("\n### Recent Decisions")
        for ns, key, agent in recent:
            lines.append(f"- [{ns}] {key} by {agent}")

    if not active and not recent:
        lines.append("\nNo active tasks or recent decisions.")

    return "\n".join(lines)
```

### 4.3 Compact COO Prompt

**File:** `backend/main.py`

Replace the massive system prompt with a compact version:

```python
def build_compact_coo_prompt(orchestrator, memory_manager) -> str:
    """Build a COMPACT orchestrator prompt. COO routes, doesn't hold details."""

    # Get swarm list (compact)
    swarms = orchestrator.list_swarms()
    teams = []
    for swarm in swarms:
        agents = list(orchestrator.swarms[swarm.name].agents.keys())
        teams.append(f"- **{swarm.name}**: {', '.join(agents)}")

    # Get compact state
    compact_state = memory_manager.get_compact_state()

    return f"""You are the Supreme Orchestrator (COO).

## Your Role
- Route tasks to the right swarm/agent
- Maintain high-level awareness ONLY
- DELEGATE all implementation to subagents
- Use Task tool to spawn agents with isolated context

## Available Teams
{chr(10).join(teams)}

## Current State
{compact_state}

## Rules
1. NEVER hold detailed implementation context
2. Use Task("swarm/agent", "instruction") to delegate
3. Subagents return summaries - trust their work
4. For parallel work, use ParallelTasks tool
5. Check memory for previous decisions

## Parallel Execution
Use ParallelTasks for concurrent work:
```json
{{"tool": "ParallelTasks", "input": {{"tasks": [
  {{"agent": "swarm_dev/researcher", "prompt": "..."}},
  {{"agent": "swarm_dev/architect", "prompt": "..."}}
]}}}}
```

## Communication
- Brief status updates to CEO
- ⚡ **DECISION REQUIRED** for approvals
- Synthesize results, don't repeat verbatim
"""
```

---

## Phase 5: Testing & Validation

### 5.1 Test Parallel Execution

```bash
# 1. Add timing logs to tools.py
import time
start = time.time()
# ... subprocess call ...
print(f"Agent {agent} completed in {time.time() - start:.2f}s")

# 2. Send a message that triggers 3 agents
# Expected: All 3 start within 0.5s of each other (parallel)
# NOT: Agent 1 finishes, then Agent 2 starts (sequential)
```

### 5.2 Test Session Continuity

```bash
# 1. Send: "Remember the number 42"
# 2. Send: "What number did I tell you?"
# 3. Expected: Claude remembers 42 (session maintained)
```

### 5.3 Test Hooks

```bash
# 1. Send a message that triggers Task tool
# 2. Check DB:
sqlite3 .claude/coordination.db "SELECT * FROM task_log"
# Expected: Entry with status 'running' or 'completed'
```

### 5.4 Measure COO Prompt Size

```python
# In build_compact_coo_prompt, add:
import tiktoken
enc = tiktoken.encoding_for_model("claude-3-opus-20240229")
tokens = len(enc.encode(prompt))
print(f"COO prompt: {tokens} tokens")
# Target: < 2000 tokens
```

---

## Updated Implementation Checklist

```markdown
## Phase 1: Parallel Execution (Day 1)
- [ ] 1.1 Parallel tool execution in run_agentic_chat()
- [ ] 1.2 Parallel subprocess spawning in ParallelTasks tool
- [ ] 1.3 Update _execute_task for concurrent subprocess

## Phase 2: Session Continuity (Day 1-2)
- [ ] 2.1 Create SessionManager class
- [ ] 2.2 Integrate with stream_claude_response()
- [ ] 2.3 Pass chat_id through WebSocket handler
- [ ] 2.4 Capture session ID from Claude output
- [ ] 2.5 Test session persistence across 5+ messages

## Phase 3: Hooks & Coordination (Day 2)
- [ ] 3.1 Update .claude/settings.json with correct paths
- [ ] 3.2 Make hook scripts executable (chmod +x)
- [ ] 3.3 Fix DB_PATH to use SWARM_ROOT env var
- [ ] 3.4 Initialize coordination.db on startup
- [ ] 3.5 Test hooks write to database

## Phase 4: Context Optimization (Day 2-3)
- [ ] 4.1 Add simple _truncate_result() (NOT LLM summarization)
- [ ] 4.2 Implement get_compact_state() in memory.py
- [ ] 4.3 Create build_compact_coo_prompt()
- [ ] 4.4 Replace massive system prompt
- [ ] 4.5 Measure COO prompt token count (target: <2000)

## Phase 5: Testing & Validation
- [ ] Test parallel execution with timing logs
- [ ] Test session continuity across messages
- [ ] Test hooks write to coordination.db
- [ ] Measure actual speedup (target: 3-5x)
```

---

## Updated Implementation Order

| Order | Task | Impact | Est. Effort |
|-------|------|--------|-------------|
| 1 | Parallel tool execution | High | 1 hour |
| 2 | Parallel subprocess in Task | **Critical** | 1 hour |
| 3 | Session continuity | **High** | 2 hours |
| 4 | Hooks system (with fixes) | Medium | 1.5 hours |
| 5 | Simple result truncation | Medium | 30 min |
| 6 | Compact COO prompt | Medium | 30 min |
| 7 | Enhanced memory layer | Medium | 1 hour |
| 8 | Connection pooling | Low (API only) | 30 min |

**Total Estimated Effort:** 8-10 hours

---

## Success Metrics

After implementation, verify:

1. **Speed**: 3-5x faster for multi-agent tasks (measured with timing logs)
2. **Context**: COO prompt < 2000 tokens
3. **Coordination**: Hooks log all Task invocations to DB
4. **Sessions**: Context persists across 5+ messages
5. **Parallel**: Multiple agents start within 0.5s of each other

---

## Quick Start

```bash
# 1. Make hooks executable
chmod +x scripts/hooks/*.py

# 2. Start with Phase 1 (parallel execution)
# Edit backend/main.py and backend/tools.py

# 3. Test parallel execution
./run.sh
# Send: "Research sparse attention AND design an implementation plan"
# Check logs for concurrent timestamps

# 4. Continue with Phase 2 (session continuity)
```
