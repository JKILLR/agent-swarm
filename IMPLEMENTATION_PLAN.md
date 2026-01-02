# Agent Swarm Architecture Improvements - Implementation Plan

**Created:** January 2, 2026
**Based on:** ARCHITECTURE_REVIEW.md findings
**Goal:** Transform the system into a "well-oiled machine" with 3-5x speed improvement

---

## Executive Summary

This plan addresses the critical bottlenecks identified in the architecture review:

| Issue | Solution | Expected Impact |
|-------|----------|-----------------|
| Sequential tool execution | `asyncio.gather()` parallel execution | 3-5x speedup |
| CLI subprocess overhead | Session continuity + connection pooling | 2-3s saved per agent |
| Context pollution | Compact COO prompt + result summarization | Cleaner context |
| No coordination | Hooks system + shared memory | Better cohesion |
| Missing verification | PreToolUse/PostToolUse hooks | Self-correcting agents |

---

## Phase 1: Parallel Execution Foundation (Priority: Critical)

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

### 1.2 ParallelTasks Tool Enhancement

**File:** `backend/tools.py`

**Current:** ParallelTasks tool exists but may not execute truly in parallel.

**Change:** Ensure the ParallelTasks tool uses `asyncio.gather()`:

```python
async def _execute_parallel_tasks(self, input: dict[str, Any]) -> str:
    """Execute multiple tasks TRULY in parallel."""
    tasks = input.get("tasks", [])

    if not tasks:
        return "No tasks provided"

    # Create all coroutines
    coroutines = [
        self._execute_task({
            "agent": t.get("agent"),
            "prompt": t.get("prompt"),
        })
        for t in tasks
    ]

    # Execute ALL in parallel
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Format results
    output_lines = ["## Parallel Execution Results\n"]
    for task, result in zip(tasks, results):
        agent = task.get("agent", "unknown")
        if isinstance(result, Exception):
            output_lines.append(f"### {agent}\n**Error:** {result}\n")
        else:
            output_lines.append(f"### {agent}\n{result}\n")

    return "\n".join(output_lines)
```

### 1.3 Connection Pooling

**File:** `backend/api_client.py` (NEW)

Create a singleton API client with connection reuse:

```python
"""Anthropic API client with connection pooling."""

import os
from typing import Optional
import anthropic

class ClaudeAPIPool:
    """Singleton API client with connection reuse."""

    _instance: Optional["ClaudeAPIPool"] = None

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(
                api_key=api_key,
                max_retries=3,
            )
        else:
            self.client = None

    @classmethod
    def get_client(cls) -> Optional[anthropic.Anthropic]:
        """Get or create the singleton client."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.client

    @classmethod
    def reset(cls):
        """Reset the connection pool."""
        cls._instance = None
```

**Integration:** Use this client in `shared/agent_executor.py` instead of creating new clients.

---

## Phase 2: Session Continuity & Coordination (Priority: High)

### 2.1 Session Manager

**File:** `backend/session_manager.py` (NEW)

Maintain persistent Claude sessions per chat:

```python
"""Session manager for persistent Claude sessions."""

import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ClaudeSession:
    session_id: str
    chat_id: str
    created_at: datetime
    last_used: datetime

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

    async def create_session(self, chat_id: str, session_id: str):
        """Register a new session."""
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

    def get_continue_flag(self, chat_id: str) -> list[str]:
        """Get --continue flag if session exists."""
        session = self.active_sessions.get(chat_id)
        if session:
            return ["--continue", session.session_id]
        return []
```

**Integration:** Modify `stream_claude_response()` in `backend/main.py` to use session continuity.

### 2.2 Hooks System

**File:** `.claude/settings.json` (NEW)

Create hooks configuration:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Task",
        "hooks": [
          {"type": "command", "command": "python3 scripts/hooks/pre_task.py"}
        ]
      },
      {
        "matcher": "Write|Edit",
        "hooks": [
          {"type": "command", "command": "python3 scripts/hooks/pre_write.py"}
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          {"type": "command", "command": "python3 scripts/hooks/agent_complete.py"}
        ]
      }
    ]
  }
}
```

**File:** `scripts/hooks/pre_task.py` (NEW)

```python
#!/usr/bin/env python3
"""Pre-task hook for agent coordination."""

import json
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / ".claude" / "coordination.db"

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
    conn.commit()
    return conn

def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except:
        sys.exit(0)  # Allow on parse error

    tool_input = hook_input.get("tool_input", {})
    agent_name = tool_input.get("agent", "unknown")
    prompt = tool_input.get("prompt", "")[:500]

    conn = init_db()

    # Log the task
    conn.execute(
        "INSERT INTO task_log (agent, prompt, started_at, status) VALUES (?, ?, ?, ?)",
        (agent_name, prompt, datetime.now().isoformat(), "starting")
    )
    conn.commit()

    # Check for conflicts
    running = conn.execute(
        "SELECT agent FROM task_log WHERE status = 'running' AND agent = ?",
        (agent_name,)
    ).fetchall()

    if running:
        print(json.dumps({
            "message": f"Note: {agent_name} already has a running task",
            "continue": True
        }))

    conn.close()
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**File:** `scripts/hooks/agent_complete.py` (NEW)

```python
#!/usr/bin/env python3
"""Agent completion hook for coordination."""

import json
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / ".claude" / "coordination.db"

def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except:
        sys.exit(0)

    agent_name = hook_input.get("agent", "unknown")

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE task_log SET status = ?, completed_at = ? WHERE agent = ? AND status = 'running'",
            ("completed", datetime.now().isoformat(), agent_name)
        )
        conn.commit()
        conn.close()
    except:
        pass

    sys.exit(0)

if __name__ == "__main__":
    main()
```

---

## Phase 3: Shared Memory & Compact Context (Priority: High)

### 3.1 Enhanced Memory Layer

**File:** `backend/memory.py`

Add decisions tracking and compact state:

```python
# Add to MemoryManager class

def _init_decisions_table(self):
    """Initialize decisions tracking table."""
    conn = sqlite3.connect(self.db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            agent TEXT,
            timestamp TEXT,
            UNIQUE(namespace, key)
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_ns ON decisions(namespace);
    """)
    conn.commit()
    conn.close()

def store_decision(self, namespace: str, key: str, value: Any, agent: str = None):
    """Store a decision that other agents can reference."""
    conn = sqlite3.connect(self.db_path)
    conn.execute(
        "INSERT OR REPLACE INTO decisions (namespace, key, value, agent, timestamp) VALUES (?, ?, ?, ?, ?)",
        (namespace, key, json.dumps(value), agent, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_decision(self, namespace: str, key: str) -> Optional[dict]:
    """Get a previous decision."""
    conn = sqlite3.connect(self.db_path)
    row = conn.execute(
        "SELECT value, agent, timestamp FROM decisions WHERE namespace = ? AND key = ?",
        (namespace, key)
    ).fetchone()
    conn.close()

    if row:
        return {"value": json.loads(row[0]), "agent": row[1], "timestamp": row[2]}
    return None

def get_compact_state(self) -> str:
    """Get compact state for COO - NOT detailed implementation."""
    conn = sqlite3.connect(self.db_path)

    recent = conn.execute(
        "SELECT namespace, key, agent FROM decisions ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()

    conn.close()

    lines = ["## Current State (Compact)"]
    if recent:
        lines.append("\n### Recent Decisions")
        for ns, key, agent in recent:
            lines.append(f"- [{ns}] {key} by {agent}")

    return "\n".join(lines)
```

### 3.2 Compact COO Prompt

**File:** `backend/main.py`

Create a helper function for compact COO prompt:

```python
def build_compact_coo_prompt(orchestrator, memory_manager) -> str:
    """
    Build a COMPACT orchestrator prompt.
    The COO routes - it doesn't hold implementation details.
    """
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
- DELEGATE all implementation details to subagents
- Use Task tool to spawn agents with isolated context

## Available Teams
{chr(10).join(teams)}

## Current State
{compact_state}

## Rules
1. NEVER hold detailed implementation context
2. Use Task("swarm/agent", "specific instruction") to delegate
3. Subagents return summaries - trust their work
4. For parallel work, spawn multiple Tasks in one message
5. Check memory for previous decisions before delegating

## Parallel Execution
When you need multiple agents, use ParallelTasks:
```json
{{"tool": "ParallelTasks", "input": {{"tasks": [
  {{"agent": "swarm_dev/researcher", "prompt": "Research..."}},
  {{"agent": "swarm_dev/architect", "prompt": "Design..."}}
]}}}}
```

## Communication
- Brief status updates to CEO
- ⚡ **DECISION REQUIRED** for approvals
- Synthesize subagent results, don't repeat them verbatim
"""
```

### 3.3 Result Summarization

**File:** `backend/tools.py`

Add auto-summarization for long results:

```python
async def _summarize_if_needed(self, result: str, max_length: int = 2000) -> str:
    """Summarize long results before returning to orchestrator."""
    if len(result) <= max_length:
        return result

    # Use quick summarization
    summary_prompt = f"""Summarize these findings in 3-5 bullet points.
Focus on: key discoveries, blockers, actionable items.

FULL RESULT:
{result[:8000]}...
"""

    # Quick summarization with haiku
    try:
        client = ClaudeAPIPool.get_client()
        if client:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            return response.content[0].text
    except:
        pass

    # Fallback: truncate with notice
    return result[:max_length] + "\n\n[Result truncated. Use Read tool to see full output.]"
```

---

## Phase 4: Subagent Definitions (Priority: Medium)

### 4.1 Create .claude/agents/ Directory

**Files to create:**

1. `.claude/agents/researcher.md`
2. `.claude/agents/architect.md`
3. `.claude/agents/implementer.md`
4. `.claude/agents/critic.md`
5. `.claude/agents/tester.md`

Use the example definitions provided in the conversation (from the user's examples).

### 4.2 Update Workflow with Subagent Patterns

**File:** `.claude/workflow.md`

Add new section:

```markdown
---

## Subagent Patterns (NEW)

### Parallel Execution
For tasks requiring multiple perspectives, use ParallelTasks:

```json
{
  "tool": "ParallelTasks",
  "input": {
    "tasks": [
      {"agent": "swarm_dev/researcher", "prompt": "Research existing patterns"},
      {"agent": "swarm_dev/architect", "prompt": "Design the solution"},
      {"agent": "swarm_dev/critic", "prompt": "Identify potential issues"}
    ]
  }
}
```

### Pipeline Chaining
For deterministic workflows, chain agents sequentially:
1. researcher → gather information
2. architect → design solution
3. implementer → write code
4. tester → verify implementation
5. critic → final review

### Context Isolation
- Each subagent has its own context window
- Return SUMMARIES, not full context
- Use shared memory for cross-agent decisions

### Session Continuity
Use `--continue` flag to maintain context across messages:
```bash
claude -p --continue <session_id> "Follow-up prompt"
```

---

## Hooks for Coordination (NEW)

Hooks in `.claude/settings.json` enable automated feedback loops:

- **PreToolUse**: Validate actions before execution
- **SubagentStop**: Coordinate handoffs between agents
- **Stop**: Finalize session state

See `scripts/hooks/` for implementations.
```

---

## Phase 5: Testing & Validation

### 5.1 Test Parallel Execution

```bash
# Test that parallel tasks actually run in parallel
# 1. Start the server
./run.sh

# 2. Send a message that triggers multiple agents
# Observe logs for concurrent execution timestamps
```

### 5.2 Test Session Continuity

```bash
# Test that sessions persist
# 1. Send a message, note the session ID
# 2. Send a follow-up, verify context is maintained
# 3. Restart server, verify session resumes
```

### 5.3 Test Hooks

```bash
# Test hooks are triggered
# 1. Check .claude/coordination.db exists after Task tool use
# 2. Verify task_log table has entries
# 3. Check status transitions (starting → running → completed)
```

---

## Implementation Order

| Order | Task | Est. Effort | Files |
|-------|------|-------------|-------|
| 1 | Parallel tool execution | 1-2 hours | `backend/main.py` |
| 2 | ParallelTasks enhancement | 1 hour | `backend/tools.py` |
| 3 | Connection pooling | 30 min | `backend/api_client.py` (NEW) |
| 4 | Session manager | 1 hour | `backend/session_manager.py` (NEW) |
| 5 | Hooks system | 2 hours | `.claude/settings.json`, `scripts/hooks/` |
| 6 | Enhanced memory | 1 hour | `backend/memory.py` |
| 7 | Compact COO prompt | 30 min | `backend/main.py` |
| 8 | Result summarization | 1 hour | `backend/tools.py` |
| 9 | Subagent definitions | 1 hour | `.claude/agents/` |
| 10 | Workflow updates | 30 min | `.claude/workflow.md` |

**Total Estimated Effort:** 10-12 hours

---

## Success Metrics

After implementation, verify:

1. **Speed**: Chat responses 3-5x faster for multi-agent tasks
2. **Context**: COO prompt stays under 2000 tokens
3. **Coordination**: Hooks log all Task tool invocations
4. **Summarization**: Long results auto-summarized
5. **Sessions**: Context persists across messages

---

## Quick Start

To begin implementation:

```bash
# 1. Start with Phase 1 (parallel execution)
# Edit backend/main.py - modify tool execution loop

# 2. Test the change
./run.sh
# Send: "Research sparse attention AND design an implementation plan"
# Verify both agents spawn concurrently

# 3. Continue with remaining phases
```
