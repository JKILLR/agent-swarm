# Agent Swarm Development Workflow

This document captures the established development rules for working on the Agent Swarm system.

## Git Workflow

### Branch Naming
Always use branches starting with `claude/` and ending with a session ID:
```
claude/feature-name-6c6nL
```

### Commit Messages
Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code cleanup without behavior change
- `docs:` - Documentation only
- `chore:` - Maintenance tasks

### Push Commands
Always use:
```bash
git push -u origin <branch-name>
```

### Protected Branches
Never push to `main` without explicit permission.

---

## Code Quality Rules

### Import Style
Use **absolute imports** (not relative) for run.sh compatibility:

```python
# In backend/main.py - add to sys.path first
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

# Then use absolute imports
from tools import ToolExecutor, get_tool_definitions
from memory import get_memory_manager
from jobs import get_job_queue, get_job_manager
```

**Why**: Relative imports break when running via `./run.sh` which executes files directly.

### Subprocess Stderr Handling
Always drain stderr to prevent buffer deadlocks:

```python
async def drain_stderr():
    """Drain stderr to prevent buffer deadlock."""
    if process.stderr:
        try:
            while True:
                chunk = await process.stderr.read(65536)
                if not chunk:
                    break
        except Exception:
            pass

# Start draining stderr concurrently
stderr_task = asyncio.create_task(drain_stderr())

# ... do work with stdout ...

# Ensure stderr task completes
try:
    await asyncio.wait_for(stderr_task, timeout=5.0)
except asyncio.TimeoutError:
    stderr_task.cancel()
```

**Why**: Claude CLI with `--verbose` writes to stderr. If the buffer fills, the process hangs indefinitely.

### Dead Code Policy
- **Remove**: Unused functions, orphaned SDK integrations, placeholder code
- **Keep**: Valid fallbacks that are actually wired up (e.g., `run_agentic_chat()` as API fallback)
- **Document**: If keeping code for future use, add a clear comment explaining why

### No Duplicate Definitions
Use shared interfaces from `lib/api.ts` or `shared/` modules. Don't redefine types locally in components.

---

## Architecture Rules

### Hierarchy
```
CEO (Human) → COO (Supreme Orchestrator) → Swarms → Agents
```
Never bypass the hierarchy. All work flows through the COO.

### Agent Delegation
Use the Task tool with qualified agent names:
```
Task(agent="swarm_name/agent_name", prompt="...")
```

Examples:
```
Task(agent="asa_research/researcher", prompt="Research attention mechanisms")
Task(agent="swarm_dev/implementer", prompt="Implement the feature")
```

### Real-Time Updates
Use WebSocket events for streaming updates to frontend:
- `tool_start` - Agent started using a tool
- `tool_complete` - Tool execution finished
- `agent_delta` - Streaming text from agent
- `agent_spawn` - New subagent activated
- `agent_complete` - Agent finished response

### Session Persistence
All chat sessions are saved to:
```
logs/sessions/{session_id}.json
```

Sessions persist across page refreshes and backend restarts.

---

## Testing Before Commit

### 1. Verify Imports
```bash
python3 -c "from backend.main import app; print('OK')"
python3 -c "from supreme.orchestrator import SupremeOrchestrator; print('OK')"
```

### 2. Test API
```bash
curl -s http://localhost:8000/api/swarms | head
```

### 3. Check Server Starts
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Should see "Application startup complete" without errors
```

---

## Key File Locations

| Purpose | Location |
|---------|----------|
| Backend hub | `backend/main.py` |
| Job queue | `backend/jobs.py` |
| Tool executor | `backend/tools.py` |
| Memory manager | `backend/memory.py` |
| Orchestrator | `supreme/orchestrator.py` |
| Swarm definitions | `swarms/{name}/swarm.yaml` |
| Agent prompts | `swarms/{name}/agents/{agent}.md` |
| Frontend API | `frontend/lib/api.ts` |
| WebSocket client | `frontend/lib/websocket.ts` |
| Chat UI | `frontend/app/chat/page.tsx` |
| Architecture docs | `ARCHITECTURE.md` |

---

## What NOT to Do

1. **Don't use relative imports** - Breaks `./run.sh`
2. **Don't leave stderr unread** - Causes subprocess hangs
3. **Don't commit SDK placeholder code** - Remove or fully implement
4. **Don't duplicate interfaces** - Use shared definitions
5. **Don't skip testing** - Always verify before push
6. **Don't bypass the hierarchy** - Work flows through COO
7. **Don't push to main** - Use feature branches

---

## Common Fixes

### "attempted relative import with no known parent package"
Change relative imports to absolute:
```python
# Wrong
from .tools import ToolExecutor

# Right
from tools import ToolExecutor
```

### Subprocess hangs after 15+ minutes
Add stderr draining (see Subprocess Stderr Handling above).

### "ModuleNotFoundError" when running backend
Ensure sys.path includes both PROJECT_ROOT and BACKEND_DIR before imports.

### WebSocket not receiving events
Check that `_process_cli_event()` in `backend/main.py` handles the event type and calls `manager.send_event()`.

---

## Parallel Execution Patterns (Architecture Enhancement)

### Parallel Tool Execution
Tool calls should execute concurrently using `asyncio.gather()`:

```python
# In run_agentic_chat() - execute tools in parallel
if tool_uses:
    results = await asyncio.gather(*[
        tool_executor.execute(tu.name, tu.input, workspace)
        for tu in tool_uses
    ], return_exceptions=True)
```

**Why**: Sequential execution of 3 agents takes 3x longer than parallel.

### ParallelTasks Tool
For multi-agent work, use the ParallelTasks tool:

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
1. `researcher` → gather information
2. `architect` → design solution
3. `implementer` → write code
4. `tester` → verify implementation
5. `critic` → final review

---

## Subagent Definitions

### Location
Subagent definitions are in `.claude/agents/`:
- `researcher.md` - Deep research and exploration
- `architect.md` - System design and planning
- `implementer.md` - Code implementation
- `critic.md` - Code review and QA
- `tester.md` - Test creation and execution

### Format
Each subagent has YAML frontmatter + markdown body:

```markdown
---
name: researcher
description: Deep research agent
tools: Read, Grep, Glob, Bash, WebSearch
model: sonnet
---

You are a Research Specialist...
```

### Context Isolation
- Each subagent has its own context window
- Return SUMMARIES, not full context
- Use shared memory for cross-agent decisions

---

## Hooks for Coordination

### Configuration
Hooks are defined in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [...],
    "SubagentStop": [...]
  }
}
```

### Hook Scripts
Located in `scripts/hooks/`:
- `pre_task.py` - Log task, check for conflicts before Task tool
- `pre_write.py` - Validate file paths before Write/Edit
- `agent_complete.py` - Update task status when agent finishes

### Coordination Database
Hooks use `.claude/coordination.db` (SQLite) to track:
- Active tasks per agent
- Task start/completion times
- Cross-agent decisions

---

## COO Prompt Guidelines

### Keep It Compact
The COO should NOT hold detailed implementation context:

```python
# BAD: Loading all swarm details into COO context
system_prompt = f"""...{all_agent_prompts}...{full_memory}..."""

# GOOD: Compact routing information only
system_prompt = f"""...{swarm_list}...{compact_state}..."""
```

### Delegation Rules
1. NEVER hold detailed implementation context
2. Use Task tool to spawn agents with isolated context
3. Trust subagent summaries
4. For parallel work, spawn multiple Tasks in one message
5. Check shared memory for previous decisions

---

## Session Continuity

### The --continue Flag
Use `--continue` to maintain context across messages:

```bash
claude -p --continue <session_id> "Follow-up prompt"
```

### Session Manager
The backend should maintain session IDs per chat:
- First message: Start new session
- Subsequent messages: Continue existing session
- Session end: Clean up resources

---

## Result Summarization

### Long Results
Subagent results over 2000 chars should be auto-summarized:

```python
if len(result) > 2000:
    result = await summarize_with_haiku(result)
```

### Summary Format
Summaries should include:
- Key findings (3-5 bullet points)
- Relevant files (if applicable)
- Next steps (actionable items)

---

## New File Locations

| Purpose | Location |
|---------|----------|
| Subagent definitions | `.claude/agents/` |
| Hook scripts | `scripts/hooks/` |
| Hooks config | `.claude/settings.json` |
| Coordination DB | `.claude/coordination.db` |
| Implementation plan | `IMPLEMENTATION_PLAN.md` |
| Architecture review | `ARCHITECTURE_REVIEW.md` |
