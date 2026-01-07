# Review: COO Delegation Behavior Failure Modes

**Date**: 2026-01-03
**Reviewer**: Quality Critic
**Status**: NEEDS_CHANGES

---

## Executive Summary

The COO (Supreme Orchestrator) delegation mechanism has **fundamental architectural issues** that prevent reliable agent delegation. The primary problem is that **the Task tool appears to work in the frontend UI but is not actually executing real sub-agents**. The delegation is happening only as a UI notification - no separate agent process is being spawned.

---

## Critical Failure Modes

### 1. CRITICAL: Task Tool Does Not Spawn Real Agent Processes

**Location**: `backend/main.py:2060-2356` (`_process_cli_event` function)

**Evidence**:
- When the COO uses the `Task` tool, the backend detects this in the streaming events (lines 2119-2139, 2238-2272)
- The backend sends `agent_spawn` and `agent_complete_subagent` WebSocket events to the frontend
- **However, no actual agent subprocess is started** - the Task tool's result comes from Claude's internal handling, not from a real separate agent process

**How it fails**:
```python
# Lines 2260-2270 - Only sends UI event, does NOT spawn agent
await manager.send_event(
    websocket,
    "agent_spawn",
    {
        "agent": agent_name,
        "description": desc,
        "parentAgent": parent_agent,
    },
)
context["agent_spawn_sent"] = True  # Just marks as "sent", no actual spawn
```

**Root Cause**:
The Task tool is a **Claude CLI built-in** that creates an internal sub-conversation. The backend simply parses the output stream but has no control over what the "sub-agent" actually is or does. There is no integration point where the backend:
1. Reads the agent's `.md` configuration
2. Injects the agent's system prompt
3. Runs the agent in its proper workspace with correct permissions

**Impact**:
- Sub-agents run with COO's context, not their specialized prompts
- No workspace isolation for delegated tasks
- No agent-specific tool permissions enforced
- Effectively, delegation is a LIE - it's just the same Claude session pretending to be different agents

---

### 2. CRITICAL: No Connection Between WebSocket Flow and AgentExecutorPool

**Location**: `backend/main.py:1861-1924` (`stream_claude_response`) vs `shared/agent_executor_pool.py`

**Evidence**:
The main WebSocket chat flow (`/ws/chat`) uses `stream_claude_response()` which directly spawns a Claude CLI subprocess:

```python
# Lines 1914-1922 - Direct subprocess, bypasses pool entirely
process = await asyncio.create_subprocess_exec(
    *cmd,
    stdin=asyncio.subprocess.DEVNULL,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=cwd,
    env=env,
)
```

Meanwhile, the `AgentExecutorPool` (lines 80-178 in `agent_executor_pool.py`) provides:
- Concurrency limits
- Workspace isolation
- Proper event streaming
- Cancellation support
- Agent-specific configuration

**But it is never used for COO execution or Task tool delegation.**

**Impact**:
- COO runs without concurrency limits
- No workspace isolation for the COO
- Task delegations don't spawn through the pool
- The pool exists but provides no value for the main chat flow

---

### 3. HIGH: Agent Stack Tracking is Purely Cosmetic

**Location**: `backend/main.py:2060-2078`, `2119-2140`, `2248-2271`, `2302-2315`

The `context["agent_stack"]` mechanism tracks which agent is "currently active" but this is only for:
1. Attributing tool events to agents in the UI
2. Sending `agent_spawn` / `agent_complete_subagent` notifications

**What it does NOT do**:
- Modify the system prompt for sub-agents
- Change the working directory
- Adjust tool permissions
- Load agent-specific configuration from `.md` files

**Evidence**:
```python
# Line 2073 - Stack is just names, no configuration
context["agent_stack"] = ["COO"]  # COO is always the base

# Lines 2248-2258 - Just pushes a name, doesn't load agent config
context["agent_stack"].append(agent_name)
context["pending_tasks"][tool_use_id] = agent_name
```

---

### 4. HIGH: Agent Definitions in .md Files Are Not Used for Delegation

**Location**: `swarms/swarm_dev/agents/*.md` (orchestrator.md, architect.md, etc.)

Each agent has a carefully crafted definition file with:
- System prompt (markdown content)
- Tool permissions (`tools: [Task, Read, Glob, Bash]`)
- Model preference (`model: opus`)
- Background/wake settings

**But when the COO uses `Task(subagent_type="architect", ...)`:
- The architect's `.md` file is NEVER read
- The architect's system prompt is NOT injected
- The tool permissions are NOT enforced
- It's just the COO pretending to be the architect

**Evidence**: Search for agent `.md` loading in `_process_cli_event`:
```bash
# No references to agent markdown files in the delegation logic
grep -n "\.md" backend/main.py  # Only references STATE.md in prompts
```

---

### 5. MEDIUM: Results Not Captured from Delegated Tasks

**Location**: `backend/main.py:2302-2315` (`content_block_stop` handling)

When a Task tool completes:
```python
if tool_name == "Task" and tool_use_id in context.get("pending_tasks", {}):
    completed_agent = context["pending_tasks"].pop(tool_use_id)
    if context["agent_stack"] and context["agent_stack"][-1] == completed_agent:
        context["agent_stack"].pop()
    # Send agent completion event
    await manager.send_event(
        websocket,
        "agent_complete_subagent",
        {
            "agent": completed_agent,
            "success": True,  # Always True - no actual result captured
        },
    )
```

**Issues**:
- `success` is hardcoded to `True` - no error detection
- The actual result content from the Task is not extracted
- No mechanism to pass Task results to the Work Ledger or Mailbox systems

---

### 6. MEDIUM: Work Ledger and Agent Mailbox Not Integrated with Delegation

**Location**:
- `shared/work_ledger.py` - Built but not called during Task execution
- `shared/agent_mailbox.py` - Built but not called during Task execution

**Evidence from STATE.md**:
```
### NOT CONNECTED: Work Ledger (`shared/work_ledger.py`) - built but not integrated
### NOT CONNECTED: Agent Mailbox (`shared/agent_mailbox.py`) - built but not integrated
```

These systems were designed to:
1. Track work items through delegation
2. Enable structured handoffs between agents
3. Persist work state for crash recovery

**But they are orphaned code** - no integration with the Task tool flow.

---

## Why Delegation "Appears" to Work

The COO claims to delegate and the UI shows delegation happening because:

1. **Claude CLI has a built-in Task tool** that creates internal sub-conversations
2. The backend detects `Task` tool usage in the stream and sends `agent_spawn` events
3. The frontend displays these as agent spawns in the Activity Panel
4. Claude continues and returns a response that appears to come from the "sub-agent"

**But this is all happening within a single Claude CLI process** with:
- The same system prompt (COO's prompt, not the sub-agent's)
- The same working directory
- The same tool permissions
- No actual isolation or specialization

---

## Evidence from Logs

The backend log shows no evidence of actual sub-agent process spawns:

```
2026-01-03 08:33:29,157 [INFO] Starting Claude CLI in /Users/jellingson/agent-swarm
# Only ONE Claude CLI process per chat - no additional spawns for delegation
```

Expected behavior if delegation worked:
```
[INFO] Starting Claude CLI for COO in /Users/jellingson/agent-swarm
[INFO] Starting Claude CLI for architect in swarms/swarm_dev/workspace
[INFO] Starting Claude CLI for implementer in swarms/swarm_dev/workspace
```

---

## Recommendations

### Immediate Fixes (Critical)

1. **Intercept Task Tool and Spawn Real Agents**

   Modify `_process_cli_event` to:
   - When Task tool is detected, pause the COO's stream
   - Read the target agent's `.md` file for configuration
   - Spawn a new Claude CLI process with the agent's system prompt
   - Use `AgentExecutorPool` for proper isolation
   - Capture the result and inject it back into COO's conversation

2. **Use AgentExecutorPool for All Execution**

   Route the main WebSocket chat flow through the pool:
   ```python
   # Instead of stream_claude_response(), use:
   pool = get_executor_pool(workspace_manager=workspace_manager)
   context = AgentExecutionContext(
       agent_name="Supreme Orchestrator",
       agent_type="orchestrator",
       swarm_name="supreme",
       workspace=PROJECT_ROOT,
       ...
   )
   async for event in pool.execute(context, user_prompt):
       await manager.send_event(websocket, event["type"], event)
   ```

### Medium-Term Fixes (High)

3. **Integrate Work Ledger with Task Execution**
   - Create WorkItem when Task tool starts
   - Update status as Task progresses
   - Complete/fail WorkItem when Task finishes

4. **Integrate Agent Mailbox for Handoffs**
   - Use mailbox for Task results
   - Enable async handoffs between agents
   - Support background agent communication

### Long-Term Fixes (Architecture)

5. **Create a Proper Task Execution Layer**
   - Define a `TaskExecutor` class that:
     - Loads agent configuration from `.md` files
     - Builds proper system prompts
     - Manages workspace isolation
     - Handles result capture and error propagation
   - Wire this into the Claude CLI's tool handling

6. **Consider Custom MCP Tool for Task**
   - Replace Claude's built-in Task tool with a custom MCP server
   - Full control over agent spawning, configuration, and isolation
   - Can integrate directly with AgentExecutorPool, Work Ledger, Mailbox

---

## Files Requiring Changes

| File | Required Change | Priority |
|------|----------------|----------|
| `backend/main.py` | Intercept Task tool, spawn real agents | Critical |
| `backend/main.py` | Route WebSocket chat through AgentExecutorPool | Critical |
| `shared/agent_executor_pool.py` | Add method to load agent from .md file | High |
| `shared/work_ledger.py` | Integration hooks for Task execution | Medium |
| `shared/agent_mailbox.py` | Integration hooks for Task handoffs | Medium |

---

## Conclusion

**The delegation system is fundamentally broken.** The COO can "use" the Task tool, but this doesn't actually spawn specialized agents with their proper configurations. It's essentially role-playing within a single Claude session.

To fix this properly requires architectural changes to:
1. Intercept Task tool invocations before Claude executes them internally
2. Actually spawn sub-agent processes with proper isolation
3. Capture and route results back to the parent agent
4. Integrate with the existing Work Ledger and Mailbox systems

The current implementation gives the **illusion** of delegation without any of the benefits of specialized agents, workspace isolation, or proper tool enforcement.
