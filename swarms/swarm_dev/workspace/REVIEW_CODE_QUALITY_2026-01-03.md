# Code Quality Review: Backend System
## Date: 2026-01-03
## Reviewer: Critic Agent

---

## Review Result: NEEDS_CHANGES

The backend system has several issues that require attention, with one critical bug that will cause runtime errors.

---

## Issues Found

### Critical (Must Fix)

#### 1. Missing Method `_get_tool_description` in AgentExecutorPool
- **File**: `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py`
- **Line**: 422
- **Severity**: CRITICAL

**Description**: The method `self._get_tool_description(tool_name, tool_input)` is called on line 422 but is never defined in the `AgentExecutorPool` class. This will cause an `AttributeError` at runtime when the pool parses tool events.

**Code**:
```python
# Line 422 in agent_executor_pool.py
description = self._get_tool_description(tool_name, tool_input)
```

**The method exists in**:
- `backend/main.py:1462` - module-level function `_get_tool_description`
- `backend/tools.py:425` - method in a different class

**Fix Required**: Either:
1. Add a `_get_tool_description` method to `AgentExecutorPool` class, OR
2. Import and use the function from `backend/main.py`, OR
3. Inline the tool description logic

**Recommended Fix** - Add method to class:
```python
def _get_tool_description(self, tool_name: str, tool_input: dict) -> str:
    """Generate human-readable description for a tool call."""
    descriptions = {
        "Read": lambda i: f"Reading {i.get('file_path', 'file')[:60]}",
        "Write": lambda i: f"Writing to {i.get('file_path', 'file')[:60]}",
        "Edit": lambda i: f"Editing {i.get('file_path', 'file')[:60]}",
        "Bash": lambda i: f"Running: {i.get('command', '')[:50]}",
        "Glob": lambda i: f"Searching for {i.get('pattern', 'files')}",
        "Grep": lambda i: f"Searching for '{i.get('pattern', '')[:40]}'",
        "Task": lambda i: f"Delegating to {i.get('subagent_type', i.get('agent', 'agent'))}",
    }
    if tool_name in descriptions:
        try:
            return descriptions[tool_name](tool_input)
        except Exception:
            pass
    return f"Using {tool_name}"
```

---

### High Priority (Should Fix)

#### 2. Unused `_running` Dict in AgentExecutorPool
- **File**: `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py`
- **Lines**: 57, 580-581, 594
- **Severity**: HIGH

**Description**: The `_running` dictionary is defined and referenced but never populated. Tasks are never added to it, making the `cancel()` method's task check (line 580-581) ineffective.

**Code**:
```python
# Line 57 - defined but never populated
self._running: dict[str, asyncio.Task] = {}

# Line 580-581 - checked but will never find anything
if execution_id in self._running:
    task = self._running[execution_id]
```

**Impact**: Cancellation via task reference will never work; only process kill will work.

**Fix Required**: Populate `_running` dict in the `execute()` method.

---

#### 3. Race Condition in Singleton Initialization
- **File**: `/Users/jellingson/agent-swarm/shared/workspace_manager.py`
- **Lines**: 296-317
- **Severity**: HIGH

**Description**: The `get_workspace_manager()` singleton getter is not thread-safe. In concurrent scenarios, multiple WorkspaceManager instances could be created.

**Code**:
```python
def get_workspace_manager(project_root: Path | None = None) -> WorkspaceManager:
    global _workspace_manager
    if _workspace_manager is None:  # TOCTOU race condition here
        if project_root is None:
            raise ValueError(...)
        _workspace_manager = WorkspaceManager(project_root)
    return _workspace_manager
```

**Note**: The `escalation_protocol.py` does this correctly with `threading.Lock()`.

**Fix Required**: Add thread-safe locking similar to `escalation_protocol.py`:
```python
_singleton_lock = threading.Lock()

def get_workspace_manager(project_root: Path | None = None) -> WorkspaceManager:
    global _workspace_manager
    if _workspace_manager is None:
        with _singleton_lock:
            if _workspace_manager is None:  # Double-check
                _workspace_manager = WorkspaceManager(project_root)
    return _workspace_manager
```

---

#### 4. Same Race Condition in AgentExecutorPool
- **File**: `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py`
- **Lines**: 612-633
- **Severity**: HIGH

**Description**: Same issue as workspace_manager - no thread-safe locking.

---

### Medium Priority (Should Address)

#### 5. Fire-and-Forget Tasks Without Error Handling
- **File**: `/Users/jellingson/agent-swarm/backend/main.py`
- **Lines**: 125, 138
- **Severity**: MEDIUM

**Description**: `asyncio.create_task()` is called without storing the task reference. If these tasks fail, errors may be silently swallowed.

**Code**:
```python
# Line 125
asyncio.create_task(_broadcast_job_update_safe(job))

# Line 138
asyncio.create_task(_broadcast_executor_event_safe(event))
```

**Note**: The `_safe` wrappers do catch exceptions, but task failures are only logged. Consider collecting tasks for graceful shutdown.

---

#### 6. Hard-coded "swarm_dev" Default in Jobs
- **File**: `/Users/jellingson/agent-swarm/backend/jobs.py`
- **Line**: 545
- **Severity**: MEDIUM

**Description**: Default swarm name is hard-coded rather than configurable.

**Code**:
```python
swarm_name = job.swarm or "swarm_dev"  # Default to swarm_dev for COO tasks
```

**Impact**: May cause unexpected behavior if other swarms expect different defaults.

---

#### 7. Imports at Non-Top-Level
- **File**: `/Users/jellingson/agent-swarm/backend/main.py`
- **Lines**: 1141-1143, 1909
- **Severity**: MEDIUM (style)

**Description**: Imports for `urllib.parse`, `urllib.request`, `re`, and `tempfile` are done mid-file rather than at the top. While functional, this violates PEP 8.

---

#### 8. Bare Exception Catches
- **File**: `/Users/jellingson/agent-swarm/backend/main.py`
- **Lines**: Multiple (561, 582, 595, 609, 1875, 1883, etc.)
- **Severity**: MEDIUM

**Description**: Many `except Exception:` catches with pass or minimal logging can hide bugs.

**Example**:
```python
except Exception:
    try:
        executor_pool_subscribers.remove(ws)
    except ValueError:
        pass
```

---

### Low Priority (Nice to Have)

#### 9. TOCTOU in Path Validation
- **File**: `/Users/jellingson/agent-swarm/shared/workspace_manager.py`
- **Lines**: 84-100
- **Severity**: LOW (documented in previous review)

**Description**: Path validation uses `resolve()` before access, but the path could change between validation and use. This is a known issue from the security review.

---

#### 10. Magic Numbers
- **File**: `/Users/jellingson/agent-swarm/backend/main.py`
- **Line**: 2074
- **Severity**: LOW

**Description**: Timeout of 3600.0 (1 hour) is hard-coded. Consider making configurable.

```python
timeout=3600.0,  # 1 hour timeout
```

---

#### 11. `allowed_tools` Not Enforced
- **Files**: Multiple
- **Severity**: LOW (documented)

**Description**: The `allowed_tools` list in permissions is informational only - Claude CLI doesn't enforce tool restrictions based on this list. This is documented in STATE.md as a known issue.

---

## Positive Observations

1. **Good Error Handling in Key Areas**: The `_broadcast_*_safe` functions properly wrap async operations with try/except.

2. **Thread-Safe Escalation Protocol**: The `escalation_protocol.py` correctly implements double-checked locking with `threading.Lock()`.

3. **Atomic File Writes**: `escalation_protocol.py` uses atomic write pattern (write to .tmp then rename).

4. **Proper Process Cleanup**: The `AgentExecutorPool` properly kills processes and waits for termination on cancel.

5. **Context Validation**: `AgentExecutionContext.__post_init__` validates input properly.

6. **Comprehensive Logging**: Good use of structured logging throughout.

7. **WebSocket Connection Management**: Proper cleanup of connections on disconnect.

---

## Summary Table

| Issue | Severity | File | Status |
|-------|----------|------|--------|
| Missing `_get_tool_description` method | CRITICAL | agent_executor_pool.py:422 | MUST FIX |
| Unused `_running` dict | HIGH | agent_executor_pool.py | Should fix |
| Race condition in singleton (workspace_manager) | HIGH | workspace_manager.py | Should fix |
| Race condition in singleton (executor_pool) | HIGH | agent_executor_pool.py | Should fix |
| Fire-and-forget tasks | MEDIUM | main.py | Consider |
| Hard-coded "swarm_dev" | MEDIUM | jobs.py:545 | Consider |
| Mid-file imports | MEDIUM | main.py | Style |
| Bare exception catches | MEDIUM | main.py | Consider |
| TOCTOU path validation | LOW | workspace_manager.py | Documented |
| Magic timeout number | LOW | main.py:2074 | Consider |
| `allowed_tools` not enforced | LOW | Multiple | Documented |

---

## Recommendations

### Immediate Actions (Before Next Deploy)
1. Fix the missing `_get_tool_description` method - this will cause runtime crashes
2. Add thread-safe locking to singleton getters

### Short-Term (This Sprint)
1. Populate and use the `_running` dict for proper task tracking
2. Extract magic numbers to configuration

### Long-Term (Technical Debt)
1. Consolidate tool description logic into shared module
2. Add proper enforcement of `allowed_tools`
3. Move all imports to top of files
4. Add more specific exception types

---

## Files Reviewed
- `/Users/jellingson/agent-swarm/backend/main.py` (2176 lines)
- `/Users/jellingson/agent-swarm/backend/jobs.py` (628 lines)
- `/Users/jellingson/agent-swarm/shared/workspace_manager.py` (318 lines)
- `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py` (634 lines)
- `/Users/jellingson/agent-swarm/shared/escalation_protocol.py` (518 lines)
- `/Users/jellingson/agent-swarm/shared/execution_context.py` (124 lines)

---

*Review completed by Critic Agent*
