# Design: Hard Enforcement of COO Delegation Rules

**Date**: 2026-01-03
**Author**: System Architect
**Status**: PROPOSED

---

## Problem Statement

The COO (Supreme Orchestrator) repeatedly violates delegation rules by directly editing files instead of delegating to implementer agents. Despite clear rules in `workspace/STATE.md`:

- COO MUST NEVER write or edit code (even 1 line)
- COO MUST NEVER create or modify files (except STATE.md)
- COO should delegate ALL implementation work to specialized agents

**Root Cause**: These rules exist only as soft guidance in system prompts. There is no technical enforcement - the COO has full access to all tools including Write, Edit, and Bash.

---

## Analysis of Current Architecture

### Current Tool Access Flow

```
User Request
    |
    v
backend/main.py:websocket_chat()
    |
    v
stream_claude_response()   <- COO spawned HERE with ALL tools
    |
    v
Claude CLI with:
  - --permission-mode acceptEdits
  - NO tool restrictions
  - Full Read/Write/Edit/Bash access
```

### Where COO is Spawned

1. **Primary: `backend/main.py:2846`**
   ```python
   process = await stream_claude_response(
       prompt=user_prompt,
       system_prompt=system_prompt,  # Contains soft rules only
       ...
   )
   ```

2. **WebSocket handler: `backend/websocket/chat_handler.py:259`**
   ```python
   process = await stream_claude_response(
       prompt=user_prompt,
       system_prompt=system_prompt,  # Also soft rules only
       ...
   )
   ```

### Why Soft Rules Fail

1. **LLM Compliance is Probabilistic**: System prompts are suggestions, not constraints
2. **Convenience Temptation**: It's "faster" for COO to do small edits directly
3. **Context Pressure**: Long conversations lead to rule forgetting
4. **No Feedback Loop**: COO doesn't get immediate rejection when violating rules

---

## Proposed Solution: Multi-Layer Enforcement

I recommend a **defense-in-depth** approach with 4 layers:

### Layer 1: Tool Access Restriction (HARD ENFORCEMENT)

**Mechanism**: Use Claude CLI's `--disallowedTools` flag to prevent COO from using Write/Edit tools.

**Implementation**:

Modify `backend/main.py:stream_claude_response()`:

```python
async def stream_claude_response(
    prompt: str,
    swarm_name: str | None = None,
    workspace: Path | None = None,
    chat_id: str | None = None,
    system_prompt: str | None = None,
    disallowed_tools: list[str] | None = None,  # NEW PARAMETER
) -> asyncio.subprocess.Process:
    cmd = [
        "claude",
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "acceptEdits",
    ]

    # NEW: Add tool restrictions for COO
    if disallowed_tools:
        for tool in disallowed_tools:
            cmd.extend(["--disallowedTools", tool])

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    cmd.append(prompt)
    # ...
```

Update COO invocation in `backend/main.py:2846`:

```python
process = await stream_claude_response(
    prompt=user_prompt,
    system_prompt=system_prompt,
    swarm_name=None,
    workspace=PROJECT_ROOT,
    chat_id=session_id,
    disallowed_tools=["Write", "Edit"],  # COO CANNOT write/edit
)
```

**Pros**:
- Absolute enforcement - Claude CLI will reject tool calls
- Simple implementation
- No performance overhead

**Cons**:
- COO cannot update STATE.md (could be handled via special path)
- Requires Claude CLI to support `--disallowedTools` (needs verification)

**Note**: If `--disallowedTools` is not available in Claude CLI, we can use Layer 2 as primary enforcement.

---

### Layer 2: Pre-Execution Hook (HARD ENFORCEMENT)

**Mechanism**: Intercept tool calls before execution and reject disallowed operations.

**Implementation**:

Create new file `/shared/coo_enforcement.py`:

```python
"""COO delegation rule enforcement.

This module provides hard enforcement of COO delegation rules,
preventing the Supreme Orchestrator from directly modifying files.
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class COOEnforcementViolation(Exception):
    """Raised when COO attempts a disallowed operation."""
    pass


# Files COO is allowed to modify
COO_ALLOWED_FILES = {
    "STATE.md",
    "workspace/STATE.md",
}

# Tools COO cannot use
COO_DISALLOWED_TOOLS = {"Write", "Edit"}

# Tool parameters that indicate file modification
MODIFICATION_PARAMS = {"file_path", "path", "content", "old_string", "new_string"}


def enforce_coo_rules(
    tool_name: str,
    tool_input: dict,
    is_coo: bool = False,
) -> None:
    """Enforce COO delegation rules on tool usage.

    Args:
        tool_name: Name of the tool being invoked
        tool_input: Tool parameters
        is_coo: Whether the caller is the COO

    Raises:
        COOEnforcementViolation: If COO attempts a disallowed operation
    """
    if not is_coo:
        return  # Rules only apply to COO

    # Check for disallowed tools
    if tool_name in COO_DISALLOWED_TOOLS:
        file_path = tool_input.get("file_path", tool_input.get("path", "unknown"))

        # Allow STATE.md modifications
        if _is_allowed_file(file_path):
            logger.debug(f"COO allowed to modify STATE.md: {file_path}")
            return

        raise COOEnforcementViolation(
            f"COO DELEGATION RULE VIOLATION: Cannot use {tool_name} tool.\n"
            f"File: {file_path}\n\n"
            f"You are an ORCHESTRATOR, not a WORKER.\n"
            f"Delegate this to an implementer agent using the Task tool:\n\n"
            f"Task(subagent_type=\"implementer\", prompt=\"...\")"
        )

    # Check Bash for file modifications (echo >, sed, etc.)
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if _is_file_modifying_command(command):
            raise COOEnforcementViolation(
                f"COO DELEGATION RULE VIOLATION: Cannot run file-modifying Bash command.\n"
                f"Command: {command[:100]}...\n\n"
                f"Delegate this to an implementer agent using the Task tool."
            )


def _is_allowed_file(file_path: str) -> bool:
    """Check if file is in COO's allowed modification list."""
    if not file_path:
        return False

    path = Path(file_path)

    # Check exact matches
    if path.name in COO_ALLOWED_FILES:
        return True

    # Check full path matches
    for allowed in COO_ALLOWED_FILES:
        if str(path).endswith(allowed):
            return True

    return False


def _is_file_modifying_command(command: str) -> bool:
    """Detect if a bash command modifies files."""
    # Patterns that indicate file modification
    modification_patterns = [
        " > ",      # Redirect to file
        " >> ",     # Append to file
        "sed -i",   # In-place sed
        "echo.*>",  # Echo to file
        "cat.*>",   # Cat to file
        "tee ",     # Write to file
        "mv ",      # Move/rename
        "rm ",      # Delete
        "touch ",   # Create/update timestamp
        "mkdir ",   # Create directory
        "cp ",      # Copy (creates new file)
    ]

    command_lower = command.lower()
    for pattern in modification_patterns:
        if pattern in command_lower:
            return True

    return False
```

**Integration Point**: Modify `parse_claude_stream()` in `backend/main.py`:

```python
async def parse_claude_stream(...):
    # ... existing code ...

    # Import enforcement
    from shared.coo_enforcement import enforce_coo_rules, COOEnforcementViolation

    # In the tool execution handling section (around line 2300):
    if event_type == "tool_use" and tool_name:
        try:
            # Check if this is COO (we're always in COO context here)
            enforce_coo_rules(
                tool_name=tool_name,
                tool_input=tool_input,
                is_coo=True,
            )
        except COOEnforcementViolation as e:
            # Send violation warning to frontend
            await manager.send_event(
                websocket,
                "enforcement_violation",
                {
                    "agent": "Supreme Orchestrator",
                    "violation": str(e),
                    "tool": tool_name,
                },
            )
            # Note: We can't actually stop Claude CLI mid-execution,
            # but we can surface the warning prominently
```

**Pros**:
- Works regardless of CLI flag support
- Can provide detailed violation messages
- Can log violations for analysis

**Cons**:
- Cannot actually prevent execution after Claude starts (CLI is already running)
- Detection is heuristic for Bash commands

---

### Layer 3: System Prompt Hardening (SOFT ENFORCEMENT)

**Mechanism**: Strengthen the system prompt with more explicit, formatted rules and consequences.

**Implementation**:

Update system prompt in both locations:
- `backend/main.py:2757-2825`
- `backend/websocket/chat_handler.py:51-119`

**New System Prompt Structure**:

```python
system_prompt = f"""You are the Supreme Orchestrator (COO) - a fully autonomous AI orchestrator.

## TOOL RESTRICTIONS - HARD ENFORCED

**YOU DO NOT HAVE ACCESS TO Write OR Edit TOOLS.**

Attempting to use these tools will fail. Do not try.

Instead, delegate ALL file modifications to specialized agents:

```
Task(subagent_type="implementer", prompt="...")
Task(subagent_type="architect", prompt="...")  // for design docs
Task(subagent_type="researcher", prompt="...")  // for research docs
```

## YOUR CAPABILITIES

You CAN use:
- **Read** - Read any file to understand context
- **Glob/Grep** - Search files and code
- **Bash** - Run read-only commands (git status, ls, cat, etc.)
- **Task** - Delegate work to specialized agents

You CANNOT use:
- **Write** - BLOCKED (delegate to implementer)
- **Edit** - BLOCKED (delegate to implementer)
- **Bash with > or >>** - BLOCKED (no file modification via bash)

## SINGLE EXCEPTION: STATE.md

You MAY update STATE.md files directly via Bash:
```bash
cat >> workspace/STATE.md << 'EOF'
### Progress Entry
...
EOF
```

## DELEGATION EXAMPLES

**Implement a feature:**
```
Task(subagent_type="implementer", prompt="Read workspace/STATE.md first. Implement feature X in file Y. Update STATE.md when done.")
```

**Design a solution:**
```
Task(subagent_type="architect", prompt="Read workspace/STATE.md first. Design the solution for problem X. Create design doc at workspace/DESIGN_X.md. Update STATE.md when done.")
```

**Review code:**
```
Task(subagent_type="critic", prompt="Read workspace/STATE.md first. Review the implementation in file X for bugs and issues. Update STATE.md with findings.")
```

## SWARM WORKSPACES
{all_swarms_str}

## PROJECT ROOT
{PROJECT_ROOT}

---

Remember: You are a CONDUCTOR, not a MUSICIAN. Conduct the orchestra, don't play the instruments.
"""
```

**Key Changes**:
1. State restrictions as fact ("YOU DO NOT HAVE ACCESS")
2. Explicit "BLOCKED" labeling
3. Provide concrete alternatives
4. Single exception clearly stated
5. Examples for each delegation type
6. Memorable metaphor at end

---

### Layer 4: Detection and Warning System (MONITORING)

**Mechanism**: Real-time detection when COO attempts to edit, with prominent UI warning.

**Implementation**:

1. **Frontend WebSocket Handler** (`frontend/lib/websocket.ts`):

Add new event type:
```typescript
interface EnforcementViolationEvent {
  type: "enforcement_violation";
  agent: string;
  violation: string;
  tool: string;
}
```

2. **ActivityPanel Update** (`frontend/components/ActivityPanel.tsx`):

Add violation display:
```typescript
// In the activity list
{violations.map((v, i) => (
  <div key={i} className="bg-red-900/50 border border-red-500 rounded p-3 mb-2">
    <div className="text-red-400 font-bold flex items-center gap-2">
      <AlertTriangle className="w-4 h-4" />
      DELEGATION RULE VIOLATION
    </div>
    <div className="text-red-200 text-sm mt-1">
      {v.violation}
    </div>
    <div className="text-red-300 text-xs mt-2">
      Tool attempted: {v.tool}
    </div>
  </div>
))}
```

3. **Backend Logging**:

Add to `shared/coo_enforcement.py`:
```python
def log_violation(tool_name: str, file_path: str, details: str):
    """Log COO delegation rule violations for analysis."""
    log_path = Path("logs/coo_violations.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "file": file_path,
        "details": details,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

---

## Recommended Implementation Priority

| Layer | Enforcement | Implementation Effort | Effectiveness |
|-------|-------------|----------------------|---------------|
| **1. Tool Restriction** | HARD | Low (if CLI supports flag) | HIGH |
| **2. Pre-Execution Hook** | HARD* | Medium | MEDIUM** |
| **3. System Prompt Hardening** | SOFT | Low | MEDIUM |
| **4. Detection/Warning** | MONITORING | Medium | LOW (after-the-fact) |

*Cannot actually block mid-execution
**Detection is heuristic

### Recommended Approach

**If Claude CLI supports `--disallowedTools`:**
1. Implement Layer 1 (Tool Restriction) - PRIMARY ENFORCEMENT
2. Implement Layer 3 (System Prompt) - REINFORCE MESSAGE
3. Implement Layer 4 (Detection/Warning) - MONITORING

**If Claude CLI does NOT support `--disallowedTools`:**
1. Implement Layer 3 (System Prompt) - PRIMARY SOFT ENFORCEMENT
2. Implement Layer 2 (Pre-Execution Hook) - DETECTION + WARNING
3. Implement Layer 4 (Detection/Warning) - MONITORING

---

## Files to Modify

### New Files

| File | Purpose |
|------|---------|
| `/shared/coo_enforcement.py` | COO rule enforcement logic |

### Modified Files

| File | Changes |
|------|---------|
| `/backend/main.py` | Add disallowed_tools to stream_claude_response(), update system prompt |
| `/backend/websocket/chat_handler.py` | Add disallowed_tools, update system prompt |
| `/frontend/lib/websocket.ts` | Add enforcement_violation event type |
| `/frontend/components/ActivityPanel.tsx` | Add violation display |

---

## Implementation Plan

### Phase 1: Investigation (30 min)
1. Verify if Claude CLI supports `--disallowedTools` flag
   - Run `claude --help` and check for tool restriction flags
   - Test with: `claude -p --disallowedTools Write "Try to write a file"`

### Phase 2: Core Enforcement (1-2 hours)
1. If CLI flag exists: Add to `stream_claude_response()`
2. If not: Create `/shared/coo_enforcement.py` with detection logic
3. Update system prompts in both locations

### Phase 3: Detection Layer (1 hour)
1. Add enforcement_violation event handling
2. Update ActivityPanel for violation display
3. Add violation logging

### Phase 4: Testing (1 hour)
1. Test COO cannot use Write/Edit
2. Test COO CAN update STATE.md (exception)
3. Test violations are properly displayed
4. Test delegation via Task still works

---

## Success Criteria

1. COO attempts to use Write/Edit on non-STATE.md files result in:
   - Immediate rejection OR prominent warning
   - Clear instructions to delegate instead

2. COO can still:
   - Read files
   - Use Glob/Grep
   - Run read-only Bash commands
   - Update STATE.md
   - Delegate via Task tool

3. Violation attempts are:
   - Logged for analysis
   - Surfaced prominently in UI
   - Include instructions for correct behavior

---

## Open Questions

1. **Does Claude CLI support `--disallowedTools`?**
   - Need to verify via CLI help or documentation
   - This determines primary vs fallback approach

2. **Should COO be allowed to append to STATE.md via Bash?**
   - Current proposal: YES (single exception)
   - Alternative: Delegate even STATE.md updates

3. **How strict on Bash detection?**
   - Could miss creative file modifications
   - False positives on legitimate read commands
   - Consider whitelist vs blacklist approach

---

## Consequences

**Pros:**
- Reliable enforcement of delegation rules
- Clear error messages guide correct behavior
- Violations logged for pattern analysis
- Defense-in-depth with multiple layers

**Cons:**
- Adds complexity to the codebase
- May need CLI flag verification
- Bash detection is heuristic
- COO convenience slightly reduced

**Mitigations:**
- Clear exception for STATE.md
- Helpful error messages with examples
- Logging enables iterative improvement

---

## References

- `workspace/STATE.md` - COO Operating Rules section
- `backend/main.py:2757-2825` - Current COO system prompt
- `backend/websocket/chat_handler.py:51-119` - WebSocket COO system prompt
- `/swarms/swarm_dev/workspace/DESIGN_DELEGATION_PATTERN.md` - Delegation pattern design
